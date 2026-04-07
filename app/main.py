import asyncio
import json
import logging
import os
import time
import uuid

import typer
import uvicorn
from agents import Runner, MaxTurnsExceeded
from agents.lifecycle import RunHooksBase
from agents.memory.session import SessionABC
from agents.items import TResponseInputItem
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents.orchestrator import orchestrator

load_dotenv()

logging.basicConfig(format="%(message)s", level=logging.INFO)

# ---------------------------------------------------------------------------
# In-memory session store (per-process; survives within a Cloud Run instance)
# ---------------------------------------------------------------------------

_sessions_store: dict[str, list[TResponseInputItem]] = {}


class InMemorySession(SessionABC):
    """Ephemeral chat history stored in a Python dict, keyed by session_id."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        if session_id not in _sessions_store:
            _sessions_store[session_id] = []

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        items = _sessions_store[self.session_id]
        return items[-limit:] if limit is not None else list(items)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        _sessions_store[self.session_id].extend(items)

    async def pop_item(self) -> TResponseInputItem | None:
        if _sessions_store[self.session_id]:
            return _sessions_store[self.session_id].pop()
        return None

    async def clear_session(self) -> None:
        _sessions_store[self.session_id] = []


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    question: str
    session_id: str | None = None  # omit to start a new session


class DatasetInfo(BaseModel):
    name: str
    start_date: str
    end_date: str
    rows: str
    description: str


class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    charts: list[dict] = []
    steps: list[str] = []


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Marketing Analytics Causal Agent",
    description="Ask marketing questions — get statistically grounded answers from GA360 data.",
    version="0.1.0",
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    index = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "ok", "message": "Marketing Analytics Causal Agent is running."}


@app.get("/datasets", response_model=list[DatasetInfo])
def list_datasets() -> list[DatasetInfo]:
    """List available datasets and their date ranges."""
    return [
        DatasetInfo(
            name="GA360 Sample — Google Merchandise Store",
            start_date="20160801",
            end_date="20170801",
            rows="~903,654 sessions",
            description=(
                "12 months of real Google Merchandise Store sessions from BigQuery public datasets. "
                "Includes traffic source, device, geography, pageviews, and transaction data."
            ),
        )
    ]


def _extract_steps(result) -> list[str]:
    """Derive which pipeline steps ran based on tool names called."""
    # run_causal_analysis now covers both EDA and Hypothesize in one step
    _TOOL_STEP: dict[str, list[str]] = {
        "fetch_channel_data": ["Collect"],
        "fetch_device_data": ["Collect"],
        "fetch_sessions": ["Collect"],
        "run_causal_analysis": ["EDA", "Hypothesize"],
        "visualize_segments": ["Visualize"],
        "visualize_ab_test": ["Visualize"],
    }
    seen: list[str] = []
    for item in result.new_items:
        raw = getattr(item, "raw_item", None)
        name = getattr(raw, "name", None)
        if name and name in _TOOL_STEP:
            for label in _TOOL_STEP[name]:
                if label not in seen:
                    seen.append(label)
    return seen


def _extract_charts(result) -> list[dict]:
    """Pull ChartJSON objects out of tool call outputs in the run trace."""
    from app.tools.visualization import ChartJSON
    charts = []
    for item in result.new_items:
        raw = getattr(item, "output", None)
        if raw is None:
            continue
        try:
            # Direct ChartJSON Pydantic object (from @function_tool on orchestrator)
            if isinstance(raw, ChartJSON):
                if raw.plotly_json and raw.plotly_json != "{}":
                    charts.append(raw.model_dump())
                continue
            # JSON string
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict) and "plotly_json" in data:
                pj = data.get("plotly_json", "{}")
                if pj and pj != "{}":
                    charts.append(data)
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return charts


_AGENT_TIMEOUT_SECS = 240  # 4 min — Cloud Run timeout is 300s
_MAX_TURNS = 20
_active_tasks: dict[str, asyncio.Task] = {}  # session_id → running task


class _TimingHooks(RunHooksBase):
    """Logs tool start/end with elapsed times to stdout."""

    def __init__(self) -> None:
        self._run_start = time.perf_counter()
        self._tool_starts: dict[str, float] = {}

    async def on_tool_start(self, context, agent, tool) -> None:
        self._tool_starts[tool.name] = time.perf_counter()
        elapsed = time.perf_counter() - self._run_start
        logging.info("[+%.1fs] ▶ %s.%s started", elapsed, agent.name, tool.name)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        start = self._tool_starts.pop(tool.name, time.perf_counter())
        tool_elapsed = time.perf_counter() - start
        total_elapsed = time.perf_counter() - self._run_start
        logging.info("[+%.1fs] ✓ %s.%s done in %.1fs", total_elapsed, agent.name, tool.name, tool_elapsed)

    async def on_agent_start(self, context, agent) -> None:
        elapsed = time.perf_counter() - self._run_start
        logging.info("[+%.1fs] ► Agent '%s' started", elapsed, agent.name)

    async def on_agent_end(self, context, agent, output) -> None:
        elapsed = time.perf_counter() - self._run_start
        logging.info("[+%.1fs] ■ Agent '%s' finished", elapsed, agent.name)


async def _run(question: str, session_id: str) -> AnalyzeResponse:
    session = InMemorySession(session_id)
    hooks = _TimingHooks()
    t0 = time.perf_counter()
    logging.info("\n─── Question: %s", question)

    async def _agent_task():
        return await asyncio.wait_for(
            Runner.run(orchestrator, input=question, session=session, max_turns=_MAX_TURNS, hooks=hooks),
            timeout=_AGENT_TIMEOUT_SECS,
        )

    task = asyncio.create_task(_agent_task())
    _active_tasks[session_id] = task
    try:
        result = await task
    except asyncio.CancelledError:
        logging.info("─── Cancelled by user after %.1fs\n", time.perf_counter() - t0)
        raise ValueError("Analysis was stopped by the user.")
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError("Agent timed out after 4 minutes.")
    except MaxTurnsExceeded:
        raise ValueError(f"Agent exceeded {_MAX_TURNS} turns without finishing.")
    finally:
        _active_tasks.pop(session_id, None)
        logging.info("─── Total: %.1fs\n", time.perf_counter() - t0)

    answer = str(result.final_output) if result.final_output else ""
    if not answer or answer == "None":
        # Agent called tools but produced no synthesis — extract from last tool output
        answer = "Analysis complete. See the pipeline steps and charts above for results."

    return AnalyzeResponse(
        question=question,
        answer=answer,
        session_id=session_id,
        charts=_extract_charts(result),
        steps=_extract_steps(result),
    )


@app.post("/cancel/{session_id}")
async def cancel_analysis(session_id: str):
    """Cancel a running analysis for the given session."""
    task = _active_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        logging.info("─── Cancel requested for session %s", session_id)
        return {"cancelled": True}
    return {"cancelled": False}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Run a marketing question through the full Collect → EDA → Hypothesize pipeline.

    Pass session_id from a previous response to continue a conversation.
    Omit session_id to start fresh.
    """
    session_id = request.session_id or str(uuid.uuid4())
    try:
        response = await _run(request.question, session_id)
        return response
    except asyncio.TimeoutError as e:
        return JSONResponse(status_code=504, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ---------------------------------------------------------------------------
# Typer CLI
# ---------------------------------------------------------------------------

cli = typer.Typer(help="Marketing Analytics Causal Agent CLI")


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)"),
) -> None:
    """Start the FastAPI server."""
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


@cli.command()
def ask(
    question: str = typer.Argument(..., help="Marketing question to analyze"),
    session_id: str = typer.Option(None, "--session", "-s", help="Session ID to continue a conversation"),
) -> None:
    """Run a question through the agent pipeline and print the answer."""
    sid = session_id or str(uuid.uuid4())
    response = asyncio.run(_run(question, sid))
    typer.echo("\n" + "=" * 60)
    typer.echo(f"Q: {response.question}")
    typer.echo(f"Session: {response.session_id}")
    typer.echo("=" * 60)
    typer.echo(response.answer)
    if response.charts:
        typer.echo(f"\n[{len(response.charts)} chart(s) generated — open the web UI to view]")
    typer.echo(f"\nTo continue this conversation: --session {response.session_id}")


if __name__ == "__main__":
    cli()
