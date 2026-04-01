import asyncio
import json
import os

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents import Runner

from app.agents.orchestrator import orchestrator

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    question: str


class DatasetInfo(BaseModel):
    name: str
    start_date: str
    end_date: str
    rows: str
    description: str


class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    charts: list[dict] = []


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


def _extract_charts(result) -> list[dict]:
    """Pull any ChartJSON objects out of tool call outputs in the run trace."""
    charts = []
    for item in result.new_items:
        raw = getattr(item, "output", None)
        if raw is None:
            continue
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict) and "plotly_json" in data:
                charts.append(data)
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return charts


async def _run(question: str) -> AnalyzeResponse:
    result = await Runner.run(orchestrator, input=question)
    return AnalyzeResponse(
        question=question,
        answer=str(result.final_output),
        charts=_extract_charts(result),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run a marketing question through the full Collect → EDA → Hypothesize pipeline."""
    return await _run(request.question)


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
) -> None:
    """Run a question through the agent pipeline and print the answer."""
    response = asyncio.run(_run(question))
    typer.echo("\n" + "=" * 60)
    typer.echo(f"Q: {response.question}")
    typer.echo("=" * 60)
    typer.echo(response.answer)
    if response.charts:
        typer.echo(f"\n[{len(response.charts)} chart(s) generated — open the web UI to view]")


if __name__ == "__main__":
    cli()
