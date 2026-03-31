# Claude.md

## Project Overview

Marketing analytics causal agent — users ask questions like "Does organic traffic convert better than paid?", and the agent queries Google Analytics data from BigQuery, runs EDA, performs statistical tests (A/B, t-test, chi-squared), and returns grounded insights with visualizations.

**Stack:** OpenAI Agents SDK + LiteLLM + FastAPI → Google Cloud Run  
**Data:** `bigquery-public-data.google_analytics_sample.ga_sessions_*` (Aug 2016–Aug 2017)

## Running & Testing

```bash
uv run python app/main.py serve   # Start server
uv run python app/main.py ask "Does mobile convert as well as desktop?"
uv run ruff check .               # Lint
```

## Architecture

```
app/
├── main.py                  # FastAPI + Typer entry point
├── agents/
│   ├── orchestrator.py      # Routes question → sub-agents, synthesizes final answer
│   ├── eda_agent.py         # EDA sub-agent (descriptive stats, segment comparison)
│   └── causal_agent.py      # Causal sub-agent (t-test, chi-sq, effect size)
├── tools/
│   ├── bigquery.py          # BigQuery queries → pandas DataFrames
│   ├── statistics.py        # Statistical tests (scipy)
│   └── visualization.py     # matplotlib → base64 PNG
└── schemas/
    └── models.py            # Pydantic structured output models
```

Multi-agent pattern: **orchestrator uses sub-agents as tools** (not handoffs). See `.claude/skills/openai-agents-sdk-multi-agent/SKILL.md`.

## Environment

Config in `.env` — see `.env.example` for required vars.  
Auth: `gcloud auth application-default login` (local). Cloud Run uses attached service account automatically.

## Code Style

- Type-hint everything. Minimal comments. Simple docstrings.
- All structured data in/out of agents: Pydantic `BaseModel` (never raw dicts).
- All tools: `@function_tool` with Google-style docstrings.
- Prices from GA data are in micros — divide by 1,000,000 for USD.

## Key Skills (reference when needed)

| Topic | Path |
|-------|------|
| Bootstrap boilerplate | `.claude/skills/bootstrap/SKILL.md` |
| Tool writing | `.claude/skills/openai-agents-sdk-tools/SKILL.md` |
| Multi-agent / sub-agents | `.claude/skills/openai-agents-sdk-multi-agent/SKILL.md` |
| Sessions | `.claude/skills/openai-agents-sdk-sessions/SKILL.md` |
| RunContext / state | `.claude/skills/openai-agents-sdk-state/SKILL.md` |
| Guardrails | `.claude/skills/openai-agents-sdk-guardrails/SKILL.md` |
