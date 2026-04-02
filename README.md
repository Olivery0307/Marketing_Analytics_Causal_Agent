# Marketing Analytics Causal Agent

An AI agent that answers marketing questions using real Google Analytics e-commerce data. Ask natural language questions and get statistically rigorous answers — not just summaries.

**Live demo:** https://marketing-analytics-causal-agent-32542646336.us-central1.run.app/

## What it does

- **Causal analysis** — runs A/B tests (chi-squared, Mann-Whitney U) to determine whether differences between segments are statistically significant, with effect sizes and confidence intervals
- **EDA** — descriptive stats and segment comparison across channels, devices, and geographies
- **Multi-agent pipeline** — orchestrator routes your question to specialized EDA and causal sub-agents, then synthesizes a final answer
- **Interactive charts** — Plotly visualizations (conversion rate bars, CI plots) rendered in the browser
- **Chat history** — session-based conversation memory; follow-up questions retain prior context
- **Structured output** — every result is a typed Pydantic schema: `EDAResult`, `ABTestResult`, `CausalResult`

**Data source:** [Google Analytics Sample](https://console.cloud.google.com/marketplace/product/obfuscated-ga360-data/obfuscated-ga360-data) — 12 months of real Google Merchandise Store sessions (Aug 2016 – Aug 2017), queried live from BigQuery.

**Example questions:**
- "Does referral traffic convert significantly better than organic search?"
- "How do all traffic channels compare in conversion rate and revenue?"
- "Do desktop users convert at a higher rate than mobile users?"
- "Is the difference between social and organic conversion significant?"

## Stack

| Layer | Technology |
|---|---|
| Agent framework | OpenAI Agents SDK + LiteLLM |
| LLM | Vertex AI — Gemini 2.5 Flash (orchestrator/causal), Gemini 2.0 Flash Lite (EDA) |
| Data | BigQuery public dataset — `bigquery-public-data.google_analytics_sample` |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS + Plotly.js |
| Deployment | Google Cloud Run |

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), a Google Cloud project with BigQuery API enabled.

```bash
# 1. Clone and install
git clone <repo>
cd data_analyst_agent
uv sync

# 2. Authenticate with Google Cloud
gcloud auth application-default login

# 3. Configure environment
cp .env.example .env
# Edit .env — set GCP_PROJECT_ID to your project

# 4. Run
uv run python -m app.main serve        # Start server at http://localhost:8000
uv run python -m app.main ask "Does mobile convert as well as desktop?"
```

**Deploy to Cloud Run:**
```bash
gcloud run deploy --source . --memory 1Gi --max-instances 1
```
Cloud Run uses the attached service account automatically — no credential file needed in the container. `--max-instances 1` ensures session continuity across requests.

## Repo structure

```
app/
├── main.py                  # FastAPI server + Typer CLI entry point
├── agents/
│   ├── orchestrator.py      # Top-level agent — routes question, synthesizes answer
│   ├── eda_agent.py         # EDA sub-agent — segment stats, top/bottom performers
│   └── causal_agent.py      # Causal sub-agent — A/B tests, effect size, CI
├── tools/
│   ├── bigquery.py          # BigQuery queries → DataFrames (sessions, channel, device)
│   ├── statistics.py        # Statistical tests — chi-squared, Mann-Whitney, Cohen's h
│   └── visualization.py     # Plotly charts → JSON for frontend rendering
└── schemas/
    └── models.py            # Pydantic models: EDAResult, ABTestResult, CausalResult

static/
└── index.html               # Chat UI — pipeline badges, Plotly charts, Learn More modals

tests/
├── test_tools.py            # Unit tests — BigQuery, EDA, A/B test, visualization (25 tests)
└── test_agents.py           # Integration tests — live LLM calls (6 tests)
```
