# Build Plan — Marketing Analytics Causal Agent

## Overview
An AI agent that performs causal analysis on Google Analytics e-commerce data (Google Merchandise Store).
Users ask marketing questions, and the agent queries BigQuery, runs EDA, performs statistical tests, and generates grounded insights.

**Data Source**: `bigquery-public-data.google_analytics_sample.ga_sessions_*`
(12 months, Aug 2016 – Aug 2017, real e-commerce data)

**Deployment**: FastAPI on Google Cloud Run

---

## Phase 0: Environment Setup
- [x] Install packages via `uv add`
- [x] Add `google-cloud-bigquery` and `scipy` to dependencies
- [x] Scaffold the `app/` directory structure:
  ```
  app/
  ├── main.py              # FastAPI app entry point
  ├── agents/
  │   ├── orchestrator.py   # Routes user question to sub-agents
  │   ├── eda_agent.py      # Exploratory data analysis
  │   └── causal_agent.py   # Statistical testing & causal inference
  ├── tools/
  │   ├── bigquery.py       # Query BigQuery for GA data
  │   ├── statistics.py     # t-test, chi-squared, effect size
  │   └── visualization.py  # Charts (matplotlib → base64 PNG)
  └── schemas/
      └── models.py         # Pydantic structured output schemas
  ```
- [x] Set up BigQuery access (service account or ADC)

---

## Phase 1: Data Collection (Step 1 — 5 pts)
- [x] Build `query_bigquery` tool:
  - Agent generates SQL based on user question
  - Queries `ga_sessions_*` tables with date range filters
  - Flattens nested fields (hits, products, trafficSource) into analysis-ready DataFrames
- [ ] Key fields to extract:
  - **Traffic**: `channelGrouping`, `trafficSource.source`, `trafficSource.medium`, `trafficSource.campaign`
  - **Behavior**: `totals.pageviews`, `totals.timeOnSite`, `totals.hits`
  - **Conversion**: `totals.transactions`, `totals.totalTransactionRevenue`
  - **Segments**: `device.deviceCategory`, `geoNetwork.country`, `geoNetwork.continent`

---

## Phase 2: EDA Agent (Step 2 — 5 pts)
- [ ] Build EDA sub-agent with tools:
  - `descriptive_stats` — summary stats, distributions, missing data
  - `segment_comparison` — group-by analysis (e.g., mobile vs desktop conversion rates)
  - `trend_analysis` — time-series patterns across date range
- [ ] Agent receives raw data, identifies patterns, surfaces anomalies

---

## Phase 3: Causal / Hypothesis Agent (Step 3 — 5 pts)
- [ ] Build causal analysis sub-agent with tools:
  - `ab_test` — two-sample t-test / chi-squared for conversion comparison
  - `effect_size` — Cohen's d, relative lift, confidence intervals
  - `power_analysis` — is the sample large enough for reliable conclusions?
  - `segment_deep_dive` — check if effect holds across subgroups (Simpson's paradox check)
- [ ] Agent takes EDA findings → formulates hypothesis → runs statistical test → reports with confidence level

---

## Phase 4: Grab-Bag (pick 2+)
- [ ] **Structured output** — Pydantic schemas for:
  - `EDAResult(summary, segments, anomalies)`
  - `CausalResult(hypothesis, test_type, p_value, effect_size, confidence_interval, conclusion)`
- [ ] **Data Visualization** — matplotlib charts:
  - Conversion rate bar charts by segment
  - Before/after comparison plots
  - Confidence interval plots

---

## Phase 5: Frontend & Deployment
- [ ] FastAPI endpoints:
  - `POST /analyze` — accepts user question, returns full analysis
  - `GET /datasets` — list available date ranges
- [ ] Simple frontend (HTML/JS) with:
  - Text input for questions
  - Results panel showing EDA → Hypothesis → Conclusion pipeline
  - Embedded charts
- [ ] Deploy to Cloud Run via `cloudbuild.yaml`

---

## Example User Queries
- "Does organic traffic convert better than paid search?"
- "Is there a significant difference in revenue between mobile and desktop users?"
- "Did the holiday season (Nov-Dec) cause higher conversion rates?"
- "Which traffic channel has the highest ROI?"
- "Is there a Simpson's paradox in our conversion data across regions?"

---

## New Dependencies Needed
```bash
uv add google-cloud-bigquery scipy matplotlib
```
