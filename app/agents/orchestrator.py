import os

from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from app.agents.causal_agent import causal_agent
from app.agents.eda_agent import eda_agent
from app.tools.bigquery import (
    SessionData,
    SessionQueryParams,
    query_channel_conversion,
    query_device_conversion,
    query_sessions,
)

_model = LitellmModel(model=os.environ.get("MODEL", "vertex_ai/gemini-2.5-flash"))

_DATE_START = os.environ.get("BQ_DATE_START", "20160801")
_DATE_END = os.environ.get("BQ_DATE_END", "20170801")


# ---------------------------------------------------------------------------
# BigQuery tools
# ---------------------------------------------------------------------------

@function_tool
def fetch_channel_data(start_date: str = _DATE_START, end_date: str = _DATE_END) -> SessionData:
    """Fetch pre-aggregated conversion and revenue metrics grouped by traffic channel.

    Use for questions about organic vs paid, channel ROI, or traffic source comparisons.

    Args:
        start_date: Start date in YYYYMMDD format, e.g. '20160801'.
        end_date: End date in YYYYMMDD format, e.g. '20170801'.

    Returns:
        SessionData with one row per channel: sessions, transactions, conversion_rate_pct, revenue.
    """
    return query_channel_conversion(SessionQueryParams(start_date=start_date, end_date=end_date))


@function_tool
def fetch_device_data(start_date: str = _DATE_START, end_date: str = _DATE_END) -> SessionData:
    """Fetch pre-aggregated conversion and revenue metrics grouped by device category.

    Use for questions about mobile vs desktop, or device-level performance comparisons.

    Args:
        start_date: Start date in YYYYMMDD format, e.g. '20160801'.
        end_date: End date in YYYYMMDD format, e.g. '20170801'.

    Returns:
        SessionData with one row per device (desktop/mobile/tablet): sessions, conversion_rate_pct, revenue.
    """
    return query_device_conversion(SessionQueryParams(start_date=start_date, end_date=end_date))


@function_tool
def fetch_sessions(
    start_date: str = _DATE_START,
    end_date: str = _DATE_END,
    limit: int = 300,
) -> SessionData:
    """Fetch raw session-level GA data. ONLY use this for time-trend or geographic questions
    that cannot be answered with fetch_channel_data or fetch_device_data.
    Always use a narrow date range (one or two months) to keep the result small.

    Args:
        start_date: Start date in YYYYMMDD format, e.g. '20161101'.
        end_date: End date in YYYYMMDD format, e.g. '20161231'.
        limit: Max rows to return, default 300. Never exceed 500.

    Returns:
        SessionData with one row per session: date, channelGrouping, device_category,
        continent, country, transactions, revenue_usd.
    """
    return query_sessions(SessionQueryParams(start_date=start_date, end_date=end_date, limit=limit))


# ---------------------------------------------------------------------------
# Sub-agents as tools
# ---------------------------------------------------------------------------

_eda_tool = eda_agent.as_tool(
    tool_name="run_eda",
    tool_description=(
        "Run exploratory data analysis on fetched session data. "
        "Pass the full data records and the user's original question. "
        "Returns segment-level patterns, top/bottom performers, anomalies, and a chart."
    ),
)

_causal_tool = causal_agent.as_tool(
    tool_name="run_causal_analysis",
    tool_description=(
        "Run causal/A/B test analysis. Pass the full data records, the EDA findings, "
        "and the user's original question. Returns hypothesis, statistical test results, "
        "power analysis, Simpson's paradox check, and a chart."
    ),
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a marketing analytics orchestrator. Users ask questions about \
Google Analytics e-commerce data from the Google Merchandise Store (Aug 2016 – Aug 2017).

Follow this pipeline for every question:

1. COLLECT — Choose the smallest dataset that can answer the question:
   - Channel/traffic questions ("organic vs paid", "which channel", "referral vs direct") \
→ fetch_channel_data ONLY. Returns 8 rows. Do not also call fetch_sessions.
   - Device questions ("mobile vs desktop", "desktop vs tablet") \
→ fetch_device_data ONLY. Returns 3 rows. Do not also call fetch_sessions.
   - Time-trend or geographic questions ONLY → fetch_sessions with a ONE-month date range \
and limit=300. Never use fetch_sessions for channel or device questions.

2. EDA — Call run_eda, passing ONLY the data.data list (not the full SessionData object) \
and the user's question.

3. CAUSAL — Call run_causal_analysis with the same data.data list, the EDA summary, \
and the user's question.

4. SYNTHESIZE — Write a final answer that:
   - Opens with a one-sentence direct answer
   - Cites key numbers from EDA (conversion rates, top/bottom segment)
   - States the statistical conclusion (p-value, effect size, significant or not)
   - Closes with a plain-language recommendation

Critical: fetch_channel_data and fetch_device_data return pre-aggregated data — \
use these whenever possible. fetch_sessions returns raw rows and must be kept small."""


orchestrator = Agent(
    name="Orchestrator",
    instructions=_SYSTEM_PROMPT,
    tools=[fetch_channel_data, fetch_device_data, fetch_sessions, _eda_tool, _causal_tool],
    model=_model,
)
