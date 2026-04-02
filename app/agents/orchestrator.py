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
from app.tools.statistics import ABTestRequest, EDARequest, run_ab_test, run_eda
from app.tools.visualization import ChartJSON, chart_ab_test, chart_eda_segments

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

@function_tool(strict_mode=False)
def visualize_segments(data: list[dict], dimension: str) -> ChartJSON:
    """Generate a side-by-side bar chart of conversion rate and revenue per segment.

    Use for overview or ranking questions where all segments should be shown at once.

    Args:
        data: Row records from a BigQuery query result.
        dimension: Column to group by, e.g. 'channel' or 'device'.

    Returns:
        ChartJSON with a Plotly figure JSON string for frontend rendering.
    """
    eda = run_eda(EDARequest(data=data, dimension=dimension))
    return chart_eda_segments(eda)


@function_tool(strict_mode=False)
def visualize_ab_test(
    data: list[dict],
    treatment: str,
    control: str,
    dimension: str,
    metric: str = "converted",
) -> ChartJSON:
    """Generate an A/B test chart with conversion rate bars and 95% confidence interval.

    Use for two-group comparison questions ('Does X outperform Y?').

    Args:
        data: Row records from a BigQuery query result.
        treatment: Treatment group value, e.g. 'Referral'.
        control: Control group value, e.g. 'Organic Search'.
        dimension: Column identifying group membership, e.g. 'channel'.
        metric: 'converted' for conversion rate or 'revenue_usd' for revenue.

    Returns:
        ChartJSON with a Plotly figure JSON string for frontend rendering.
    """
    result = run_ab_test(ABTestRequest(
        data=data, treatment=treatment, control=control,
        dimension=dimension, metric=metric,
    ))
    return chart_ab_test(result)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a marketing analytics orchestrator. Users ask questions about \
Google Analytics e-commerce data from the Google Merchandise Store (Aug 2016 – Aug 2017).

If the user asks a conversational or clarifying question (e.g. "what does p-value mean?", \
"can you explain that?"), answer directly without calling any tools.

For data analysis questions, follow this pipeline:

1. COLLECT — Choose the smallest dataset that answers the question:
   - Channel/traffic questions → fetch_channel_data ONLY (returns 8 rows).
   - Device questions → fetch_device_data ONLY (returns 3 rows).
   - Time-trend or geographic questions → fetch_sessions with a ONE-month range, limit=300.
   Never call fetch_sessions for channel or device questions.

2. EDA — Call run_eda with:
   - data: the data.data list from step 1 (a Python list of dicts — NOT a JSON string)
   - dimension: "channel" for channel data, "device_category" for device data

3. CAUSAL — Call run_causal_analysis with:
   - data: the same data.data list (a Python list of dicts — NOT a JSON string)
   - The EDA summary and user question

4. VISUALIZE — Call exactly one chart tool after step 3:
   - Two-group comparison ("does X beat Y", "X vs Y", "better than") \
→ visualize_ab_test(data=<list>, treatment=<str>, control=<str>, dimension=<str>)
   - Overview or ranking ("compare all", "how do channels compare", "which is best") \
→ visualize_segments(data=<list>, dimension=<str>)
   CRITICAL: The data argument must be a Python list of dicts, never a JSON string. \
Pass the same list you used in steps 2 and 3.

5. SYNTHESIZE — Write a final answer that:
   - Opens with a one-sentence direct answer
   - Cites key numbers (conversion rates, top/bottom segment)
   - States the statistical conclusion (p-value, effect size, significant or not)
   - Closes with a plain-language recommendation"""


orchestrator = Agent(
    name="Orchestrator",
    instructions=_SYSTEM_PROMPT,
    tools=[fetch_channel_data, fetch_device_data, fetch_sessions, _eda_tool, _causal_tool, visualize_segments, visualize_ab_test],
    model=_model,
)
