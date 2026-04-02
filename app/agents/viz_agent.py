import os

from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from app.tools.statistics import ABTestRequest, EDARequest, run_ab_test, run_eda
from app.tools.visualization import ChartJSON, chart_ab_test, chart_eda_segments

_model = LitellmModel(model=os.environ.get("MODEL", "vertex_ai/gemini-2.5-flash"))


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@function_tool(strict_mode=False)
def make_segment_chart(data: list[dict], dimension: str) -> ChartJSON:
    """Generate a segment comparison chart (conversion rate + revenue per session by group).

    Use for overview questions, ranking questions, or when the user wants to compare
    all segments at once — not a specific treatment vs control comparison.

    Args:
        data: Row records from BigQuery result.
        dimension: Column to group by, e.g. 'channel' or 'device'.

    Returns:
        ChartJSON with a Plotly figure JSON string.
    """
    eda = run_eda(EDARequest(data=data, dimension=dimension))
    return chart_eda_segments(eda)


@function_tool(strict_mode=False)
def make_ab_chart(
    data: list[dict],
    treatment: str,
    control: str,
    dimension: str,
    metric: str = "converted",
) -> ChartJSON:
    """Generate an A/B test results chart with conversion rates and 95% confidence interval.

    Use when the question compares exactly two groups (e.g. 'Does X outperform Y?').

    Args:
        data: Row records from BigQuery result.
        treatment: Treatment group value, e.g. 'Organic Search'.
        control: Control group value, e.g. 'Paid Search'.
        dimension: Column identifying group membership.
        metric: 'converted' for conversion rate or 'revenue_usd' for revenue.

    Returns:
        ChartJSON with a Plotly figure JSON string.
    """
    result = run_ab_test(ABTestRequest(
        data=data, treatment=treatment, control=control,
        dimension=dimension, metric=metric,
    ))
    return chart_ab_test(result)


@function_tool
def skip_chart(reason: str) -> ChartJSON:
    """Return an empty chart signal when visualization would not add value.

    Use when the question is purely factual, the data has only one segment,
    or the answer is already fully conveyed in text.

    Args:
        reason: Brief explanation of why no chart is needed.

    Returns:
        ChartJSON with an empty plotly_json to signal no chart.
    """
    return ChartJSON(title=f"No chart: {reason}", plotly_json="{}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a data visualization specialist. Given a user question and analysis results, \
decide whether a chart would add value and generate the most appropriate one.

Decision rules:
- TWO-GROUP COMPARISON ("Does X outperform Y?", "Is X better than Y?") → use make_ab_chart with the \
  treatment and control groups identified from the question or eda_summary
- RANKING OR OVERVIEW ("Which channel is best?", "How do all channels compare?", "Show me performance") \
  → use make_segment_chart
- SINGLE-VALUE or FACTUAL ("What is the conversion rate of X?") → use skip_chart

Always call exactly one tool. Do not call multiple chart tools."""


viz_agent = Agent(
    name="Viz Agent",
    instructions=_SYSTEM_PROMPT,
    tools=[make_segment_chart, make_ab_chart, skip_chart],
    model=_model,
)
