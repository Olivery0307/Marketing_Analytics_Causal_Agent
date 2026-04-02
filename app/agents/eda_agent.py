import os

from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from app.schemas.models import EDAResult
from app.tools.statistics import EDARequest, run_eda
from app.tools.visualization import ChartJSON, chart_eda_segments

_model = LitellmModel(model=os.environ.get("EDA_MODEL", "vertex_ai/gemini-2.0-flash-lite"))


@function_tool(strict_mode=False)
def descriptive_stats(data: list[dict], dimension: str) -> EDAResult:
    """Compute per-segment descriptive statistics for an overview of the data.

    Args:
        data: Row records from a BigQuery query result.
        dimension: Column to group by, e.g. 'channelGrouping' or 'device_category'.

    Returns:
        EDAResult with sessions, transactions, conversion rate, and revenue per segment.
    """
    return run_eda(EDARequest(data=data, dimension=dimension))


@function_tool(strict_mode=False)
def segment_comparison(data: list[dict], dimension: str) -> EDAResult:
    """Compare conversion rates and revenue across segments to identify top and bottom performers.

    Args:
        data: Row records from a BigQuery query result.
        dimension: Column to segment by, e.g. 'channelGrouping', 'device_category', 'continent'.

    Returns:
        EDAResult with ranked segments, top/bottom performers, and a summary.
    """
    return run_eda(EDARequest(data=data, dimension=dimension))


@function_tool(strict_mode=False)
def trend_analysis(data: list[dict]) -> EDAResult:
    """Analyze conversion and revenue trends over time by grouping sessions by date.

    Args:
        data: Row records from a BigQuery query result. Must include a 'date' column.

    Returns:
        EDAResult with date-level conversion rates showing temporal patterns.
    """
    return run_eda(EDARequest(data=data, dimension="date"))


@function_tool(strict_mode=False)
def visualize_segments(data: list[dict], dimension: str) -> ChartJSON:
    """Generate a Plotly chart of conversion rate and revenue per session by segment.

    Args:
        data: Row records from a BigQuery query result.
        dimension: Column to segment by, matching what was used in the EDA analysis.

    Returns:
        ChartJSON with a Plotly figure JSON string for frontend rendering.
    """
    eda = run_eda(EDARequest(data=data, dimension=dimension))
    return chart_eda_segments(eda)


_SYSTEM_PROMPT = """You are an exploratory data analyst specializing in web analytics.

You receive Google Analytics session data and a marketing question. Surface specific patterns, \
anomalies, and segment differences that directly address the question — not a generic summary.

Choose the right tool based on the question type:
- Time-based questions ("Did holiday season affect conversions?") → use trend_analysis
- Channel, device, or geographic questions ("Does organic convert better than paid?") → use segment_comparison
- General distribution or overview questions → use descriptive_stats

After your analysis, always call visualize_segments to generate a chart.

Your final response must cite actual numbers: name the top/bottom segments, report specific \
conversion rates, and flag any anomalies. End with a clear EDA finding that can be tested statistically."""


eda_agent = Agent(
    name="EDA Agent",
    instructions=_SYSTEM_PROMPT,
    tools=[descriptive_stats, segment_comparison, trend_analysis, visualize_segments],
    model=_model,
)
