import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

from app.schemas.models import ABTestResult, EDAResult


_PALETTE = px.colors.qualitative.Plotly


class ChartJSON(BaseModel):
    title: str
    plotly_json: str = Field(description="Plotly figure serialized as JSON, pass to Plotly.react() on the frontend")


def chart_eda_segments(eda: EDAResult) -> ChartJSON:
    """Build a horizontal bar chart of conversion rate by segment.

    Args:
        eda: EDA result with per-segment stats.

    Returns:
        ChartJSON with Plotly figure JSON.
    """
    segments = sorted(eda.segments, key=lambda s: s.conversion_rate_pct)
    labels = [s.segment for s in segments]
    conv_rates = [s.conversion_rate_pct for s in segments]
    rev_per_session = [s.revenue_per_session for s in segments]
    sessions = [s.sessions for s in segments]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Conversion Rate (%)", "Revenue per Session (USD)"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            y=labels,
            x=conv_rates,
            orientation="h",
            marker_color=_PALETTE[0],
            text=[f"{v:.2f}%" for v in conv_rates],
            textposition="outside",
            customdata=sessions,
            hovertemplate="<b>%{y}</b><br>Conversion: %{x:.2f}%<br>Sessions: %{customdata:,}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            y=labels,
            x=rev_per_session,
            orientation="h",
            marker_color=_PALETTE[1],
            text=[f"${v:.2f}" for v in rev_per_session],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Rev/Session: $%{x:.2f}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Channel Performance by {eda.dimension.replace('_', ' ').title()}",
        height=420,
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor="#1c1c20",
        paper_bgcolor="#1c1c20",
        font=dict(family="system-ui, sans-serif", size=12, color="#a1a1aa"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2e2e33", color="#a1a1aa")
    fig.update_yaxes(showgrid=False, color="#a1a1aa")

    title = f"EDA: {eda.dimension} — {eda.summary[:80]}..."
    return ChartJSON(title=title, plotly_json=fig.to_json())


def chart_ab_test(ab: ABTestResult) -> ChartJSON:
    """Build an A/B test results chart with conversion rates and confidence interval.

    Args:
        ab: A/B test result with rates, CI, p-value, and effect size.

    Returns:
        ChartJSON with Plotly figure JSON.
    """
    groups = [ab.treatment, ab.control]
    rates = [ab.treatment_rate, ab.control_rate]
    colors = [_PALETTE[2] if ab.significant else _PALETTE[4], _PALETTE[7]]

    diff = ab.treatment_rate - ab.control_rate
    ci_low, ci_high = ab.confidence_interval_95

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Conversion Rate (%)", "Difference with 95% CI"),
        horizontal_spacing=0.15,
    )

    # Left: grouped bar chart
    fig.add_trace(
        go.Bar(
            x=groups,
            y=rates,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in rates],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Rate: %{y:.4f}%<extra></extra>",
        ),
        row=1, col=1,
    )

    # Right: point estimate + CI
    sig_color = "#2ecc71" if ab.significant and ab.relative_lift_pct > 0 else (
        "#e74c3c" if ab.significant else "#95a5a6"
    )
    fig.add_trace(
        go.Scatter(
            x=[diff],
            y=["difference"],
            mode="markers",
            marker=dict(size=14, color=sig_color),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[ci_high - diff],
                arrayminus=[diff - ci_low],
                color=sig_color,
                thickness=2,
                width=8,
            ),
            hovertemplate=f"Diff: {diff:+.4f}%<br>95% CI: [{ci_low:.4f}, {ci_high:.4f}]<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#bdc3c7", row=1, col=2)

    sig_icon = "✓" if ab.significant else "✗"
    sig_label = f"{sig_icon} {'Significant' if ab.significant else 'Not significant'}  ·  p = {ab.p_value:.4f}"
    lift_label = f"Lift {ab.relative_lift_pct:+.1f}%  ·  Effect: {ab.effect_size_label}"

    fig.update_layout(
        title=dict(
            text=f"<b>{ab.treatment} vs {ab.control}</b>",
            font=dict(size=14, color="#f4f4f5"),
            x=0.0,
            xanchor="left",
        ),
        annotations=[
            # keep subplot titles (index 0 and 1 are added by make_subplots)
            *fig.layout.annotations,
            dict(
                text=f"{sig_label}    {lift_label}",
                xref="paper", yref="paper",
                x=0.0, y=1.12,
                xanchor="left", yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#71717a"),
            ),
        ],
        height=440,
        showlegend=False,
        margin=dict(l=20, r=20, t=110, b=20),
        plot_bgcolor="#1c1c20",
        paper_bgcolor="#1c1c20",
        font=dict(family="system-ui, sans-serif", size=12, color="#a1a1aa"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2e2e33", color="#a1a1aa")
    fig.update_yaxes(showgrid=False, color="#a1a1aa")

    return ChartJSON(
        title=f"A/B Test: {ab.treatment} vs {ab.control}",
        plotly_json=fig.to_json(),
    )
