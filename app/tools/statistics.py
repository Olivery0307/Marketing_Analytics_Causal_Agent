import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field

from app.schemas.models import ABTestResult, EDAResult, SegmentStats


class EDARequest(BaseModel):
    data: list[dict] = Field(description="Row records from BigQuery query result")
    dimension: str = Field(description="Column to group by, e.g. 'channelGrouping' or 'device_category'")


class ABTestRequest(BaseModel):
    data: list[dict] = Field(description="Row records from BigQuery query result")
    treatment: str = Field(description="The treatment group value, e.g. 'Organic Search'")
    control: str = Field(description="The control group value, e.g. 'Paid Search'")
    dimension: str = Field(description="Column that identifies group membership")
    metric: str = Field(default="converted", description="Metric to test: 'converted' (binary) or 'revenue_usd'")


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns and normalize pre-aggregated vs raw column names."""
    # Alias pre-aggregated column names to standard names
    if "total_transactions" in df.columns and "transactions" not in df.columns:
        df["transactions"] = df["total_transactions"]
    if "total_revenue_usd" in df.columns and "revenue_usd" not in df.columns:
        df["revenue_usd"] = df["total_revenue_usd"]
    for col in ("transactions", "revenue_usd", "sessions", "visits"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "sessions" not in df.columns:
        df["sessions"] = df["visits"] if "visits" in df.columns else 1
    return df


def _effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def run_eda(request: EDARequest) -> EDAResult:
    """Compute descriptive stats grouped by a dimension column.

    Args:
        request: Contains row data and the dimension to group by.

    Returns:
        EDAResult with per-segment stats, top/bottom performers, and a summary.
    """
    df = _normalize_df(pd.DataFrame(request.data))

    dim = request.dimension
    if dim not in df.columns:
        raise ValueError(f"Column '{dim}' not found. Available: {list(df.columns)}")

    grouped = (
        df.groupby(dim)
        .agg(
            sessions=("sessions", "sum"),
            transactions=("transactions", "sum"),
            total_revenue_usd=("revenue_usd", "sum"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["sessions"] > 0]
    grouped["conversion_rate_pct"] = (grouped["transactions"] / grouped["sessions"] * 100).round(4)
    grouped["revenue_per_session"] = (grouped["total_revenue_usd"] / grouped["sessions"]).round(4)
    grouped = grouped.sort_values("conversion_rate_pct", ascending=False)

    segments = [
        SegmentStats(
            segment=str(row[dim]),
            sessions=int(row["sessions"]),
            transactions=int(row["transactions"]),
            conversion_rate_pct=float(row["conversion_rate_pct"]),
            total_revenue_usd=float(round(row["total_revenue_usd"], 2)),
            revenue_per_session=float(row["revenue_per_session"]),
        )
        for _, row in grouped.iterrows()
    ]

    top = segments[0].segment
    bottom = segments[-1].segment
    summary = (
        f"Across {len(segments)} segments of '{dim}', '{top}' has the highest conversion rate "
        f"({segments[0].conversion_rate_pct:.2f}%) and '{bottom}' the lowest "
        f"({segments[-1].conversion_rate_pct:.2f}%)."
    )

    return EDAResult(
        dimension=dim,
        segments=segments,
        top_segment=top,
        bottom_segment=bottom,
        summary=summary,
    )


def run_ab_test(request: ABTestRequest) -> ABTestResult:
    """Run a two-sample statistical test between treatment and control groups.

    Uses chi-squared test for binary conversion metric and Mann-Whitney U
    for continuous revenue metric. Also computes Cohen's d effect size and
    95% confidence interval for the difference in rates.

    Args:
        request: Specifies treatment/control groups, dimension column, and metric.

    Returns:
        ABTestResult with p-value, effect size, confidence interval, and conclusion.
    """
    df = _normalize_df(pd.DataFrame(request.data))

    dim = request.dimension
    treat_df = df[df[dim] == request.treatment]
    ctrl_df = df[df[dim] == request.control]

    if treat_df.empty or ctrl_df.empty:
        raise ValueError(
            f"One or both groups not found in '{dim}'. "
            f"Available values: {df[dim].unique().tolist()}"
        )

    if request.metric == "revenue_usd":
        treat_vals = treat_df["revenue_usd"].values
        ctrl_vals = ctrl_df["revenue_usd"].values
        stat, p_value = stats.mannwhitneyu(treat_vals, ctrl_vals, alternative="two-sided")
        test_type = "Mann-Whitney U"
        treat_rate = float(np.mean(treat_vals))
        ctrl_rate = float(np.mean(ctrl_vals))
        pooled_std = float(np.std(np.concatenate([treat_vals, ctrl_vals])))
        cohens_d = (treat_rate - ctrl_rate) / pooled_std if pooled_std > 0 else 0.0
        se = float(np.sqrt(np.var(treat_vals) / len(treat_vals) + np.var(ctrl_vals) / len(ctrl_vals)))
        diff = treat_rate - ctrl_rate
        ci = (round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4))
    else:
        # Binary conversion: chi-squared on a 2x2 contingency table
        treat_conv = int(treat_df["transactions"].sum())
        treat_no = int(treat_df["sessions"].sum()) - treat_conv
        ctrl_conv = int(ctrl_df["transactions"].sum())
        ctrl_no = int(ctrl_df["sessions"].sum()) - ctrl_conv
        contingency = np.array([[treat_conv, treat_no], [ctrl_conv, ctrl_no]])
        stat, p_value, _, _ = stats.chi2_contingency(contingency)
        test_type = "chi-squared"
        treat_rate = round(treat_conv / (treat_conv + treat_no) * 100, 4) if (treat_conv + treat_no) > 0 else 0.0
        ctrl_rate = round(ctrl_conv / (ctrl_conv + ctrl_no) * 100, 4) if (ctrl_conv + ctrl_no) > 0 else 0.0
        # Cohen's h for proportions
        p1, p2 = treat_rate / 100, ctrl_rate / 100
        cohens_d = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        n1, n2 = treat_conv + treat_no, ctrl_conv + ctrl_no
        se = float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)) * 100
        diff = treat_rate - ctrl_rate
        ci = (round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4))

    significant = bool(p_value < 0.05)
    lift = round((treat_rate - ctrl_rate) / ctrl_rate * 100, 2) if ctrl_rate != 0 else 0.0
    label = _effect_size_label(cohens_d)

    conclusion = (
        f"'{request.treatment}' {'significantly' if significant else 'does not significantly'} "
        f"{'outperforms' if treat_rate > ctrl_rate else 'underperforms'} '{request.control}' "
        f"(p={p_value:.4f}, {label} effect). "
        f"Relative lift: {lift:+.1f}%."
    )

    return ABTestResult(
        treatment=request.treatment,
        control=request.control,
        metric=request.metric,
        treatment_rate=treat_rate,
        control_rate=ctrl_rate,
        relative_lift_pct=lift,
        test_type=test_type,
        statistic=float(stat),
        p_value=float(p_value),
        significant=significant,
        effect_size=round(float(cohens_d), 4),
        effect_size_label=label,
        confidence_interval_95=ci,
        conclusion=conclusion,
    )
