import os

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.stats.power import NormalIndPower

from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from app.schemas.models import ABTestResult
from app.tools.statistics import ABTestRequest, run_ab_test
from app.tools.visualization import ChartJSON, chart_ab_test

_model = LitellmModel(model=os.environ.get("MODEL", "vertex_ai/gemini-2.5-flash"))


# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------

class EffectSizeResult(BaseModel):
    treatment_rate: float
    control_rate: float
    absolute_diff: float
    relative_lift_pct: float
    cohens_h: float
    effect_size_label: str
    ci_low_95: float
    ci_high_95: float


class PowerAnalysisResult(BaseModel):
    n_treatment: int
    n_control: int
    effect_size: float
    alpha: float
    power: float
    adequate: bool
    conclusion: str


class SubgroupResult(BaseModel):
    subgroup: str
    n_treatment: int
    n_control: int
    treatment_rate: float
    control_rate: float
    relative_lift_pct: float
    p_value: float
    significant: bool


class SegmentDeepDiveResult(BaseModel):
    dimension: str
    subgroup_dimension: str
    overall_lift_pct: float
    subgroups: list[SubgroupResult]
    simpsons_paradox_detected: bool
    conclusion: str


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@function_tool(strict_mode=False)
def ab_test(
    data: list[dict],
    treatment: str,
    control: str,
    dimension: str,
    metric: str = "converted",
) -> ABTestResult:
    """Run a two-sample statistical test between treatment and control groups.

    Uses chi-squared for binary conversion and Mann-Whitney U for revenue.
    Returns p-value, effect size, 95% CI, relative lift, and a plain-language conclusion.

    Args:
        data: Row records from a BigQuery query result.
        treatment: The treatment group value, e.g. 'Organic Search'.
        control: The control group value, e.g. 'Paid Search'.
        dimension: Column that identifies group membership, e.g. 'channelGrouping'.
        metric: Metric to test — 'converted' (binary conversion) or 'revenue_usd'.

    Returns:
        ABTestResult with test type, p-value, effect size, CI, and conclusion.
    """
    return run_ab_test(ABTestRequest(
        data=data, treatment=treatment, control=control,
        dimension=dimension, metric=metric,
    ))


@function_tool
def effect_size(
    treatment_rate: float,
    control_rate: float,
    n_treatment: int,
    n_control: int,
) -> EffectSizeResult:
    """Compute Cohen's h effect size, relative lift, and 95% CI from summary conversion rates.

    Useful when you already have pre-aggregated rates (e.g. from EDA segment stats)
    and want effect size without re-running the full A/B test on raw rows.

    Args:
        treatment_rate: Conversion rate of the treatment group as a percentage (e.g. 3.2).
        control_rate: Conversion rate of the control group as a percentage (e.g. 1.8).
        n_treatment: Number of sessions in the treatment group.
        n_control: Number of sessions in the control group.

    Returns:
        EffectSizeResult with Cohen's h, relative lift, and 95% confidence interval.
    """
    p1, p2 = treatment_rate / 100, control_rate / 100
    cohens_h = float(2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))))
    abs_h = abs(cohens_h)
    if abs_h < 0.2:
        label = "negligible"
    elif abs_h < 0.5:
        label = "small"
    elif abs_h < 0.8:
        label = "medium"
    else:
        label = "large"

    se = float(np.sqrt(p1 * (1 - p1) / n_treatment + p2 * (1 - p2) / n_control)) * 100
    diff = treatment_rate - control_rate
    lift = round((diff / control_rate) * 100, 2) if control_rate != 0 else 0.0

    return EffectSizeResult(
        treatment_rate=treatment_rate,
        control_rate=control_rate,
        absolute_diff=round(diff, 4),
        relative_lift_pct=lift,
        cohens_h=round(cohens_h, 4),
        effect_size_label=label,
        ci_low_95=round(diff - 1.96 * se, 4),
        ci_high_95=round(diff + 1.96 * se, 4),
    )


@function_tool
def power_analysis(
    n_treatment: int,
    n_control: int,
    effect_size_value: float,
    alpha: float = 0.05,
) -> PowerAnalysisResult:
    """Check whether the sample is large enough to reliably detect the observed effect.

    Args:
        n_treatment: Number of sessions in the treatment group.
        n_control: Number of sessions in the control group.
        effect_size_value: Standardized effect size (Cohen's h or d) from the A/B test.
        alpha: Significance level, default 0.05.

    Returns:
        PowerAnalysisResult with achieved statistical power and adequacy assessment.
    """
    ratio = n_control / n_treatment if n_treatment > 0 else 1.0
    power = float(NormalIndPower().power(
        effect_size=abs(effect_size_value),
        nobs1=n_treatment,
        alpha=alpha,
        ratio=ratio,
    ))
    adequate = power >= 0.8
    conclusion = (
        f"With n={n_treatment:,} (treatment) and n={n_control:,} (control), "
        f"the test achieves {power:.1%} power to detect an effect of size "
        f"{effect_size_value:.3f} at α={alpha}. "
        f"{'✓ Adequately powered (≥80%).' if adequate else '⚠ Underpowered (<80%) — interpret results with caution.'}"
    )
    return PowerAnalysisResult(
        n_treatment=n_treatment,
        n_control=n_control,
        effect_size=effect_size_value,
        alpha=alpha,
        power=round(power, 4),
        adequate=adequate,
        conclusion=conclusion,
    )


@function_tool(strict_mode=False)
def segment_deep_dive(
    data: list[dict],
    treatment: str,
    control: str,
    dimension: str,
    subgroup_dimension: str,
    metric: str = "converted",
) -> SegmentDeepDiveResult:
    """Check whether the treatment effect holds across subgroups (Simpson's paradox check).

    Runs the A/B test separately for each value of subgroup_dimension and compares
    the direction of lift to the overall result.

    Args:
        data: Row records from a BigQuery query result.
        treatment: The treatment group value, e.g. 'Organic Search'.
        control: The control group value, e.g. 'Paid Search'.
        dimension: Column identifying treatment/control membership.
        subgroup_dimension: Column to split by, e.g. 'device_category' or 'continent'.
        metric: Metric to test — 'converted' or 'revenue_usd'.

    Returns:
        SegmentDeepDiveResult with per-subgroup results and a Simpson's paradox flag.
    """
    df = pd.DataFrame(data)

    overall = run_ab_test(ABTestRequest(
        data=data, treatment=treatment, control=control,
        dimension=dimension, metric=metric,
    ))

    subgroups: list[SubgroupResult] = []
    for val in df[subgroup_dimension].dropna().unique():
        subset = df[df[subgroup_dimension] == val].to_dict(orient="records")
        try:
            r = run_ab_test(ABTestRequest(
                data=subset, treatment=treatment, control=control,
                dimension=dimension, metric=metric,
            ))
            treat_n = int(df[(df[subgroup_dimension] == val) & (df[dimension] == treatment)].shape[0])
            ctrl_n = int(df[(df[subgroup_dimension] == val) & (df[dimension] == control)].shape[0])
            subgroups.append(SubgroupResult(
                subgroup=str(val),
                n_treatment=treat_n,
                n_control=ctrl_n,
                treatment_rate=r.treatment_rate,
                control_rate=r.control_rate,
                relative_lift_pct=r.relative_lift_pct,
                p_value=r.p_value,
                significant=r.significant,
            ))
        except (ValueError, ZeroDivisionError):
            continue

    overall_positive = overall.relative_lift_pct > 0
    if subgroups:
        n_positive = sum(s.relative_lift_pct > 0 for s in subgroups)
        majority_positive = n_positive > len(subgroups) / 2
        simpsons_paradox = len(subgroups) >= 2 and (overall_positive != majority_positive)
    else:
        simpsons_paradox = False

    if simpsons_paradox:
        conclusion = (
            f"⚠ Simpson's Paradox detected: overall {treatment} shows "
            f"{'positive' if overall_positive else 'negative'} lift "
            f"({overall.relative_lift_pct:+.1f}%), but the majority of "
            f"{subgroup_dimension} subgroups show the opposite direction."
        )
    else:
        conclusion = (
            f"No Simpson's Paradox. The {overall.relative_lift_pct:+.1f}% lift for "
            f"{treatment} is consistent across {subgroup_dimension} subgroups."
        )

    return SegmentDeepDiveResult(
        dimension=dimension,
        subgroup_dimension=subgroup_dimension,
        overall_lift_pct=overall.relative_lift_pct,
        subgroups=subgroups,
        simpsons_paradox_detected=simpsons_paradox,
        conclusion=conclusion,
    )


@function_tool(strict_mode=False)
def visualize_ab_test(
    data: list[dict],
    treatment: str,
    control: str,
    dimension: str,
    metric: str = "converted",
) -> ChartJSON:
    """Generate a Plotly chart comparing treatment vs control with a confidence interval plot.

    Args:
        data: Row records from a BigQuery query result.
        treatment: The treatment group value.
        control: The control group value.
        dimension: Column identifying group membership.
        metric: Metric to visualize — 'converted' or 'revenue_usd'.

    Returns:
        ChartJSON with a Plotly figure JSON string for frontend rendering.
    """
    result = run_ab_test(ABTestRequest(
        data=data, treatment=treatment, control=control,
        dimension=dimension, metric=metric,
    ))
    return chart_ab_test(result)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a data analyst and statistician specializing in causal inference for marketing analytics.

You receive EDA findings and raw session data. Work through these steps in order:

1. HYPOTHESIZE — Form a narrative hypothesis that proposes a mechanism, not just a direction.
   Write it in first person, like an analyst presenting to a stakeholder:
   "I hypothesize that [group A] has a [higher/lower] [metric] than [group B] because [proposed
   mechanism grounded in user behavior or intent]."
   Then cite the specific EDA numbers that motivated this hypothesis.
   Example: "I hypothesize that Organic Search has a fundamentally higher conversion rate than
   Paid Search because it captures users with higher 'active intent' who are specifically seeking
   the Google brand, rather than users responding to generic display ads. The EDA shows Organic
   converts at 3.2% vs Paid's 1.1% — a 3x gap that suggests this is structural, not random."

2. FORMALIZE — State H0 and H1 to frame the statistical test.

3. TEST — Run ab_test with the appropriate metric ('converted' for conversion rate, 'revenue_usd' for revenue).

4. POWER — Run power_analysis using the effect size and group sizes from the ab_test result.

5. DEEP DIVE — Run segment_deep_dive across a relevant subgroup (e.g. device_category or continent)
   to check whether the effect holds or reveals a Simpson's Paradox.

6. VISUALIZE — Call visualize_ab_test to generate a chart.

Your final answer must:
- Lead with the narrative hypothesis and the EDA evidence that motivated it
- State whether the statistical test confirms or challenges the proposed mechanism
- Report test type, p-value, effect size label, and 95% CI
- Note power level and any Simpson's Paradox finding
- Close with a plain-language recommendation a marketer could act on"""


causal_agent = Agent(
    name="Causal Agent",
    instructions=_SYSTEM_PROMPT,
    tools=[ab_test, effect_size, power_analysis, segment_deep_dive, visualize_ab_test],
    model=_model,
)
