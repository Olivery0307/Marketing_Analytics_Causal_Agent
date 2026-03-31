from pydantic import BaseModel, Field


class SegmentStats(BaseModel):
    segment: str
    sessions: int
    transactions: int
    conversion_rate_pct: float
    total_revenue_usd: float
    revenue_per_session: float


class EDAResult(BaseModel):
    dimension: str = Field(description="The dimension analyzed, e.g. 'channelGrouping'")
    segments: list[SegmentStats]
    top_segment: str
    bottom_segment: str
    summary: str


class ABTestResult(BaseModel):
    treatment: str
    control: str
    metric: str
    treatment_rate: float
    control_rate: float
    relative_lift_pct: float
    test_type: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_label: str
    confidence_interval_95: tuple[float, float]
    conclusion: str


class CausalResult(BaseModel):
    question: str
    hypothesis: str
    eda: EDAResult
    ab_test: ABTestResult
    final_answer: str
