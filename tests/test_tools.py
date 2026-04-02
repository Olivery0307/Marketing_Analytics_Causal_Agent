"""Unit tests for tools layer: bigquery, statistics, visualization.

Run with:
    uv run pytest tests/test_tools.py -v
"""

import json
import pytest

from app.tools.bigquery import SessionQueryParams, query_channel_conversion, query_device_conversion
from app.tools.statistics import ABTestRequest, EDARequest, run_ab_test, run_eda
from app.tools.visualization import chart_ab_test, chart_eda_segments


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ONE_MONTH = SessionQueryParams(start_date="20160801", end_date="20160831")


@pytest.fixture(scope="module")
def channel_data():
    return query_channel_conversion(ONE_MONTH)


@pytest.fixture(scope="module")
def device_data():
    return query_device_conversion(ONE_MONTH)


@pytest.fixture(scope="module")
def channel_eda(channel_data):
    return run_eda(EDARequest(data=channel_data.data, dimension="channel"))


@pytest.fixture(scope="module")
def organic_vs_paid(channel_data):
    return run_ab_test(ABTestRequest(
        data=channel_data.data,
        treatment="Organic Search",
        control="Paid Search",
        dimension="channel",
        metric="converted",
    ))


@pytest.fixture(scope="module")
def referral_vs_organic(channel_data):
    return run_ab_test(ABTestRequest(
        data=channel_data.data,
        treatment="Referral",
        control="Organic Search",
        dimension="channel",
        metric="converted",
    ))


# ---------------------------------------------------------------------------
# BigQuery tests
# ---------------------------------------------------------------------------

class TestBigQuery:
    def test_channel_data_returns_rows(self, channel_data):
        assert channel_data.rows > 0

    def test_channel_data_has_expected_columns(self, channel_data):
        expected = {"channel", "sessions", "total_transactions", "conversion_rate_pct", "total_revenue_usd"}
        assert expected.issubset(set(channel_data.columns))

    def test_channel_data_has_known_channels(self, channel_data):
        channels = {row["channel"] for row in channel_data.data}
        assert "Organic Search" in channels
        assert "Paid Search" in channels

    def test_device_data_has_three_categories(self, device_data):
        devices = {row["device"] for row in device_data.data}
        assert devices == {"desktop", "mobile", "tablet"}

    def test_sessions_are_positive(self, channel_data):
        for row in channel_data.data:
            assert row["sessions"] > 0


# ---------------------------------------------------------------------------
# EDA tests
# ---------------------------------------------------------------------------

class TestEDA:
    def test_eda_returns_all_channels(self, channel_data, channel_eda):
        # filter out (Other) which has 0 sessions after normalization
        bq_channels = {r["channel"] for r in channel_data.data if r["sessions"] > 0}
        eda_channels = {s.segment for s in channel_eda.segments}
        assert bq_channels == eda_channels

    def test_eda_top_segment_has_highest_conversion(self, channel_eda):
        rates = [s.conversion_rate_pct for s in channel_eda.segments]
        top = next(s for s in channel_eda.segments if s.segment == channel_eda.top_segment)
        assert top.conversion_rate_pct == max(rates)

    def test_eda_bottom_segment_has_lowest_conversion(self, channel_eda):
        rates = [s.conversion_rate_pct for s in channel_eda.segments]
        bottom = next(s for s in channel_eda.segments if s.segment == channel_eda.bottom_segment)
        assert bottom.conversion_rate_pct == min(rates)

    def test_eda_conversion_rates_are_percentages(self, channel_eda):
        for s in channel_eda.segments:
            assert 0.0 <= s.conversion_rate_pct <= 100.0

    def test_eda_revenue_non_negative(self, channel_eda):
        for s in channel_eda.segments:
            assert s.total_revenue_usd >= 0.0

    def test_eda_summary_mentions_dimension(self, channel_eda):
        assert "channel" in channel_eda.summary.lower()

    def test_eda_device_dimension(self, device_data):
        eda = run_eda(EDARequest(data=device_data.data, dimension="device"))
        assert eda.top_segment in {"desktop", "mobile", "tablet"}

    def test_eda_invalid_dimension_raises(self, channel_data):
        with pytest.raises(ValueError, match="not found"):
            run_eda(EDARequest(data=channel_data.data, dimension="nonexistent_col"))


# ---------------------------------------------------------------------------
# A/B test tests
# ---------------------------------------------------------------------------

class TestABTest:
    def test_ab_result_has_valid_p_value(self, organic_vs_paid):
        assert 0.0 <= organic_vs_paid.p_value <= 1.0

    def test_ab_significant_flag_consistent_with_p_value(self, organic_vs_paid):
        if organic_vs_paid.p_value < 0.05:
            assert organic_vs_paid.significant is True
        else:
            assert organic_vs_paid.significant is False

    def test_ab_confidence_interval_ordered(self, organic_vs_paid):
        low, high = organic_vs_paid.confidence_interval_95
        assert low <= high

    def test_ab_effect_size_label_valid(self, organic_vs_paid):
        assert organic_vs_paid.effect_size_label in {"negligible", "small", "medium", "large"}

    def test_referral_significantly_beats_organic(self, referral_vs_organic):
        # Referral converts at ~6% vs Organic ~1.5% — should always be significant
        assert referral_vs_organic.significant is True
        assert referral_vs_organic.relative_lift_pct > 0

    def test_ab_lift_calculation(self, channel_data):
        result = run_ab_test(ABTestRequest(
            data=channel_data.data,
            treatment="Direct",
            control="Organic Search",
            dimension="channel",
            metric="converted",
        ))
        expected_lift = round(
            (result.treatment_rate - result.control_rate) / result.control_rate * 100, 2
        )
        assert abs(result.relative_lift_pct - expected_lift) < 0.1

    def test_ab_unknown_group_raises(self, channel_data):
        with pytest.raises(ValueError, match="not found"):
            run_ab_test(ABTestRequest(
                data=channel_data.data,
                treatment="Unknown Channel",
                control="Paid Search",
                dimension="channel",
            ))


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_eda_chart_returns_valid_json(self, channel_eda):
        chart = chart_eda_segments(channel_eda)
        parsed = json.loads(chart.plotly_json)
        assert "data" in parsed
        assert "layout" in parsed

    def test_eda_chart_has_two_traces(self, channel_eda):
        chart = chart_eda_segments(channel_eda)
        parsed = json.loads(chart.plotly_json)
        assert len(parsed["data"]) == 2

    def test_ab_chart_returns_valid_json(self, referral_vs_organic):
        chart = chart_ab_test(referral_vs_organic)
        parsed = json.loads(chart.plotly_json)
        assert "data" in parsed
        assert "layout" in parsed

    def test_ab_chart_title_contains_groups(self, referral_vs_organic):
        chart = chart_ab_test(referral_vs_organic)
        assert "Referral" in chart.title
        assert "Organic Search" in chart.title

    def test_chart_json_is_non_empty(self, channel_eda):
        chart = chart_eda_segments(channel_eda)
        assert len(chart.plotly_json) > 1000
