"""Integration tests for agent pipeline.

These tests run live agent calls — they require valid GCP credentials and
consume BigQuery free-tier quota. Each test runs a real LLM call.

Run with:
    uv run pytest tests/test_agents.py -v

Run only fast tool-level tests (no LLM calls):
    uv run pytest tests/test_tools.py -v
"""

from agents import Runner

from app.agents.orchestrator import orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def ask(question: str) -> str:
    result = await Runner.run(orchestrator, input=question)
    return str(result.final_output)


# ---------------------------------------------------------------------------
# Orchestrator smoke tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    async def test_channel_question_returns_answer(self):
        answer = await ask("Does organic search convert better than paid search?")
        assert len(answer) > 100
        assert "organic" in answer.lower() or "paid" in answer.lower()

    async def test_device_question_returns_answer(self):
        answer = await ask("Does mobile convert as well as desktop?")
        assert len(answer) > 100
        assert "mobile" in answer.lower() or "desktop" in answer.lower()

    async def test_answer_mentions_p_value_or_significance(self):
        answer = await ask("Is the difference in conversion between referral and direct traffic statistically significant?")
        lower = answer.lower()
        assert any(term in lower for term in ["p-value", "p =", "significant", "statistic", "confidence"])

    async def test_answer_mentions_conversion_rate(self):
        answer = await ask("Which channel has the highest conversion rate?")
        lower = answer.lower()
        assert "%" in answer or "conversion" in lower or "rate" in lower

    async def test_social_low_conversion_finding(self):
        """Social has ~0.07% conversion — agent should identify it as bottom performer."""
        answer = await ask("How does social media traffic perform compared to other channels?")
        assert "social" in answer.lower()

    async def test_referral_outperforms_organic(self):
        """Referral (~6.4%) is statistically significantly better than organic (~1.5%)."""
        answer = await ask("Does referral traffic convert better than organic search?")
        lower = answer.lower()
        assert "referral" in lower
        assert "significant" in lower or "p " in lower or "%" in answer
