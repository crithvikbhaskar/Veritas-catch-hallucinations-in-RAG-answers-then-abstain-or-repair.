"""Unit tests for the policy engine."""

import pytest

from groundcheck.policy import PolicyEngine
from groundcheck.types import (
    Claim,
    ClaimVerdict,
    ClaimVerification,
    PolicyAction,
    PolicyConfig,
)


@pytest.fixture
def sample_claim():
    return Claim(
        id="c1",
        text="The sky is blue.",
        start_char=0,
        end_char=16,
        sentence_idx=0,
    )


def test_policy_all_supported(sample_claim):
    engine = PolicyEngine()
    cv = ClaimVerification(
        claim=sample_claim,
        verdict=ClaimVerdict.SUPPORTED,
        entailment=0.92,
        contradiction=0.01,
        neutral=0.07,
    )
    result = engine.evaluate([cv], "The sky is blue.")
    assert result.passed is True
    assert result.verdict_counts["SUPPORTED"] == 1
    assert len(result.unsupported_claims) == 0


def test_policy_unsupported_pass_action(sample_claim):
    config = PolicyConfig(action=PolicyAction.PASS)
    engine = PolicyEngine(config)
    cv = ClaimVerification(
        claim=sample_claim,
        verdict=ClaimVerdict.UNSUPPORTED,
        entailment=0.10,
        contradiction=0.15,
        neutral=0.75,
    )
    result = engine.evaluate([cv], "The sky is blue.")
    assert result.passed is False
    assert result.overall_action == PolicyAction.PASS
    assert len(result.unsupported_claims) == 1
    assert result.repaired_answer is None


def test_policy_abstain_action(sample_claim):
    config = PolicyConfig(action=PolicyAction.ABSTAIN)
    engine = PolicyEngine(config)
    cv = ClaimVerification(
        claim=sample_claim,
        verdict=ClaimVerdict.UNSUPPORTED,
        entailment=0.10,
    )
    result = engine.evaluate([cv], "The sky is blue.")
    assert result.passed is False
    assert result.overall_action == PolicyAction.ABSTAIN
    assert "Abstained" in (result.repaired_answer or "")


def test_policy_weak_claim_handling(sample_claim):
    # Default: abstain_on_weak is False -> WEAK is tolerated
    config = PolicyConfig(abstain_on_weak=False)
    engine = PolicyEngine(config)
    cv = ClaimVerification(
        claim=sample_claim,
        verdict=ClaimVerdict.WEAK,
        entailment=0.55,
    )
    result = engine.evaluate([cv], "The sky is blue.")
    assert result.passed is True

    # With abstain_on_weak = True -> WEAK is treated as unsupported
    strict_config = PolicyConfig(abstain_on_weak=True)
    strict_engine = PolicyEngine(strict_config)
    strict_result = strict_engine.evaluate([cv], "The sky is blue.")
    assert strict_result.passed is False
    assert len(strict_result.unsupported_claims) == 1
