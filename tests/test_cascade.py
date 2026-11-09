"""Unit tests for the cascade verifier."""

from groundcheck.cascade import CascadeVerifier
from groundcheck.types import Claim, ClaimVerdict, PolicyConfig
from groundcheck.verify import CrossEncoderVerifier


class DummyVerifier(CrossEncoderVerifier):
    def __init__(self, name: str, fixed_score: float):
        self.model_name = name
        self.fixed_score = fixed_score

    def score_pairs(self, premises, hypotheses):
        return [
            {"entailment": self.fixed_score, "neutral": 1.0 - self.fixed_score, "contradiction": 0.0}
            for _ in premises
        ]


def test_cascade_escalates_uncertain_claim():
    # Cheap verifier produces 0.50 (in uncertainty band [0.35, 0.75])
    cheap = DummyVerifier("cheap-model", 0.50)
    # Strong verifier resolves it to 0.90
    strong = DummyVerifier("strong-model", 0.90)

    cascade = CascadeVerifier(cheap_verifier=cheap, strong_verifier=strong)

    claim = Claim(id="c1", text="Uncertain claim text", start_char=0, end_char=20, sentence_idx=0)
    chunks = ["Context chunk"]
    policy = PolicyConfig()

    result = cascade.verify_claim(claim, chunks, policy)

    assert result.escalated is True
    assert result.model_name == "strong-model"
    assert result.entailment == 0.90
    assert result.verdict == ClaimVerdict.SUPPORTED


def test_cascade_skips_escalation_for_confident_claim():
    # Cheap verifier produces 0.95 (outside uncertainty band)
    cheap = DummyVerifier("cheap-model", 0.95)
    strong = DummyVerifier("strong-model", 0.10)

    cascade = CascadeVerifier(cheap_verifier=cheap, strong_verifier=strong)

    claim = Claim(id="c1", text="Confident claim text", start_char=0, end_char=20, sentence_idx=0)
    chunks = ["Context chunk"]
    policy = PolicyConfig()

    result = cascade.verify_claim(claim, chunks, policy)

    assert result.escalated is False
    assert result.model_name == "cheap-model"
    assert result.entailment == 0.95
