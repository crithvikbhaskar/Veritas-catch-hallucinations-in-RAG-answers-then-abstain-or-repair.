"""Unit tests for the verifier engine and verdict evaluation."""

from groundcheck.types import Claim, ClaimVerdict, PolicyConfig
from groundcheck.verify import CrossEncoderVerifier


def test_verdict_evaluation_thresholds():
    policy = PolicyConfig(supported_threshold=0.70, weak_threshold=0.40, contradiction_penalty=0.30)

    # High entailment -> SUPPORTED
    v1 = CrossEncoderVerifier._evaluate_verdict(entailment=0.85, contradiction=0.05, policy=policy)
    assert v1 == ClaimVerdict.SUPPORTED

    # Moderate entailment -> WEAK
    v2 = CrossEncoderVerifier._evaluate_verdict(entailment=0.55, contradiction=0.05, policy=policy)
    assert v2 == ClaimVerdict.WEAK

    # Low entailment -> UNSUPPORTED
    v3 = CrossEncoderVerifier._evaluate_verdict(entailment=0.20, contradiction=0.05, policy=policy)
    assert v3 == ClaimVerdict.UNSUPPORTED

    # High contradiction overrides entailment -> UNSUPPORTED
    v4 = CrossEncoderVerifier._evaluate_verdict(entailment=0.35, contradiction=0.60, policy=policy)
    assert v4 == ClaimVerdict.UNSUPPORTED


def test_verify_batch_mock_scoring(monkeypatch):
    verifier = CrossEncoderVerifier()

    # Mock score_pairs to return deterministic scores without downloading model
    def mock_score_pairs(premises, hypotheses):
        # Return entailment=0.9 for match, 0.1 for mismatch
        results = []
        for p, h in zip(premises, hypotheses):
            if "Paris" in p and "Paris" in h:
                results.append({"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})
            elif "Mars" in p and "Mars" in h:
                results.append({"entailment": 0.92, "neutral": 0.05, "contradiction": 0.03})
            else:
                results.append({"entailment": 0.05, "neutral": 0.15, "contradiction": 0.80})
        return results

    monkeypatch.setattr(verifier, "score_pairs", mock_score_pairs)

    claim1 = Claim(id="c1", text="Paris is the capital of France.", start_char=0, end_char=32, sentence_idx=0)
    claim2 = Claim(id="c2", text="Jupiter has rings.", start_char=33, end_char=51, sentence_idx=1)

    chunks = [
        "France is in Europe and Paris is its capital.",
        "Mars is known as the red planet.",
    ]

    policy = PolicyConfig()
    results = verifier.verify_batch([claim1, claim2], chunks, policy)

    assert len(results) == 2
    assert results[0].verdict == ClaimVerdict.SUPPORTED
    assert results[0].best_chunk_idx == 0
    assert results[0].entailment == 0.95

    assert results[1].verdict == ClaimVerdict.UNSUPPORTED
    assert results[1].contradiction == 0.80
