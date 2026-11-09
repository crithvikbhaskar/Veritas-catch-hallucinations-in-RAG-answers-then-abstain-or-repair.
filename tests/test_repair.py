"""Unit tests for the claim repair engine."""

from groundcheck.repair import ClaimRepairer
from groundcheck.types import (
    Claim,
    ClaimVerdict,
    ClaimVerification,
    PolicyConfig,
)


def test_repair_drops_single_unsupported_sentence():
    text = "The earth is round. The moon is made of green cheese. Water boils at 100C."
    # Offsets:
    # "The earth is round. " (0-20)
    # "The moon is made of green cheese. " (20-54)
    # "Water boils at 100C." (54-74)
    c1 = Claim(id="1", text="The earth is round.", start_char=0, end_char=19, sentence_idx=0)
    c2 = Claim(id="2", text="The moon is made of green cheese.", start_char=20, end_char=53, sentence_idx=1)
    c3 = Claim(id="3", text="Water boils at 100C.", start_char=54, end_char=74, sentence_idx=2)

    verifications = [
        ClaimVerification(claim=c1, verdict=ClaimVerdict.SUPPORTED, entailment=0.9),
        ClaimVerification(claim=c2, verdict=ClaimVerdict.UNSUPPORTED, entailment=0.05),
        ClaimVerification(claim=c3, verdict=ClaimVerdict.SUPPORTED, entailment=0.9),
    ]

    repairer = ClaimRepairer()
    repaired, diff = repairer.repair(text, verifications, PolicyConfig())

    assert "The moon is made of green cheese" not in repaired
    assert "The earth is round" in repaired
    assert "Water boils at 100C" in repaired
    assert "--- original_answer" in diff
    assert "+++ repaired_answer" in diff


def test_repair_all_unsupported_yields_empty():
    text = "Aliens built the pyramids."
    c1 = Claim(id="1", text="Aliens built the pyramids.", start_char=0, end_char=26, sentence_idx=0)
    verifications = [
        ClaimVerification(claim=c1, verdict=ClaimVerdict.UNSUPPORTED, entailment=0.01)
    ]
    repairer = ClaimRepairer()
    repaired, diff = repairer.repair(text, verifications, PolicyConfig())
    assert repaired == ""
    assert "-Aliens built the pyramids." in diff


def test_repair_all_supported_no_change():
    text = "Python is a popular programming language."
    c1 = Claim(id="1", text=text, start_char=0, end_char=len(text), sentence_idx=0)
    verifications = [
        ClaimVerification(claim=c1, verdict=ClaimVerdict.SUPPORTED, entailment=0.99)
    ]
    repairer = ClaimRepairer()
    repaired, diff = repairer.repair(text, verifications, PolicyConfig())
    assert repaired == text
    assert diff == ""
