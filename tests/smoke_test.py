"""Manual end-to-end smoke test of groundcheck verification and repair."""

from groundcheck.claims import ClaimSplitter
from groundcheck.policy import PolicyEngine
from groundcheck.types import PolicyAction, PolicyConfig
from groundcheck.verify import CrossEncoderVerifier


def run_smoke_test():
    print("=== STARTING GROUNDCHECK SMOKE TEST ===")

    # Test Sample
    chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. Constructed from 1887 to 1889, it was named after engineer Gustave Eiffel."
    ]
    answer = (
        "The Eiffel Tower is located on the Champ de Mars in Paris, France. "
        "It was designed by James Watt in London during the Industrial Revolution."
    )

    print(f"\n1. Input Chunks: {len(chunks)} chunks")
    print(f"2. Original Answer:\n   \"{answer}\"")

    # 1. Claim Splitting
    splitter = ClaimSplitter()
    claims = splitter.split(answer)
    print(f"\n3. Claim Splitting: Identified {len(claims)} atomic claims:")
    for i, c in enumerate(claims):
        print(f"   [{i}] (chars {c.start_char}-{c.end_char}): \"{c.text}\"")

    assert len(claims) == 2, f"Expected 2 claims, got {len(claims)}"

    # 2. Local NLI Verification (Cheap Model)
    print("\n4. Running NLI Cross-Encoder Verifier...")
    verifier = CrossEncoderVerifier(model_name=CrossEncoderVerifier.CHEAP_MODEL)
    policy = PolicyConfig(
        supported_threshold=0.65,
        weak_threshold=0.35,
        contradiction_penalty=0.35,
        action=PolicyAction.REPAIR,
    )

    verifications = verifier.verify_batch(claims, chunks, policy)
    for i, cv in enumerate(verifications):
        print(
            f"   Claim [{i}]: Verdict={cv.verdict.value} | "
            f"Entailment={cv.entailment:.3f} | Contradiction={cv.contradiction:.3f}"
        )

    # Claim 0 must be SUPPORTED
    assert verifications[0].verdict.value == "SUPPORTED", f"Claim 0 should be SUPPORTED, got {verifications[0].verdict}"
    # Claim 1 must be UNSUPPORTED
    assert verifications[1].verdict.value == "UNSUPPORTED", f"Claim 1 should be UNSUPPORTED, got {verifications[1].verdict}"

    # 3. Policy Engine Evaluation (Repair Action)
    print("\n5. Testing Action: REPAIR...")
    engine = PolicyEngine(policy)
    repair_result = engine.evaluate(verifications, answer, action_override=PolicyAction.REPAIR)

    print(f"   Passed: {repair_result.passed}")
    print(f"   Unsupported Claims: {repair_result.unsupported_claims}")
    print(f"   Repaired Answer:\n      \"{repair_result.repaired_answer}\"")
    print(f"   Unified Diff:\n{repair_result.diff}")

    assert not repair_result.passed
    assert "James Watt" not in (repair_result.repaired_answer or "")
    assert "Champ de Mars in Paris" in (repair_result.repaired_answer or "")
    assert "--- original_answer" in (repair_result.diff or "")

    # 4. Policy Engine Evaluation (Abstain Action)
    print("\n6. Testing Action: ABSTAIN...")
    abstain_result = engine.evaluate(verifications, answer, action_override=PolicyAction.ABSTAIN)
    print(f"   Passed: {abstain_result.passed}")
    print(f"   Abstain Notice: {abstain_result.repaired_answer}")

    assert not abstain_result.passed
    assert "Abstained" in (abstain_result.repaired_answer or "")

    print("\n=== ALL SMOKE TEST CHECKS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_smoke_test()
