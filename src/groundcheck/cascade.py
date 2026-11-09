"""Cascade verifier: cheap model first, escalating uncertain claims to strong model."""

import time

from groundcheck.policy import PolicyEngine
from groundcheck.types import (
    Claim,
    ClaimVerification,
    PolicyConfig,
    VerificationResult,
)
from groundcheck.verify import BaseVerifier, CrossEncoderVerifier


class CascadeVerifier(BaseVerifier):
    """Tiered verification: cheap model runs first; claims in uncertainty band escalate to strong model."""

    def __init__(
        self,
        cheap_verifier: CrossEncoderVerifier | None = None,
        strong_verifier: CrossEncoderVerifier | None = None,
        uncertainty_lower: float = 0.35,
        uncertainty_upper: float = 0.75,
    ):
        self.cheap_verifier = cheap_verifier or CrossEncoderVerifier(
            model_name=CrossEncoderVerifier.CHEAP_MODEL
        )
        self.strong_verifier = strong_verifier or CrossEncoderVerifier(
            model_name=CrossEncoderVerifier.STRONG_MODEL
        )
        self.uncertainty_lower = uncertainty_lower
        self.uncertainty_upper = uncertainty_upper

    def is_uncertain(self, verification: ClaimVerification) -> bool:
        """Check if cheap verifier's confidence falls within uncertainty band."""
        entailment = verification.entailment
        contradiction = verification.contradiction
        
        # If contradiction is very high, it's definitely contradictory, not uncertain
        if contradiction > 0.8:
            return False
            
        # Borderline entailment score falls in the uncertainty band
        return self.uncertainty_lower <= entailment <= self.uncertainty_upper

    def verify_claim(
        self, claim: Claim, chunks: list[str], policy: PolicyConfig
    ) -> ClaimVerification:
        """Verify single claim using cascade logic."""
        results = self.verify_batch([claim], chunks, policy)
        return results[0]

    def verify_batch(
        self, claims: list[Claim], chunks: list[str], policy: PolicyConfig
    ) -> list[ClaimVerification]:
        """Verify batch of claims through the cheap model, escalating uncertain ones."""
        if not claims or not chunks:
            return []

        # Step 1: Run cheap tier on all claims
        cheap_results = self.cheap_verifier.verify_batch(claims, chunks, policy)

        # Step 2: Identify uncertain claims for escalation
        escalate_indices = [
            i for i, cv in enumerate(cheap_results) if self.is_uncertain(cv)
        ]

        if not escalate_indices:
            return cheap_results

        # Step 3: Run strong tier on uncertain claims
        escalate_claims = [claims[i] for i in escalate_indices]
        strong_results = self.strong_verifier.verify_batch(escalate_claims, chunks, policy)

        # Merge results
        final_results = list(cheap_results)
        for idx, strong_cv in zip(escalate_indices, strong_results):
            strong_cv.escalated = True
            final_results[idx] = strong_cv

        return final_results

    def verify_text(
        self,
        text: str,
        chunks: list[str],
        policy: PolicyConfig | None = None,
        splitter=None,
    ) -> VerificationResult:
        """Full end-to-end verification pipeline on raw text answer."""
        t0 = time.perf_counter()
        policy = policy or PolicyConfig()

        if splitter is None:
            from groundcheck.claims import ClaimSplitter
            splitter = ClaimSplitter()

        claims = splitter.split(text)
        if not claims:
            return VerificationResult(
                claims=[],
                verdict_counts={},
                overall_action=policy.action,
                passed=True,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                escalation_rate=0.0,
            )

        verifications = self.verify_batch(claims, chunks, policy)
        escalated_count = sum(1 for cv in verifications if cv.escalated)
        escalation_rate = escalated_count / len(verifications) if verifications else 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        engine = PolicyEngine(policy)
        result = engine.evaluate(
            claims=verifications,
            original_answer=text,
            action_override=policy.action,
            latency_ms=latency_ms,
            escalation_rate=escalation_rate,
        )
        return result
