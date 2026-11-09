"""Policy engine for groundcheck: determines overall pass/abstain/repair actions."""


from groundcheck.types import (
    ClaimVerdict,
    ClaimVerification,
    PolicyAction,
    PolicyConfig,
    VerificationResult,
)


class PolicyEngine:
    """Evaluates claim verifications against policy thresholds and decides actions."""

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def evaluate(
        self,
        claims: list[ClaimVerification],
        original_answer: str,
        action_override: PolicyAction | None = None,
        latency_ms: float = 0.0,
        escalation_rate: float = 0.0,
    ) -> VerificationResult:
        """Evaluate claims and return VerificationResult with decision, repair/diff if requested."""
        action = action_override or self.config.action

        # Count verdicts
        counts: dict[str, int] = {
            ClaimVerdict.SUPPORTED.value: 0,
            ClaimVerdict.WEAK.value: 0,
            ClaimVerdict.UNSUPPORTED.value: 0,
        }
        unsupported_claims: list[str] = []

        for cv in claims:
            counts[cv.verdict.value] = counts.get(cv.verdict.value, 0) + 1
            if cv.verdict == ClaimVerdict.UNSUPPORTED or cv.verdict == ClaimVerdict.WEAK and self.config.abstain_on_weak:
                unsupported_claims.append(cv.claim.text)

        has_unsupported = len(unsupported_claims) > 0
        passed = not has_unsupported

        repaired_answer = None
        diff = None

        if action == PolicyAction.REPAIR and has_unsupported:
            from groundcheck.repair import ClaimRepairer
            repairer = ClaimRepairer()
            repaired_answer, diff = repairer.repair(original_answer, claims, self.config)
        elif action == PolicyAction.ABSTAIN and has_unsupported:
            repaired_answer = (
                "Abstained: Response contains claims unsupported by the provided context."
            )

        return VerificationResult(
            claims=claims,
            verdict_counts=counts,
            overall_action=action,
            passed=passed,
            repaired_answer=repaired_answer,
            diff=diff,
            unsupported_claims=unsupported_claims,
            latency_ms=latency_ms,
            escalation_rate=escalation_rate,
        )
