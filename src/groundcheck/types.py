"""Data models and type definitions for groundcheck."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClaimVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    WEAK = "WEAK"
    UNSUPPORTED = "UNSUPPORTED"


class PolicyAction(str, Enum):
    PASS = "pass"
    ABSTAIN = "abstain"
    REPAIR = "repair"


class Claim(BaseModel):
    id: str
    text: str
    start_char: int
    end_char: int
    sentence_idx: int


class ChunkScore(BaseModel):
    chunk_idx: int
    chunk_text: str
    entailment: float
    neutral: float
    contradiction: float


class ClaimVerification(BaseModel):
    claim: Claim
    verdict: ClaimVerdict
    best_chunk_idx: int | None = None
    best_chunk_score: float | None = None
    entailment: float = 0.0
    contradiction: float = 0.0
    neutral: float = 0.0
    model_name: str = ""
    escalated: bool = False
    chunk_scores: list[ChunkScore] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    supported_threshold: float = Field(
        default=0.70,
        description="Minimum entailment score to consider a claim SUPPORTED.",
    )
    weak_threshold: float = Field(
        default=0.40,
        description="Minimum entailment score below which a claim is considered UNSUPPORTED.",
    )
    contradiction_penalty: float = Field(
        default=0.30,
        description="Maximum contradiction score before demoting a claim to UNSUPPORTED.",
    )
    action: PolicyAction = Field(
        default=PolicyAction.PASS,
        description="Default action to take: pass, abstain, or repair.",
    )
    abstain_on_weak: bool = Field(
        default=False,
        description="If True, WEAK claims also trigger abstention/repair.",
    )


class VerificationResult(BaseModel):
    claims: list[ClaimVerification]
    verdict_counts: dict[str, int]
    overall_action: PolicyAction
    passed: bool
    repaired_answer: str | None = None
    diff: str | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    escalation_rate: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifyRequest(BaseModel):
    question: str | None = None
    chunks: list[str] = Field(
        ..., min_length=1, description="Source context chunks to verify against."
    )
    answer: str = Field(..., min_length=1, description="Generated answer to verify.")
    policy: PolicyConfig | None = None
    action: PolicyAction | None = None
    use_cascade: bool = False


class VerifyResponse(BaseModel):
    status: str
    claims: list[ClaimVerification]
    verdict_counts: dict[str, int]
    action_taken: PolicyAction
    passed: bool
    final_answer: str
    repaired_answer: str | None = None
    diff: str | None = None
    unsupported_claims: list[str]
    latency_ms: float
    escalation_rate: float = 0.0
