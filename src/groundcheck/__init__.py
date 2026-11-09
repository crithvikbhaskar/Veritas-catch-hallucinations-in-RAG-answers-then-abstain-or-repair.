"""groundcheck: Drop-in middleware that verifies RAG claims against source chunks."""

from groundcheck.types import (
    Claim,
    ClaimVerdict,
    PolicyAction,
    PolicyConfig,
    VerificationResult,
    VerifyRequest,
    VerifyResponse,
)

__version__ = "0.1.0"

__all__ = [
    "Claim",
    "ClaimVerdict",
    "PolicyAction",
    "PolicyConfig",
    "VerificationResult",
    "VerifyRequest",
    "VerifyResponse",
]

