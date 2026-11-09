"""FastAPI application for groundcheck verification middleware."""

import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from groundcheck.cascade import CascadeVerifier
from groundcheck.claims import ClaimSplitter
from groundcheck.policy import PolicyEngine
from groundcheck.types import (
    PolicyAction,
    PolicyConfig,
    VerifyRequest,
    VerifyResponse,
)
from groundcheck.verify import CrossEncoderVerifier

app = FastAPI(
    title="groundcheck API",
    description="Drop-in middleware checking RAG claims against source chunks",
    version="0.1.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons (lazy-loaded on first request)
_splitter = ClaimSplitter()
_cheap_verifier: CrossEncoderVerifier | None = None
_strong_verifier: CrossEncoderVerifier | None = None
_cascade_verifier: CascadeVerifier | None = None


def get_verifiers():
    global _cheap_verifier, _strong_verifier, _cascade_verifier
    if _cheap_verifier is None:
        _cheap_verifier = CrossEncoderVerifier(model_name=CrossEncoderVerifier.CHEAP_MODEL)
    if _strong_verifier is None:
        _strong_verifier = CrossEncoderVerifier(model_name=CrossEncoderVerifier.STRONG_MODEL)
    if _cascade_verifier is None:
        _cascade_verifier = CascadeVerifier(
            cheap_verifier=_cheap_verifier, strong_verifier=_strong_verifier
        )
    return _cheap_verifier, _cascade_verifier


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/sample")
def sample_payload():
    """Return a representative sample payload for instant testing."""
    return {
        "question": "Where is the Eiffel Tower and who designed it?",
        "chunks": [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. Constructed from 1887 to 1889, it was named after engineer Gustave Eiffel."
        ],
        "answer": "The Eiffel Tower is located on the Champ de Mars in Paris, France. It was designed by James Watt in London during the Industrial Revolution.",
        "policy": {
            "supported_threshold": 0.70,
            "weak_threshold": 0.40,
            "contradiction_penalty": 0.30,
            "action": "repair",
            "abstain_on_weak": False,
        },
        "use_cascade": True,
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest):
    """Verify claims in answer against source chunks."""
    if not request.answer or not request.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    if not request.chunks:
        raise HTTPException(status_code=400, detail="Chunks cannot be empty")

    t0 = time.perf_counter()
    policy = request.policy or PolicyConfig()
    if request.action is not None:
        policy.action = request.action

    cheap, cascade = get_verifiers()
    active_verifier = cascade if request.use_cascade else cheap

    # 1. Split into claims
    claims = _splitter.split(request.answer)
    if not claims:
        return VerifyResponse(
            status="ok",
            claims=[],
            verdict_counts={},
            action_taken=policy.action,
            passed=True,
            final_answer=request.answer,
            repaired_answer=None,
            diff=None,
            unsupported_claims=[],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            escalation_rate=0.0,
        )

    # 2. Verify claims
    verifications = active_verifier.verify_batch(claims, request.chunks, policy)
    escalated_count = sum(1 for cv in verifications if cv.escalated)
    esc_rate = escalated_count / len(verifications) if verifications else 0.0

    # 3. Policy evaluation & action
    engine = PolicyEngine(policy)
    eval_result = engine.evaluate(
        claims=verifications,
        original_answer=request.answer,
        action_override=policy.action,
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Determine final returned answer based on action taken
    final_answer = request.answer
    if policy.action == PolicyAction.REPAIR and eval_result.repaired_answer is not None or policy.action == PolicyAction.ABSTAIN and eval_result.repaired_answer is not None:
        final_answer = eval_result.repaired_answer

    return VerifyResponse(
        status="ok",
        claims=eval_result.claims,
        verdict_counts=eval_result.verdict_counts,
        action_taken=eval_result.overall_action,
        passed=eval_result.passed,
        final_answer=final_answer,
        repaired_answer=eval_result.repaired_answer,
        diff=eval_result.diff,
        unsupported_claims=eval_result.unsupported_claims,
        latency_ms=latency_ms,
        escalation_rate=esc_rate,
    )
