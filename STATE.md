# Project State: groundcheck

Start Time: 2026-11-09 14:55:08

## Phase Checklist

### P1 (0-25 min): Scaffold + Core Verification Engine
- [x] Initialize repository structure, pyproject.toml with uv, ruff, pytest (Elapsed: 7 min)
- [x] Implement `src/groundcheck/types.py` (Claim, VerificationResult, PolicyDecision, etc.) (Elapsed: 8 min)
- [x] Implement `src/groundcheck/claims.py` (sentence/clause claim splitter) (Elapsed: 9 min)
- [x] Implement `src/groundcheck/verify.py` (NLI cross-encoder verifier on CPU) (Elapsed: 11 min)
- [x] Implement `src/groundcheck/policy.py` (SUPPORTED / WEAK / UNSUPPORTED policy rules) (Elapsed: 12 min)
- [x] Write and pass unit tests for P1 (`tests/test_claims.py`, `tests/test_policy.py`, `tests/test_verify.py`) (Elapsed: 15 min)

### P2 (25-45 min): Eval Harness + Baseline & Single-Tier
- [x] Create eval dataset loader (`eval/data.py` - HF subset or curated benchmark dataset) (Elapsed: 18 min)
- [x] Implement metrics computation (`eval/metrics.py`) (precision, recall, F1, hallucination rate, answer rate, latency) (Elapsed: 20 min)
- [x] Implement eval runner (`eval/run.py`) and run baseline vs single-tier (Elapsed: 22 min)
- [x] Generate initial eval artifacts (Elapsed: 25 min)

### P3 (45-60 min): Cascade + Repair Engine
- [x] Implement `src/groundcheck/repair.py` (drop unsupported claims, generate diff) (Elapsed: 12 min)
- [x] Implement `src/groundcheck/cascade.py` (cheap model first, escalate uncertain claims to strong model) (Elapsed: 14 min)
- [x] Implement plot generation (`eval/plots.py`) for risk-coverage curve (Elapsed: 18 min)
- [x] Run full eval (baseline vs single-tier vs cascade) and save `docs/results/report.md` + `docs/results/risk_coverage.png` (Elapsed: 26 min)
- [x] Unit tests for repair and cascade (`tests/test_repair.py`, `tests/test_cascade.py`) (Elapsed: 15 min)

### P4 (60-80 min): FastAPI + Next.js Demo
- [x] Implement FastAPI server (`api/main.py`) with `POST /verify` (Elapsed: 22 min)
- [x] Add API integration test (`tests/test_api.py`) (Elapsed: 24 min)
- [x] Scaffold Next.js + Tailwind + TypeScript single-page app in `web/` (Elapsed: 23 min)
- [x] Build demo UI: Question/chunks/answer inputs, threshold sliders, colored claims highlight, diff view (Elapsed: 25 min)
- [x] End-to-end smoke test of API and web demo (Elapsed: 27 min)

### P5 (80-90 min): Documentation & Final Verification
- [x] Write `docs/design-decisions.md` (Elapsed: 24 min)
- [x] Create `Makefile` with targets (`install`, `test`, `eval`, `api`, `web`) (Elapsed: 22 min)
- [x] Write `README.md` (problem, quickstart, real results table, limitations, Mermaid diagram) (Elapsed: 25 min)
- [x] Add `LICENSE` (MIT) (Elapsed: 22 min)
- [x] Final run of `pytest` and verification of all outputs (Elapsed: 28 min)
