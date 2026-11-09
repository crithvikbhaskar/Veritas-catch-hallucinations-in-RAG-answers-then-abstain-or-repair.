"""Evaluation runner: executes baseline, single-tier, and cascade evaluations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.data import EvalExample, load_eval_dataset
from eval.metrics import EvalMetrics, compute_metrics
from eval.plots import plot_risk_coverage
from groundcheck.cascade import CascadeVerifier
from groundcheck.claims import ClaimSplitter
from groundcheck.policy import PolicyEngine
from groundcheck.types import (
    ClaimVerdict,
    ClaimVerification,
    PolicyAction,
    PolicyConfig,
)
from groundcheck.verify import CrossEncoderVerifier

TupleMetrics = tuple[EvalMetrics, list[dict[str, Any]]]


def evaluate_baseline(examples: list[EvalExample]) -> TupleMetrics:
    """Baseline: No guardrail applied (every answer passed as-is)."""
    t0 = time.perf_counter()
    splitter = ClaimSplitter()
    predictions = []

    for ex in examples:
        claims = splitter.split(ex.answer)
        # All claims marked SUPPORTED by baseline
        verifications = [
            ClaimVerification(
                claim=c,
                verdict=ClaimVerdict.SUPPORTED,
                entailment=1.0,
                model_name="baseline_none",
            )
            for c in claims
        ]
        predictions.append(
            {
                "claims": verifications,
                "passed": True,
                "repaired_answer": ex.answer,
            }
        )

    latency_ms = ((time.perf_counter() - t0) / len(examples)) * 1000.0 if examples else 0.0
    metrics = compute_metrics(
        examples, predictions, mean_latency_ms=latency_ms, escalation_rate=0.0, action="pass"
    )
    return metrics, predictions


def evaluate_verifier(
    verifier,
    examples: list[EvalExample],
    policy: PolicyConfig,
    is_cascade: bool = False,
    action: str = "abstain",
) -> TupleMetrics:
    """Evaluate a verifier (single-tier or cascade) across the dataset."""
    splitter = ClaimSplitter()
    engine = PolicyEngine(policy)
    predictions = []
    latencies = []
    escalated_claims = 0
    total_claims = 0

    # Warm up model so load time is not attributed to query latency
    if hasattr(verifier, "_load_model"):
        verifier._load_model()
    elif hasattr(verifier, "cheap_verifier") and hasattr(verifier.cheap_verifier, "_load_model"):
        verifier.cheap_verifier._load_model()
        if hasattr(verifier, "strong_verifier") and hasattr(verifier.strong_verifier, "_load_model"):
            verifier.strong_verifier._load_model()

    for ex in examples:
        t0 = time.perf_counter()
        claims = splitter.split(ex.answer)
        total_claims += len(claims)

        if not claims:
            verifications = []
        else:
            verifications = verifier.verify_batch(claims, ex.chunks, policy)

        if is_cascade:
            escalated_claims += sum(1 for cv in verifications if cv.escalated)

        result = engine.evaluate(
            claims=verifications,
            original_answer=ex.answer,
            action_override=PolicyAction(action),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        predictions.append(
            {
                "claims": verifications,
                "passed": result.passed,
                "repaired_answer": result.repaired_answer,
                "unsupported_claims": result.unsupported_claims,
            }
        )

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    esc_rate = (escalated_claims / total_claims) if total_claims > 0 else 0.0

    metrics = compute_metrics(
        examples,
        predictions,
        mean_latency_ms=mean_latency,
        escalation_rate=esc_rate,
        action=action,
    )
    return metrics, predictions


TupleMetrics = tuple[EvalMetrics, list[dict[str, Any]]]


def run_evaluation(
    dataset_name: str = "mini",
    max_examples: int = 60,
    output_dir: str = "docs/results",
):
    """Run full comparative evaluation and produce report and plots."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading evaluation dataset '{dataset_name}'...")
    examples = load_eval_dataset(dataset_name=dataset_name, max_examples=max_examples)
    print(f"Loaded {len(examples)} examples. Hallucination rate in ground truth: "
          f"{sum(1 for ex in examples if ex.has_hallucination) / len(examples):.2%}")

    policy = PolicyConfig(
        supported_threshold=0.65,
        weak_threshold=0.35,
        contradiction_penalty=0.35,
        action=PolicyAction.ABSTAIN,
    )

    print("\n--- 1. Evaluating Baseline (No Guardrail) ---")
    base_metrics, _ = evaluate_baseline(examples)
    print(f"Baseline - Hallucination Rate: {base_metrics.hallucination_rate:.2%}, Answer Rate: {base_metrics.answer_rate:.2%}")

    print("\n--- 2. Evaluating Single-Tier: Cheap Model (cross-encoder/nli-deberta-v3-small) ---")
    cheap_verifier = CrossEncoderVerifier(model_name=CrossEncoderVerifier.CHEAP_MODEL)
    cheap_metrics, _ = evaluate_verifier(cheap_verifier, examples, policy, is_cascade=False, action="abstain")
    print(f"Cheap - F1: {cheap_metrics.claim_f1:.3f}, Hallucination Rate: {cheap_metrics.hallucination_rate:.2%}, "
          f"Answer Rate: {cheap_metrics.answer_rate:.2%}, Latency: {cheap_metrics.mean_latency_ms:.1f}ms")

    print("\n--- 3. Evaluating Single-Tier: Strong Model (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) ---")
    strong_verifier = CrossEncoderVerifier(model_name=CrossEncoderVerifier.STRONG_MODEL)
    strong_metrics, _ = evaluate_verifier(strong_verifier, examples, policy, is_cascade=False, action="abstain")
    print(f"Strong - F1: {strong_metrics.claim_f1:.3f}, Hallucination Rate: {strong_metrics.hallucination_rate:.2%}, "
          f"Answer Rate: {strong_metrics.answer_rate:.2%}, Latency: {strong_metrics.mean_latency_ms:.1f}ms")

    print("\n--- 4. Evaluating Cascade Verifier (Cheap -> Strong on Uncertainty) ---")
    cascade_verifier = CascadeVerifier(cheap_verifier=cheap_verifier, strong_verifier=strong_verifier)
    cascade_metrics, _ = evaluate_verifier(cascade_verifier, examples, policy, is_cascade=True, action="abstain")
    print(f"Cascade - F1: {cascade_metrics.claim_f1:.3f}, Hallucination Rate: {cascade_metrics.hallucination_rate:.2%}, "
          f"Answer Rate: {cascade_metrics.answer_rate:.2%}, Escalated: {cascade_metrics.escalation_rate:.2%}, "
          f"Latency: {cascade_metrics.mean_latency_ms:.1f}ms")

    # Generate Risk-Coverage plot
    coverage_risk_points = [
        (base_metrics.answer_rate, base_metrics.hallucination_rate, "Baseline"),
        (cheap_metrics.answer_rate, cheap_metrics.hallucination_rate, "Cheap (Small)"),
        (strong_metrics.answer_rate, strong_metrics.hallucination_rate, "Strong (Base)"),
        (cascade_metrics.answer_rate, cascade_metrics.hallucination_rate, "Cascade"),
    ]
    plot_path = str(out_path / "risk_coverage.png")
    plot_risk_coverage(coverage_risk_points, plot_path)
    print(f"\nGenerated risk-coverage plot: {plot_path}")

    # Generate Markdown Report
    report_content = f"""# groundcheck Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: `{dataset_name}` ({len(examples)} examples, seed=42)

## Executive Summary

We evaluated `groundcheck` across three operational configurations against an ungoverned RAG baseline:
1. **Baseline**: Pass all answers without verification.
2. **Cheap Tier**: Single cross-encoder `cross-encoder/nli-deberta-v3-small` (CPU).
3. **Strong Tier**: Single cross-encoder `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (CPU).
4. **Cascade Tier**: Two-stage execution where `cheap` runs on all claims, and claims falling in uncertainty band `[0.35, 0.75]` escalate to `strong`.

## Quantitative Results

| Configuration | Claim Precision | Claim Recall | Claim F1 | Answer Rate | Final Hallucination Rate | Mean Latency (ms) | Escalation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No Guardrail)** | 0.0% | 0.0% | 0.000 | {base_metrics.answer_rate * 100:.1f}% | {base_metrics.hallucination_rate * 100:.1f}% | {base_metrics.mean_latency_ms:.1f} ms | 0.0% |
| **Cheap Tier (DeBERTa-v3-small)** | {cheap_metrics.claim_precision * 100:.1f}% | {cheap_metrics.claim_recall * 100:.1f}% | {cheap_metrics.claim_f1:.3f} | {cheap_metrics.answer_rate * 100:.1f}% | {cheap_metrics.hallucination_rate * 100:.1f}% | {cheap_metrics.mean_latency_ms:.1f} ms | 0.0% |
| **Strong Tier (DeBERTa-v3-base)** | {strong_metrics.claim_precision * 100:.1f}% | {strong_metrics.claim_recall * 100:.1f}% | {strong_metrics.claim_f1:.3f} | {strong_metrics.answer_rate * 100:.1f}% | {strong_metrics.hallucination_rate * 100:.1f}% | {strong_metrics.mean_latency_ms:.1f} ms | 0.0% |
| **Cascade (Cheap &rarr; Strong)** | {cascade_metrics.claim_precision * 100:.1f}% | {cascade_metrics.claim_recall * 100:.1f}% | {cascade_metrics.claim_f1:.3f} | {cascade_metrics.answer_rate * 100:.1f}% | {cascade_metrics.hallucination_rate * 100:.1f}% | {cascade_metrics.mean_latency_ms:.1f} ms | {cascade_metrics.escalation_rate * 100:.1f}% |

## Key Findings

1. **Hallucination Suppression**: Groundcheck suppresses ungrounded hallucination from **{base_metrics.hallucination_rate * 100:.1f}%** down to **{cascade_metrics.hallucination_rate * 100:.1f}%**.
2. **Cascade Efficiency**: The cascade architecture achieves within **{(strong_metrics.claim_f1 - cascade_metrics.claim_f1) * 100:.1f}%** F1 of the strong tier while escalating only **{cascade_metrics.escalation_rate * 100:.1f}%** of claims.
3. **Latency Profile on CPU**: All inference runs purely on CPU with standard HuggingFace cross-encoders.

## Risk-Coverage Curve

![Risk-Coverage Curve](risk_coverage.png)
"""
    report_file = out_path / "report.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"Generated report: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mini")
    parser.add_argument("--max-examples", type=int, default=60)
    args = parser.parse_args()
    run_evaluation(dataset_name=args.dataset, max_examples=args.max_examples)
