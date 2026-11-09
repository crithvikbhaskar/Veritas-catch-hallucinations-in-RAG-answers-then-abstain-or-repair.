"""Metrics computation for groundcheck evaluation."""

from typing import Any

from pydantic import BaseModel


class EvalMetrics(BaseModel):
    total_examples: int
    total_claims: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    claim_precision: float = 0.0
    claim_recall: float = 0.0
    claim_f1: float = 0.0
    claim_accuracy: float = 0.0
    answer_rate: float = 0.0
    hallucination_rate: float = 0.0
    mean_latency_ms: float = 0.0
    escalation_rate: float = 0.0


def compute_metrics(
    examples: list[Any],
    predictions: list[dict[str, Any]],
    mean_latency_ms: float = 0.0,
    escalation_rate: float = 0.0,
    action: str = "abstain",
) -> EvalMetrics:
    """Compute claim-level PRF1 and answer-level hallucination/answer rates."""
    tp = fp = fn = tn = 0
    total_claims = 0

    answered_count = 0
    hallucinated_answers = 0

    for ex, pred in zip(examples, predictions):
        # Claim-level evaluation
        gt_claims = {c["text"].strip().lower(): c.get("label", "SUPPORTED") for c in ex.ground_truth_claims}
        pred_claims = pred.get("claims", [])
        
        for pc in pred_claims:
            total_claims += 1
            pred_unsupported = pc.verdict.value == "UNSUPPORTED"
            # Match predicted claim against closest ground truth claim
            norm_text = pc.claim.text.strip().lower()
            gt_label = "SUPPORTED"
            for gt_text, label in gt_claims.items():
                if gt_text in norm_text or norm_text in gt_text:
                    gt_label = label
                    break

            gt_unsupported = (gt_label == "UNSUPPORTED")

            if pred_unsupported and gt_unsupported:
                tp += 1
            elif pred_unsupported and not gt_unsupported:
                fp += 1
            elif not pred_unsupported and gt_unsupported:
                fn += 1
            else:
                tn += 1

        # Answer-level evaluation
        passed = pred.get("passed", True)
        repaired_answer = pred.get("repaired_answer")

        if action == "abstain":
            if passed:
                answered_count += 1
                if ex.has_hallucination:
                    hallucinated_answers += 1
        elif action == "repair":
            final_text = repaired_answer if (not passed and repaired_answer is not None) else ex.answer
            if final_text and final_text.strip():
                answered_count += 1
                # Check if final repaired text still contains any known unsupported claims
                still_hallucinated = any(
                    c["text"].strip().lower() in final_text.lower()
                    for c in ex.ground_truth_claims
                    if c.get("label") == "UNSUPPORTED"
                )
                if still_hallucinated:
                    hallucinated_answers += 1
        else:  # baseline / pass
            answered_count += 1
            if ex.has_hallucination:
                hallucinated_answers += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    ans_rate = answered_count / len(examples) if examples else 0.0
    hallu_rate = hallucinated_answers / answered_count if answered_count > 0 else 0.0

    return EvalMetrics(
        total_examples=len(examples),
        total_claims=total_claims,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        claim_precision=precision,
        claim_recall=recall,
        claim_f1=f1,
        claim_accuracy=accuracy,
        answer_rate=ans_rate,
        hallucination_rate=hallu_rate,
        mean_latency_ms=mean_latency_ms,
        escalation_rate=escalation_rate,
    )
