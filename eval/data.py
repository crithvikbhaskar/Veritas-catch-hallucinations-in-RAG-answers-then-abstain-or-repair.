"""Dataset loader for RAG evaluation benchmarks."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class EvalExample(BaseModel):
    id: str
    question: str | None = ""
    chunks: list[str]
    answer: str
    ground_truth_claims: list[dict[str, Any]]
    has_hallucination: bool


def load_mini_dataset(file_path: str | None = None) -> list[EvalExample]:
    """Load the curated 60-example evaluation benchmark from local JSONL."""
    if file_path is None:
        file_path = str(Path(__file__).parent / "data" / "mini.jsonl")

    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(
                EvalExample(
                    id=data["id"],
                    question=data.get("question", ""),
                    chunks=data["chunks"],
                    answer=data["answer"],
                    ground_truth_claims=data.get("claims", []),
                    has_hallucination=data.get(
                        "has_hallucination",
                        any(c.get("label") == "UNSUPPORTED" for c in data.get("claims", [])),
                    ),
                )
            )
    return examples


def load_eval_dataset(
    dataset_name: str = "mini", max_examples: int = 60, seed: int = 42
) -> list[EvalExample]:
    """Load evaluation benchmark (either 'mini' curated dataset or HF dataset with fallback)."""
    if dataset_name == "mini":
        return load_mini_dataset()[:max_examples]

    # Attempt HuggingFace dataset loading if requested
    try:
        import numpy as np
        from datasets import load_dataset

        np.random.seed(seed)
        # Attempt to load RAGTruth or HaluEval
        if dataset_name == "ragtruth":
            ds = load_dataset("jon-tow/ragtruth", split="train", streaming=True)
        elif dataset_name == "halueval":
            ds = load_dataset("pandalab/HaluEval", "qa", split="data", streaming=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        examples = []
        for row in ds:
            # Transform to standard schema
            chunks = row.get("source_passage") or row.get("knowledge") or [row.get("context", "")]
            if isinstance(chunks, str):
                chunks = [chunks]
            answer = row.get("response") or row.get("output") or row.get("hallucinated_answer", "")
            is_hallu = bool(row.get("is_hallucination", row.get("label", 0)))
            examples.append(
                EvalExample(
                    id=f"hf_{len(examples)}",
                    question=row.get("question", ""),
                    chunks=chunks,
                    answer=answer,
                    ground_truth_claims=[],
                    has_hallucination=is_hallu,
                )
            )
            if len(examples) >= max_examples:
                break
        return examples
    except Exception as e:  # noqa: BLE001
        # Fallback to local mini dataset per specification
        print(f"Dataset download/streaming fallback ({e}). Using curated mini dataset.")
        return load_mini_dataset()[:max_examples]
