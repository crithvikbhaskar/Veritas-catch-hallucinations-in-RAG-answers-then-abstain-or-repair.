"""NLI Cross-Encoder Verification Engine for groundcheck."""

from abc import ABC, abstractmethod

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from groundcheck.types import (
    ChunkScore,
    Claim,
    ClaimVerdict,
    ClaimVerification,
    PolicyConfig,
)


class BaseVerifier(ABC):
    """Abstract interface for claim verifiers."""

    @abstractmethod
    def verify_claim(
        self, claim: Claim, chunks: list[str], policy: PolicyConfig
    ) -> ClaimVerification:
        """Verify a single claim against a list of source chunks."""

    @abstractmethod
    def verify_batch(
        self, claims: list[Claim], chunks: list[str], policy: PolicyConfig
    ) -> list[ClaimVerification]:
        """Verify multiple claims against source chunks."""


class LLMJudge(ABC):
    """Optional interface for external LLM-as-a-judge verifiers (zero API requirement by default)."""

    @abstractmethod
    def judge(self, claim: str, chunk: str) -> dict[str, float]:
        """Return entailment, neutral, contradiction scores in [0.0, 1.0]."""


class CrossEncoderVerifier(BaseVerifier):
    """Local HuggingFace NLI Cross-Encoder verifier running on CPU."""

    CHEAP_MODEL = "cross-encoder/nli-deberta-v3-small"
    STRONG_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def __init__(
        self,
        model_name: str = CHEAP_MODEL,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._label_map: dict[str, int] = {}

    def _load_model(self):
        """Lazy load model and tokenizer on first use to ensure fast module imports."""
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            config = AutoConfig.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, config=config
            )
            self._model.to(self.device)
            self._model.eval()

            # Dynamically map model label indices to 'entailment', 'contradiction', 'neutral'
            id2label = config.id2label or {}
            self._label_map = {}
            for idx, label_name in id2label.items():
                lname = str(label_name).lower()
                if "entail" in lname:
                    self._label_map["entailment"] = int(idx)
                elif "contra" in lname:
                    self._label_map["contradiction"] = int(idx)
                elif "neut" in lname:
                    self._label_map["neutral"] = int(idx)

            # Fallbacks if label names aren't standard
            if "entailment" not in self._label_map:
                self._label_map["entailment"] = 1
            if "contradiction" not in self._label_map:
                self._label_map["contradiction"] = 0
            if "neutral" not in self._label_map:
                self._label_map["neutral"] = 2

    def score_pairs(self, premises: list[str], hypotheses: list[str]) -> list[dict[str, float]]:
        """Score (premise, hypothesis) pairs returning probabilities for each pair."""
        if not premises:
            return []

        self._load_model()
        results: list[dict[str, float]] = []

        ent_idx = self._label_map["entailment"]
        contra_idx = self._label_map["contradiction"]
        neut_idx = self._label_map["neutral"]

        # Process in batches
        for i in range(0, len(premises), self.batch_size):
            batch_prem = premises[i : i + self.batch_size]
            batch_hypo = hypotheses[i : i + self.batch_size]

            inputs = self._tokenizer(
                batch_prem,
                batch_hypo,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for row in probs:
                results.append(
                    {
                        "entailment": float(row[ent_idx]),
                        "contradiction": float(row[contra_idx]),
                        "neutral": float(row[neut_idx]),
                    }
                )

        return results

    def verify_claim(
        self, claim: Claim, chunks: list[str], policy: PolicyConfig
    ) -> ClaimVerification:
        """Verify single claim against chunks by taking max entailment over chunks."""
        batch_results = self.verify_batch([claim], chunks, policy)
        return batch_results[0]

    def verify_batch(
        self, claims: list[Claim], chunks: list[str], policy: PolicyConfig
    ) -> list[ClaimVerification]:
        """Verify multiple claims against source chunks with max-pooling over chunks."""
        if not claims or not chunks:
            return []

        # Prepare all pairs (premise=chunk, hypothesis=claim.text)
        pair_premises: list[str] = []
        pair_hypotheses: list[str] = []
        pair_claim_indices: list[int] = []
        pair_chunk_indices: list[int] = []

        for c_idx, claim in enumerate(claims):
            for ch_idx, chunk in enumerate(chunks):
                pair_premises.append(chunk)
                pair_hypotheses.append(claim.text)
                pair_claim_indices.append(c_idx)
                pair_chunk_indices.append(ch_idx)

        # Batch score all pairs
        raw_scores = self.score_pairs(pair_premises, pair_hypotheses)

        # Aggregate per claim: organize scores by claim
        claim_chunk_scores: list[list[ChunkScore]] = [[] for _ in range(len(claims))]
        for idx, score in enumerate(raw_scores):
            c_idx = pair_claim_indices[idx]
            ch_idx = pair_chunk_indices[idx]
            claim_chunk_scores[c_idx].append(
                ChunkScore(
                    chunk_idx=ch_idx,
                    chunk_text=chunks[ch_idx],
                    entailment=score["entailment"],
                    neutral=score["neutral"],
                    contradiction=score["contradiction"],
                )
            )

        # Apply max-entailment aggregation and policy verdict
        verifications: list[ClaimVerification] = []
        for c_idx, claim in enumerate(claims):
            scores_for_claim = claim_chunk_scores[c_idx]
            if not scores_for_claim:
                verifications.append(
                    ClaimVerification(
                        claim=claim,
                        verdict=ClaimVerdict.UNSUPPORTED,
                        model_name=self.model_name,
                    )
                )
                continue

            # Best chunk has highest entailment
            best_chunk = max(scores_for_claim, key=lambda s: s.entailment)
            
            # Policy evaluation
            verdict = self._evaluate_verdict(
                entailment=best_chunk.entailment,
                contradiction=best_chunk.contradiction,
                policy=policy,
            )

            verifications.append(
                ClaimVerification(
                    claim=claim,
                    verdict=verdict,
                    best_chunk_idx=best_chunk.chunk_idx,
                    best_chunk_score=best_chunk.entailment,
                    entailment=best_chunk.entailment,
                    contradiction=best_chunk.contradiction,
                    neutral=best_chunk.neutral,
                    model_name=self.model_name,
                    chunk_scores=scores_for_claim,
                )
            )

        return verifications

    @staticmethod
    def _evaluate_verdict(
        entailment: float, contradiction: float, policy: PolicyConfig
    ) -> ClaimVerdict:
        """Determine ClaimVerdict given scores and policy thresholds."""
        # Contradiction override: high contradiction forces UNSUPPORTED
        if contradiction >= policy.contradiction_penalty and contradiction > entailment:
            return ClaimVerdict.UNSUPPORTED

        if entailment >= policy.supported_threshold:
            return ClaimVerdict.SUPPORTED
        elif entailment >= policy.weak_threshold:
            return ClaimVerdict.WEAK
        else:
            return ClaimVerdict.UNSUPPORTED
