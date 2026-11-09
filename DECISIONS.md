# Architectural & Design Decisions

This log records all pragmatic technical decisions made during the autonomous build of `groundcheck`.

## 1. Initial Setup & Environment
- **Decision**: Use `uv` for lightning-fast virtual environment and dependency management.
- **Rationale**: Python 3.11+, reliable dependency resolution, zero overhead.
## 4. Evaluation Benchmark Selection
- **Decision**: Curated a deterministic, 60-example multi-domain RAG evaluation dataset (`eval/data/mini.jsonl`) spanning science, geography, tech, history, and medicine, with explicit per-claim ground-truth labels and 50% baseline hallucination rate.
- **Rationale**: Meets the anti-drift protocol requirement for CPU execution within 10 minutes without depending on external HF download stability or rate limits, while keeping dynamic HF fallback option available.
- **Date**: 2026-11-09

## 5. Non-Destructive Repair Strategy
- **Decision**: Implemented claim repair via reverse-offset character slicing and whitespace normalization, generating standard unified diffs via `difflib`.
- **Rationale**: Preserves formatting of supported sentences while completely excising unsupported assertions and providing clear provenance via diffs.
- **Date**: 2026-11-09

## 6. Cascade Uncertainty Band
- **Decision**: Configured the uncertainty band at `[0.35, 0.75]`. Claims with cheap-tier entailment outside this range terminate immediately; only borderline claims escalate to the strong tier.
- **Rationale**: Balances CPU execution speed with high precision/recall on ambiguous or adversarial assertions.
- **Date**: 2026-11-09

## 7. Next.js Single-Page Demo Architecture
- **Decision**: Built a client-rendered Next.js + Tailwind UI featuring live threshold sliders, sentence-level color highlighting with evidence tooltips, and an interactive diff viewer.
- **Rationale**: Provides intuitive visual feedback matching the three policy actions (pass, abstain, repair).
- **Date**: 2026-11-09


## 2. Claim Splitting Rules & Offsets
- **Decision**: Implemented regex-based sentence and semicolon-clause splitting with offset tracking and abbreviation suppression (titles like "Dr.", "Mr." suppressed; abbreviations followed by uppercase treated as sentence boundary).
- **Rationale**: Preserves exact character offsets for non-destructive diff generation and frontend span highlighting.
## 4. Evaluation Benchmark Selection
- **Decision**: Curated a deterministic, 60-example multi-domain RAG evaluation dataset (`eval/data/mini.jsonl`) spanning science, geography, tech, history, and medicine, with explicit per-claim ground-truth labels and 50% baseline hallucination rate.
- **Rationale**: Meets the anti-drift protocol requirement for CPU execution within 10 minutes without depending on external HF download stability or rate limits, while keeping dynamic HF fallback option available.
- **Date**: 2026-11-09

## 5. Non-Destructive Repair Strategy
- **Decision**: Implemented claim repair via reverse-offset character slicing and whitespace normalization, generating standard unified diffs via `difflib`.
- **Rationale**: Preserves formatting of supported sentences while completely excising unsupported assertions and providing clear provenance via diffs.
- **Date**: 2026-11-09

## 6. Cascade Uncertainty Band
- **Decision**: Configured the uncertainty band at `[0.35, 0.75]`. Claims with cheap-tier entailment outside this range terminate immediately; only borderline claims escalate to the strong tier.
- **Rationale**: Balances CPU execution speed with high precision/recall on ambiguous or adversarial assertions.
- **Date**: 2026-11-09

## 7. Next.js Single-Page Demo Architecture
- **Decision**: Built a client-rendered Next.js + Tailwind UI featuring live threshold sliders, sentence-level color highlighting with evidence tooltips, and an interactive diff viewer.
- **Rationale**: Provides intuitive visual feedback matching the three policy actions (pass, abstain, repair).
- **Date**: 2026-11-09


## 3. NLI Label Mapping and Aggregation
- **Decision**: Dynamically inspect HuggingFace `config.id2label` to handle varied label conventions (entailment, neutral, contradiction). For multiple source chunks, use max-entailment pooling, with contradiction penalty override when contradiction dominates.
- **Rationale**: Robust across cross-encoder models without hardcoded assumptions on label ordering.
## 4. Evaluation Benchmark Selection
- **Decision**: Curated a deterministic, 60-example multi-domain RAG evaluation dataset (`eval/data/mini.jsonl`) spanning science, geography, tech, history, and medicine, with explicit per-claim ground-truth labels and 50% baseline hallucination rate.
- **Rationale**: Meets the anti-drift protocol requirement for CPU execution within 10 minutes without depending on external HF download stability or rate limits, while keeping dynamic HF fallback option available.
- **Date**: 2026-11-09

## 5. Non-Destructive Repair Strategy
- **Decision**: Implemented claim repair via reverse-offset character slicing and whitespace normalization, generating standard unified diffs via `difflib`.
- **Rationale**: Preserves formatting of supported sentences while completely excising unsupported assertions and providing clear provenance via diffs.
- **Date**: 2026-11-09

## 6. Cascade Uncertainty Band
- **Decision**: Configured the uncertainty band at `[0.35, 0.75]`. Claims with cheap-tier entailment outside this range terminate immediately; only borderline claims escalate to the strong tier.
- **Rationale**: Balances CPU execution speed with high precision/recall on ambiguous or adversarial assertions.
- **Date**: 2026-11-09

## 7. Next.js Single-Page Demo Architecture
- **Decision**: Built a client-rendered Next.js + Tailwind UI featuring live threshold sliders, sentence-level color highlighting with evidence tooltips, and an interactive diff viewer.
- **Rationale**: Provides intuitive visual feedback matching the three policy actions (pass, abstain, repair).
- **Date**: 2026-11-09


