# groundcheck: Design Decisions & Architectural Rationale

This document details the architectural decisions, trade-offs, and design principles behind `groundcheck`.

---

## 1. Core Problem: RAG Faithfulness & The Need for Verification

Retrieval-Augmented Generation (RAG) is prone to two primary failure modes:
1. **Extrinsic Hallucination**: The LLM introduces facts not present anywhere in the retrieved context chunks.
2. **Intrinsic Contradiction**: The LLM directly contradicts statements made in the retrieved chunks.

Existing solutions often rely on generative LLM-as-a-judge prompting (e.g., GPT-4 or Claude inspecting the answer). This approach has severe drawbacks for production systems:
- **High latency**: Adding 1.5–4.0 seconds per query.
- **Cost**: Doubling token costs for every generated response.
- **Evaluation Hallucinations**: Generative judges themselves suffer from probabilistic inconsistencies.
- **Opacity**: LLMs return freeform explanations without deterministic calibrated thresholds.

`groundcheck` was engineered as a **deterministic, local, zero-API NLI middleware** that operates directly on CPU in tens of milliseconds per sentence.

---

## 2. NLI Cross-Encoder Selection (CPU-Only, Under 1GB)

### Cheap Tier: `cross-encoder/nli-deberta-v3-small` (~140MB)
- **Architecture**: DeBERTa-v3 with disentangled attention.
- **Training**: Fine-tuned on MNLI.
- **Role**: High-throughput first-pass screener. Evaluates claims in ~10–25ms on CPU. Highly accurate for obvious entailments and clear contradictions.

### Strong Tier: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (~500MB)
- **Architecture**: DeBERTa-v3 base model.
- **Training**: Trained on MNLI, FEVER (fact extraction & verification), and ANLI (adversarial NLI).
- **Role**: High-capacity verifier for nuanced, syntactically inverted, or borderline claims.

### Optional LLMJudge Interface
To preserve modularity, `groundcheck.verify.LLMJudge` is provided as an abstract interface. However, out-of-the-box operation strictly requires no external APIs or network calls for inference.

---

## 3. Claim Splitting & Offset Preservation

A crucial design requirement for both UI highlighting and text repair is **non-destructive span tracking**:
- Standard sentence splitters (like naive regex or splitlines) lose character positions or produce inaccurate boundaries around abbreviations (e.g., `Dr.`, `U.S.`, `3.14 PM`).
- `ClaimSplitter` maintains exact `(start_char, end_char)` offsets relative to the original string.
- Semicolons are supported as atomic clause delimiters to verify compound statements independently without conflating valid and invalid assertions.

---

## 4. Aggregation Over Chunks & Policy Engine

In RAG, an answer claim only needs to be supported by **at least one** source chunk:
1. **Max-Pooling**: For claim $C$ and context chunks $\{K_1, \dots, K_n\}$, we compute $s_i = \text{NLI}(K_i, C)$ and take the chunk with maximum entailment:
   $$\text{best\_chunk} = \arg\max_{i} \text{entailment}(K_i, C)$$
2. **Contradiction Penalty Override**: If any chunk directly contradicts the claim with high confidence ($\text{contradiction} \ge 0.30$ and $\text{contradiction} > \text{entailment}$), the claim is demoted to `UNSUPPORTED`.
3. **Verdict Categories**:
   - `SUPPORTED`: $\text{entailment} \ge \tau_{\text{supported}}$ (default 0.70)
   - `WEAK`: $\tau_{\text{weak}} \le \text{entailment} < \tau_{\text{supported}}$ (default 0.40–0.70)
   - `UNSUPPORTED`: $\text{entailment} < \tau_{\text{weak}}$ or contradiction override

---

## 5. Policy Actions: Pass, Abstain, and Repair

When unsupported claims are detected, applications have different tolerances:
- **`pass`**: Non-blocking audit mode. Tags claims with verdicts and returns telemetry without altering output.
- **`abstain`**: Refuses to answer when evidence is missing. Returns a transparent refusal listing the ungrounded claims.
- **`repair`**: Surgically deletes unsupported claims from the answer text using offset ranges. Emits a clean unified diff (`difflib`) and the repaired string, allowing partial answers when evidence partially exists.

---

## 6. Two-Tier Uncertainty Cascade

Running the strong 500MB model on every claim increases CPU latency. However, running only the cheap model risks borderline false positives.

`groundcheck.cascade.CascadeVerifier` introduces an uncertainty band $[\tau_{\text{lower}}, \tau_{\text{upper}}] = [0.35, 0.75]$:
1. The **Cheap Model** evaluates all claims in batch.
2. Claims with $\text{entailment} > 0.75$ (confidently supported) or $\text{entailment} < 0.35$ (confidently unsupported) terminate immediately.
3. Only claims in the uncertainty band $[0.35, 0.75]$ escalate to the **Strong Model**.
4. The system logs the latency delta and the exact `escalation_rate` fraction.

---

## 7. Trade-offs and Limitations

| Choice | Advantage | Trade-off |
| :--- | :--- | :--- |
| **CPU Inference** | Zero GPU requirement, zero API bills, private execution | Single-thread inference takes ~50–150ms per sentence |
| **Sentence-Level Splitting** | Granular attribution, pinpoint diffs | Compound sentences without semicolons are verified as a single unit |
| **Heuristic Semicolon Clauses** | Breaks compound assertions | May occasionally separate dependent clauses |
| **Local NLI Weights** | Deterministic outputs, no drift | Initial download requires ~650MB disk cache |

