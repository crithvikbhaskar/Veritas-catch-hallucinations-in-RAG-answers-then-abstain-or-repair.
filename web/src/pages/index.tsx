import React, { useState } from "react";
import Head from "next/head";
import {
  ShieldAlert,
  ShieldCheck,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Zap,
  ArrowRight,
  GitCompare,
  Sliders,
  Sparkles,
  Info,
} from "lucide-react";

interface ChunkScore {
  chunk_idx: number;
  chunk_text: string;
  entailment: number;
  neutral: number;
  contradiction: number;
}

interface ClaimVerification {
  claim: {
    id: string;
    text: string;
    start_char: number;
    end_char: number;
    sentence_idx: number;
  };
  verdict: "SUPPORTED" | "WEAK" | "UNSUPPORTED";
  best_chunk_idx: number | null;
  best_chunk_score: number | null;
  entailment: number;
  contradiction: number;
  neutral: number;
  model_name: string;
  escalated: boolean;
  chunk_scores: ChunkScore[];
}

interface VerifyResponse {
  status: string;
  claims: ClaimVerification[];
  verdict_counts: Record<string, number>;
  action_taken: "pass" | "abstain" | "repair";
  passed: boolean;
  final_answer: string;
  repaired_answer?: string | null;
  diff?: string | null;
  unsupported_claims: string[];
  latency_ms: number;
  escalation_rate: number;
}

const DEFAULT_SAMPLE = {
  question: "Where is the Eiffel Tower and who designed it?",
  chunks: [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. Constructed from 1887 to 1889, it was named after engineer Gustave Eiffel.",
    "Gustave Eiffel's company designed and built the tower as the entrance arch to the 1889 World's Fair.",
  ],
  answer:
    "The Eiffel Tower is located on the Champ de Mars in Paris, France. It was designed by James Watt in London during the Industrial Revolution. It serves as an iconic cultural symbol.",
};

export default function Home() {
  const [question, setQuestion] = useState(DEFAULT_SAMPLE.question);
  const [chunksText, setChunksText] = useState(DEFAULT_SAMPLE.chunks.join("\n\n---\n\n"));
  const [answer, setAnswer] = useState(DEFAULT_SAMPLE.answer);

  // Policy Settings
  const [action, setAction] = useState<"pass" | "abstain" | "repair">("repair");
  const [useCascade, setUseCascade] = useState(true);
  const [supportedThreshold, setSupportedThreshold] = useState(0.70);
  const [weakThreshold, setWeakThreshold] = useState(0.40);
  const [contradictionPenalty, setContradictionPenalty] = useState(0.30);
  const [abstainOnWeak, setAbstainOnWeak] = useState(false);

  // Results State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedClaimIdx, setSelectedClaimIdx] = useState<number | null>(null);

  const handleLoadSample = async () => {
    try {
      const res = await fetch("http://localhost:8000/sample");
      if (res.ok) {
        const data = await res.json();
        setQuestion(data.question || "");
        setChunksText(data.chunks.join("\n\n---\n\n"));
        setAnswer(data.answer);
        if (data.policy?.action) setAction(data.policy.action);
        return;
      }
    } catch {
      // Fallback to embedded default sample
    }
    setQuestion(DEFAULT_SAMPLE.question);
    setChunksText(DEFAULT_SAMPLE.chunks.join("\n\n---\n\n"));
    setAnswer(DEFAULT_SAMPLE.answer);
  };

  const handleVerify = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedClaimIdx(null);

    const chunks = chunksText
      .split("\n\n---\n\n")
      .map((c) => c.trim())
      .filter((c) => c.length > 0);

    if (chunks.length === 0) {
      setError("Please provide at least one source chunk.");
      setLoading(false);
      return;
    }
    if (!answer.trim()) {
      setError("Please provide an answer to verify.");
      setLoading(false);
      return;
    }

    const payload = {
      question: question.trim(),
      chunks,
      answer: answer.trim(),
      action,
      use_cascade: useCascade,
      policy: {
        supported_threshold: supportedThreshold,
        weak_threshold: weakThreshold,
        contradiction_penalty: contradictionPenalty,
        action,
        abstain_on_weak: abstainOnWeak,
      },
    };

    try {
      const res = await fetch("http://localhost:8000/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server responded with status ${res.status}`);
      }

      const data: VerifyResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to connect to groundcheck backend.");
    } finally {
      setLoading(false);
    }
  };

  const getVerdictBadge = (verdict: string) => {
    switch (verdict) {
      case "SUPPORTED":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-300">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-600" /> SUPPORTED
          </span>
        );
      case "WEAK":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-300">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-600" /> WEAK
          </span>
        );
      case "UNSUPPORTED":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-300">
            <XCircle className="w-3.5 h-3.5 text-rose-600" /> UNSUPPORTED
          </span>
        );
      default:
        return null;
    }
  };

  const getHighlightClass = (verdict: string) => {
    switch (verdict) {
      case "SUPPORTED":
        return "bg-emerald-100 border-b-2 border-emerald-500 hover:bg-emerald-200 transition-colors cursor-pointer";
      case "WEAK":
        return "bg-amber-100 border-b-2 border-amber-500 hover:bg-amber-200 transition-colors cursor-pointer";
      case "UNSUPPORTED":
        return "bg-rose-100 border-b-2 border-rose-500 hover:bg-rose-200 transition-colors cursor-pointer text-rose-950 font-medium";
      default:
        return "";
    }
  };

  return (
    <>
      <Head>
        <title>groundcheck | RAG Hallucination & Grounding Guardrail</title>
        <meta
          name="description"
          content="Drop-in middleware checking every claim against source chunks, abstaining or repairing when evidence is lacking."
        />
      </Head>

      <main className="min-h-screen bg-slate-50 text-slate-900 pb-20">
        {/* Top Header */}
        <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-black text-xl shadow-md">
                G
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl font-bold tracking-tight text-slate-900">groundcheck</h1>
                  <span className="text-xs font-semibold bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded border border-indigo-200">
                    NLI Guardrail
                  </span>
                </div>
                <p className="text-xs text-slate-500 hidden sm:block">
                  Per-claim source chunk verification &amp; non-destructive repair
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={handleLoadSample}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors"
              >
                <Sparkles className="w-3.5 h-3.5 text-indigo-600" /> Load Sample
              </button>
            </div>
          </div>
        </header>

        {/* Content Container */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
          {/* Settings Bar */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm mb-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-4 flex-wrap">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                  <Sliders className="w-3.5 h-3.5" /> Policy Action:
                </span>
                <div className="inline-flex bg-slate-100 p-1 rounded-lg">
                  {(["pass", "abstain", "repair"] as const).map((act) => (
                    <button
                      key={act}
                      onClick={() => setAction(act)}
                      className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                        action === act
                          ? "bg-white text-indigo-700 shadow-sm"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      {act.toUpperCase()}
                    </button>
                  ))}
                </div>

                <div className="h-4 w-px bg-slate-200 mx-1 hidden sm:block" />

                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={useCascade}
                    onChange={(e) => setUseCascade(e.target.checked)}
                    className="w-4 h-4 rounded text-indigo-600 focus:ring-indigo-500"
                  />
                  <span className="text-xs font-medium text-slate-700 flex items-center gap-1">
                    <Zap className="w-3.5 h-3.5 text-amber-500 fill-amber-500" /> Cascade Model (Cheap &rarr; Strong)
                  </span>
                </label>
              </div>

              {/* Threshold Sliders */}
              <div className="flex items-center gap-6 flex-wrap text-xs">
                <div className="flex items-center gap-2">
                  <span className="text-slate-600 font-medium">Supported &ge;</span>
                  <input
                    type="range"
                    min="0.50"
                    max="0.95"
                    step="0.05"
                    value={supportedThreshold}
                    onChange={(e) => setSupportedThreshold(parseFloat(e.target.value))}
                    className="w-20 accent-indigo-600"
                  />
                  <span className="font-mono font-bold text-slate-800">{supportedThreshold.toFixed(2)}</span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-slate-600 font-medium">Weak &ge;</span>
                  <input
                    type="range"
                    min="0.20"
                    max="0.60"
                    step="0.05"
                    value={weakThreshold}
                    onChange={(e) => setWeakThreshold(parseFloat(e.target.value))}
                    className="w-20 accent-amber-500"
                  />
                  <span className="font-mono font-bold text-slate-800">{weakThreshold.toFixed(2)}</span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-slate-600 font-medium">Contradiction Penalty</span>
                  <input
                    type="range"
                    min="0.10"
                    max="0.60"
                    step="0.05"
                    value={contradictionPenalty}
                    onChange={(e) => setContradictionPenalty(parseFloat(e.target.value))}
                    className="w-20 accent-rose-500"
                  />
                  <span className="font-mono font-bold text-slate-800">{contradictionPenalty.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Grid: Inputs vs Results */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Input Panels */}
            <div className="space-y-4">
              {/* Question */}
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-2">
                  User Question
                </label>
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="e.g. Where is the Eiffel Tower and who designed it?"
                  className="w-full text-sm px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>

              {/* Source Context Chunks */}
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-600">
                    Source Chunks (Evidence)
                  </label>
                  <span className="text-[11px] text-slate-400">Separate multiple chunks with: ---</span>
                </div>
                <textarea
                  rows={5}
                  value={chunksText}
                  onChange={(e) => setChunksText(e.target.value)}
                  placeholder="Paste retrieved passages here..."
                  className="w-full text-xs font-mono p-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 leading-relaxed"
                />
              </div>

              {/* Generated Answer to Verify */}
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-2">
                  RAG Generated Answer (To Check)
                </label>
                <textarea
                  rows={4}
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Generated response to check for hallucinations..."
                  className="w-full text-sm p-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 leading-relaxed"
                />

                <div className="mt-4 flex items-center justify-between">
                  <button
                    type="button"
                    onClick={handleVerify}
                    disabled={loading}
                    className="w-full inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm rounded-lg shadow-sm disabled:opacity-50 transition-all"
                  >
                    {loading ? (
                      <>
                        <span className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                        Running NLI Verification Engine...
                      </>
                    ) : (
                      <>
                        <ShieldCheck className="w-4 h-4" /> Verify Grounding
                      </>
                    )}
                  </button>
                </div>
              </div>

              {error && (
                <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-sm flex items-start gap-2">
                  <XCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
                  <div>
                    <strong className="font-semibold">Verification Error:</strong> {error}
                  </div>
                </div>
              )}
            </div>

            {/* Right: Results & Claim Highlights */}
            <div className="space-y-4">
              {!result && !loading && (
                <div className="bg-white border border-dashed border-slate-300 rounded-xl p-12 text-center text-slate-400">
                  <ShieldCheck className="w-12 h-12 mx-auto text-slate-300 mb-3" />
                  <h3 className="font-semibold text-slate-600 text-sm">No Verification Run Yet</h3>
                  <p className="text-xs mt-1">
                    Click &ldquo;Verify Grounding&rdquo; to test the answer claims against retrieved source chunks.
                  </p>
                </div>
              )}

              {loading && (
                <div className="bg-white border border-slate-200 rounded-xl p-12 text-center text-slate-600 shadow-sm animate-pulse">
                  <Zap className="w-10 h-10 mx-auto text-amber-500 mb-3 animate-bounce" />
                  <h3 className="font-bold text-slate-800 text-sm">Evaluating Claims with DeBERTa NLI</h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Splitting claims, pooling max entailment across chunks, evaluating policy...
                  </p>
                </div>
              )}

              {result && (
                <div className="space-y-4">
                  {/* Status Banner */}
                  <div
                    className={`p-4 rounded-xl border flex items-center justify-between ${
                      result.passed
                        ? "bg-emerald-50 border-emerald-200 text-emerald-900"
                        : "bg-rose-50 border-rose-200 text-rose-900"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      {result.passed ? (
                        <ShieldCheck className="w-7 h-7 text-emerald-600 shrink-0" />
                      ) : (
                        <ShieldAlert className="w-7 h-7 text-rose-600 shrink-0" />
                      )}
                      <div>
                        <div className="font-bold text-sm">
                          {result.passed ? "PASSED: All Claims Supported" : "FLAGGED: Unsupported Claims Detected"}
                        </div>
                        <div className="text-xs opacity-90 mt-0.5">
                          Action: <span className="font-semibold uppercase">{result.action_taken}</span> |{" "}
                          {result.verdict_counts["SUPPORTED"] || 0} Supported,{" "}
                          {result.verdict_counts["WEAK"] || 0} Weak,{" "}
                          {result.verdict_counts["UNSUPPORTED"] || 0} Unsupported
                        </div>
                      </div>
                    </div>

                    <div className="text-right text-xs">
                      <div className="font-mono font-bold">{result.latency_ms.toFixed(1)} ms</div>
                      {result.escalation_rate > 0 && (
                        <div className="text-amber-700 font-semibold text-[11px]">
                          Escalated: {(result.escalation_rate * 100).toFixed(0)}%
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Sentence-Level Highlight View */}
                  <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600">
                        Sentence-Level Claim Grounding
                      </h3>
                      <div className="flex items-center gap-2 text-[11px]">
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Supported
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Weak
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-2.5 rounded-full bg-rose-500" /> Unsupported
                        </span>
                      </div>
                    </div>

                    <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 text-sm leading-relaxed space-y-2">
                      {result.claims.map((cv, idx) => (
                        <span
                          key={cv.claim.id}
                          onClick={() => setSelectedClaimIdx(selectedClaimIdx === idx ? null : idx)}
                          className={`inline px-1 py-0.5 mx-0.5 rounded ${getHighlightClass(
                            cv.verdict
                          )} ${selectedClaimIdx === idx ? "ring-2 ring-indigo-600 font-semibold" : ""}`}
                          title={`Entailment: ${(cv.entailment * 100).toFixed(1)}% | Contradiction: ${(
                            cv.contradiction * 100
                          ).toFixed(1)}%`}
                        >
                          {cv.claim.text}{" "}
                        </span>
                      ))}
                    </div>

                    {/* Inspection of selected claim */}
                    {selectedClaimIdx !== null && result.claims[selectedClaimIdx] && (
                      <div className="mt-3 p-3 bg-indigo-50 border border-indigo-200 rounded-lg text-xs space-y-1">
                        <div className="font-semibold text-indigo-950 flex items-center justify-between">
                          <span>Claim #{selectedClaimIdx + 1} Evidence Breakdown</span>
                          {getVerdictBadge(result.claims[selectedClaimIdx].verdict)}
                        </div>
                        <div className="text-slate-700 font-mono text-[11px]">
                          &ldquo;{result.claims[selectedClaimIdx].claim.text}&rdquo;
                        </div>
                        <div className="flex gap-4 pt-1 text-slate-600">
                          <span>Entailment: <strong>{(result.claims[selectedClaimIdx].entailment * 100).toFixed(1)}%</strong></span>
                          <span>Contradiction: <strong>{(result.claims[selectedClaimIdx].contradiction * 100).toFixed(1)}%</strong></span>
                          <span>Model: <strong>{result.claims[selectedClaimIdx].model_name}</strong></span>
                        </div>
                        {result.claims[selectedClaimIdx].best_chunk_score !== null && (
                          <div className="text-slate-600 pt-1">
                            Best Chunk #{result.claims[selectedClaimIdx].best_chunk_idx}:{" "}
                            <span className="italic">
                              &ldquo;{result.claims[selectedClaimIdx].chunk_scores[result.claims[selectedClaimIdx].best_chunk_idx || 0]?.chunk_text.slice(0, 100)}...&rdquo;
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Diff View for Repair Action */}
                  {result.action_taken === "repair" && result.diff && (
                    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                      <div className="flex items-center gap-2 mb-3 text-xs font-bold uppercase tracking-wider text-slate-700">
                        <GitCompare className="w-4 h-4 text-indigo-600" /> Unified Repair Diff
                      </div>
                      <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre-wrap leading-relaxed">
                        {result.diff.split("\n").map((line, i) => {
                          let lineClass = "text-slate-300";
                          if (line.startsWith("-")) lineClass = "text-rose-400 bg-rose-950/40 px-1 rounded";
                          if (line.startsWith("+")) lineClass = "text-emerald-400 bg-emerald-950/40 px-1 rounded";
                          if (line.startsWith("@")) lineClass = "text-indigo-400 font-semibold";
                          return (
                            <div key={i} className={lineClass}>
                              {line}
                            </div>
                          );
                        })}
                      </pre>

                      <div className="mt-3">
                        <div className="text-xs font-bold text-slate-700 mb-1">Repaired Final Answer:</div>
                        <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-950 leading-relaxed">
                          {result.repaired_answer || "(All unsupported claims removed. Answer suppressed.)"}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Abstain View */}
                  {result.action_taken === "abstain" && !result.passed && (
                    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                      <div className="flex items-center gap-2 mb-2 text-xs font-bold uppercase tracking-wider text-rose-700">
                        <AlertTriangle className="w-4 h-4" /> Abstention Notice
                      </div>
                      <p className="text-xs text-slate-700 leading-relaxed">
                        The response was abstained because it contained the following unsupported claims:
                      </p>
                      <ul className="mt-2 space-y-1">
                        {result.unsupported_claims.map((claimText, i) => (
                          <li
                            key={i}
                            className="text-xs text-rose-800 bg-rose-50 px-2.5 py-1 rounded border border-rose-200"
                          >
                            &bull; {claimText}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

