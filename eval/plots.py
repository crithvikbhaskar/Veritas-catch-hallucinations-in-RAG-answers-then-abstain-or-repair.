"""Plotting utilities for risk-coverage curves."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_risk_coverage(
    coverage_risk_points: list[tuple[float, float, str]],
    output_path: str = "docs/results/risk_coverage.png",
):
    """Generate a clean risk-coverage curve comparing baseline, cheap, strong, and cascade.
    
    Each point/series is (coverage, risk, label) where coverage = answer rate, risk = hallucination rate.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6), dpi=150)
    
    # Plot styled points
    markers = {"Baseline": "x", "Cheap (Small)": "o", "Strong (Base)": "s", "Cascade": "D"}
    colors = {"Baseline": "#e74c3c", "Cheap (Small)": "#3498db", "Strong (Base)": "#9b59b6", "Cascade": "#2ecc71"}

    for cov, risk, label in coverage_risk_points:
        m = markers.get(label, "o")
        c = colors.get(label, "#333333")
        plt.scatter(
            cov * 100.0,
            risk * 100.0,
            s=120,
            marker=m,
            color=c,
            label=f"{label} (Coverage: {cov*100:.1f}%, Risk: {risk*100:.1f}%)",
            zorder=4,
        )

    plt.title("Risk-Coverage Tradeoff: RAG Hallucination Guardrails", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Coverage (% of Queries Answered)", fontsize=11, labelpad=8)
    plt.ylabel("Risk (% Hallucinated Answers Among Answered)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5, zorder=1)
    plt.xlim(0, 105)
    plt.ylim(-2, 60)
    plt.legend(loc="upper left", frameon=True, fontsize=10)

    # Ideal operating corner annotation
    plt.annotate(
        "Ideal (100% Coverage, 0% Risk)",
        xy=(100, 0),
        xytext=(65, 8),
        arrowprops={"facecolor": "#27ae60", "shrink": 0.08, "width": 1.5, "headwidth": 6},
        fontsize=9,
        color="#27ae60",
        fontweight="semibold",
    )

    plt.tight_layout()
    plt.savefig(str(output_file))
    plt.close()
