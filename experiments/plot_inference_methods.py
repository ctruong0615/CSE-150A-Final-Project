"""Plot matched error and runtime results for the three inference methods."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABELS = {
    "rejection": "Rejection sampling",
    "likelihood_weighting": "Likelihood weighting",
    "gibbs": "Gibbs sampling",
}

def label_for(evidence):
    return ", ".join(f"{key}={value}" for key, value in evidence.items()) or "No evidence"

def plot(result, output):
    by_evidence = {}
    for row in result["results"]:
        by_evidence.setdefault(label_for(row["evidence"]), []).append(row)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (evidence_label, rows) in zip(axes.flat, by_evidence.items()):
        for method in result["methods"]:
            series = sorted((row for row in rows if row["method"] == method),
                            key=lambda row: row["sample_count"])
            if not series:
                continue
            label = METHOD_LABELS.get(method, method)
            if method == "gibbs":
                label += f" (burn-in {result['gibbs_burn_in_steps']:,})"
            ax.errorbar(
                [row["sample_count"] for row in series],
                [row["mean_total_variation_distance"] for row in series],
                yerr=[row["tv_standard_error"] or 0 for row in series],
                marker="o", capsize=3, label=label,
            )
        ax.set_title(evidence_label.replace(", ", "\n"), fontsize=10)
        ax.set_xlabel("Nominal sample size")
        ax.set_ylabel("Mean total variation error vs exact")
        counts = sorted({row["sample_count"] for row in rows})
        ax.set_xscale("log")
        ax.set_xticks(counts, [f"{n:,}" for n in counts])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Approximation error vs exact inference\nError bars: one standard error across matched seeds")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

def plot_runtime(result, output):
    by_evidence = {}
    for row in result["results"]:
        by_evidence.setdefault(label_for(row["evidence"]), []).append(row)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (evidence_label, rows) in zip(axes.flat, by_evidence.items()):
        for method in result["methods"]:
            series = sorted((row for row in rows if row["method"] == method),
                            key=lambda row: row["sample_count"])
            if not series:
                continue
            label = METHOD_LABELS.get(method, method)
            if method == "gibbs":
                label += f" (burn-in {result['gibbs_burn_in_steps']:,})"
            ax.plot([row["sample_count"] for row in series],
                    [1000 * row["mean_runtime_seconds"] for row in series],
                    marker="o", label=label)
        ax.set_title(evidence_label.replace(", ", "\n"), fontsize=10)
        ax.set_xlabel("Nominal sample size")
        ax.set_ylabel("Mean sampler runtime (ms)")
        counts = sorted({row["sample_count"] for row in rows})
        ax.set_xscale("log")
        ax.set_xticks(counts, [f"{n:,}" for n in counts])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Runtime for matched inference experiments")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_convergence(result, output):
    groups = {}
    for row in result["results"]:
        groups.setdefault(label_for(row["evidence"]), []).append(row)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (evidence_label, rows) in zip(axes.flat, groups.items()):
        exact = rows[0]["exact_posterior"]["yes"]
        for method in result["methods"]:
            series = sorted((row for row in rows if row["method"] == method),
                            key=lambda row: row["sample_count"])
            if not series:
                continue
            xs, means, errors = [], [], []
            for row in series:
                estimates = [run["posterior"]["yes"] for run in row["runs"]
                             if run["posterior"] is not None]
                if not estimates:
                    continue
                xs.append(row["sample_count"])
                means.append(sum(estimates) / len(estimates))
                if len(estimates) > 1:
                    mean = means[-1]
                    variance = sum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1)
                    errors.append(variance ** 0.5 / len(estimates) ** 0.5)
                else:
                    errors.append(0.0)
            if not xs:
                continue
            label = METHOD_LABELS.get(method, method)
            if method == "gibbs":
                label += f" (burn-in {result['gibbs_burn_in_steps']:,})"
            ax.errorbar(xs, means, yerr=errors, marker="o", capsize=3, label=label)
        ax.axhline(exact, color="black", linestyle="--", label=f"Exact = {exact:.3f}")
        ax.set_title(evidence_label.replace(", ", "\n"), fontsize=10)
        ax.set_xlabel("Sample size")
        ax.set_ylabel("Estimated P(success=yes)")
        ax.set_xscale("log")
        counts = sorted({row["sample_count"] for row in rows})
        ax.set_xticks(counts, [f"{n:,}" for n in counts])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Posterior estimates by sample size vs exact inference\n"
                 "Error bars: one standard error across seeds")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/inference/all_methods_comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("results/inference/all_methods_error.png"))
    parser.add_argument("--runtime-output", type=Path, default=Path("results/inference/all_methods_runtime.png"))
    parser.add_argument("--convergence-output", type=Path,
                        default=Path("results/inference/all_methods_convergence.png"))
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    plot(result, args.output)
    plot_runtime(result, args.runtime_output)
    plot_convergence(result, args.convergence_output)

if __name__ == "__main__":
    main()
