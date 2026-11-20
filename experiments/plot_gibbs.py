"""Plot saved Gibbs sampling error, runtime, and convergence experiments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def evidence_label(evidence):
    return ", ".join(f"{key}={value}" for key, value in evidence.items()) or "No evidence"

def grouped_results(result):
    groups = {}
    for row in result["results"]:
        groups.setdefault(evidence_label(row["evidence"]), []).append(row)
    return groups

def plot_comparison(result, output):
    groups = grouped_results(result)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    burn_ins = sorted({row["burn_in_steps"] for row in result["results"]})
    for ax, (label, rows) in zip(axes.flat, groups.items()):
        for burn_in in burn_ins:
            series = sorted((row for row in rows if row["burn_in_steps"] == burn_in),
                            key=lambda row: row["sample_count"])
            if not series:
                continue
            ax.errorbar(
                [row["sample_count"] for row in series],
                [row["mean_total_variation_distance"] for row in series],
                yerr=[row["tv_standard_error"] or 0 for row in series],
                marker="o", capsize=3, label=f"Burn-in {burn_in:,}",
            )
        ax.set_title(label.replace(", ", "\n"), fontsize=10)
        ax.set_xlabel("Retained Gibbs samples")
        ax.set_ylabel("Mean total variation error")
        ax.set_xscale("log")
        ax.set_xticks(sorted({row["sample_count"] for row in rows}),
                      [f"{n:,}" for n in sorted({row["sample_count"] for row in rows})])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(handles)))
    fig.suptitle("Gibbs approximation error vs exact inference\nError bars: one standard error across seeds")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

def plot_convergence(result, output, sample_count=None, burn_in=None):
    groups = grouped_results(result)
    sample_count = sample_count or max(row["sample_count"] for row in result["results"])
    burn_in = burn_in if burn_in is not None else max(row["burn_in_steps"] for row in result["results"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (label, rows) in zip(axes.flat, groups.items()):
        selected = [row for row in rows
                    if row["sample_count"] == sample_count and row["burn_in_steps"] == burn_in]
        if not selected:
            ax.set_visible(False)
            continue
        row = selected[0]
        exact = row["exact_posterior"]["yes"]
        trajectories = [run["convergence_checkpoints"] for run in row["runs"]]
        checkpoints = min(map(len, trajectories))
        xs, means, errors = [], [], []
        for index in range(checkpoints):
            points = [trace[index] for trace in trajectories]
            xs.append(sum(point["retained_samples"] for point in points) / len(points))
            values = [point["cumulative_posterior"]["yes"] for point in points]
            means.append(sum(values) / len(values))
            if len(values) > 1:
                mean = means[-1]
                variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                errors.append(variance ** 0.5 / len(values) ** 0.5)
            else:
                errors.append(0)
        ax.plot(xs, means, marker="o", markersize=3, label="Mean cumulative estimate")
        ax.fill_between(xs, [max(0, m-e) for m, e in zip(means, errors)],
                        [min(1, m+e) for m, e in zip(means, errors)], alpha=0.2,
                        label="±1 SE across seeds")
        ax.axhline(exact, color="black", linestyle="--", label=f"Exact = {exact:.3f}")
        ax.set_title(label.replace(", ", "\n"), fontsize=10)
        ax.set_xlabel("Retained samples")
        ax.set_ylabel("Cumulative P(success=yes)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"Gibbs convergence | {sample_count:,} samples, {burn_in:,} burn-in updates")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

def plot_runtime(result, output):
    groups = grouped_results(result)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    burn_ins = sorted({row["burn_in_steps"] for row in result["results"]})
    for ax, (label, rows) in zip(axes.flat, groups.items()):
        for burn_in in burn_ins:
            series = sorted((row for row in rows if row["burn_in_steps"] == burn_in),
                            key=lambda row: row["sample_count"])
            if not series:
                continue
            ax.plot([row["sample_count"] for row in series],
                    [1000 * row["mean_runtime_seconds"] for row in series],
                    marker="o", label=f"Burn-in {burn_in:,}")
        ax.set_title(label.replace(", ", "\n"), fontsize=10)
        ax.set_xlabel("Retained Gibbs samples")
        ax.set_ylabel("Mean sampler runtime (ms)")
        ax.set_xscale("log")
        counts = sorted({row["sample_count"] for row in rows})
        ax.set_xticks(counts, [f"{n:,}" for n in counts])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(handles)))
    fig.suptitle("Gibbs sampling runtime by retained sample count and burn-in")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/inference/gibbs_comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("results/inference/gibbs_comparison.png"))
    parser.add_argument("--convergence-output", type=Path,
                        default=Path("results/inference/gibbs_convergence.png"))
    parser.add_argument("--runtime-output", type=Path,
                        default=Path("results/inference/gibbs_runtime.png"))
    parser.add_argument("--sample-count", type=int,
                        help="Sample budget for convergence plot (default: largest saved budget)")
    parser.add_argument("--burn-in", type=int,
                        help="Burn-in for convergence plot (default: largest saved burn-in)")
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    plot_comparison(result, args.output)
    plot_convergence(result, args.convergence_output, args.sample_count, args.burn_in)
    plot_runtime(result, args.runtime_output)

if __name__ == "__main__":
    main()
