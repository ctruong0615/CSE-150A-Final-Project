"""Run a matched exact-reference comparison of rejection, likelihood weighting, and Gibbs"""
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from bayesnet.exact_inference import posterior as exact_posterior
from bayesnet.gibbs_sampling import sample as gibbs_sample
from bayesnet.likelihood_weighting import sample as likelihood_sample
from bayesnet.rejection_sampling import sample as rejection_sample

METHODS = ("rejection", "likelihood_weighting", "gibbs")

def trace_checkpoints(trace, states, checkpoint_count=20):
    # Summarize Gibbs
    positions = {max(1, (len(trace) * i + checkpoint_count - 1) // checkpoint_count)
                 for i in range(1, checkpoint_count + 1)}
    counts = {state: 0 for state in states}
    checkpoints = []
    for index, value in enumerate(trace, start=1):
        counts[value] += 1
        if index in positions:
            checkpoints.append({
                "retained_samples": index,
                "cumulative_posterior": {state: counts[state] / index for state in states},
            })
    return checkpoints

def compare(model, queries, *, sample_sizes=(1_000, 5_000, 10_000),
            seeds=tuple(range(42, 52)), gibbs_burn_in=1_000, thinning=1):
    if not sample_sizes or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sample_sizes):
        raise ValueError("Sample sizes must be positive integers")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide at least one unique seed")
    if gibbs_burn_in < 0 or thinning < 1:
        raise ValueError("Gibbs burn-in must be nonnegative and thinning must be positive")

    results = []
    for scenario in queries:
        evidence = scenario["evidence"]
        exact = exact_posterior(model, "success", evidence)
        if any(not math.isclose(exact[state], scenario["posterior"][state], abs_tol=1e-10)
               for state in exact):
            raise ValueError(f"Saved exact reference differs for evidence {evidence}; regenerate exact results")
        for sample_count in sample_sizes:
            runs_by_method = {method: [] for method in METHODS}
            # Interleave method order across seeds to reduce systematic timing-order bias
            for seed_index, seed in enumerate(seeds):
                method_order = METHODS[seed_index % len(METHODS):] + METHODS[:seed_index % len(METHODS)]
                for method in method_order:
                    started = time.perf_counter()
                    if method == "rejection":
                        estimate = rejection_sample(model, "success", evidence,
                                                    samples=sample_count, seed=seed)
                    elif method == "likelihood_weighting":
                        estimate = likelihood_sample(model, "success", evidence,
                                                     samples=sample_count, seed=seed)
                    else:
                        estimate = gibbs_sample(model, "success", evidence,
                                                samples=sample_count, burn_in=gibbs_burn_in,
                                                thinning=thinning, seed=seed, return_trace=True)
                    elapsed = time.perf_counter() - started
                    posterior = estimate["posterior"]
                    errors = ({state: abs(posterior[state] - exact[state]) for state in exact}
                              if posterior is not None else None)
                    tv = 0.5 * sum(errors.values()) if errors is not None else None
                    runs_by_method[method].append({
                        "seed": seed, "posterior": posterior, "absolute_error": errors,
                        "total_variation_distance": tv, "runtime_seconds": elapsed,
                        "status": estimate.get("status", "ok"),
                        **({"convergence_checkpoints": trace_checkpoints(
                                estimate["trace"], list(exact))}
                           if method == "gibbs" else {}),
                        **({"accepted_samples": estimate["accepted_samples"],
                            "acceptance_rate": estimate["acceptance_rate"]}
                           if method == "rejection" else {}),
                        **({"effective_sample_size": estimate["effective_sample_size"],
                            "effective_sample_fraction": estimate["effective_sample_fraction"]}
                           if method == "likelihood_weighting" else {}),
                    })
            for method in METHODS:
                runs = runs_by_method[method]
                valid_tvs = [run["total_variation_distance"] for run in runs
                             if run["total_variation_distance"] is not None]
                row = {
                    "method": method, "evidence": evidence, "sample_count": sample_count,
                    "gibbs_burn_in_steps": gibbs_burn_in if method == "gibbs" else None,
                    "gibbs_thinning": thinning if method == "gibbs" else None,
                    "seeds": list(seeds), "exact_posterior": exact,
                    "successful_runs": len(valid_tvs), "failed_runs": len(runs) - len(valid_tvs),
                    "mean_total_variation_distance": statistics.mean(valid_tvs) if valid_tvs else None,
                    "tv_standard_error": (statistics.stdev(valid_tvs) / math.sqrt(len(valid_tvs))
                                          if len(valid_tvs) > 1 else None),
                    "mean_runtime_seconds": statistics.mean(run["runtime_seconds"] for run in runs),
                    "runs": runs,
                }
                if method == "rejection":
                    row["mean_acceptance_rate"] = statistics.mean(r["acceptance_rate"] for r in runs)
                elif method == "likelihood_weighting":
                    row["mean_effective_sample_size"] = statistics.mean(r["effective_sample_size"] for r in runs)
                else:
                    trajectories = [run["convergence_checkpoints"] for run in runs]
                    checkpoint_count = min(map(len, trajectories))
                    convergence = []
                    for index in range(checkpoint_count):
                        points = [trajectory[index] for trajectory in trajectories]
                        estimates = [point["cumulative_posterior"]["yes"] for point in points]
                        convergence.append({
                            "retained_samples": points[0]["retained_samples"],
                            "mean_posterior_yes": statistics.mean(estimates),
                            "standard_error": (statistics.stdev(estimates) / math.sqrt(len(estimates))
                                               if len(estimates) > 1 else 0.0),
                        })
                    row["mean_convergence_checkpoints"] = convergence
                results.append(row)
    return {
        "query": "success", "methods": list(METHODS),
        "seeds": list(seeds), "sample_sizes": list(sample_sizes),
        "gibbs_burn_in_steps": gibbs_burn_in, "gibbs_thinning": thinning,
        "budget_note": ("Rejection and likelihood weighting use sample_count complete assignments. "
                        "Gibbs uses sample_count retained updates plus burn-in updates; each update "
                        "changes one non-evidence node."),
        "timing_note": "Methods were interleaved by seed in one process; runtimes include sampler setup and validation.",
        "results": results,
    }

def write_csv(path, result):
    rows = []
    for condition in result["results"]:
        for run in condition["runs"]:
            rows.append({
                "method": condition["method"],
                "evidence": json.dumps(condition["evidence"], sort_keys=True),
                "sample_count": condition["sample_count"],
                "gibbs_burn_in_steps": condition["gibbs_burn_in_steps"],
                "gibbs_thinning": condition["gibbs_thinning"],
                "seed": run["seed"],
                "exact_success_probability": condition["exact_posterior"]["yes"],
                "estimated_success_probability": run["posterior"]["yes"] if run["posterior"] is not None else None,
                "total_variation_distance": run["total_variation_distance"],
                "runtime_seconds": run["runtime_seconds"],
                "status": run["status"],
                "accepted_samples": run.get("accepted_samples"),
                "acceptance_rate": run.get("acceptance_rate"),
                "effective_sample_size": run.get("effective_sample_size"),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=Path("results/inference/trained_network.json"))
    parser.add_argument("--reference", type=Path, default=Path("results/inference/exact_results.json"))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[1_000, 5_000, 10_000])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--gibbs-burn-in", type=int, default=1_000)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/inference/all_methods_comparison.json"))
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    queries = json.loads(args.reference.read_text(encoding="utf-8"))["queries"]
    result = compare(model, queries, sample_sizes=args.sample_sizes, seeds=args.seeds,
                     gibbs_burn_in=args.gibbs_burn_in, thinning=args.thinning)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), result)
    for row in result["results"]:
        print(f"{row['method']:20s} N={row['sample_count']:5d} "
              f"TV={row['mean_total_variation_distance']} "
              f"runtime={row['mean_runtime_seconds']:.4f}s evidence={row['evidence']}")

if __name__ == "__main__":
    main()
