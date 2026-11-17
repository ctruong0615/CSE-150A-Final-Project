# Compare Gibbs sampling estimates with exact Bayesian network inference
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from exact_inference import posterior as exact_posterior
from gibbs_sampling import sample

SCENARIOS = [
    {},
    {"release_era": "2021_plus"},
    {"release_era": "2021_plus", "is_rpg": "yes"},
    {"price_band": "10_to_30", "is_rpg": "yes", "is_indie": "yes"},
]

def trace_batches(trace, states, batch_count=10):
    checkpoints = []
    counts = {state: 0 for state in states}
    next_batch = 1
    for index, value in enumerate(trace, start=1):
        counts[value] += 1
        if next_batch <= batch_count and index >= len(trace) * next_batch // batch_count:
            checkpoints.append({"retained_samples": index,
                                "cumulative_posterior": {state: count / index
                                                          for state, count in counts.items()}})
            next_batch += 1
    return checkpoints

def compare(model, *, sample_sizes=(1_000, 5_000, 10_000),
            burn_in_periods=(0, 1_000, 5_000), thinning=1,
            seeds=(42, 43, 44)):
    # Report Gibbs error, convergence, stability, and runtime vs exact posteriors
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide at least one unique random seed")
    if not sample_sizes or any(n < 1 for n in sample_sizes):
        raise ValueError("each sample size must be positive")
    if not burn_in_periods or any(n < 0 for n in burn_in_periods):
        raise ValueError("burn-in periods must be nonnegative")
    results = []
    for evidence in SCENARIOS:
        exact = exact_posterior(model, "success", evidence)
        for burn_in in burn_in_periods:
            for sample_count in sample_sizes:
                runs = []
                for run_seed in seeds:
                    started = time.perf_counter()
                    estimate = sample(model, "success", evidence, samples=sample_count,
                                      burn_in=burn_in, thinning=thinning,
                                      seed=run_seed, return_trace=True)
                    elapsed = time.perf_counter() - started
                    sampled = estimate["posterior"]
                    errors = {state: sampled[state] - exact[state] for state in exact}
                    runs.append({"seed": run_seed, "posterior": sampled,
                                 "absolute_error": {s: abs(e) for s, e in errors.items()},
                                 "total_variation_distance": 0.5 * sum(abs(e) for e in errors.values()),
                                 "runtime_seconds": elapsed,
                                 "convergence_checkpoints": trace_batches(estimate["trace"], list(exact))})
                tvs = [run["total_variation_distance"] for run in runs]
                results.append({
                    "evidence": evidence, "sample_count": sample_count,
                    "burn_in_steps": burn_in, "thinning": thinning,
                    "seeds": list(seeds), "exact_posterior": exact,
                    "mean_estimated_posterior": {
                        state: statistics.mean(run["posterior"][state] for run in runs)
                        for state in exact
                    },
                    "mean_absolute_error": {
                        state: statistics.mean(run["absolute_error"][state] for run in runs)
                        for state in exact
                    },
                    "mean_total_variation_distance": statistics.mean(tvs),
                    "tv_standard_error": (statistics.stdev(tvs) / math.sqrt(len(seeds))
                                          if len(seeds) > 1 else None),
                    "mean_runtime_seconds": statistics.mean(run["runtime_seconds"] for run in runs),
                    "runs": runs,
                })
    return {"query": "success", "method": "random-scan Gibbs sampling",
            "seeds": list(seeds), "results": results}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[1_000, 5_000, 10_000])
    parser.add_argument("--burn-in-periods", type=int, nargs="+", default=[0, 1_000, 5_000])
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Explicit random seeds to run for every experiment condition")
    parser.add_argument("--output", type=Path, default=Path("gibbs_comparison.json"))
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    result = compare(model, sample_sizes=args.sample_sizes, burn_in_periods=args.burn_in_periods,
                     thinning=args.thinning, seeds=args.seeds)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
