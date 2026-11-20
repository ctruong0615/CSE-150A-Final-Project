"""Compare likelihood weighting with exact inference using Kai's experiment settings."""
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from bayesnet.exact_inference import posterior as exact_posterior
from bayesnet.likelihood_weighting import sample


def compare(model, queries, sample_sizes=(1000, 5000, 10000), seeds=tuple(range(42, 52))):
    if not sample_sizes or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sample_sizes):
        raise ValueError('Sample sizes must be positive integers')
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Provide at least one unique seed')
    results = []
    for scenario in queries:
        evidence = scenario['evidence']
        started = time.perf_counter()
        exact = exact_posterior(model, 'success', evidence)
        exact_runtime = time.perf_counter() - started
        if any(not math.isclose(exact[s], scenario['posterior'][s], abs_tol=1e-10) for s in exact):
            raise ValueError('Saved reference differs from current model; regenerate exact results')
        for n in sample_sizes:
            runs = []
            for seed in seeds:
                started = time.perf_counter()
                estimate = sample(model, 'success', evidence, samples=n, seed=seed)
                runtime = time.perf_counter() - started
                posterior = estimate['posterior']
                errors = {s: abs(posterior[s] - exact[s]) for s in exact} if posterior else None
                runs.append({**estimate, 'absolute_error': errors,
                             'total_variation_distance': 0.5 * sum(errors.values()) if errors else None,
                             'runtime_seconds': runtime})
            valid = [r for r in runs if r['posterior'] is not None]
            tvs = [r['total_variation_distance'] for r in valid]
            results.append({
                'evidence': evidence, 'sample_count': n, 'seeds': list(seeds),
                'exact_posterior': exact, 'exact_runtime_seconds': exact_runtime,
                'successful_runs': len(valid), 'failed_runs': len(runs) - len(valid),
                'mean_estimated_posterior': {s: statistics.mean(r['posterior'][s] for r in valid) for s in exact} if valid else None,
                'mean_absolute_error': {s: statistics.mean(r['absolute_error'][s] for r in valid) for s in exact} if valid else None,
                'mean_total_variation_distance': statistics.mean(tvs) if valid else None,
                'tv_standard_error': statistics.stdev(tvs) / math.sqrt(len(tvs)) if len(tvs) > 1 else None,
                'mean_runtime_seconds': statistics.mean(r['runtime_seconds'] for r in runs),
                'mean_effective_sample_size': statistics.mean(r['effective_sample_size'] for r in runs),
                'mean_effective_sample_fraction': statistics.mean(r['effective_sample_fraction'] for r in runs),
                'runs': runs,
            })
    return {'query': 'success', 'method': 'likelihood weighting',
            'sample_budget_definition': 'weighted complete assignments with fixed evidence',
            'seeds': list(seeds), 'results': results}


def write_csv(path, result):
    rows = []
    for condition in result['results']:
        for run in condition['runs']:
            rows.append({
                'method': 'likelihood_weighting', 'evidence': json.dumps(condition['evidence'], sort_keys=True),
                'sample_count': condition['sample_count'], 'seed': run['seed'],
                'exact_success_probability': condition['exact_posterior']['yes'],
                'estimated_success_probability': run['posterior']['yes'] if run['posterior'] else None,
                'absolute_error': run['absolute_error']['yes'] if run['absolute_error'] else None,
                'runtime_seconds': run['runtime_seconds'],
                'effective_sample_size': run['effective_sample_size'],
                'effective_sample_fraction': run['effective_sample_fraction'], 'zero_weight_samples': run['zero_weight_samples'], 'status': run['status'],
            })
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--reference', type=Path, default=Path('results/inference/exact_results.json'))
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=[1000, 5000, 10000])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    parser.add_argument('--output', type=Path, default=Path('results/inference/likelihood_comparison.json'))
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding='utf-8'))
    queries = json.loads(args.reference.read_text(encoding='utf-8'))['queries']
    result = compare(model, queries, args.sample_sizes, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_csv(args.output.with_suffix('.csv'), result)
    for row in result['results']:
        print(f"N={row['sample_count']:5d} ESS={row['mean_effective_sample_size']:8.1f} "
              f"MAE={row['mean_total_variation_distance']} "
              f"runtime={row['mean_runtime_seconds']:.4f}s evidence={row['evidence']}")


if __name__ == '__main__':
    main()
