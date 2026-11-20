"""Likelihood weighting for discrete Bayesian networks (standard library only).

Evidence is fixed; each sample is weighted by the product of the evidence CPT
probabilities given its sampled parents. Log weights prevent product underflow.
"""
import argparse
import json
import math
import random
from pathlib import Path

from .rejection_sampling import prepare


def sample(model, query, evidence=None, *, samples=10_000, seed=None):
    """Estimate P(query | evidence), returning weight-degeneracy diagnostics.

    Effective sample size is (sum weights)**2 / sum(weights**2); it diagnoses
    weight concentration, not model accuracy. A zero-weight run has no estimate.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError('samples must be a positive integer')
    evidence = dict(evidence or {})
    order, tables = prepare(model, query, evidence)
    states = model['states']
    evidence_indices = {node: states[node].index(value) for node, value in evidence.items()}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        assignment = {}
        log_weight = 0.0
        for node in order:
            key = tuple(assignment[p] for p in model['parents'][node])
            probabilities = tables[node][key]
            if node in evidence:
                assignment[node] = evidence[node]
                probability = probabilities[evidence_indices[node]]
                log_weight += math.log(probability) if probability > 0 else -math.inf
            else:
                assignment[node] = rng.choices(states[node], weights=probabilities, k=1)[0]
        draws.append((assignment[query], log_weight))
    maximum = max(weight for _, weight in draws)
    positive_samples = sum(math.isfinite(weight) for _, weight in draws)
    if maximum == -math.inf:
        return {'posterior': None, 'samples': samples, 'positive_weight_samples': 0,
                'zero_weight_samples': samples, 'effective_sample_size': 0.0,
                'effective_sample_fraction': 0.0, 'log_total_weight': None,
                'seed': seed, 'status': 'zero_total_weight'}
    # A common scaling factor cancels from both normalized probabilities and ESS.
    scaled = [(value, math.exp(weight - maximum)) for value, weight in draws]
    totals = {state: math.fsum(weight for value, weight in scaled if value == state)
              for state in states[query]}
    total = math.fsum(totals.values())
    sum_squares = math.fsum(weight * weight for _, weight in scaled)
    ess = min(float(samples), total * total / sum_squares)
    return {'posterior': {state: weight / total for state, weight in totals.items()},
            'samples': samples, 'positive_weight_samples': positive_samples,
            'zero_weight_samples': samples - positive_samples,
            'effective_sample_size': ess, 'effective_sample_fraction': ess / samples,
            'log_total_weight': maximum + math.log(total),
            'seed': seed, 'status': 'ok'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--query', default='success')
    parser.add_argument('--evidence', default='{}', help='JSON object of evidence assignments')
    parser.add_argument('--samples', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding='utf-8'))
    result = sample(model, args.query, json.loads(args.evidence), samples=args.samples, seed=args.seed)
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
