"""Rejection sampling for discrete Bayesian networks (standard library only).

Draw complete assignments from the prior; retain only assignments matching evidence.
The budget counts attempted assignments, not accepted assignments.
"""
import argparse
import json
import math
import random
from itertools import product
from pathlib import Path


def prepare(model, query, evidence):
    nodes, states, parents = model['nodes'], model['states'], model['parents']
    if len(nodes) != len(set(nodes)) or set(nodes) != set(states):
        raise ValueError('Model nodes and state definitions must match')
    if query not in nodes or query in evidence:
        raise ValueError('Query must be a model variable outside the evidence')
    for node, value in evidence.items():
        if node not in states or value not in states[node]:
            raise ValueError(f'Invalid evidence: {node}={value}')
    # Compute a topological order rather than relying on JSON node ordering.
    order = []
    pending = list(nodes)
    while pending:
        ready = [node for node in pending if set(parents[node]) <= set(order)]
        if not ready:
            raise ValueError('Model contains a cycle or an unknown parent')
        order.extend(ready)
        pending = [node for node in pending if node not in ready]
    tables = {}
    for node in nodes:
        values = states[node]
        if not values or len(values) != len(set(values)):
            raise ValueError(f'Invalid states for {node}')
        if len(parents[node]) != len(set(parents[node])):
            raise ValueError(f'Duplicate parents for {node}')
        table = {}
        for row in model['cpts'][node]:
            key = tuple(row['parent_values'])
            probabilities = row['probabilities']
            if key in table or set(probabilities) != set(values):
                raise ValueError(f'Duplicate CPT row or invalid probability states for {node}')
            weights = [probabilities[value] for value in values]
            if any(not math.isfinite(p) or p < 0 for p in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-9):
                raise ValueError(f'Invalid CPT probabilities for {node}')
            table[key] = weights
        expected = set(product(*(states[p] for p in parents[node])))
        if set(table) != expected:
            raise ValueError(f'Incomplete or invalid CPT parent assignments for {node}')
        tables[node] = table
    return order, tables


def sample(model, query, evidence=None, *, samples=10_000, seed=None):
    """Return a posterior or null when no assignments survive rejection.

    No accepted samples means insufficient evidence support at this budget;
    it does not establish that the evidence is impossible.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError('samples must be a positive integer')
    evidence = dict(evidence or {})
    order, tables = prepare(model, query, evidence)
    rng = random.Random(seed)
    counts = {state: 0 for state in model['states'][query]}
    accepted = 0
    for _ in range(samples):
        assignment = {}
        for node in order:
            key = tuple(assignment[p] for p in model['parents'][node])
            assignment[node] = rng.choices(model['states'][node], weights=tables[node][key], k=1)[0]
        if all(assignment[node] == value for node, value in evidence.items()):
            accepted += 1
            counts[assignment[query]] += 1
    return {
        'posterior': {state: count / accepted for state, count in counts.items()} if accepted else None,
        'counts': counts, 'samples': samples, 'attempted_samples': samples,
        'accepted_samples': accepted, 'rejected_samples': samples - accepted,
        'acceptance_rate': accepted / samples, 'seed': seed,
        'status': 'ok' if accepted else 'no_accepted_samples',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--query', default='success')
    parser.add_argument('--evidence', default='{}', help='JSON object of evidence assignments')
    parser.add_argument('--samples', type=int, default=10_000, help='Number of attempted complete assignments')
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
