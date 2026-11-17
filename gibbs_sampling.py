# Gibbs sampling for discrete Bayesian network posterior inference
import argparse
import json
import random
from pathlib import Path

def validate(model, query, evidence):
    states = model["states"]
    nodes = model["nodes"]
    if query not in states or query in evidence:
        raise ValueError("Query must be a model variable outside the evidence")
    if len(nodes) != len(set(nodes)) or set(nodes) != set(states):
        raise ValueError("Model nodes and state definitions must match")
    for node, value in evidence.items():
        if node not in states or value not in states[node]:
            raise ValueError(f"Invalid evidence: {node}={value}")
    cpt_maps = {}
    for node in nodes:
        parents = model["parents"][node]
        cpt_maps[node] = {
            tuple(row["parent_values"]): row["probabilities"]
            for row in model["cpts"][node]
        }
    return cpt_maps

def conditional(model, cpt_maps, node, assignment):
    weights = []
    original_value = assignment[node]
    parents = model["parents"][node]
    for value in model["states"][node]:
        assignment[node] = value
        probability = cpt_maps[node][tuple(assignment[parent] for parent in parents)][value]
        for child in model["nodes"]:
            if node in model["parents"][child]:
                child_parents = model["parents"][child]
                probability *= cpt_maps[child][tuple(assignment[parent] for parent in child_parents)][assignment[child]]
        weights.append(probability)
    assignment[node] = original_value
    total = sum(weights)
    if total <= 0:
        raise ValueError(
            "No positive-probability values are available in this Markov blanket; "
            "the assignment may conflict with deterministic CPT constraints or evidence."
        )
    return [weight / total for weight in weights]

def draw(states, probabilities, rng):
    threshold = rng.random()
    cumulative = 0.0
    for state, probability in zip(states, probabilities):
        cumulative += probability
        if threshold < cumulative:
            return state
    return states[-1]

def sample(model, query, evidence=None, *, samples=10_000, burn_in=1_000,
           thinning=1, seed=None, initial_state=None, return_trace=False):
    evidence = dict(evidence or {})
    if samples < 1 or burn_in < 0 or thinning < 1:
        raise ValueError("samples must be positive; burn_in nonnegative; thinning positive")
    cpt_maps = validate(model, query, evidence)
    states = model["states"]
    free_nodes = [node for node in model["nodes"] if node not in evidence]
    rng = random.Random(seed)

    assignment = {}
    for node in model["nodes"]:
        if node in evidence:
            assignment[node] = evidence[node]
        elif initial_state and node in initial_state:
            value = initial_state[node]
            if value not in states[node]:
                raise ValueError(f"Invalid initial state: {node}={value}")
            assignment[node] = value
        else:
            parents = model["parents"][node]
            row = cpt_maps[node][tuple(assignment[parent] for parent in parents)]
            assignment[node] = draw(states[node], [row[value] for value in states[node]], rng)

    counts = {value: 0 for value in states[query]}
    trace = []
    total_steps = burn_in + samples * thinning
    for step in range(total_steps):
        # Random-scan Gibbs
        node = rng.choice(free_nodes)
        probabilities = conditional(model, cpt_maps, node, assignment)
        assignment[node] = draw(states[node], probabilities, rng)
        if step >= burn_in and (step - burn_in + 1) % thinning == 0:
            counts[assignment[query]] += 1
            if return_trace:
                trace.append(assignment[query])

    result = {"posterior": {value: count / samples for value, count in counts.items()},
              "counts": counts, "samples": samples, "burn_in": burn_in,
              "thinning": thinning, "seed": seed}
    if return_trace:
        result["trace"] = trace
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--query", default="success")
    parser.add_argument("--evidence", default="{}", help="JSON object of evidence assignments")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    result = sample(model, args.query, json.loads(args.evidence), samples=args.samples,
                    burn_in=args.burn_in, thinning=args.thinning, seed=args.seed)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

if __name__ == "__main__":
    main()
