# Compute exact posteriors and evaluate success predictions
import argparse
import json
import math
from functools import lru_cache
from itertools import product
from pathlib import Path
from .parameter_learning import load_split

def multiply(factors, states):
    scope = tuple(dict.fromkeys(node for names, _ in factors for node in names))
    positions = [
        tuple(scope.index(node) for node in names) for names, _ in factors
    ]
    table = {}
    for values in product(*(states[node] for node in scope)):
        table[values] = math.prod(
            factor[tuple(values[index] for index in indexes)]
            for (_, factor), indexes in zip(factors, positions)
        )
    return scope, table

def posterior(model, query, evidence):
    states = model["states"]
    if query not in states or query in evidence:
        raise ValueError("Query must be a model variable outside the evidence")
    for node, value in evidence.items():
        if node not in states or value not in states[node]:
            raise ValueError(f"Invalid evidence: {node}={value}")

    factors = []
    for node in model["nodes"]:
        scope = (*model["parents"][node], node)
        fixed = [(i, evidence[name]) for i, name in enumerate(scope) if name in evidence]
        free = [i for i, name in enumerate(scope) if name not in evidence]
        table = {}
        for row in model["cpts"][node]:
            for value, probability in row["probabilities"].items():
                assignment = (*row["parent_values"], value)
                if all(assignment[i] == state for i, state in fixed):
                    table[tuple(assignment[i] for i in free)] = probability
        factors.append((tuple(scope[i] for i in free), table))

    for node in reversed(model["nodes"]):
        if node == query or node in evidence:
            continue
        related = [factor for factor in factors if node in factor[0]]
        factors = [factor for factor in factors if node not in factor[0]]
        scope, table = multiply(related, states)
        index = scope.index(node)
        reduced = {}
        for assignment, probability in table.items():
            key = assignment[:index] + assignment[index + 1:]
            reduced[key] = reduced.get(key, 0.0) + probability
        factors.append((scope[:index] + scope[index + 1:], reduced))

    _, table = multiply(factors, states)
    total = sum(table.values())
    if total == 0:
        raise ValueError("Evidence has zero probability under the model")
    return {value: table[value,] / total for value in states[query]}

def evaluate(model, train, test):
    # Uses only metadata available w/o owner and player counts
    features = ("release_era", "price_band", "description_length", "is_indie", "is_rpg")
    prior = sum(row["success"] == "yes" for row in train) / len(train)

    @lru_cache(maxsize=None)
    def predict(values):
        return posterior(model, "success", dict(zip(features, values)))["yes"]

    correct = baseline_correct = scored = 0
    loss = baseline_loss = 0.0
    for row in test:
        actual = row["success"] == "yes"
        values = tuple(row[feature] for feature in features)
        try:
            probability = predict(values)
        except ValueError as error:
            if str(error) != "Evidence has zero probability under the model":
                raise
            continue
        correct += (probability >= 0.5) == actual
        baseline_correct += (prior >= 0.5) == actual
        actual_probability = probability if actual else 1 - probability
        loss += -math.log(actual_probability) if actual_probability else math.inf
        baseline_loss -= math.log(prior if actual else 1 - prior)
        scored += 1

    return {
        "evidence_features": features,
        "held_out_games": len(test),
        "scored_games": scored,
        "zero_probability_evidence": len(test) - scored,
        "accuracy": correct / scored if scored else None,
        "log_loss": loss / scored if scored and math.isfinite(loss) else (
            "infinite" if scored else None
        ),
        "baseline_accuracy": baseline_correct / scored if scored else None,
        "baseline_log_loss": baseline_loss / scored if scored else None,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("results/inference/exact_results.json"))
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    train, test = load_split(args.repo)
    scenarios = [
        {},
        {"release_era": "2021_plus"},
        {"release_era": "2021_plus", "is_rpg": "yes"},
        {"price_band": "10_to_30", "is_rpg": "yes", "is_indie": "yes"},
    ]
    results = {
        "queries": [
            {"evidence": evidence, "posterior": posterior(model, "success", evidence)}
            for evidence in scenarios
        ],
        "evaluation": evaluate(model, train, test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()