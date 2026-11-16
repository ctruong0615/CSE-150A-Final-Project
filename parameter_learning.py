# Fit discrete Bayesian network CPTs on the training split
import argparse
import csv
import json
import runpy
from collections import Counter
from itertools import product
from pathlib import Path

def load_split(repo):
    split_rows = runpy.run_path(str(repo / "preprocess.py"))["split_rows"]
    with (repo / "cleaned_games.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["app_id"] = int(row["app_id"])
    return split_rows(rows, test_fraction=0.2, seed=42)

def fit(graph, train):
    cpts = {}
    for node in graph["nodes"]:
        parents = graph["parents"][node]
        states = graph["states"][node]
        counts = Counter(
            (tuple(row[parent] for parent in parents), row[node]) for row in train
        )
        cpts[node] = []
        for assignment in product(*(graph["states"][parent] for parent in parents)):
            frequencies = [counts[assignment, state] for state in states]
            total = sum(frequencies)
            probabilities = (
                [count / total for count in frequencies]
                if total else [1 / len(states)] * len(states)
            )
            cpts[node].append({
                "parent_values": assignment,
                "count": total,
                "probabilities": dict(zip(states, probabilities)),
            })
    return {**graph, "cpts": cpts}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--graph", choices=["learned", "manual"], default="learned")
    parser.add_argument("--output", type=Path, default=Path("trained_network.json"))
    args = parser.parse_args()
    graph_path = args.repo / "results" / "structure" / f"{args.graph}_network.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    train, _ = load_split(args.repo)
    model = fit(graph, train)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()