"""Small hand-checkable examples. Run: python test_structure_learning.py"""
import math
from itertools import combinations
from pathlib import Path

from structure_learning import local_score, learn_structure, load_data, MANUAL_EDGES, NODE_ORDER


def run_checks():
    states = {'A': ['0', '1'], 'B': ['0', '1'], 'C': ['0', '1']}
    # A and B always agree; C is balanced for either value of A.
    rows = []
    for _ in range(100):
        for a in ['0', '1']:
            for c in ['0', '1']:
                rows.append({'A': a, 'B': a, 'C': c})
    likelihood, parameters, bic = local_score(rows, 'A', (), states)
    assert abs(likelihood - 400 * math.log(0.5)) < 1e-8
    assert parameters == 1
    assert abs(bic - (likelihood - 0.5 * math.log(400))) < 1e-8
    likelihood, parameters, bic = local_score(rows, 'B', ['A'], states)
    assert likelihood == 0.0
    assert parameters == 2
    assert abs(bic + math.log(400)) < 1e-8
    edges = learn_structure(rows, ['A', 'B', 'C'], 2, states)
    assert edges == [('A', 'B')], edges
    # Independently check all allowed parent sets beat neither selected local score.
    for position, node in enumerate(states):
        selected = [p for p, c in edges if c == node]
        selected_bic = local_score(rows, node, selected, states)[2]
        for size in range(position + 1):
            for parents in combinations(list(states)[:position], size):
                assert local_score(rows, node, parents, states)[2] <= selected_bic + 1e-8
    train, test = load_data(Path(__file__).resolve().with_name('cleaned_games.csv'))
    assert (len(train), len(test)) == (18244, 4560)
    assert not ({r['app_id'] for r in train} & {r['app_id'] for r in test})
    assert all(NODE_ORDER.index(p) < NODE_ORDER.index(c) for p, c in MANUAL_EDGES)
    print('Passed: hand-calculated likelihood/BIC, dependency recovery, parent-set selection, split separation, and manual node order.')


if __name__ == '__main__':
    run_checks()
