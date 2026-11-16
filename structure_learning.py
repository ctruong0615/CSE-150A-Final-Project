"""Compare a manual Bayesian network with one learned using a fixed node order."""
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from preprocess import STATES, split_rows

# Parents must occur before their children. This guarantees an acyclic graph.
NODE_ORDER = ['release_era', 'is_indie', 'is_rpg', 'price_band',
              'description_length', 'success', 'owners_band', 'peak_ccu_band']
MAX_PARENTS = 3
MANUAL_EDGES = [
    ('release_era', 'price_band'), ('is_indie', 'price_band'),
    ('is_rpg', 'price_band'), ('is_rpg', 'description_length'),
    ('is_indie', 'description_length'), ('price_band', 'success'),
    ('is_indie', 'success'), ('is_rpg', 'success'),
    ('release_era', 'owners_band'), ('price_band', 'owners_band'),
    ('success', 'owners_band'), ('owners_band', 'peak_ccu_band'),
    ('release_era', 'peak_ccu_band'), ('success', 'peak_ccu_band'),
]


def load_data(path):
    """Reproduce the preprocessing split; keep test games out of structure selection."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('The cleaned dataset is empty')
    ids = set()
    for row in rows:
        row['app_id'] = int(row['app_id'])
        if row['app_id'] in ids:
            raise ValueError('Duplicate app ID')
        ids.add(row['app_id'])
        for node in NODE_ORDER:
            if row[node] not in STATES[node]:
                raise ValueError('Invalid state for ' + node)
    return split_rows(rows, test_fraction=0.2, seed=42)


def local_score(rows, node, parents, states=STATES):
    """Compute count-based maximum log likelihood and a BIC complexity penalty."""
    if not rows:
        raise ValueError('No training rows')
    counts = {}
    for row in rows:
        parent_values = tuple(row[p] for p in parents)
        if parent_values not in counts:
            counts[parent_values] = {}
        value = row[node]
        counts[parent_values][value] = counts[parent_values].get(value, 0) + 1

    log_likelihood = 0.0
    for value_counts in counts.values():
        total = sum(value_counts.values())
        for count in value_counts.values():
            probability = count / total
            log_likelihood += count * math.log(probability)

    parent_configurations = 1
    for parent in parents:
        parent_configurations *= len(states[parent])
    parameters = (len(states[node]) - 1) * parent_configurations
    bic = log_likelihood - 0.5 * parameters * math.log(len(rows))
    return log_likelihood, parameters, bic


def learn_structure(rows, order=NODE_ORDER, max_parents=MAX_PARENTS, states=STATES):
    """Try every allowed parent set for each node and choose the highest BIC."""
    edges = []
    for position, node in enumerate(order):
        possible_parents = order[:position]
        best_parents = ()
        best_bic = local_score(rows, node, (), states)[2]
        for size in range(1, min(max_parents, len(possible_parents)) + 1):
            for parents in combinations(possible_parents, size):
                bic = local_score(rows, node, parents, states)[2]
                if bic > best_bic + 1e-8:
                    best_bic = bic
                    best_parents = parents
        for parent in best_parents:
            edges.append((parent, node))
    return edges


def compare_structure(rows, edges, order=NODE_ORDER, states=STATES):
    log_likelihood = 0.0
    parameters = 0
    for node in order:
        parents = [parent for parent, child in edges if child == node]
        node_likelihood, node_parameters, _ = local_score(rows, node, parents, states)
        log_likelihood += node_likelihood
        parameters += node_parameters
    return {'edges': len(edges), 'parameters': parameters,
            'log_likelihood': log_likelihood,
            'bic': log_likelihood - 0.5 * parameters * math.log(len(rows))}


def save_network(path, edges):
    network = {'nodes': NODE_ORDER, 'states': STATES,
               'edges': sorted(edges),
               'parents': {node: [p for p, c in edges if c == node] for node in NODE_ORDER}}
    path.write_text(json.dumps(network, indent=2) + '\n', encoding='utf-8')


def draw_network(path, edges, title):
    graph = nx.DiGraph()
    graph.add_nodes_from(NODE_ORDER)
    graph.add_edges_from(edges)
    # Fixed positions make the two diagrams directly comparable.
    positions = {'release_era': (0, 3), 'is_indie': (2, 3), 'is_rpg': (4, 3),
                 'price_band': (1, 2), 'description_length': (4, 2),
                 'success': (3, 1), 'owners_band': (0, 0), 'peak_ccu_band': (4, 0)}
    labels = {node: node.replace('_', '\n') for node in NODE_ORDER}
    plt.figure(figsize=(11, 9))
    nx.draw_networkx(graph, pos=positions, labels=labels, node_size=4000,
                     node_color='#dbeafe', font_size=10, arrowsize=20,
                     connectionstyle='arc3,rad=0.08', edge_color='#475569')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def run():
    root = Path(__file__).resolve().parent
    train, test = load_data(root / 'cleaned_games.csv')
    learned_edges = learn_structure(train)
    output = root / 'results' / 'structure'
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for name, edges in [('manual', MANUAL_EDGES), ('learned', learned_edges)]:
        comparisons.append({'structure': name, **compare_structure(train, edges)})
        save_network(output / (name + '_network.json'), edges)
        draw_network(output / (name + '_network.png'), edges, name.title() + ' Bayesian network')
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print('Training games:', len(train), '| Reserved test games:', len(test))
    for result in comparisons:
        print(result)
    print('Learned edges:', sorted(learned_edges))
    return comparisons


if __name__ == '__main__':
    run()
