"""Plot saved rejection sampling experiments; requires matplotlib."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/inference/rejection_comparison.json'))
    parser.add_argument('--output', type=Path, default=Path('results/inference/rejection_comparison.png'))
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    groups = {}
    for row in result['results']:
        label = ', '.join(f'{k}={v}' for k, v in row['evidence'].items()) or 'No evidence'
        groups.setdefault(label, []).append(row)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for label, rows in groups.items():
        rows.sort(key=lambda r: r['sample_count'])
        ns = [r['sample_count'] for r in rows]
        axes[0].errorbar(ns, [r['mean_total_variation_distance'] for r in rows],
                         yerr=[r['tv_standard_error'] or 0 for r in rows], marker='o', capsize=3, label=label)
        axes[1].plot(ns, [100*r['mean_acceptance_rate'] for r in rows], marker='o', label=label)
        axes[2].plot(ns, [1000*r['mean_runtime_seconds'] for r in rows], marker='o', label=label)
    for ax, title, ylabel in zip(axes,
        ['Approximation error (mean +/- SE)', 'Evidence acceptance', 'End-to-end sampling runtime'],
        ['Absolute error in P(success=yes)', 'Accepted attempts (%)', 'Runtime (ms)']):
        ax.set(title=title, xlabel='Attempted complete assignments', ylabel=ylabel)
        ax.set_xscale('log')
        ax.set_xticks([1000, 5000, 10000], ['1,000', '5,000', '10,000'])
        ax.grid(alpha=0.25)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 105)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', fontsize=8, ncol=2)
    fig.suptitle('Rejection sampling on the trained Steam Bayesian network | seeds ' + ', '.join(map(str, result['seeds'])))
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
