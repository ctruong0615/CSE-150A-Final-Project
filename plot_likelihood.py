"""Compare saved likelihood-weighting and rejection estimates against exact inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--likelihood', type=Path, default=Path('results/inference/likelihood_comparison.json'))
    parser.add_argument('--rejection', type=Path, default=Path('results/inference/rejection_comparison.json'))
    parser.add_argument('--output', type=Path, default=Path('results/inference/likelihood_comparison.png'))
    args = parser.parse_args()
    lw = json.loads(args.likelihood.read_text())
    rejection = json.loads(args.rejection.read_text())
    if lw['seeds'] != rejection['seeds']:
        raise ValueError('Comparison seed lists differ')
    def key(row):
        return json.dumps(row['evidence'], sort_keys=True), row['sample_count']
    refs = {key(r): r for r in rejection['results']}
    if set(refs) != {key(r) for r in lw['results']}:
        raise ValueError('Comparison settings differ')
    for row in lw['results']:
        if row['exact_posterior'] != refs[key(row)]['exact_posterior']:
            raise ValueError('Exact references differ')
    groups = {}
    for row in lw['results']:
        label = ', '.join(f'{k}={v}' for k, v in row['evidence'].items()) or 'No evidence'
        groups.setdefault(label, []).append(row)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (label, rows) in zip(axes.flat, groups.items()):
        rows.sort(key=lambda r: r['sample_count'])
        for name, data in [('Rejection sampling', [refs[key(r)] for r in rows]), ('Likelihood weighting', rows)]:
            ax.errorbar([r['sample_count'] for r in data], [r['mean_absolute_error']['yes'] for r in data],
                        yerr=[r['tv_standard_error'] or 0 for r in data], marker='o', capsize=3, label=name)
        ax.set_title(label.replace(', ', '\n'), fontsize=10)
        ax.set_xlabel('Attempted / weighted complete assignments')
        ax.set_ylabel('Mean absolute error in P(success=yes)')
        ax.set_xscale('log')
        ax.set_xticks([1000, 5000, 10000], ['1,000', '5,000', '10,000'])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    fig.suptitle('Approximate inference vs exact probabilities | 10 seeds (42-51)\nError bars: one standard error across runs', fontsize=12)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
