# Rejection sampling: implementation and experiments

This document describes the rejection-sampling implementation and its saved ten-seed experiments.

## New files to add to your repository

- rejection_sampling.py: prior sampling followed by evidence rejection; compatible sample(...) interface and JSON model with Kai's implementation.
- test_rejection_sampling.py: eight tests, including analytic Bayes calculation, lecture network, invalid CPTs, reproducibility, impossible evidence, and unsorted nodes.
- compare_rejection.py: four existing exact queries x three sample budgets x ten seeds; JSON summaries and a CSV of individual runs.
- plot_rejection.py: plot saved comparison results (requires matplotlib).
- results/inference/rejection_*: generated experiment results, one CLI example, and comparison plot.

The implementation was developed and tested against supporting files and the trained network/exact results from ctruong0615's project (repository owner: ctruong0615), commit 223838a2f33b60b7c4f5a952854968c5a31b4f79. Kai's Gibbs files were inspected for interface/settings compatibility; her code was not modified. test_rejection_sampling.py imports the small network fixtures from her existing test_gibbs_sampling.py. Keep that file available when running tests.

## Run from this folder or the repository root after copying the new files

```bash
python3 -m unittest test_rejection_sampling -v
python3 rejection_sampling.py results/inference/trained_network.json --samples 10000 --seed 42
python3 rejection_sampling.py results/inference/trained_network.json --evidence '{"price_band":"10_to_30","is_rpg":"yes","is_indie":"yes"}' --samples 10000 --seed 42
python3 compare_rejection.py results/inference/trained_network.json
python3 plot_rejection.py
```

The sampler, comparison script, and tests use only Python's standard library. Only plotting requires matplotlib. No cleaned CSV or retraining is needed for these commands. Both the command-line and Python function defaults use ten seeds, 42 through 51. Override them with `--seeds` if needed.

## How the algorithm works

1. Validate the model and compute a parent-before-child order.
2. For each attempt, generate a complete assignment. Each variable is drawn from its CPT conditional on the already sampled parents. Evidence variables are sampled too, not forced to their observed values.
3. Reject assignments that disagree with any evidence value.
4. Estimate each query-state probability using its count divided by the number of accepted assignments.

The samples argument is the number of attempted complete assignments. With no evidence, every attempt is accepted. With no accepted assignments, posterior is null and status is no_accepted_samples. This is not a proof of impossible evidence, and the program does not invent a zero posterior.

The compare script recomputes exact posteriors and checks that they match saved exact_results.json before running experiments. Runtime includes sampler setup/validation; it excludes file loading and result serialization. Exact timings are single observations, not a benchmark average. Timing results depend on the machine and current load. Seeds make posterior/count results reproducible, not timings.

## Observed results

All eight new rejection tests and four existing Gibbs tests passed. All 120 experiment runs completed with accepted samples.

Mean absolute error in P(success=yes), across seeds 42 through 51:

| Evidence | 1,000 attempts | 5,000 attempts | 10,000 attempts | Acceptance at 10,000 |
|---|---:|---:|---:|---:|
| None | 0.01130 | 0.00655 | 0.00405 | 100.0% |
| release_era=2021_plus | 0.01683 | 0.00612 | 0.00532 | 39.1% |
| release_era=2021_plus, is_rpg=yes | 0.02808 | 0.01554 | 0.00962 | 11.5% |
| price_band=10_to_30, is_rpg=yes, is_indie=yes | 0.10659 | 0.03179 | 0.02606 | 2.3% |

For the last query, 10,000 attempts retained about 227 assignments on average. Lower acceptance leaves fewer useful observations and greater sampling variability. Mean absolute error fell from 0.10659 at 1,000 attempts to 0.02606 at 10,000 attempts. Individual runs need not improve monotonically. Plot error bars show one standard error across ten runs, not 95% confidence intervals. For a binary query, total variation distance equals the absolute error for either state.

## Scope and learning

Likelihood weighting is not implemented in this package. Model CPTs were not changed or smoothed. These experiments evaluate agreement with the model's exact posterior, not held-out game classification accuracy.

Read through and adapt this implementation before submitting so you can explain it. The supplied course guidelines ask you not to copy AI output verbatim and to include a brief AI-use statement. In your own words, describe the assistance actually used: drafting implementation/tests, checking algorithm behavior, and running comparison experiments. This guide is explanatory material, not a finished report section.
