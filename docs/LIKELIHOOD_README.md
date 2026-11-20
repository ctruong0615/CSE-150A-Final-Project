# Likelihood weighting

## Files and dependencies

- `bayesnet/likelihood_weighting.py`: weighted posterior estimates, effective sample size (ESS), and zero-weight diagnostics.
- `experiments/compare_likelihood.py`: exact comparisons for the four existing queries, budgets 1,000/5,000/10,000, and seeds 42-51 (120 runs).
- `tests/test_likelihood_weighting.py`: nine tests including analytic Bayes calculations, lecture network, zero weights, underflow, and repeatability.
- `experiments/plot_likelihood.py`: compare saved rejection and likelihood-weighting errors.
- results/inference/likelihood_*: JSON, CSV, plot, and one CLI example.

The sampler reuses prepare() from `bayesnet/rejection_sampling.py` for validation and topological ordering. Tests use network fixtures from `tests/test_gibbs_sampling.py`. The comparison imports `bayesnet/exact_inference.py`, which imports `bayesnet/parameter_learning.py`. All are already in the project. Core code uses the standard library; plotting additionally requires matplotlib. No dataset reload or retraining is necessary.

## Commands (run from the repository root)

```bash
python3 -m unittest tests.test_likelihood_weighting -v
python3 -m bayesnet.likelihood_weighting results/inference/trained_network.json --samples 10000 --seed 42
python3 -m bayesnet.likelihood_weighting results/inference/trained_network.json --evidence '{"price_band":"10_to_30","is_rpg":"yes","is_indie":"yes"}' --samples 10000 --seed 42
python3 -m experiments.compare_likelihood results/inference/trained_network.json
python3 -m experiments.plot_likelihood
```

Both comparison defaults use ten seeds. The plot uses the previously saved rejection results. The underlying model and exact-reference JSON are unchanged.

## Algorithm

For every weighted sample, visit variables in parent-before-child order. Draw non-evidence variables from their CPTs. Fix each evidence variable at its observed value and multiply the sample weight by the CPT probability of that value given its current parent assignment. Sum weights by query state, then normalize across states.

For example, if evidence B=yes has probability 0.8 under one sampled parent assignment and 0.1 under another, the first assignment gets eight times as much weight. Evidence is not randomly drawn and then rejected.

Products are accumulated as log weights and rescaled before normalization, avoiding numerical underflow when many evidence probabilities are tiny. Zero probabilities produce zero weights. If every sampled weight is zero, posterior is null and status is zero_total_weight; that does not prove the evidence impossible because a finite run may miss positive-support assignments.

ESS = (sum of weights)^2 / sum of squared weights. Equal weights give ESS equal to the number of samples; concentrated weights give a smaller ESS. This diagnoses weight concentration and does not establish accuracy. The code uses O(N) storage for query values and log weights; much larger budgets would benefit from a streaming accumulator.

## Verification and observed results

All nine likelihood tests, eight rejection tests, and four existing Gibbs tests passed (21 total). All 120 likelihood runs completed successfully. Saved posteriors normalize to one, ESS lies in (0, N], and CSV and JSON contain the expected runs. The comparison recomputes exact probabilities and checks them against the saved references before sampling.

At 10,000 samples/attempts, mean absolute errors across seeds 42-51 were:

| Evidence | Likelihood weighting | Rejection sampling | Mean LW ESS |
|---|---:|---:|---:|
| None | 0.00405 | 0.00405 | 10,000 |
| release_era=2021_plus | 0.00411 | 0.00532 | 10,000 |
| release_era=2021_plus, is_rpg=yes | 0.00249 | 0.00962 | 9,951 |
| price_band=10_to_30, is_rpg=yes, is_indie=yes | 0.00627 | 0.02606 | 6,883 |

With no evidence, both implementations use identical prior draws for the same seed, so their estimates coincide. With only root evidence, likelihood weights are constant and ESS equals N. With more evidence, weight variation reduces ESS. For the most restrictive query, rejection retained only about 227 of 10,000 attempts, while likelihood weighting achieved ESS about 6,883 and smaller observed error.

These are results for this trained eight-node network and these queries; they do not show that likelihood weighting always wins. Unfavorable evidence can concentrate weights severely. Error bars are one standard error of run-level absolute errors across ten seeds, not 95% confidence intervals. Budgets within each algorithm reuse the same initial random sequence for a seed, so different budgets are not independent repetitions.

Runtime is measured around each sampler, including setup and validation but excluding file I/O; likelihood timing also includes log-weight normalization and ESS. Rejection timings were recorded in an earlier session; rerun both in a common benchmark before making precise speed claims. No Gibbs results were changed. This measures approximation to the model, not held-out prediction quality.
