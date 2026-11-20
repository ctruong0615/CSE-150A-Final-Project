# CSE 150A Final Project

## Project layout

- `bayesnet/` contains preprocessing, learning, exact inference, and sampling code.
- `experiments/` contains comparison runners and plotting scripts. Run them as modules from the project root.
- `tests/` contains the unit tests; run them with `python3 -m unittest discover -s tests -v`.
- `data/` contains the cleaned dataset. Generated outputs stay under `results/`.
- `docs/` contains the method notes for rejection sampling and likelihood weighting.

## Project progress

- Downloaded the [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
- Added preprocessing to check invalid values and duplicates, fix a merged CSV header, and convert features into categories.
- Kept games with **at least 100 reviews**; labeled them successful if **80% or more reviews are positive**. These thresholds are provisional.
- Generated **22,804 eligible games**, split into **18,244 training** and **4,560 test** rows, plus feature-distribution charts.

The source has 125,855 games: 103,050 were below the review minimum and one was released after the cutoff. Of the eligible games, 13,137 have `success=yes` and 9,667 have `success=no`.

The inputs are price, release era, estimated owners, peak concurrent players, description length, and Indie/RPG genre flags. The target is `success`. The network structures are defined below. Learning their probability tables and running inference are separate parts of the project.

Open [cleaned_games.csv](data/cleaned_games.csv) directly in Excel, or load it in a Jupyter notebook opened from the project folder (with pandas installed):

```python
import pandas as pd
games = pd.read_csv("data/cleaned_games.csv")
games.head()
```

## Bayesian network structure learning

This part looks at how the game features are connected. The code starts with a network based on assumptions about relationships between features, such as price, genre, and game success. It then learns another network from the training data and compares the two.

For the learned network, the variables are put in a fixed order. Each variable can have up to three parents, chosen from the variables before it. The code tries the possible parent groups and uses BIC to choose between them. BIC rewards a better fit to the data but also adds a penalty for larger probability tables.

The learned network has 11 connections, compared with 14 in the manual network, and gets a better BIC score on the training data. The test data is kept separate. These results compare the network structures; they do not measure prediction accuracy or show that one feature causes another.

The graphs, network diagrams, and comparison table are saved in `results/structure/`. The JSON files list the connections and parents for each variable so they can be used in the parameter-learning part.

## Parameter learning and exact inference

- [parameter_learning.py](bayesnet/parameter_learning.py) fits CPTs for the learned network using maximum likelihood on the training split. Unseen parent configurations use a uniform distribution.
- [exact_inference.py](bayesnet/exact_inference.py) computes exact success probabilities by variable elimination. It scored 59.00% accuracy on 4,559 held-out games. One game's evidence had zero probability and was excluded.
- For rejection sampling, likelihood weighting, and Gibbs sampling, load [trained_network.json](results/inference/trained_network.json) and use the evidence and exact posteriors in the `queries` array of [exact_results.json](results/inference/exact_results.json) for comparison.

## Gibbs sampling

- [gibbs_sampling.py](bayesnet/gibbs_sampling.py) estimates Bayesian network posteriors using random-scan Gibbs sampling. Evidence variables stay fixed; each update samples one non-evidence variable from its Markov blanket conditional.
- [compare_gibbs.py](experiments/compare_gibbs.py) compares Gibbs estimates of `success` with exact inference on the same original MLE network and the four evidence scenarios listed in `exact_results.json`. It varies retained sample counts (1,000, 5,000, and 10,000), burn-in update counts (0, 1,000, and 5,000), and explicit random seeds (default: 42, 43, and 44). The JSON reports per-seed estimates, exact posteriors, approximation error, convergence checkpoints, and runtime.
- [test_gibbs_sampling.py](tests/test_gibbs_sampling.py) checks estimates against exact inference on a two-node network and the lecture example, checks repeatability with a fixed seed, and rejects invalid sample settings.

When no evidence is provided, all network variables are free to be updated. The sampler initializes values from the CPTs, then repeatedly selects a variable at random and resamples it using its Markov blanket. After burn-in and thinning, it estimates the query’s prior probability from the frequency of its sampled values.

Generate the saved Gibbs error, convergence, and runtime figures from the comparison JSON:

```bash
python3 -m experiments.plot_gibbs
```

This writes `gibbs_comparison.png` (error by sample count and burn-in), `gibbs_convergence.png` (cumulative estimates against exact inference), and `gibbs_runtime.png` under `results/inference/`.

### Experimental analysis

Run the same four exact-reference queries, sample sizes (1,000, 5,000, and 10,000), and seeds (42-51) through rejection sampling, likelihood weighting, and Gibbs sampling:

```bash
python3 -m experiments.compare_inference_methods
python3 -m experiments.plot_inference_methods
```

The plotter saves `all_methods_error.png`, `all_methods_runtime.png`, and `all_methods_convergence.png` under `results/inference/`. The convergence chart compares rejection, likelihood-weighting, and Gibbs posterior estimates at the shared sample sizes against the exact posterior. The separate Gibbs chart shows the cumulative Gibbs trace.

The combined comparison checks every exact posterior against `exact_results.json` and saves per-run JSON and CSV results under `results/inference/`. It fixes Gibbs burn-in at 1,000 updates (change with `--gibbs-burn-in`) and thinning at 1 by default. The JSON includes 20 cumulative Gibbs checkpoints per seed; `experiments.plot_gibbs` uses them for the detailed Gibbs chain trace. For Gibbs, the sample size counts retained single-node updates, in addition to burn-in; rejection and likelihood weighting process complete assignments. Runtime includes each sampler's setup and validation and is measured in one interleaved run. The combined plotter compares all three methods' error, posterior estimates by sample size, and runtime. Plotting requires `matplotlib`.