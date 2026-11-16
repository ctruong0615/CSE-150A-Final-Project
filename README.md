## progress/what has been done

- Downloaded the [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
- Added preprocessing to check invalid values and duplicates, fix a merged CSV header, and convert features into categories.
- Kept games with **at least 100 reviews**; labeled them successful if **80% or more reviews are positive**. These thresholds are provisional.
- Generated **22,804 eligible games**, split into **18,244 training** and **4,560 test** rows, plus feature-distribution charts.

The source has 125,855 games: 103,050 were below the review minimum and one was released after the cutoff. Of the eligible games, 13,137 have `success=yes` and 9,667 have `success=no`.

The inputs are price, release era, estimated owners, peak concurrent players, description length, and Indie/RPG genre flags. The target is `success`. No Bayesian network has been trained yet.

## Get started

Use **Python 3.10+**. No extra packages are needed.

From the project folder, download the data once, then preprocess it:

```sh
python download_data.py --version 59
python preprocess.py --input data/raw/games.csv
```

Skip the download if `data/raw/games.csv` already exists. On Windows, try `py -3` if `python` is unavailable. Raw and generated data are excluded from Git, so teammates cloning the repo need to run these commands.

## Start your part

Use `data/processed/train.csv` to learn the network and its probabilities. Reserve `test.csv` for final evaluation. Both files contain categorical features and the `success` label.

| Lead | Next work |
| --- | --- |
| Calvin | Review preprocessing choices and charts |
| Daniel | Define and learn the network structure |
| Nathan | Learn conditional probability tables and implement exact inference |
| Kai / Minyoung | Implement rejection sampling, likelihood weighting, and Gibbs sampling; agree on the individual split |

Calvin, Daniel, and Nathan are collaborating on the first three parts.

`data/processed/cleaned_games.csv` keeps extra details for inspection; `metadata.json` records settings, category states, filtering counts, and the source-file hash; `figures/` contains charts. Download provenance is saved in `data/raw/source.json`.

## Preprocessing choices

The script keeps the first record for each AppID, rejects invalid required values, and excludes release dates after **November 15, 2026**. It fixes the source's merged `DiscountDLC count` header while reading, checks row widths, and leaves the raw CSV unchanged. Owner ranges use their midpoint; descriptions become word counts after removing HTML. Missing optional values become `unknown`.

| Column | Categories |
| --- | --- |
| `price_band` | free; above $0 to under $10; $10 to under $30; $30+ |
| `release_era` | through 2010; 2011-2015; 2016-2020; 2021+ |
| `owners_band` | under 20,000; 20,000 to under 100,000; 100,000+ |
| `peak_ccu_band` | under 100; 100-999; 1,000+ |
| `description_length` | under 100; 100-299; 300+ words |
| `is_indie`, `is_rpg` | yes/no genre membership |
| `success` | yes/no using the review threshold above |

Use the exact state strings in `preprocess.STATES` or generated `metadata.json`. The split preserves approximate success proportions with random seed **42**. Use training/validation data to choose settings; keep the test set for final evaluation.

Success here means review reception among games with enough reviews, not profitability. Review counts and positive fractions are excluded from model inputs because they define the label. Owners and player counts are post-release observations, so predictions using them describe existing games. Genre coverage currently uses only Indie/RPG flags. Network edges alone do not establish causation.

## Connecting the methods

Agree on one shared model with nodes, ordered states, ordered parents, and conditional probability tables. Include all allowed states, even those absent from training; check for cycles and complete, normalized probability tables. Record the training-data hash, structure choice, and any smoothing.

Each inference method should take the same model, query, and evidence, and return probabilities plus runtime. Samplers also need a seed, sample count, and diagnostics. Report impossible evidence or zero usable samples clearly.

Start with `P(success=yes)`, then try evidence such as `price_band=under_10`, or `is_indie=yes` together with `is_rpg=yes`. Compare samplers against exact inference on the same model with 1,000, 5,000, and 10,000 samples across multiple seeds. Track error, runtime, acceptance/weight quality, and Gibbs burn-in/convergence as appropriate. These methods are still to be implemented.

To check preprocessing:

```sh
python -m unittest discover -s tests -v
```

This AI-assisted starter needs team review and adaptation. Record AI use and actual contributions in the final report.
