## progress/what has been done

- Downloaded the [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
- Added preprocessing to check invalid values and duplicates, fix a merged CSV header, and convert features into categories.
- Kept games with **at least 100 reviews**; labeled them successful if **80% or more reviews are positive**. These thresholds are provisional.
- Generated **22,804 eligible games**, split into **18,244 training** and **4,560 test** rows, plus feature-distribution charts.

The source has 125,855 games: 103,050 were below the review minimum and one was released after the cutoff. Of the eligible games, 13,137 have `success=yes` and 9,667 have `success=no`.

The inputs are price, release era, estimated owners, peak concurrent players, description length, and Indie/RPG genre flags. The target is `success`. We haven't trained a  Bayesian network yet.

## View the data

The prepared CSVs are included in the repo. Open them in Excel, a text editor, or your own notebook; no download script or extra packages are needed to access them.

| File | What to use it for |
| --- | --- |
| [cleaned_games.csv](data/processed/cleaned_games.csv) | Browse all 22,804 eligible games, including names, prices, genres, review information, and categories |
| [train.csv](data/processed/train.csv) | Learn the network and its probabilities from 18,244 games |
| [test.csv](data/processed/test.csv) | Evaluate the model on the remaining 4,560 games |
| [metadata.json](data/processed/metadata.json) | Check dataset version, category names, settings, and filtering counts |
| [figures](data/processed/figures) | View the feature-distribution charts |

Use the existing train/test split. The matching `train_ids.csv` and `test_ids.csv` identify the games in each file in row order. Review counts and positive fractions are inspection fields, not model inputs: they define the success label. This label measures review reception, not profitability; owners and player counts describe games after release.

The full original 125,855-row dataset is kept locally in `data/raw/games.csv` and excluded from Git because of its size. The CSV handoff does not remove any more games or columns from the existing processed files. The 100-review cutoff excludes games with little review evidence; it does not mean they failed.

`preprocess.py` keeps the cleaning and feature choices reproducible. To rebuild the CSVs, manually download version 59 from the [dataset page](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/versions/59), extract `games.csv` to `data/raw/`, and run:

```sh
python preprocess.py --input data/raw/games.csv
```

The script uses only Python's built-in modules.
