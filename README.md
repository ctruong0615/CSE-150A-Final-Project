## progress/what has been done

- Downloaded the [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
- Added preprocessing to check invalid values and duplicates, fix a merged CSV header, and convert features into categories.
- Kept games with **at least 100 reviews**; labeled them successful if **80% or more reviews are positive**. These thresholds are provisional.
- Generated **22,804 eligible games**, split into **18,244 training** and **4,560 test** rows, plus feature-distribution charts.

The source has 125,855 games: 103,050 were below the review minimum and one was released after the cutoff. Of the eligible games, 13,137 have `success=yes` and 9,667 have `success=no`.

The inputs are price, release era, estimated owners, peak concurrent players, description length, and Indie/RPG genre flags. The target is `success`. The network structures are defined below. Learning their probability tables and running inference are separate parts of the project.

Open [cleaned_games.csv](cleaned_games.csv) directly in Excel, or load it in a Jupyter notebook opened from the project folder (with pandas installed):

```python
import pandas as pd
games = pd.read_csv("cleaned_games.csv")
games.head()
```

## Bayesian network structure learning

This part looks at how the game features are connected. The code starts with a network based on assumptions about relationships between features, such as price, genre, and game success. It then learns another network from the training data and compares the two.

For the learned network, the variables are put in a fixed order. Each variable can have up to three parents, chosen from the variables before it. The code tries the possible parent groups and uses BIC to choose between them. BIC rewards a better fit to the data but also adds a penalty for larger probability tables.

The learned network has 11 connections, compared with 14 in the manual network, and gets a better BIC score on the training data. The test data is kept separate. These results compare the network structures; they do not measure prediction accuracy or show that one feature causes another.

The graphs, network diagrams, and comparison table are saved in `results/structure/`. The JSON files list the connections and parents for each variable so they can be used in the parameter-learning part.

## Parameter learning and exact inference

- [parameter_learning.py](parameter_learning.py) fits conditional probability tables for the learned network using maximum likelihood on the training split. Unseen parent configurations use a uniform distribution.
- [exact_inference.py](exact_inference.py) computes exact success probabilities by variable elimination. It scored 59.00% accuracy on 4,559 held-out games. One game's evidence had zero probability and was excluded.
- For rejection sampling, likelihood weighting, and Gibbs sampling, load [trained_network.json](results/inference/trained_network.json) and use the evidence and exact posteriors in the `queries` array of [exact_results.json](results/inference/exact_results.json) for comparison.
