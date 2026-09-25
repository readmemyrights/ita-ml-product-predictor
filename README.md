# ita-ml-product-predictor

A machine learning model to classify products into categories
based on title, description and other features.

## Project structure

* `data/`: training data.
* `exploration.ipynb`: the notebook with explorative data analysis and model comparison
* `train_model.py`: trains a model on the full dataset and dumps the full pipeline
* `features.py`: feature engineering transformer helper class
* `model/random-forest-category-predictor.pkl`: output of `train_model.py`: full pipeline
* `category_predictor.py`: sample script for interactive testing of the model.

## How to run

For quick interactive tests,
one can use the provided `category_predictor.py` script.
The script takes an optional argument
to an alternative model in `.pkl` format.
The only thing the script assumes about the model
is that it takes a title column.

The provided `model/random-forest-category-predictor.pkl` file
can be used in other scripts,
but it requires the class `TitleStatsExtractor`
from the `features.py` module to be imported.
