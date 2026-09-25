import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TitleStatsExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        titles = np.asarray(X).ravel()

        lengths = np.array([len(t) for t in titles])
        n_digits = np.array([sum(c.isdigit() for c in t) for t in titles])

        return np.column_stack([lengths, n_digits])

    def get_feature_names_out(self, input_features=None):
        return np.array(["length", "number_of_digits"])


