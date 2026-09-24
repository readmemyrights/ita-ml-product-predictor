import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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

df = pd.read_csv("data/products.csv")
# Originalni skup podataka je imao ekstra razmak u imenu kolone Category Label
df = df.filter(["Product Title", " Category Label"])
df = df.rename(columns={
    "Product Title": "title",
    " Category Label": "category"
})
df = df.dropna()
df.title = df.title.astype(str)
df.category = df.category.astype('category')
df.category = df.category.astype(str).replace({
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
    "fridge": "Fridge Freezers",
    "Fridges": "Fridge Freezers",
    "Freezers": "Fridge Freezers"
}).astype('category')
X = df[["title"]]
y = df.category
stats_process = Pipeline([
    ("Extraction", TitleStatsExtractor()),
    ("Scaler", MinMaxScaler())
])
preprocess = ColumnTransformer([
    ("Tfidf", TfidfVectorizer(), "title"),
    ("Stats", stats_process, "title")
])
pipeline = Pipeline([
    ("Preprocess", preprocess),
    ("Classify", RandomForestClassifier())
])
pipeline.fit(X, y)
joblib.dump(pipeline, "model/random-forest-category-predictor.pkl")
