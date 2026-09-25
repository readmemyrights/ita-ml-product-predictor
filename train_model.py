import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from features import TitleStatsExtractor

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
