from features import TitleStatsExtractor
import joblib
import pandas as pd
import sys

pipeline = joblib.load(sys.argv[1] if len(sys.argv) >= 2 else "model/random-forest-category-predictor.pkl")
print("model učitan: exit da izađete")
while True:
    title = input("? ")
    if title == "exit":
        break
    df = pd.DataFrame([{"title": title}])
    print(pipeline.predict(df)[0])
