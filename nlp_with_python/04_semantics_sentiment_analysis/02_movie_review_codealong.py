import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("../TextFiles/moviereviews.tsv", sep="\t")
df.dropna(inplace=True)
blanks = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)

df.drop(blanks, inplace=True)

print(df["label"].value_counts())

sid = SentimentIntensityAnalyzer()
df["scores"] = df["review"].apply(lambda r: sid.polarity_scores(r))
df["compound"] = df["scores"].apply(lambda d: d["compound"])
df["comp score"] = df["compound"].apply(lambda s: "pos" if s >= 0 else "neg")

display(df.head())

print(classification_report(df["label"], df["comp score"]))
print(confusion_matrix(df["label"], df["comp score"]))
