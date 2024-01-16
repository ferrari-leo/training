import nltk
import pandas as pd

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report

sid = SentimentIntensityAnalyzer()

a = "This is a good movie"
print(sid.polarity_scores(a))

b = "This was the best, most awesome movie EVER MADE!!!"
print(sid.polarity_scores(b))

c = "This was the WORST movie that has ever disgraced the screen"
print(sid.polarity_scores(c))

df = pd.read_csv("../TextFiles/amazonreviews.tsv", sep="\t")
display(df.head())
display(df["label"].value_counts())

df.dropna(inplace=True)

blanks = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)

print(sid.polarity_scores(df.iloc[0]["review"]))

df["scores"] = df["review"].apply(lambda review: sid.polarity_scores(review))
df["compound"] = df["scores"].apply(lambda d: d["compound"])
df["comp_score"] = df["compound"].apply(lambda score: "pos" if score >= 0 else "neg")
display(df.head())

print(classification_report(df["label"], df["comp_score"]))
print(confusion_matrix(df["label"], df["comp_score"]))
