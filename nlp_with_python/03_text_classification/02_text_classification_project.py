import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("../TextFiles/moviereviews.tsv", sep="\t")
display(df.head())
print(df["review"][0])
display(df.isnull().sum())
df.dropna(inplace=True)

# remove blanks
blanks = []
for i, lb, rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)

df.drop(blanks, inplace=True)

print(len(df))

X = df["review"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])

text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
