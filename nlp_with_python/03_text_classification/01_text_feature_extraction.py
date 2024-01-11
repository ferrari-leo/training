import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

# region Part 1
df = pd.read_csv("../TextFiles/smsspamcollection.tsv", sep="\t")
display(df.head())
display(df["label"].value_counts())

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
count_vect = CountVectorizer()

# fit  count vectorizer: build vocab, count words, etc
X_train_counts = count_vect.fit_transform(X_train)
# endregion
# region Part 2
# fit tfidf transformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# fit both at same time
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# fit classifier
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)

# pipeline
text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])

text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

text_clf.predict(["Hi how are you doing today?"])
text_clf.predict(
    [
        "Congratulations! You've been selected as a winner. TEXT WON to 4425 congratulations free entry to contest"
    ]
)
# endregion
