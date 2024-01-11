import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics

# region Part 1
df = pd.read_csv("../TextFiles/smsspamcollection.tsv", sep="\t")
display(df.head())
display(df.isnull().sum())
print(len(df))
display(df["label"].unique())
display(df["label"].value_counts())


plt.xscale("log")
bins = 1.15 ** (np.arange(0, 50))
plt.hist(df[df["label"] == "ham"]["length"], bins=bins, alpha=0.8)
plt.hist(df[df["label"] == "spam"]["length"], bins=bins, alpha=0.8)
plt.legend(("ham", "spam"))
plt.show()

plt.xscale("log")
bins = 1.5 ** (np.arange(0, 15))
plt.hist(df[df["label"] == "ham"]["punct"], bins=bins, alpha=0.8)
plt.hist(df[df["label"] == "spam"]["punct"], bins=bins, alpha=0.8)
plt.legend(("ham", "spam"))
plt.show()

# Split into train and test
X = df[["length", "punct"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lr_model = LogisticRegression(solver="lbfgs")
lr_model.fit(X_train, y_train)
# endregion
# region Part 2
predictions = lr_model.predict(X_test)
df = pd.DataFrame(
    metrics.confusion_matrix(y_test, predictions),
    index=["ham", "spam"],
    columns=["ham", "spam"],
)
display(df)
print(metrics.classification_report(y_test, predictions))

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

predictions = nb_model.predict(X_test)
df = pd.DataFrame(
    metrics.confusion_matrix(y_test, predictions),
    index=["ham", "spam"],
    columns=["ham", "spam"],
)
display(df)
print(metrics.classification_report(y_test, predictions))

svc_model = SVC()
svc_model.fit(X_train, y_train)

predictions = svc_model.predict(X_test)
df = pd.DataFrame(
    metrics.confusion_matrix(y_test, predictions),
    index=["ham", "spam"],
    columns=["ham", "spam"],
)
display(df)
print(metrics.classification_report(y_test, predictions))

# endregion
