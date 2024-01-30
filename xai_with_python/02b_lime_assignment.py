from utils import DataLoader
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer

# load in text data for all classes
newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")
# make class_names shorter
class_names = [
    x.split(".")[-1] if "misc" not in x else ".".join(x.split(".")[-2:])
    for x in newsgroups_train.target_names
]
class_names[3] = "pc.hardware"
class_names[4] = "mac.hardware"

# vectorize data
vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

# Train classifier
nb = MultinomialNB(alpha=0.1)
nb.fit(train_vectors, newsgroups_train.target)

# Evaluate classifier
pred = nb.predict(test_vectors)
print(classification_report(newsgroups_test.target, pred))

# make pipeline for predicting on raw text files
c = make_pipeline(vectorizer, nb)

# Implement text explainer
explainer = LimeTextExplainer(class_names=class_names)

# Select text to explain
ix = 1340
print(newsgroups_test.data[ix])

exp = explainer.explain_instance(
    newsgroups_test.data[ix], c.predict_proba, num_features=6, labels=[0, 17]
)

print(f"Document ID {ix}")
print(
    f"Predicted class = {class_names[nb.predict(test_vectors[ix]).reshape(1,-1)[0,0]]}"
)
print(f"True class: {class_names[newsgroups_test.target[ix]]}")

print(f"Explanation for class {class_names[0]}")
print("\n".join(map(str, exp.as_list(label=0))))
print()
print(f"Explanation for class {class_names[17]}")
print("\n".join(map(str, exp.as_list(label=17))))


exp = explainer.explain_instance(
    newsgroups_test.data[ix], c.predict_proba, num_features=6, top_labels=2
)
print(exp.available_labels())

# explain in notebook
exp.show_in_notebook(text=True)

# repeat but remove headers, footers, and quotes
newsgroups_train = fetch_20newsgroups(
    subset="train", remove=("headers", "footers", "quotes")
)
newsgroups_test = fetch_20newsgroups(
    subset="test", remove=("headers", "footers", "quotes")
)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
nb = MultinomialNB(alpha=0.1)
nb.fit(train_vectors, newsgroups_train.target)
c = make_pipeline(vectorizer, nb)
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(
    newsgroups_test.data[ix], c.predict_proba, num_features=6, top_labels=2
)
print(exp.available_labels())

exp.show_in_notebook(text=False)
