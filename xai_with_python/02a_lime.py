from utils import DataLoader
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from interpret.blackbox import LimeTabular
from interpret import show
from lime.lime_text import LimeTextExplainer

# region LIME on Stroke dataset
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
X_train, X_test, y_train, y_test = data_loader.get_data_split()
X_train, y_train = data_loader.oversample(X_train, y_train)

# train RF classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Implement LIME
lime = LimeTabular(model=rf.predict_proba, data=X_train, random_state=1)

lime_local = lime.explain_local(X_test[-20:], y_test[-20:], name="LIME")

show(lime_local)
# endregion
# region LIME on text data
categories = ["sci.electronics", "sci.space"]
newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)
newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)
class_names = ["electronics", "space"]

vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

# AutoML experiment:
best = 0
clf = "None"
scores = dict()

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=500),
    # 'Bagging': BaggingClassifier(KNeighborsClassifier(), n_estimators=500,max_samples=0.5, max_features=0.5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1, n_estimators=500),
    "Decicion Tree": DecisionTreeClassifier(random_state=1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=500, random_state=1),
}

for model_name, model in classifiers.items():
    model.fit(train_vectors, newsgroups_train.target)
    pred = model.predict(test_vectors)
    score = f1_score(newsgroups_test.target, pred, average="binary")
    scores[model_name] = score
    if score > best:
        best = score
        clf = model_name
        print(f"{model_name} achieved top score so far: {score}")
    else:
        print(f"{model_name} score: {score}")

best_model = classifiers[clf]
best_model.fit(train_vectors, newsgroups_train.target)
pred = best_model.predict(test_vectors)
print(classification_report(newsgroups_test.target, pred))

# make pipeline for predicting on raw text files
c = make_pipeline(vectorizer, best_model)

# Implement text explainer
explainer = LimeTextExplainer(class_names=class_names)

# Select text to explain
ix = 6
print(newsgroups_test.data[ix])

exp = explainer.explain_instance(
    newsgroups_test.data[ix], c.predict_proba, num_features=10
)

print(f"Document ID {ix}")
print(f"Probability(space) = {c.predict_proba([newsgroups_test.data[ix]])[0,1]}")
print(f"True class: {class_names[newsgroups_test.target[ix]]}")

# explain as a list
for l in exp.as_list():
    print(l)

# explain as plot
fig = exp.as_pyplot_figure()

# explain in notebook
exp.show_in_notebook(text=True)

# endregion
