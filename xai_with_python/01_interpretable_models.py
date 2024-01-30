from utils import DataLoader
from interpret.glassbox import (
    LogisticRegression,
    ClassificationTree,
    ExplainableBoostingClassifier,
)
from interpret import show
from sklearn.metrics import classification_report

data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

X_train, X_test, y_train, y_test = data_loader.get_data_split()
X_train, y_train = data_loader.oversample(X_train, y_train)

# region Logistic Regression
lr = LogisticRegression(
    random_state=2021, feature_names=X_train.columns, penalty="l1", solver="liblinear"
)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))

lr_local = lr.explain_local(X_test[:100], y_test[:100], name="Logistic Regression")
show(lr_local)

lr_global = lr.explain_global(name="Logistic Regression")

show(lr_global)
# endregion
# region DT Classifier
tree = ClassificationTree()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(classification_report(y_test, y_pred))

tree_local = tree.explain_local(X_test[:100], y_test[:100], name="Classifier")
show(tree_local)

tree_global = tree.explain_global(name="Classifier")
show(tree_global)
# endregion
# region Explainable Boosting Classifier
ebm = ExplainableBoostingClassifier(random_state=2021)
ebm.fit(X_train, y_train)
y_pred = ebm.predict(X_test)
print(classification_report(y_test, y_pred))

ebm_local = ebm.explain_local(X_test[:100], y_test[:100], name="EBM")
show(ebm_local)

ebm_global = ebm.explain_global(name="EBM")
show(ebm_global)

# endregion
