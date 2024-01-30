from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

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

# Local SHAP values
explainer = shap.TreeExplainer(rf)
start_index = 1
end_index = 100
shap_values = explainer.shap_values(X_test[start_index:end_index])
print(X_test[start_index:end_index])

print(shap_values[0].shape)
print(shap_values[0])

shap.initjs()

pred = rf.predict(X_test[start_index:end_index])[0]
print(f"RF predicts: {pred}")
shap.force_plot(
    explainer.expected_value[1], shap_values[1], X_test[start_index:end_index]
)

# global SHAP values
shap.summary_plot(shap_values, X_test)
