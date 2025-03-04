import dice_ml
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
X_train, X_test, y_train, y_test = data_loader.get_data_split()
X_train, y_train = data_loader.oversample(X_train, y_train)

# train model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Create counterfactual explanations
# Dataset
data_dice = dice_ml.Data(
    dataframe=data_loader.data,
    continuous_features=["age", "avg_glucose_level", "bmi"],
    outcome_name="stroke",
)
# Model
rf_dice = dice_ml.Model(model=rf, backend="sklearn")
explainer = dice_ml.Dice(data_dice, rf_dice, method="random")

# Create explanation
input_datapoint = X_test[0:1]
cf = explainer.generate_counterfactuals(
    input_datapoint, total_CFs=3, desired_class="opposite"
)

# Visualise explaantion
cf.visualize_as_dataframe(show_only_changes=True)

# Create feasible counterfactuals
features_to_vary = ["avg_glucose_level", "bmi", "smoking_status_smokes"]
permitted_range = {"avg_glucose_level": [50, 250], "bmi": [18, 25]}

# Generate explanations using new ranges
cf = explainer.generate_counterfactuals(
    input_datapoint,
    total_CFs=3,
    desired_class="opposite",
    permitted_range=permitted_range,
    features_to_vary=features_to_vary,
)

cf.visualize_as_dataframe(show_only_changes=True)
