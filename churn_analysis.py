
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

# 1. Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce",
)

df.loc[
    df["TotalCharges"].isna() & (df["tenure"] == 0),
    "TotalCharges",
] = 0

# 3. Prepare target and features
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns=["Churn"])
y = df["Churn"]

# 4. Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 5. Preprocess numeric and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ]
)

# 6. Baseline model: always predicts the majority class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

# 7. XGBoost model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
    ),
])

model.fit(X_train, y_train)

# 8. Predictions and evaluation
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

def show_metrics(name, actual, predicted, probabilities=None):
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(actual, predicted):.4f}")
    print(f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(actual, predicted, zero_division=0):.4f}")

    if probabilities is not None:
        print(f"ROC-AUC:   {roc_auc_score(actual, probabilities):.4f}")

show_metrics("BASELINE MODEL", y_test, baseline_predictions)
show_metrics("XGBOOST MODEL", y_test, predictions, probabilities)

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# 9. Save confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("XGBoost - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 10. Save feature importance plot
fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()
importance = fitted_classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance,
}).sort_values("Importance", ascending=False)

importance_df.head(15).sort_values("Importance").plot(
    x="Feature",
    y="Importance",
    kind="barh",
    legend=False,
)
plt.title("Top 15 XGBoost Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# 11. SHAP explanations on a sample of test rows
X_test_transformed = fitted_preprocessor.transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

sample_size = min(200, X_test_transformed.shape[0])
X_shap = X_test_transformed[:sample_size]

explainer = shap.TreeExplainer(fitted_classifier)
shap_values = explainer.shap_values(X_shap)

shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    max_display=15,
    show=False,
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# 12. Save test predictions
results = X_test.copy()
results["ActualChurn"] = y_test
results["PredictedChurn"] = predictions
results["ChurnProbability"] = probabilities
results.to_csv("churn_predictions.csv", index=False)

# 13. Save model pipeline
import joblib
joblib.dump(model, "churn_model.joblib")

print("\nSaved files:")
print("- confusion_matrix.png")
print("- feature_importance.png")
print("- shap_summary.png")
print("- churn_predictions.csv")
print("- churn_model.joblib")
print("\nProject completed successfully.")


