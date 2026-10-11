# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave a service.

## Project Overview

This project uses the IBM Telco Customer Churn dataset to train a classification model that identifies customers who may churn.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib

## Project Workflow

1. Load and explore the customer dataset.
2. Clean missing values and preprocess the data.
3. Encode categorical features and prepare numerical features.
4. Split the dataset into training and testing sets.
5. Train an XGBoost classification model.
6. Evaluate the model using classification metrics.
7. Analyse feature importance and SHAP explanations.

## Model Performance

The XGBoost model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 80.41% |
| Precision | 66.78% |
| Recall | 52.14% |
| F1-score | 58.56% |
| ROC-AUC | 84.14% |

These results show the model's performance on the held-out test data. Performance may vary with different data and settings.

## Visualisations

The project includes:

- `confusion_matrix.png` — compares predicted and actual classes.
- `feature_importance.png` — displays influential model features.
- `shap_summary.png` — helps interpret feature contributions to predictions.

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-xgboost
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

```bash
python churn_analysis.py
```

## Disclaimer

This project is for educational and analytical purposes. Predictions should be validated before being used for real business decisions.
