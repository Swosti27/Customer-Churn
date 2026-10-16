import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

from src.data_preprocessing import TelcoDataPreprocessor, FEATURE_NAMES


def train_model(data_path: str = None, artifact_dir: str = "artifacts"):
    """
    Trains the Customer Churn Prediction model using SMOTE, Logistic Regression, and GridSearchCV
    as established in customer_churn.py, then saves trained model & preprocessor artifacts.
    Supports dataset lookup in datasets/ or root.
    """
    candidate_paths = [
        data_path,
        "datasets/Telco-Customer-Churn.csv",
        "Telco-Customer-Churn.csv",
        "datasets/cleaned_dataset.csv",
        "cleaned_dataset.csv"
    ]

    selected_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            selected_path = p
            break

    if not selected_path:
        raise FileNotFoundError("Could not locate Telco-Customer-Churn.csv or cleaned_dataset.csv in datasets/ or root.")

    print(f"--- Loading Dataset from {selected_path} ---")
    df = pd.read_csv(selected_path)

    # Initialize Preprocessor
    preprocessor = TelcoDataPreprocessor()

    if 'TotalCharges_yj' in df.columns and 'Churn_Yes' in df.columns:
        # Pre-cleaned dataset format
        print("Detected pre-cleaned dataset structure.")
        X = df.drop(columns=['Churn_Yes'])
        y = df['Churn_Yes']
        if 'Unnamed: 0' in X.columns:
            X.drop(columns=['Unnamed: 0'], inplace=True)
        if '' in X.columns:
            X.drop(columns=[''], inplace=True)

        raw_candidate = "datasets/Telco-Customer-Churn.csv" if os.path.exists("datasets/Telco-Customer-Churn.csv") else "Telco-Customer-Churn.csv"
        raw_df = pd.read_csv(raw_candidate) if os.path.exists(raw_candidate) else df
        X, y = preprocessor.preprocess_train_data(raw_df)
    else:
        print("Processing raw dataset...")
        X, y = preprocessor.preprocess_train_data(df)

    print(f"Dataset shape after preprocessing: X={X.shape}, y={y.shape}")

    # Train/Test Split (80/20 stratified split as in customer_churn.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("--- Training Pipeline with SMOTE & Hyperparameter Tuning ---")
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 5.0, 10.0],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'saga']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"True Best Parameters: {grid_search.best_params_}")

    # Model Evaluation on Test Set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n================ Model Performance ================")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    os.makedirs(artifact_dir, exist_ok=True)
    preprocessor.save_artifacts(artifact_dir)

    model_path = os.path.join(artifact_dir, "model.pkl")
    joblib.dump(best_model, model_path)

    features_path = os.path.join(artifact_dir, "feature_names.json")
    with open(features_path, 'w') as f:
        json.dump(FEATURE_NAMES, f, indent=4)

    print(f"--- Artifacts successfully saved to {artifact_dir}/ ---")
    return best_model, preprocessor


if __name__ == "__main__":
    train_model()
