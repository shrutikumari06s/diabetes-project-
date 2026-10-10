"""
train_model.py
==============
Trains and evaluates two ML models (Logistic Regression and Random Forest)
on the Diabetes Prediction Dataset.

Run this script once before starting the dashboard:
    python train_model.py

The best pipeline is saved to: models/best_pipeline.joblib
Metrics are saved to:          models/model_metrics.json
"""

import os
import sys
import json

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------
# 1. PATHS
# ---------------------------------------------
DATASET_PATH = os.path.join("data", "diabetes_prediction_dataset.csv")
MODELS_DIR   = "models"
PIPELINE_PATH  = os.path.join(MODELS_DIR, "best_pipeline.joblib")
METRICS_PATH   = os.path.join(MODELS_DIR, "model_metrics.json")

REQUIRED_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes",
]
TARGET_COLUMN = "diabetes"
RANDOM_STATE  = 42

# ---------------------------------------------
# 2. LOAD & VALIDATE DATASET
# ---------------------------------------------
print("=" * 60)
print("  AI-Based Diabetes Risk Prediction — Model Training")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    sys.exit(
        f"\n[ERROR] Dataset not found at '{DATASET_PATH}'.\n"
        "Please place 'diabetes_prediction_dataset.csv' inside the 'data/' folder.\n"
    )

print(f"\n[1/6] Loading dataset from '{DATASET_PATH}' ...")
df = pd.read_csv(DATASET_PATH)

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    sys.exit(
        f"\n[ERROR] The following required columns are missing from the CSV: {missing_cols}\n"
        "Please check the dataset and column names.\n"
    )

# ---------------------------------------------
# 3. DATASET INSPECTION
# ---------------------------------------------
print("\n[2/6] Dataset Inspection")
print("-" * 40)
print(f"  Shape                : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n  Column data types:")
for col, dtype in df.dtypes.items():
    print(f"    {col:<25} {dtype}")

print(f"\n  Missing values per column:")
missing = df.isnull().sum()
any_missing = False
for col, n in missing.items():
    if n > 0:
        any_missing = True
        print(f"    {col:<25} {n}")
if not any_missing:
    print("    None — dataset is complete.")

duplicates = df.duplicated().sum()
print(f"\n  Duplicate rows       : {duplicates}")

print(f"\n  Target class distribution ('{TARGET_COLUMN}'):")
dist = df[TARGET_COLUMN].value_counts().sort_index()
total = len(df)
for cls, count in dist.items():
    label = "Positive (Diabetic)" if cls == 1 else "Negative (Non-Diabetic)"
    print(f"    Class {cls} [{label}]: {count:,} ({count/total*100:.1f}%)")

# ---------------------------------------------
# 4. PREPROCESSING SETUP
# ---------------------------------------------
print("\n[3/6] Preparing features and preprocessing pipeline ...")

# Drop duplicate rows (keep first occurrence)
df_clean = df.drop_duplicates()
print(f"  Rows after removing duplicates: {len(df_clean):,}")

X = df_clean.drop(columns=[TARGET_COLUMN])
y = df_clean[TARGET_COLUMN]

# Feature groups
numerical_features    = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
binary_features       = ["hypertension", "heart_disease"]   # already 0/1 — pass through
categorical_features  = ["gender", "smoking_history"]

# Stratified train/test split — preserves class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"  Training set : {len(X_train):,} rows")
print(f"  Test set     : {len(X_test):,}  rows")

# Column transformer
#   - Numerical  -> StandardScaler
#   - Binary     -> pass through (already numeric 0/1)
#   - Categorical -> OneHotEncoder (handle unknown categories gracefully)
preprocessor = ColumnTransformer(
    transformers=[
        ("num",  StandardScaler(),
         numerical_features),
        ("bin",  "passthrough",
         binary_features),
        ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_features),
    ],
    remainder="drop",
)

# ---------------------------------------------
# 5. MODEL TRAINING & EVALUATION
# ---------------------------------------------
# Class imbalance: ~91.5% negative, ~8.5% positive.
# We use class_weight='balanced' so that the minority class
# (diabetic = 1) receives proportionally more weight during training.
# This improves Recall for the positive class without oversampling.

print("\n[4/6] Training and evaluating models ...")
print("  (class_weight='balanced' used to handle 91.5%/8.5% imbalance)\n")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

results = {}

for name, model in models.items():
    print(f"  -- {name} --")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])

    # Fit ONLY on training data to prevent data leakage
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    y_prob = None
    roc_auc = None
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_prob  = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))

    acc       = float(accuracy_score(y_test, y_pred))
    prec      = float(precision_score(y_test, y_pred, zero_division=0))
    rec       = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))
    cm        = confusion_matrix(y_test, y_pred).tolist()

    results[name] = {
        "accuracy":   round(acc, 4),
        "precision":  round(prec, 4),
        "recall":     round(rec, 4),
        "f1_score":   round(f1, 4),
        "roc_auc":    round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "pipeline":   pipeline,   # stored temporarily for saving
    }

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-score  : {f1:.4f}")
    if roc_auc is not None:
        print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"      FN={cm[1][0]}  TP={cm[1][1]}")
    print()
    print(f"    Classification Report:")
    print("    " + classification_report(y_test, y_pred,
            target_names=["Negative (0)", "Positive (1)"]).replace("\n", "\n    "))

# ---------------------------------------------
# 6. MODEL SELECTION
# ---------------------------------------------
# Selection criterion: F1-score for the positive class (diabetes=1).
# Accuracy alone is misleading with an imbalanced dataset (~91.5% negative).
# F1-score balances Precision and Recall for the minority class.
print("[5/6] Selecting best model ...")
print("  Criterion: F1-score for positive class (diabetes = 1)")

best_name = max(results, key=lambda n: results[n]["f1_score"])
print(f"  [OK] Best model: {best_name}")
print(f"    F1-score = {results[best_name]['f1_score']:.4f}")

# ---------------------------------------------
# 7. SAVE PIPELINE & METRICS
# ---------------------------------------------
print(f"\n[6/6] Saving pipeline and metrics ...")
os.makedirs(MODELS_DIR, exist_ok=True)

best_pipeline = results[best_name]["pipeline"]
joblib.dump(best_pipeline, PIPELINE_PATH)
print(f"  Pipeline saved -> {PIPELINE_PATH}")

# Build metrics dict (no pipeline objects)
metrics_to_save = {
    "best_model": best_name,
    "selection_criterion": "F1-score (positive class)",
    "models": {},
}
for name, res in results.items():
    metrics_to_save["models"][name] = {
        k: v for k, v in res.items() if k != "pipeline"
    }

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_to_save, f, indent=2)
print(f"  Metrics saved  -> {METRICS_PATH}")

print("\n" + "=" * 60)
print("  Training complete! You can now run the dashboard:")
print("  python -m streamlit run app.py")
print("=" * 60 + "\n")
