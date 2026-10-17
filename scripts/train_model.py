"""
Step 2: Train a Fraud Detection Model
======================================
Model      : XGBoost (gradient-boosted decision trees)
Imbalance  : SMOTE oversampling on the minority (fraud) class in training split
Evaluation : ROC-AUC, Precision, Recall, F1, Confusion Matrix
Outputs    : models/fraud_model.pkl, reports/classification_report.txt,
             reports/confusion_matrix.png, reports/roc_curve.png,
             reports/feature_importance.png

--- Why XGBoost? ---
Fraud detection on tabular data is a structured classification problem.
XGBoost consistently outperforms simpler models here because it:
  - handles mixed feature types (numeric + encoded categoricals) natively
  - is robust to feature scale differences (no need to normalise)
  - natively supports class_weight via scale_pos_weight
  - provides interpretable feature importances

--- Why SMOTE? ---
The dataset has a 2 % fraud rate. Without correction, a naive model can
achieve 98 % accuracy by predicting "legit" every time. SMOTE synthesises
new minority-class samples in feature space so the model actually learns
fraud patterns rather than the majority prior.

--- Assumptions ---
  1. All features in the CSV are available at inference time (no leakage).
  2. The fraud label (is_fraud) is the ground truth target.
  3. A 70/30 stratified train/test split is used; SMOTE is applied only to
     the training fold to avoid data leakage into the test set.
  4. Threshold is left at 0.5 for the main report; in production you would
     tune this to balance precision vs. recall to match business cost.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RANDOM_SEED = 42
DATA_PATH   = "data/transactions.csv"
MODEL_PATH  = "models/fraud_model.pkl"
REPORT_DIR  = "reports"

# --------------------------------------------------------------------------- #
# 1. Load & Prepare Data
# --------------------------------------------------------------------------- #

print("=" * 60)
print("STEP 2: FRAUD DETECTION MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Loaded {len(df):,} rows | Fraud rate: {df['is_fraud'].mean():.1%}")

# Drop non-predictive columns
df = df.drop(columns=["transaction_id", "timestamp"])

# Encode categorical columns
categorical_cols = ["merchant_category", "card_type", "country"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

feature_cols = [c for c in df.columns if c != "is_fraud"]
X = df[feature_cols].values
y = df["is_fraud"].values

print(f"[1] Features ({len(feature_cols)}): {feature_cols}")

# --------------------------------------------------------------------------- #
# 2. Train / Test Split  (stratified to preserve fraud ratio)
# --------------------------------------------------------------------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
print(f"\n[2] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"    Train fraud rate: {y_train.mean():.1%} | Test fraud rate: {y_test.mean():.1%}")

# --------------------------------------------------------------------------- #
# 3. Handle Class Imbalance with SMOTE (training set only)
# --------------------------------------------------------------------------- #

smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\n[3] After SMOTE — Train: {len(X_train_res):,} rows | "
      f"Fraud rate: {y_train_res.mean():.1%}")

# --------------------------------------------------------------------------- #
# 4. Train XGBoost Classifier
# --------------------------------------------------------------------------- #

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

print("\n[4] Training XGBoost ...")
model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("    Training complete.")

# --------------------------------------------------------------------------- #
# 5. Cross-Validation on Original (pre-SMOTE) Training Data
# --------------------------------------------------------------------------- #

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"\n[5] 5-Fold CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# --------------------------------------------------------------------------- #
# 6. Evaluation on Held-Out Test Set
# --------------------------------------------------------------------------- #

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

roc_auc  = roc_auc_score(y_test, y_pred_prob)
avg_prec = average_precision_score(y_test, y_pred_prob)

print(f"\n[6] Test Set Results")
print(f"    ROC-AUC             : {roc_auc:.4f}")
print(f"    Avg Precision (PR)  : {avg_prec:.4f}")
print(f"\n    Classification Report:\n")
report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
print(report)

# Save text report
os.makedirs(REPORT_DIR, exist_ok=True)
with open(f"{REPORT_DIR}/classification_report.txt", "w") as f:
    f.write("FRAUD DETECTION MODEL — EVALUATION REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"ROC-AUC           : {roc_auc:.4f}\n")
    f.write(f"Avg Precision (PR): {avg_prec:.4f}\n")
    f.write(f"5-Fold CV AUC     : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}\n\n")
    f.write(report)

# --------------------------------------------------------------------------- #
# 7. Plots
# --------------------------------------------------------------------------- #

# -- Confusion Matrix --------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print(f"\n[7] Saved confusion_matrix.png")

# -- ROC Curve ---------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"XGBoost (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Fraud Detection")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/roc_curve.png", dpi=150)
plt.close()
print(f"[7] Saved roc_curve.png")

# -- Precision-Recall Curve --------------------------------------------------
prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color="darkorange", lw=2, label=f"AP = {avg_prec:.3f}")
ax.axhline(y=df["is_fraud"].mean(), color="grey", linestyle="--", label="Baseline (random)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Fraud Detection")
ax.legend()
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/pr_curve.png", dpi=150)
plt.close()
print(f"[7] Saved pr_curve.png")

# -- Feature Importances -----------------------------------------------------
importances = model.feature_importances_
feat_df = (
    pd.DataFrame({"feature": feature_cols, "importance": importances})
    .sort_values("importance", ascending=True)
)
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
ax.set_xlabel("Importance (gain)")
ax.set_title("XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/feature_importance.png", dpi=150)
plt.close()
print(f"[7] Saved feature_importance.png")

# --------------------------------------------------------------------------- #
# 8. Save Model + Metadata
# --------------------------------------------------------------------------- #

artifact = {
    "model": model,
    "label_encoders": label_encoders,
    "feature_cols": feature_cols,
    "roc_auc": roc_auc,
    "avg_precision": avg_prec,
}
joblib.dump(artifact, MODEL_PATH)
print(f"\n[8] Model saved to {MODEL_PATH}")
print("\nDone.")
