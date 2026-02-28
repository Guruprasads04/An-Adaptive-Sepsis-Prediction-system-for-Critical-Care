import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==============================
# 1. CONFIG
# ==============================
DATA_PATH = "/dataset/processed/Labeled _data.csv"  # change if needed
TARGET_COL = "sepsis_label"

TEST_SIZE = 0.30
RANDOM_STATE = 42

# ==============================
# 1.5 FIND DATASET FILE
# ==============================
parser = argparse.ArgumentParser(description='Train RF and XG models on labeled dataset')
parser.add_argument('--path', '-p', help='Path to labeled CSV', default=None)
args = parser.parse_args()

def find_labeled_data(provided_path=None):
    candidates = []
    if provided_path:
        candidates.append(provided_path)
    # check common locations
    candidates.extend([
        'Labeled_data.csv',
        'labeled_data.csv',
        'dataset/processed/Labeled_data.csv',
        'dataset/Labeled_data.csv',
        'dataset/labeled_data.csv',
        'src/Labeled_data.csv',
        'src/labeled_data.csv',
    ])
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.exists():
            return str(p)
    return None

DATA_FILE = find_labeled_data(args.path)
if DATA_FILE is None:
    print('\n❌ ERROR: Labeled dataset not found!')
    print('\nTried these locations:')
    print('  - Labeled_data.csv')
    print('  - labeled_data.csv')
    print('  - dataset/processed/Labeled_data.csv')
    print('  - dataset/Labeled_data.csv')
    print('  - dataset/labeled_data.csv')
    print('  - src/Labeled_data.csv')
    print('  - src/labeled_data.csv')
    print('\nNext steps:')
    print('  1. Run `python src/label.py` to generate Labeled_data.csv')
    print('  2. Place the file in one of the locations above')
    print('  3. Or pass --path: python src/RF_and_XG.py --path <your_file.csv>')
    sys.exit(1)

print(f"\n📂 Loading dataset: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print("✅ Shape:", df.shape)
print("✅ Columns:", df.columns.tolist())

if TARGET_COL not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COL}' not found!")

# Drop ID-like columns (patient-wise leakage)
drop_cols = []
for col in ["patient_id", "HospAdmTime"]:
    if col in df.columns:
        drop_cols.append(col)

if drop_cols:
    print("🧹 Dropping ID-like columns:", drop_cols)
    df = df.drop(columns=drop_cols)

# Separate X, y
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

print("✅ Final feature shape:", X.shape)

# ==============================
# 3. TRAIN–TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n✅ Train shape:", X_train.shape)
print("✅ Test shape :", X_test.shape)

# Class imbalance ratio
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"\n⚖️ Class ratio (train) Non-sepsis : Sepsis = {ratio:.2f} : 1")

# ==============================
# 4. DEFINE MODELS
# ==============================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# ==============================
# 5. TRAIN INDIVIDUAL MODELS
# ==============================
print("\n🚀 Training Random Forest...")
rf.fit(X_train, y_train)

print("🚀 Training XGBoost...")
xgb.fit(X_train, y_train)

# ==============================
# 6. HELPER: EVAL FUNCTION
# ==============================
def evaluate_model(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n📊 {name} PERFORMANCE:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }

# ==============================
# 7. EVALUATE RF
# ==============================
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)
rf_metrics = evaluate_model("Random Forest", y_test, rf_pred, rf_prob)

# ==============================
# 8. EVALUATE XGBOOST
# ==============================
xgb_prob = xgb.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_prob >= 0.5).astype(int)
xgb_metrics = evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob)

# ==============================
# 9. ENSEMBLE: PROBABILITY AVERAGING
# ==============================
ensemble_prob = (rf_prob + xgb_prob) / 2.0

# Default threshold 0.5
ensemble_pred_05 = (ensemble_prob >= 0.5).astype(int)
ens_metrics_05 = evaluate_model("Ensemble (avg prob, thr=0.5)", y_test, ensemble_pred_05, ensemble_prob)

# ==============================
# 10. OPTIONAL: THRESHOLD TUNING FOR F1
# ==============================
print("\n🔎 Searching best threshold for F1 (Ensemble)...")
best_thr = 0.5
best_f1 = 0.0

for thr in np.arange(0.2, 0.9, 0.02):
    preds = (ensemble_prob >= thr).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"✅ Best threshold by F1: {best_thr:.2f} (F1 = {best_f1:.4f})")

ensemble_pred_opt = (ensemble_prob >= best_thr).astype(int)
ens_metrics_opt = evaluate_model(
    f"Ensemble (avg prob, thr={best_thr:.2f})", y_test, ensemble_pred_opt, ensemble_prob
)

# ==============================
# 11. SUMMARY TABLE
# ==============================
summary = pd.DataFrame([
    rf_metrics,
    xgb_metrics,
    ens_metrics_05,
    ens_metrics_opt
])

print("\n📊 FINAL COMPARISON SUMMARY:")
print(summary)

summary.to_csv("ensemble_model_comparison.csv", index=False)
print("\n💾 Saved comparison summary to: ensemble_model_comparison.csv")
