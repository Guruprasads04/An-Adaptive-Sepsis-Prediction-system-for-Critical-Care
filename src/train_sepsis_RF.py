"""
Train Random Forest for 6-Hour Sepsis Prediction
================================================
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)

# ===============================
# 1. LOAD SPLIT DATASETS
# ===============================
TRAIN_PATH = "dataset/sepsis_data/train_sepsis.csv"
VAL_PATH   = "dataset/sepsis_data/val_sepsis.csv"
TEST_PATH  = "dataset/sepsis_data/test_sepsis.csv"

print("📂 Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"✅ Train shape: {train_df.shape}")
print(f"✅ Validation shape: {val_df.shape}")
print(f"✅ Test shape: {test_df.shape}")

TARGET = "sepsis_next_6h"

# ===============================
# 2. SPLIT FEATURES & LABEL
# ===============================
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ===============================
# 3. CLASS DISTRIBUTION CHECK
# ===============================
print("\n📊 CLASS DISTRIBUTION:")
print(f"Train: {y_train.value_counts().to_dict()}")
print(f"   Sepsis rate: {y_train.mean()*100:.2f}%")
print(f"Validation: {y_val.value_counts().to_dict()}")
print(f"Test: {y_test.value_counts().to_dict()}")

# ===============================
# 4. TRAIN RANDOM FOREST
# ===============================
# Note: Class distribution is nearly balanced (41%), so less aggressive balancing needed
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n🚀 Training Random Forest Model for Sepsis Prediction...")
rf_model.fit(X_train, y_train)

# ===============================
# 5. VALIDATION - FIND OPTIMAL THRESHOLD
# ===============================
val_prob = rf_model.predict_proba(X_val)[:, 1]

# Find optimal threshold using F1 score
thresholds = np.arange(0.3, 0.7, 0.02)
best_f1 = 0
best_threshold = 0.5

print("\n🔍 Finding optimal threshold on validation set...")
for thresh in thresholds:
    val_pred_thresh = (val_prob >= thresh).astype(int)
    f1 = f1_score(y_val, val_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"✅ Optimal Threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

val_pred = (val_prob >= best_threshold).astype(int)

print("\n✅ VALIDATION PERFORMANCE:")
print(f"Accuracy: {accuracy_score(y_val, val_pred):.4f}")
print(classification_report(y_val, val_pred, target_names=['No Sepsis', 'Sepsis']))

# ===============================
# 6. TEST PERFORMANCE
# ===============================
test_prob = rf_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\n" + "="*55)
print(f"✅ FINAL TEST PERFORMANCE - 6-Hour Sepsis Prediction")
print(f"   (Threshold: {best_threshold:.2f})")
print("="*55)
print(f"Accuracy:           {accuracy_score(y_test, test_pred):.4f}")
print(f"F1 Score (Sepsis):  {f1_score(y_test, test_pred):.4f}")
print(f"Precision (Sepsis): {precision_score(y_test, test_pred):.4f}")
print(f"Recall (Sepsis):    {recall_score(y_test, test_pred):.4f}")
print(f"ROC-AUC:            {roc_auc_score(y_test, test_prob):.4f}")
print(f"PR-AUC:             {average_precision_score(y_test, test_prob):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, test_pred, target_names=['No Sepsis', 'Sepsis']))

# ===============================
# 7. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Sepsis", "Sepsis"],
            yticklabels=["No Sepsis", "Sepsis"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Sepsis Prediction - Confusion Matrix\n(6-Hour Horizon, Threshold: {best_threshold:.2f})")
plt.tight_layout()
plt.savefig("model/sepsis_confusion_matrix.png", dpi=150)
plt.show()

# ===============================
# 8. ROC & PR CURVES
# ===============================
fpr, tpr, _ = roc_curve(y_test, test_prob)
roc_auc = roc_auc_score(y_test, test_prob)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_prob)
pr_auc = average_precision_score(y_test, test_prob)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}", color='red', linewidth=2)
axes[0].plot([0, 1], [0, 1], linestyle="--", color='gray')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve - Sepsis Prediction")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Precision-Recall Curve
axes[1].plot(recall_curve, precision_curve, label=f"PR-AUC = {pr_auc:.3f}", color='darkred', linewidth=2)
axes[1].axhline(y=y_test.mean(), linestyle="--", color='gray', label=f"Baseline = {y_test.mean():.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve - Sepsis Prediction")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("model/sepsis_roc_pr_curves.png", dpi=150)
plt.show()

print(f"\n✅ ROC-AUC Score: {roc_auc:.4f}")
print(f"✅ PR-AUC Score: {pr_auc:.4f}")

# ===============================
# 9. FEATURE IMPORTANCE
# ===============================
importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n✅ TOP 15 IMPORTANT FEATURES FOR SEPSIS PREDICTION:")
print(importances.head(15).to_string(index=False))

plt.figure(figsize=(10, 7))
sns.barplot(x="importance", y="feature", data=importances.head(15), palette="Reds_r")
plt.title("Top 15 Feature Importances - Sepsis Prediction (6-Hour Horizon)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("model/sepsis_feature_importance.png", dpi=150)
plt.show()

# ===============================
# 10. SAVE MODEL & THRESHOLD
# ===============================
model_dir = Path("model")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "sepsis_rf_model.pkl"
joblib.dump(rf_model, str(model_path))

threshold_path = model_dir / "sepsis_optimal_threshold.txt"
with open(threshold_path, 'w') as f:
    f.write(f"{best_threshold}")

print(f"\n💾 Sepsis model saved as: {model_path}")
print(f"💾 Optimal threshold saved as: {threshold_path}")

# ===============================
# SUMMARY
# ===============================
print("\n" + "="*55)
print("📋 SEPSIS PREDICTION MODEL SUMMARY")
print("="*55)
print(f"Prediction Horizon: 6 hours")
print(f"Training samples: {len(X_train)}")
print(f"Optimal threshold: {best_threshold:.2f}")
print(f"Test F1 (Sepsis): {f1_score(y_test, test_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, test_pred):.4f}")
print(f"Test Recall: {recall_score(y_test, test_pred):.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")
print(f"Test PR-AUC: {pr_auc:.4f}")
