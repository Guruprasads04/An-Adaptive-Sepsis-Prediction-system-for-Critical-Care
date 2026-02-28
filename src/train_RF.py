import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
from sklearn.utils import resample

# ===============================
# 1. LOAD SPLIT DATASETS
# ===============================
TRAIN_PATH = "dataset/t_data/train_data.csv"
VAL_PATH   = "dataset/t_data/val_data.csv"
TEST_PATH  = "dataset/t_data/test_data.csv"

print("📂 Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("✅ Train shape:", train_df.shape)
print("✅ Validation shape:", val_df.shape)
print("✅ Test shape:", test_df.shape)

TARGET = "deterioration_next_12h"

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
# 2.5 CLASS DISTRIBUTION CHECK
# ===============================
print("\n📊 CLASS DISTRIBUTION:")
print("Train:", y_train.value_counts().to_dict())
print(f"   Class 1 ratio: {y_train.mean()*100:.2f}%")
print("Validation:", y_val.value_counts().to_dict())
print("Test:", y_test.value_counts().to_dict())

# ===============================
# 3. OVERSAMPLE MINORITY CLASS
# ===============================
print("\n⚖️ Oversampling minority class to balance training data...")

# Separate majority and minority classes
train_majority = train_df[train_df[TARGET] == 0]
train_minority = train_df[train_df[TARGET] == 1]

# Upsample minority class
train_minority_upsampled = resample(
    train_minority,
    replace=True,
    n_samples=len(train_majority),
    random_state=42
)

# Combine majority and upsampled minority
train_balanced = pd.concat([train_majority, train_minority_upsampled])
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_resampled = train_balanced.drop(columns=[TARGET])
y_train_resampled = train_balanced[TARGET]

print(f"✅ Before oversampling: {len(X_train)} samples")
print(f"✅ After oversampling: {len(X_train_resampled)} samples")
print(f"   New class distribution: {y_train_resampled.value_counts().to_dict()}")

# ===============================
# 4. TRAIN RANDOM FOREST
# ===============================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\n🚀 Training Random Forest Model...")
rf_model.fit(X_train_resampled, y_train_resampled)

# ===============================
# 5. VALIDATION - FIND OPTIMAL THRESHOLD
# ===============================
val_prob = rf_model.predict_proba(X_val)[:, 1]

# Find optimal threshold using F1 score
thresholds = np.arange(0.1, 0.9, 0.05)
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

print("\n✅ VALIDATION PERFORMANCE (with optimal threshold):")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

# ===============================
# 6. TEST PERFORMANCE
# ===============================
test_prob = rf_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\n" + "="*50)
print("✅ FINAL TEST PERFORMANCE (Threshold: {:.2f})".format(best_threshold))
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
print(f"F1 Score (Class 1): {f1_score(y_test, test_pred):.4f}")
print(f"Precision (Class 1): {precision_score(y_test, test_pred):.4f}")
print(f"Recall (Class 1): {recall_score(y_test, test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, test_prob):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, test_pred))

# ===============================
# 7. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Deterioration", "Deterioration"],
            yticklabels=["No Deterioration", "Deterioration"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Threshold: {best_threshold:.2f})")
plt.tight_layout()
plt.show()

# ===============================
# 8. ROC-AUC CURVE
# ===============================
fpr, tpr, _ = roc_curve(y_test, test_prob)
roc_auc = roc_auc_score(y_test, test_prob)

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_prob)
pr_auc = average_precision_score(y_test, test_prob)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}", color='blue')
axes[0].plot([0, 1], [0, 1], linestyle="--", color='gray')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

# Precision-Recall Curve
axes[1].plot(recall_curve, precision_curve, label=f"PR-AUC = {pr_auc:.3f}", color='green')
axes[1].axhline(y=y_test.mean(), linestyle="--", color='gray', label=f"Baseline = {y_test.mean():.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
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

print("\n✅ TOP 15 IMPORTANT FEATURES:")
print(importances.head(15))

plt.figure(figsize=(8, 6))
sns.barplot(x="importance", y="feature", data=importances.head(15))
plt.title("Top 15 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# ===============================
# 10. SAVE MODEL & THRESHOLD
# ===============================
model_dir = Path("model")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "rf_model.pkl"
joblib.dump(rf_model, str(model_path))

# Save optimal threshold too
threshold_path = model_dir / "optimal_threshold.txt"
with open(threshold_path, 'w') as f:
    f.write(f"{best_threshold}")

print(f"\n💾 Trained model saved as: {model_path}")
print(f"💾 Optimal threshold saved as: {threshold_path}")

# ===============================
# SUMMARY
# ===============================
print("\n" + "="*50)
print("📋 MODEL SUMMARY")
print("="*50)
print(f"Training samples (after SMOTE): {len(X_train_resampled)}")
print(f"Optimal threshold: {best_threshold:.2f}")
print(f"Test F1 (Class 1): {f1_score(y_test, test_pred):.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")
print(f"Test PR-AUC: {pr_auc:.4f}")
