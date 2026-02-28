import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

from xgboost import XGBClassifier

# =====================================
# 1. LOAD SPLIT DATA
# =====================================
TRAIN_PATH = "dataset/t_data/train_data.csv"
VAL_PATH   = "dataset/t_data/val_data.csv"
TEST_PATH  = "dataset/t_data/test_data.csv"

print("📂 Loading datasets...")
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

TARGET = "deterioration_next_12h"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print("✅ Shapes - Train:", X_train.shape, "Test:", X_test.shape)

# =====================================
# 2. DEFINE MODELS
# =====================================
pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

models = {
    "Logistic\nRegression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    ),
    "SVM\n(RBF)": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    ),
    "Random\nForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# =====================================
# 3. TRAIN & EVALUATE
# =====================================
rows = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )
    prec_1 = report["1"]["precision"]
    rec_1  = report["1"]["recall"]
    f1_1   = report["1"]["f1-score"]

    rows.append([name, acc, prec_1, rec_1, f1_1, auc])

# =====================================
# 4. RESULTS TABLE
# =====================================
results = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "Accuracy",
        "Precision (class 1)",
        "Recall (class 1)",
        "F1 (class 1)",
        "ROC-AUC"
    ]
)

print("\n✅ FINAL COMPARISON TABLE:\n")
print(results)

results.to_csv("model_comparison_results.csv", index=False)
print("\n💾 Saved: model_comparison_results.csv")

# =====================================
# 5. PLOTS FOR REVIEW SLIDES
# =====================================
# Set model labels for x-axis
models_list = results["Model"].tolist()
x = np.arange(len(models_list))

# --- ROC-AUC BAR PLOT ---
plt.figure()
plt.bar(x, results["ROC-AUC"])
plt.xticks(x, models_list)
plt.ylabel("ROC-AUC")
plt.ylim(0.5, 1.0)
plt.title("Model Comparison - ROC-AUC")
plt.tight_layout()
plt.show()

# --- RECALL (CLASS 1) BAR PLOT ---
plt.figure()
plt.bar(x, results["Recall (class 1)"])
plt.xticks(x, models_list)
plt.ylabel("Recall (Deterioration class)")
plt.ylim(0.0, 1.0)
plt.title("Model Comparison - Recall (Deterioration)")
plt.tight_layout()
plt.show()

# --- F1 (CLASS 1) BAR PLOT ---
plt.figure()
plt.bar(x, results["F1 (class 1)"])
plt.xticks(x, models_list)
plt.ylabel("F1-score (Deterioration class)")
plt.ylim(0.0, 1.0)
plt.title("Model Comparison - F1-score (Deterioration)")
plt.tight_layout()
plt.show()
print("✅ Plots displayed.")