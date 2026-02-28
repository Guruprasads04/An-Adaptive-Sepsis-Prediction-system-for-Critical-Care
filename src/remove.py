import pandas as pd

# Load data
df = pd.read_csv("hospital_deterioration_ml_ready.csv")

# Columns to drop from FEATURES (but keep target)
cols_to_drop = ["sepsis_risk_score"]

# Target column
target_col = "deterioration_next_12h"

# ✅ X = features, y = label
X = df.drop(columns=cols_to_drop + [target_col])
y = df[target_col]

print("✅ Original shape:", df.shape)
print("✅ Feature matrix shape:", X.shape)
print("✅ Target vector shape:", y.shape)

# (Optional) Save cleaned feature set + target together to a new file
clean_df = pd.concat([X, y], axis=1)
clean_df.to_csv("hospital_deterioration_FINAL.csv", index=False)
print("✅ Saved cleaned dataset to hospital_deterioration_FINAL.csv")
