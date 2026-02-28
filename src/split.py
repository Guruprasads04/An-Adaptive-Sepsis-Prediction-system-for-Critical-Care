import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================
# 1. CONFIG
# ============================
INPUT_CSV = "dataset/ICU_SIRS_data.csv"
SAVE_DIR = "dataset/t_data"

TARGET_COL = "deterioration_next_12h"

# ============================
# 2. CREATE SAVE DIRECTORY
# ============================
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================
# 3. LOAD DATA
# ============================
print("📂 Loading dataset...")
df = pd.read_csv(INPUT_CSV)

print("✅ Total dataset shape:", df.shape)

# ============================
# 4. FEATURE & TARGET SPLIT
# ============================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ============================
# 5. 70–30 SPLIT (TRAIN + TEMP)
# ============================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ============================
# 6. 10–20 SPLIT (VAL + TEST)
# ============================
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=2/3,   # 20% test from total
    random_state=42,
    stratify=y_temp
)

# ============================
# 7. COMBINE X + y BACK
# ============================
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# ============================
# 8. SAVE FILES
# ============================
train_df.to_csv(f"{SAVE_DIR}/train_data.csv", index=False)
val_df.to_csv(f"{SAVE_DIR}/val_data.csv", index=False)
test_df.to_csv(f"{SAVE_DIR}/test_data.csv", index=False)

# ============================
# 9. FINAL CONFIRMATION
# ============================
print("\n✅ DATASET SPLIT COMPLETED SUCCESSFULLY!")
print("🧪 Train shape:", train_df.shape)
print("🧪 Validation shape:", val_df.shape)
print("🧪 Test shape:", test_df.shape)

print(f"\n💾 Files saved to: {SAVE_DIR}/")
print("   - train_data.csv")
print("   - val_data.csv")
print("   - test_data.csv")
