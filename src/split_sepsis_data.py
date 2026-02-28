"""
Split Sepsis 6-hour Prediction Dataset
======================================
Train/Validation/Test split for sepsis prediction
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ===============================
# 1. LOAD PROCESSED DATA
# ===============================
print("📂 Loading sepsis data...")
df = pd.read_csv("dataset/processed/sepsis_6h_data.csv")
print(f"✅ Dataset shape: {df.shape}")

TARGET = "sepsis_next_6h"

# ===============================
# 2. SPLIT DATA (70/15/15)
# ===============================
print("\n✂️ Splitting data...")

# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df[TARGET]
)

# Second split: 50/50 of temp -> 15% val, 15% test
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df[TARGET]
)

print(f"✅ Train: {train_df.shape}")
print(f"✅ Validation: {val_df.shape}")
print(f"✅ Test: {test_df.shape}")

# ===============================
# 3. CLASS DISTRIBUTION CHECK
# ===============================
print("\n📊 CLASS DISTRIBUTION:")
print(f"Train - Class 1: {train_df[TARGET].mean()*100:.2f}%")
print(f"Val   - Class 1: {val_df[TARGET].mean()*100:.2f}%")
print(f"Test  - Class 1: {test_df[TARGET].mean()*100:.2f}%")

# ===============================
# 4. SAVE SPLITS
# ===============================
output_dir = Path("dataset/sepsis_data")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train_sepsis.csv", index=False)
val_df.to_csv(output_dir / "val_sepsis.csv", index=False)
test_df.to_csv(output_dir / "test_sepsis.csv", index=False)

print(f"\n💾 Saved to: {output_dir}/")
print("   - train_sepsis.csv")
print("   - val_sepsis.csv")
print("   - test_sepsis.csv")
