import pandas as pd

# ==============================
# 0. CONFIG
# ==============================
INPUT_CSV = "HD_Processed.csv"
OUTPUT_CSV = "ICU_SIRS_data.csv"

# ==============================
# 1. LOAD DATA
# ==============================
print(f"📂 Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

print("✅ Loaded shape:", df.shape)

# ==============================
# 2. CREATE SIRS COMPONENTS
# ==============================

# Fever / Hypothermia
df["sirs_temp"] = ((df["temperature_c"] > 38) | (df["temperature_c"] < 36)).astype(int)

# Tachycardia
df["sirs_hr"] = (df["heart_rate"] > 90).astype(int)

# Tachypnea
df["sirs_rr"] = (df["respiratory_rate"] > 20).astype(int)

# Abnormal WBC
df["sirs_wbc"] = ((df["wbc_count"] > 12) | (df["wbc_count"] < 4)).astype(int)

# ==============================
# 3. TOTAL SIRS SCORE
# ==============================
df["sirs_score"] = (
    df["sirs_temp"] +
    df["sirs_hr"] +
    df["sirs_rr"] +
    df["sirs_wbc"]
)

# ==============================
# 4. SIRS POSITIVE LABEL (≥ 2)
# ==============================
df["sirs_positive"] = (df["sirs_score"] >= 2).astype(int)

# ==============================
# 5. DROP INTERMEDIATE FLAGS (OPTIONAL BUT CLEAN)
# ==============================
df = df.drop(columns=["sirs_temp", "sirs_hr", "sirs_rr", "sirs_wbc"])

# ==============================
# 6. FINAL CHECK & SAVE
# ==============================
print("\n📌 SIRS Distribution:")
print(df["sirs_score"].value_counts().sort_index())

print("\n📌 SIRS Positive Distribution:")
print(df["sirs_positive"].value_counts())

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾 Saved SIRS-enhanced dataset to: {OUTPUT_CSV}")
print("✅ Final shape:", df.shape)