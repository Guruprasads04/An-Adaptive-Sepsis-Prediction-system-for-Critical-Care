import pandas as pd

INPUT_CSV  = "dataset/ICU_SIRS_data.csv"        # 🔁 change to your current file
OUTPUT_CSV = "dataset/labeled _data.csv"

print(f"📂 Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print("✅ Shape:", df.shape)
print("✅ Columns:", df.columns.tolist())

required_cols = [
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "wbc_count",
    "lactate",
    "crp_level",
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")

# Ensure numeric types
num_cols = ["heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
            "systolic_bp", "wbc_count", "lactate", "crp_level"]
df[num_cols] = df[num_cols].astype(float)

# ===============================
# 1. SIRS FLAGS
# ===============================
df["sirs_temp_flag"] = ((df["temperature_c"] > 38.0) | (df["temperature_c"] < 36.0)).astype(int)
df["sirs_hr_flag"]   = (df["heart_rate"] > 90).astype(int)
df["sirs_rr_flag"]   = (df["respiratory_rate"] > 20).astype(int)
df["sirs_wbc_high_flag"] = (df["wbc_count"] > 12).astype(int)
df["sirs_wbc_low_flag"]  = (df["wbc_count"] < 4).astype(int)

# Any WBC abnormal (high or low)
df["sirs_wbc_abnormal_flag"] = (
    (df["sirs_wbc_high_flag"] == 1) | (df["sirs_wbc_low_flag"] == 1)
).astype(int)

# SIRS score = temp + HR + RR + WBC abnormal
df["sirs_score"] = (
    df["sirs_temp_flag"] +
    df["sirs_hr_flag"] +
    df["sirs_rr_flag"] +
    df["sirs_wbc_abnormal_flag"]
)

df["sirs_positive"] = (df["sirs_score"] >= 2).astype(int)

# ===============================
# 2. ORGAN DYSFUNCTION / INFECTION SURROGATE
# ===============================
lactate_cond = df["lactate"] >= 2.0
crp_cond     = df["crp_level"] >= 10.0
sbp_cond     = df["systolic_bp"] < 90.0
spo2_cond    = df["spo2_pct"] < 92.0

organ_dysfunction = (lactate_cond | crp_cond | sbp_cond | spo2_cond)

# ===============================
# 3. FINAL SEPSIS LABEL
# ===============================
df["sepsis_label"] = ((df["sirs_positive"] == 1) & (organ_dysfunction)).astype(int)

print("\n📌 SIRS Score Distribution:")
print(df["sirs_score"].value_counts().sort_index())

print("\n📌 SIRS Positive Distribution:")
print(df["sirs_positive"].value_counts())

print("\n📌 Sepsis Label Distribution:")
print(df["sepsis_label"].value_counts())

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾 Saved sepsis-augmented dataset to: {OUTPUT_CSV}")
