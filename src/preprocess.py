import pandas as pd

# ==============================
# 0. CONFIG
# ==============================
INPUT_CSV = "HD_dataset.csv"  # <-- put your file name here
OUTPUT_CSV = "hospital_deterioration_preprocessed.csv"

TARGET_COL = "deterioration_next_12h"

# ==============================
# 1. LOAD DATA
# ==============================
print(f"📂 Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

print("✅ Loaded shape:", df.shape)
print("✅ Columns:", list(df.columns))

# ==============================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================
categorical_cols = ["oxygen_device", "gender", "admission_type"]

# Safety: only encode if column exists
categorical_cols = [c for c in categorical_cols if c in df.columns]

print("\n🔤 Categorical columns to encode:", categorical_cols)

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True  # avoid dummy trap
)

print("✅ After get_dummies, shape:", df.shape)

# ==============================
# 3. DATA TYPE CORRECTION (TO FLOAT FOR VITALS/LABS)
# ==============================
float_cols = [
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_flow",
    "wbc_count",
    "lactate",
    "creatinine",
    "crp_level",
    "hemoglobin",
]

float_cols = [c for c in float_cols if c in df.columns]

print("\n🔢 Converting to float:", float_cols)
df[float_cols] = df[float_cols].astype(float)

# ==============================
# 4. OUTLIER CLIPPING (CLINICAL RANGES)
# ==============================
clip_ranges = {
    "heart_rate": (30, 220),
    "respiratory_rate": (5, 60),
    "spo2_pct": (50, 100),
    "temperature_c": (34, 42),
    "systolic_bp": (50, 250),
    "diastolic_bp": (30, 150),
    "wbc_count": (0.5, 40),
    "lactate": (0.2, 20),
    "creatinine": (0.2, 15),
    "crp_level": (0, 50),
    "hemoglobin": (4, 20),
}

print("\n✂️ Clipping outliers for:")

for col, (low, high) in clip_ranges.items():
    if col in df.columns:
        print(f"  - {col}: [{low}, {high}]")
        df[col] = df[col].clip(lower=low, upper=high)

# ==============================
# 5. FINAL CHECK & SAVE
# ==============================
print("\n📌 Final dtypes:")
print(df.dtypes)

print("\n✅ Final shape before save:", df.shape)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾 Saved preprocessed dataset to: {OUTPUT_CSV}")
