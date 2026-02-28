"""
Prepare dataset for 6-hour Sepsis Prediction
=============================================
Creates target: sepsis_next_6h - will patient develop sepsis in next 6 hours?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# 1. LOAD RAW DATA
# ===============================
print("📂 Loading raw data...")
df = pd.read_csv("dataset/raw/hospital_deterioration_ml_ready.csv")
print(f"✅ Raw data shape: {df.shape}")

# ===============================
# 2. ASSIGN PATIENT IDs
# ===============================
print("\n👥 Identifying patients...")
df['new_patient'] = (df['hour_from_admission'].diff() < 0) | (df['hour_from_admission'].diff().isna())
df['patient_id'] = df['new_patient'].cumsum()
df.drop(columns=['new_patient'], inplace=True)

print(f"✅ Total patients: {df['patient_id'].nunique()}")

# ===============================
# 3. DEFINE SEPSIS (Clinical Threshold)
# ===============================
# Using sepsis_risk_score >= 0.7 as clinical sepsis threshold
# Alternatively, we can use SIRS criteria + suspected infection

SEPSIS_THRESHOLD = 0.7
df['sepsis'] = (df['sepsis_risk_score'] >= SEPSIS_THRESHOLD).astype(int)

print(f"\n🦠 Sepsis Definition: sepsis_risk_score >= {SEPSIS_THRESHOLD}")
print(f"   Current sepsis cases: {df['sepsis'].sum()} ({df['sepsis'].mean()*100:.2f}%)")

# ===============================
# 4. CREATE 6-HOUR PREDICTION TARGET
# ===============================
print("\n⏱️ Creating 6-hour prediction horizon...")

PREDICTION_HORIZON = 6  # hours

def create_future_sepsis_label(group):
    """For each row, check if sepsis occurs in next 6 hours"""
    sepsis_next_6h = []
    sepsis_values = group['sepsis'].values
    hours = group['hour_from_admission'].values
    
    for i in range(len(group)):
        current_hour = hours[i]
        # Look ahead 6 hours
        future_mask = (hours > current_hour) & (hours <= current_hour + PREDICTION_HORIZON)
        future_sepsis = sepsis_values[future_mask]
        
        # Label = 1 if ANY sepsis in next 6 hours
        if len(future_sepsis) > 0 and future_sepsis.max() == 1:
            sepsis_next_6h.append(1)
        else:
            sepsis_next_6h.append(0)
    
    group['sepsis_next_6h'] = sepsis_next_6h
    return group

# Apply to each patient
df = df.groupby('patient_id', group_keys=False).apply(create_future_sepsis_label)

print(f"✅ Created target: sepsis_next_6h")
print(f"   Positive cases: {df['sepsis_next_6h'].sum()} ({df['sepsis_next_6h'].mean()*100:.2f}%)")

# ===============================
# 5. COMPUTE SIRS SCORE (if not present)
# ===============================
print("\n📊 Computing SIRS criteria...")

# SIRS Criteria:
# 1. Temperature >38°C or <36°C
# 2. Heart rate >90 bpm
# 3. Respiratory rate >20 or PaCO2 <32 mmHg
# 4. WBC >12,000 or <4,000

df['sirs_temp'] = ((df['temperature_c'] > 38) | (df['temperature_c'] < 36)).astype(int)
df['sirs_hr'] = (df['heart_rate'] > 90).astype(int)
df['sirs_rr'] = (df['respiratory_rate'] > 20).astype(int)
df['sirs_wbc'] = ((df['wbc_count'] > 12) | (df['wbc_count'] < 4)).astype(int)

df['sirs_score'] = df['sirs_temp'] + df['sirs_hr'] + df['sirs_rr'] + df['sirs_wbc']
df['sirs_positive'] = (df['sirs_score'] >= 2).astype(int)

print(f"   SIRS positive (≥2 criteria): {df['sirs_positive'].sum()} ({df['sirs_positive'].mean()*100:.2f}%)")

# ===============================
# 6. FEATURE ENGINEERING
# ===============================
print("\n🔧 Engineering features...")

# Vital signs change rates (per patient)
def add_change_features(group):
    for col in ['heart_rate', 'respiratory_rate', 'temperature_c', 'systolic_bp', 'lactate']:
        group[f'{col}_change'] = group[col].diff().fillna(0)
        group[f'{col}_rolling_mean'] = group[col].rolling(window=3, min_periods=1).mean()
    return group

df = df.groupby('patient_id', group_keys=False).apply(add_change_features)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['oxygen_device', 'gender', 'admission_type'], drop_first=False)

print("✅ Feature engineering complete")

# ===============================
# 7. SELECT FINAL FEATURES
# ===============================
# Remove unnecessary columns
drop_cols = ['patient_id', 'sepsis', 'sepsis_risk_score', 'new_patient', 
             'sirs_temp', 'sirs_hr', 'sirs_rr', 'sirs_wbc', 'deterioration_next_12h']
drop_cols = [c for c in drop_cols if c in df.columns]

df_final = df.drop(columns=drop_cols)

print(f"\n📋 Final dataset shape: {df_final.shape}")
print(f"   Features: {df_final.shape[1] - 1}")
print(f"   Target: sepsis_next_6h")

# ===============================
# 8. CLASS DISTRIBUTION
# ===============================
print("\n📊 CLASS DISTRIBUTION:")
print(df_final['sepsis_next_6h'].value_counts())
print(f"   Positive rate: {df_final['sepsis_next_6h'].mean()*100:.2f}%")

# ===============================
# 9. SAVE PROCESSED DATA
# ===============================
output_path = Path("dataset/processed/sepsis_6h_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(output_path, index=False)

print(f"\n💾 Saved to: {output_path}")
print(f"   Columns: {list(df_final.columns)}")
