"""
Sepsis Prediction Model - Interactive Tester
=============================================
Test the model with key clinical parameters
"""

import pandas as pd
import numpy as np
import joblib

# Load model and threshold
model = joblib.load('model/sepsis_rf_model.pkl')
with open('model/sepsis_optimal_threshold.txt', 'r') as f:
    THRESHOLD = float(f.read().strip())

print("="*60)
print("🦠 SEPSIS PREDICTION MODEL - 6 Hour Horizon")
print("="*60)
print(f"Model loaded | Threshold: {THRESHOLD}")
print()

# Get training data statistics for defaults
train_df = pd.read_csv('dataset/sepsis_data/train_sepsis.csv')
feature_means = train_df.drop(columns=['sepsis_next_6h']).mean()

def predict_sepsis(
    # KEY PARAMETERS (Top predictors)
    lactate: float,
    wbc_count: float,
    crp_level: float,
    creatinine: float,
    
    # VITAL SIGNS
    heart_rate: float = 80,
    respiratory_rate: float = 16,
    spo2_pct: float = 96,
    temperature_c: float = 37.0,
    systolic_bp: float = 120,
    diastolic_bp: float = 80,
    
    # PATIENT INFO
    age: int = 50,
    gender: str = 'M',  # 'M' or 'F'
    comorbidity_index: int = 2,
    admission_type: str = 'ED',  # 'ED', 'Elective', 'Transfer'
    
    # OTHER (defaults based on training data)
    hour_from_admission: int = 12,
    oxygen_device: str = 'none',  # 'none', 'nasal', 'mask', 'niv', 'hfnc'
    oxygen_flow: float = 0,
    mobility_score: int = 2,
    nurse_alert: int = 0,
    hemoglobin: float = 13.5,
):
    """
    Predict sepsis risk for a patient.
    
    KEY PARAMETERS (most important):
    - lactate: Lactate level (mmol/L) - Normal: 0.5-2.0, Elevated: >2.0
    - wbc_count: White blood cell count (x10^9/L) - Normal: 4-11, High: >12, Low: <4
    - crp_level: C-reactive protein (mg/L) - Normal: <10, Elevated: >10
    - creatinine: Creatinine level (mg/dL) - Normal: 0.6-1.2
    """
    
    # Compute SIRS criteria
    sirs_temp = 1 if (temperature_c > 38 or temperature_c < 36) else 0
    sirs_hr = 1 if heart_rate > 90 else 0
    sirs_rr = 1 if respiratory_rate > 20 else 0
    sirs_wbc = 1 if (wbc_count > 12 or wbc_count < 4) else 0
    sirs_score = sirs_temp + sirs_hr + sirs_rr + sirs_wbc
    sirs_positive = 1 if sirs_score >= 2 else 0
    
    # Use defaults for change/rolling features (assuming stable)
    lactate_change = 0
    lactate_rolling_mean = lactate
    heart_rate_change = 0
    heart_rate_rolling_mean = heart_rate
    respiratory_rate_change = 0
    respiratory_rate_rolling_mean = respiratory_rate
    temperature_c_change = 0
    temperature_c_rolling_mean = temperature_c
    systolic_bp_change = 0
    systolic_bp_rolling_mean = systolic_bp
    
    # One-hot encode categorical variables
    oxygen_device_hfnc = 1 if oxygen_device == 'hfnc' else 0
    oxygen_device_mask = 1 if oxygen_device == 'mask' else 0
    oxygen_device_nasal = 1 if oxygen_device == 'nasal' else 0
    oxygen_device_niv = 1 if oxygen_device == 'niv' else 0
    oxygen_device_none = 1 if oxygen_device == 'none' else 0
    
    gender_F = 1 if gender == 'F' else 0
    gender_M = 1 if gender == 'M' else 0
    
    admission_type_ED = 1 if admission_type == 'ED' else 0
    admission_type_Elective = 1 if admission_type == 'Elective' else 0
    admission_type_Transfer = 1 if admission_type == 'Transfer' else 0
    
    # Build feature vector (must match training order!)
    features = pd.DataFrame([{
        'hour_from_admission': hour_from_admission,
        'heart_rate': heart_rate,
        'respiratory_rate': respiratory_rate,
        'spo2_pct': spo2_pct,
        'temperature_c': temperature_c,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'oxygen_flow': oxygen_flow,
        'mobility_score': mobility_score,
        'nurse_alert': nurse_alert,
        'wbc_count': wbc_count,
        'lactate': lactate,
        'creatinine': creatinine,
        'crp_level': crp_level,
        'hemoglobin': hemoglobin,
        'age': age,
        'comorbidity_index': comorbidity_index,
        'sirs_score': sirs_score,
        'sirs_positive': sirs_positive,
        'heart_rate_change': heart_rate_change,
        'heart_rate_rolling_mean': heart_rate_rolling_mean,
        'respiratory_rate_change': respiratory_rate_change,
        'respiratory_rate_rolling_mean': respiratory_rate_rolling_mean,
        'temperature_c_change': temperature_c_change,
        'temperature_c_rolling_mean': temperature_c_rolling_mean,
        'systolic_bp_change': systolic_bp_change,
        'systolic_bp_rolling_mean': systolic_bp_rolling_mean,
        'lactate_change': lactate_change,
        'lactate_rolling_mean': lactate_rolling_mean,
        'oxygen_device_hfnc': oxygen_device_hfnc,
        'oxygen_device_mask': oxygen_device_mask,
        'oxygen_device_nasal': oxygen_device_nasal,
        'oxygen_device_niv': oxygen_device_niv,
        'oxygen_device_none': oxygen_device_none,
        'gender_F': gender_F,
        'gender_M': gender_M,
        'admission_type_ED': admission_type_ED,
        'admission_type_Elective': admission_type_Elective,
        'admission_type_Transfer': admission_type_Transfer,
    }])
    
    # Predict
    prob = model.predict_proba(features)[0, 1]
    prediction = 1 if prob >= THRESHOLD else 0
    
    return prediction, prob, sirs_score


def display_result(prediction, probability, sirs_score):
    """Display prediction result with clinical context"""
    print()
    print("-"*50)
    
    # Determine risk level based on probability AND SIRS
    if prediction == 1:  # Above threshold (0.47)
        print("🚨 SEPSIS ALERT: HIGH RISK")
        print(f"   Probability: {probability*100:.1f}%")
        print("   Action: Immediate clinical evaluation required")
    elif probability >= 0.30 or sirs_score >= 2:  # Moderate risk
        print("⚠️  MODERATE SEPSIS RISK - MONITOR CLOSELY")
        print(f"   Probability: {probability*100:.1f}%")
        if sirs_score >= 2:
            print("   Note: SIRS criteria met - increased vigilance needed")
        print("   Action: Re-evaluate in 1-2 hours, consider labs")
    else:  # Low risk
        print("✅ LOW SEPSIS RISK")
        print(f"   Probability: {probability*100:.1f}%")
        print("   Action: Continue routine monitoring")
    
    print(f"   SIRS Score: {sirs_score}/4", end="")
    if sirs_score >= 2:
        print(" (SIRS Positive)")
    else:
        print(" (SIRS Negative)")
    print("-"*50)


# ========================================
# TEST CASES
# ========================================

print("\n" + "="*60)
print("TEST CASE 1: Healthy Patient")
print("="*60)
pred, prob, sirs = predict_sepsis(
    lactate=1.0,       # Normal
    wbc_count=7.0,     # Normal
    crp_level=5.0,     # Normal
    creatinine=1.0,    # Normal
    heart_rate=75,
    respiratory_rate=14,
    temperature_c=36.8,
    age=45
)
print("Input: Lactate=1.0, WBC=7.0, CRP=5.0, Creatinine=1.0")
display_result(pred, prob, sirs)


print("\n" + "="*60)
print("TEST CASE 2: Early Sepsis Signs")
print("="*60)
pred, prob, sirs = predict_sepsis(
    lactate=2.5,       # Elevated
    wbc_count=14.0,    # High
    crp_level=45.0,    # Elevated
    creatinine=1.4,    # Slightly elevated
    heart_rate=95,     # Tachycardia
    respiratory_rate=22,  # Elevated
    temperature_c=38.2,   # Fever
    age=65
)
print("Input: Lactate=2.5, WBC=14.0, CRP=45.0, Creatinine=1.4")
print("       HR=95, RR=22, Temp=38.2°C")
display_result(pred, prob, sirs)


print("\n" + "="*60)
print("TEST CASE 3: Severe Sepsis")
print("="*60)
pred, prob, sirs = predict_sepsis(
    lactate=4.5,       # Very high
    wbc_count=18.0,    # Very high
    crp_level=120.0,   # Very elevated
    creatinine=2.5,    # Kidney dysfunction
    heart_rate=115,    # Tachycardia
    respiratory_rate=28,  # Tachypnea
    temperature_c=39.1,   # High fever
    systolic_bp=90,       # Hypotension
    spo2_pct=92,          # Low oxygen
    age=70,
    comorbidity_index=4
)
print("Input: Lactate=4.5, WBC=18.0, CRP=120.0, Creatinine=2.5")
print("       HR=115, RR=28, Temp=39.1°C, BP=90, SpO2=92%")
display_result(pred, prob, sirs)


print("\n" + "="*60)
print("TEST CASE 4: Borderline Case")
print("="*60)
pred, prob, sirs = predict_sepsis(
    lactate=2.0,       # Upper normal
    wbc_count=11.5,    # Upper normal
    crp_level=25.0,    # Mildly elevated
    creatinine=1.2,    # Normal
    heart_rate=88,
    respiratory_rate=18,
    temperature_c=37.8,
    age=55
)
print("Input: Lactate=2.0, WBC=11.5, CRP=25.0, Creatinine=1.2")
display_result(pred, prob, sirs)


# ========================================
# INTERACTIVE MODE
# ========================================
print("\n" + "="*60)
print("🔬 INTERACTIVE MODE - Enter Your Own Values")
print("="*60)
print("Enter values for key parameters (press Enter for defaults):\n")

try:
    lactate = input("Lactate (mmol/L) [Normal: 0.5-2.0]: ").strip()
    lactate = float(lactate) if lactate else 1.5
    
    wbc = input("WBC Count (x10^9/L) [Normal: 4-11]: ").strip()
    wbc = float(wbc) if wbc else 7.0
    
    crp = input("CRP Level (mg/L) [Normal: <10]: ").strip()
    crp = float(crp) if crp else 8.0
    
    creatinine = input("Creatinine (mg/dL) [Normal: 0.6-1.2]: ").strip()
    creatinine = float(creatinine) if creatinine else 1.0
    
    hr = input("Heart Rate (bpm) [Normal: 60-100]: ").strip()
    hr = float(hr) if hr else 80
    
    temp = input("Temperature (°C) [Normal: 36.5-37.5]: ").strip()
    temp = float(temp) if temp else 37.0
    
    print("\nPredicting...")
    pred, prob, sirs = predict_sepsis(
        lactate=lactate,
        wbc_count=wbc,
        crp_level=crp,
        creatinine=creatinine,
        heart_rate=hr,
        temperature_c=temp
    )
    display_result(pred, prob, sirs)
    
except KeyboardInterrupt:
    print("\n\nExited interactive mode.")
except Exception as e:
    print(f"\nError: {e}")
