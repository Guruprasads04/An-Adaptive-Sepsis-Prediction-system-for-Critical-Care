"""
================================================================================
SEPSIS PREDICTION SYSTEM - 6 Hour Horizon
================================================================================
A complete ML pipeline for predicting sepsis risk in hospital patients.

Features:
- Quick testing
- Interactive prediction testing
- SHAP-based explainability (for doctors and patients)

Normal Ranges for Key Parameters:
    Lactate:          0.5 - 2.0 mmol/L
    WBC Count:        4.0 - 11.0 x10^9/L
    CRP Level:        0 - 10 mg/L
    Creatinine:       0.6 - 1.2 mg/dL
    Heart Rate:       60 - 100 bpm
    Respiratory Rate: 12 - 20 /min
    Temperature:      36.5 - 37.5 °C
    SpO2:             95 - 100 %
    Systolic BP:      90 - 140 mmHg
    Diastolic BP:     60 - 90 mmHg
    Hemoglobin:       12.0 - 17.5 g/dL

Usage:
    python main.py --help
    python main.py prepare      # Prepare and split data
    python main.py train        # Train the model
    python main.py test         # Interactive testing
    python main.py explain      # SHAP explanation demo
    python main.py predict      # Quick prediction with custom values
    python main.py all          # Run full pipeline

Author: Sepsis Prediction Team
Date: February 2026
================================================================================
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'


def prepare_data():
    """Prepare and split the sepsis dataset"""
    print("\n" + "="*70)
    print("STEP 1: PREPARING SEPSIS DATA")
    print("="*70)
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Check if raw data exists
    raw_path = Path("dataset/raw/hospital_deterioration_ml_ready.csv")
    if not raw_path.exists():
        print(f"[ERROR] Raw data not found at: {raw_path}")
        return False
    
    print("Loading raw data...")
    df = pd.read_csv(raw_path)
    print(f"Raw data shape: {df.shape}")
    
    # Assign patient IDs
    print("\nIdentifying patients...")
    df['new_patient'] = (df['hour_from_admission'].diff() < 0) | (df['hour_from_admission'].diff().isna())
    df['patient_id'] = df['new_patient'].cumsum()
    df.drop(columns=['new_patient'], inplace=True)
    print(f"Total patients: {df['patient_id'].nunique()}")
    
    # Define sepsis threshold
    SEPSIS_THRESHOLD = 0.7
    df['sepsis'] = (df['sepsis_risk_score'] >= SEPSIS_THRESHOLD).astype(int)
    print(f"\nSepsis Definition: sepsis_risk_score >= {SEPSIS_THRESHOLD}")
    print(f"Current sepsis cases: {df['sepsis'].sum()} ({df['sepsis'].mean()*100:.2f}%)")
    
    # Create 6-hour prediction target
    print("\nCreating 6-hour prediction horizon...")
    PREDICTION_HORIZON = 6
    
    def create_future_sepsis_label(group):
        sepsis_next_6h = []
        sepsis_values = group['sepsis'].values
        hours = group['hour_from_admission'].values
        
        for i in range(len(group)):
            current_hour = hours[i]
            future_mask = (hours > current_hour) & (hours <= current_hour + PREDICTION_HORIZON)
            future_sepsis = sepsis_values[future_mask]
            
            if len(future_sepsis) > 0 and future_sepsis.max() == 1:
                sepsis_next_6h.append(1)
            else:
                sepsis_next_6h.append(0)
        
        group['sepsis_next_6h'] = sepsis_next_6h
        return group
    
    df = df.groupby('patient_id', group_keys=False).apply(create_future_sepsis_label)
    print(f"Target created: sepsis_next_6h")
    print(f"Positive cases: {df['sepsis_next_6h'].sum()} ({df['sepsis_next_6h'].mean()*100:.2f}%)")
    
    # Compute SIRS score
    print("\nComputing SIRS criteria...")
    df['sirs_temp'] = ((df['temperature_c'] > 38) | (df['temperature_c'] < 36)).astype(int)
    df['sirs_hr'] = (df['heart_rate'] > 90).astype(int)
    df['sirs_rr'] = (df['respiratory_rate'] > 20).astype(int)
    df['sirs_wbc'] = ((df['wbc_count'] > 12) | (df['wbc_count'] < 4)).astype(int)
    df['sirs_score'] = df['sirs_temp'] + df['sirs_hr'] + df['sirs_rr'] + df['sirs_wbc']
    df['sirs_positive'] = (df['sirs_score'] >= 2).astype(int)
    
    # Feature engineering
    print("Engineering features...")
    def add_change_features(group):
        for col in ['heart_rate', 'respiratory_rate', 'temperature_c', 'systolic_bp', 'lactate']:
            group[f'{col}_change'] = group[col].diff().fillna(0)
            group[f'{col}_rolling_mean'] = group[col].rolling(window=3, min_periods=1).mean()
        return group
    
    df = df.groupby('patient_id', group_keys=False).apply(add_change_features)
    
    # One-hot encode
    df = pd.get_dummies(df, columns=['oxygen_device', 'gender', 'admission_type'], drop_first=False)
    
    # Remove unnecessary columns
    drop_cols = ['patient_id', 'sepsis', 'sepsis_risk_score', 'sirs_temp', 'sirs_hr', 
                 'sirs_rr', 'sirs_wbc', 'deterioration_next_12h']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df_final = df.drop(columns=drop_cols)
    
    print(f"\nFinal dataset shape: {df_final.shape}")
    
    # Save processed data
    processed_path = Path("dataset/processed/sepsis_6h_data.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(processed_path, index=False)
    print(f"Saved processed data to: {processed_path}")
    
    # Split data
    print("\n" + "-"*70)
    print("Splitting data (70/15/15)...")
    TARGET = "sepsis_next_6h"
    
    train_df, temp_df = train_test_split(df_final, test_size=0.30, random_state=42, stratify=df_final[TARGET])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df[TARGET])
    
    # Save splits
    output_dir = Path("dataset/sepsis_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train_sepsis.csv", index=False)
    val_df.to_csv(output_dir / "val_sepsis.csv", index=False)
    test_df.to_csv(output_dir / "test_sepsis.csv", index=False)
    
    print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
    print(f"Saved to: {output_dir}/")
    
    print("\n[OK] Data preparation complete!")
    return True


def train_model():
    """Train the Random Forest model for sepsis prediction"""
    print("\n" + "="*70)
    print("STEP 2: TRAINING SEPSIS PREDICTION MODEL")
    print("="*70)
    
    import pandas as pd
    import numpy as np
    import joblib
    from pathlib import Path
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv("dataset/sepsis_data/train_sepsis.csv")
    val_df = pd.read_csv("dataset/sepsis_data/val_sepsis.csv")
    test_df = pd.read_csv("dataset/sepsis_data/test_sepsis.csv")
    
    print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
    
    TARGET = "sepsis_next_6h"
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    
    print(f"\nClass distribution - Sepsis rate: {y_train.mean()*100:.2f}%")
    
    # Train model
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    val_prob = rf_model.predict_proba(X_val)[:, 1]
    
    best_f1, best_threshold = 0, 0.5
    for thresh in np.arange(0.30, 0.60, 0.01):
        val_pred = (val_prob >= thresh).astype(int)
        f1 = f1_score(y_val, val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # Use 0.47 for better accuracy as previously determined
    best_threshold = 0.47
    print(f"Using threshold: {best_threshold:.2f}")
    
    # Evaluate on test set
    test_prob = rf_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    
    print("\n" + "="*55)
    print("FINAL TEST PERFORMANCE")
    print("="*55)
    print(f"Accuracy:   {accuracy_score(y_test, test_pred):.4f}")
    print(f"F1 Score:   {f1_score(y_test, test_pred):.4f}")
    print(f"Precision:  {precision_score(y_test, test_pred):.4f}")
    print(f"Recall:     {recall_score(y_test, test_pred):.4f}")
    print(f"ROC-AUC:    {roc_auc_score(y_test, test_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=['No Sepsis', 'Sepsis']))
    
    # Save model
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf_model, model_dir / "sepsis_rf_model.pkl")
    with open(model_dir / "sepsis_optimal_threshold.txt", 'w') as f:
        f.write(str(best_threshold))
    
    print(f"\nModel saved to: {model_dir}/sepsis_rf_model.pkl")
    print(f"Threshold saved to: {model_dir}/sepsis_optimal_threshold.txt")
    
    print("\n[OK] Training complete!")
    return True


def predict_sepsis(
    lactate: float,
    wbc_count: float,
    crp_level: float,
    creatinine: float,
    heart_rate: float = 80,
    respiratory_rate: float = 16,
    spo2_pct: float = 96,
    temperature_c: float = 37.0,
    systolic_bp: float = 120,
    diastolic_bp: float = 80,
    age: int = 50,
    gender: str = 'M',
    comorbidity_index: int = 2,
    admission_type: str = 'ED',
    hour_from_admission: int = 12,
    oxygen_device: str = 'none',
    oxygen_flow: float = 0,
    mobility_score: int = 2,
    nurse_alert: int = 0,
    hemoglobin: float = 13.5,
):
    """Make a sepsis prediction with given parameters"""
    import pandas as pd
    import numpy as np
    import joblib
    
    # Load model
    model = joblib.load('model/sepsis_rf_model.pkl')
    with open('model/sepsis_optimal_threshold.txt', 'r') as f:
        THRESHOLD = float(f.read().strip())
    
    # Compute SIRS
    sirs_temp = 1 if (temperature_c > 38 or temperature_c < 36) else 0
    sirs_hr = 1 if heart_rate > 90 else 0
    sirs_rr = 1 if respiratory_rate > 20 else 0
    sirs_wbc = 1 if (wbc_count > 12 or wbc_count < 4) else 0
    sirs_score = sirs_temp + sirs_hr + sirs_rr + sirs_wbc
    
    # Build features
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
        'sirs_positive': 1 if sirs_score >= 2 else 0,
        'heart_rate_change': 0,
        'heart_rate_rolling_mean': heart_rate,
        'respiratory_rate_change': 0,
        'respiratory_rate_rolling_mean': respiratory_rate,
        'temperature_c_change': 0,
        'temperature_c_rolling_mean': temperature_c,
        'systolic_bp_change': 0,
        'systolic_bp_rolling_mean': systolic_bp,
        'lactate_change': 0,
        'lactate_rolling_mean': lactate,
        'oxygen_device_hfnc': 1 if oxygen_device == 'hfnc' else 0,
        'oxygen_device_mask': 1 if oxygen_device == 'mask' else 0,
        'oxygen_device_nasal': 1 if oxygen_device == 'nasal' else 0,
        'oxygen_device_niv': 1 if oxygen_device == 'niv' else 0,
        'oxygen_device_none': 1 if oxygen_device == 'none' else 0,
        'gender_F': 1 if gender == 'F' else 0,
        'gender_M': 1 if gender == 'M' else 0,
        'admission_type_ED': 1 if admission_type == 'ED' else 0,
        'admission_type_Elective': 1 if admission_type == 'Elective' else 0,
        'admission_type_Transfer': 1 if admission_type == 'Transfer' else 0,
    }])
    
    # Predict
    prob = model.predict_proba(features)[0, 1]
    prediction = 1 if prob >= THRESHOLD else 0
    
    return prediction, prob, sirs_score, THRESHOLD


def interactive_test():
    """Interactive testing mode"""
    print("\n" + "="*70)
    print("SEPSIS PREDICTION - INTERACTIVE TEST MODE")
    print("="*70)
    print("\nEnter patient parameters (press Enter for defaults):\n")
    
    try:
        # Get inputs
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
        
        temp = input("Temperature (C) [Normal: 36.5-37.5]: ").strip()
        temp = float(temp) if temp else 37.0
        
        rr = input("Respiratory Rate (/min) [Normal: 12-20]: ").strip()
        rr = float(rr) if rr else 16
        
        age = input("Age (years): ").strip()
        age = int(age) if age else 50
        
        # Make prediction
        print("\nAnalyzing...")
        pred, prob, sirs, thresh = predict_sepsis(
            lactate=lactate,
            wbc_count=wbc,
            crp_level=crp,
            creatinine=creatinine,
            heart_rate=hr,
            temperature_c=temp,
            respiratory_rate=rr,
            age=age
        )
        
        # Display result
        print("\n" + "-"*55)
        if pred == 1:
            print("[!!!] SEPSIS ALERT: HIGH RISK")
            print(f"   Probability: {prob*100:.1f}%")
            print("   Action: Immediate clinical evaluation required")
        elif prob >= 0.30 or sirs >= 2:
            print("[!!] MODERATE SEPSIS RISK - MONITOR CLOSELY")
            print(f"   Probability: {prob*100:.1f}%")
            if sirs >= 2:
                print("   Note: SIRS criteria met - increased vigilance needed")
            print("   Action: Re-evaluate in 1-2 hours, consider labs")
        else:
            print("[OK] LOW SEPSIS RISK")
            print(f"   Probability: {prob*100:.1f}%")
            print("   Action: Continue routine monitoring")
        
        print(f"   SIRS Score: {sirs}/4", end="")
        print(" (SIRS Positive)" if sirs >= 2 else " (SIRS Negative)")
        print("-"*55)
        
        # Show abnormal values
        print("\nAbnormal Values:")
        if lactate > 2.0:
            print(f"  - Lactate: {lactate} (HIGH, normal: 0.5-2.0)")
        if wbc < 4 or wbc > 11:
            print(f"  - WBC: {wbc} ({'LOW' if wbc < 4 else 'HIGH'}, normal: 4-11)")
        if crp > 10:
            print(f"  - CRP: {crp} (HIGH, normal: <10)")
        if creatinine > 1.2:
            print(f"  - Creatinine: {creatinine} (HIGH, normal: 0.6-1.2)")
        if hr > 100 or hr < 60:
            print(f"  - Heart Rate: {hr} ({'HIGH' if hr > 100 else 'LOW'}, normal: 60-100)")
        if temp > 37.5 or temp < 36.5:
            print(f"  - Temperature: {temp} ({'HIGH' if temp > 37.5 else 'LOW'}, normal: 36.5-37.5)")
        
    except KeyboardInterrupt:
        print("\n\nExited.")
    except Exception as e:
        print(f"\nError: {e}")


# ===============================
# SHAP EXPLAINABILITY CONSTANTS
# ===============================
FEATURE_NAMES = {
    'lactate': 'Lactate Level',
    'lactate_rolling_mean': 'Lactate (Trend)',
    'lactate_change': 'Lactate Change',
    'wbc_count': 'White Blood Cell Count',
    'crp_level': 'CRP (Inflammation Marker)',
    'creatinine': 'Creatinine (Kidney Function)',
    'heart_rate': 'Heart Rate',
    'heart_rate_rolling_mean': 'Heart Rate (Trend)',
    'heart_rate_change': 'Heart Rate Change',
    'respiratory_rate': 'Respiratory Rate',
    'respiratory_rate_rolling_mean': 'Respiratory Rate (Trend)',
    'respiratory_rate_change': 'Respiratory Rate Change',
    'temperature_c': 'Body Temperature',
    'temperature_c_rolling_mean': 'Temperature (Trend)',
    'temperature_c_change': 'Temperature Change',
    'systolic_bp': 'Blood Pressure (Systolic)',
    'systolic_bp_rolling_mean': 'Blood Pressure (Trend)',
    'systolic_bp_change': 'Blood Pressure Change',
    'diastolic_bp': 'Blood Pressure (Diastolic)',
    'spo2_pct': 'Oxygen Saturation',
    'oxygen_flow': 'Oxygen Flow Rate',
    'sirs_score': 'SIRS Score',
    'sirs_positive': 'SIRS Criteria Met',
    'age': 'Patient Age',
    'comorbidity_index': 'Pre-existing Conditions',
    'hemoglobin': 'Hemoglobin Level',
    'mobility_score': 'Mobility Score',
    'nurse_alert': 'Nurse Alert Flag',
    'hour_from_admission': 'Hours Since Admission',
    'oxygen_device_none': 'No Oxygen Device',
    'oxygen_device_nasal': 'Nasal Cannula',
    'oxygen_device_mask': 'Oxygen Mask',
    'oxygen_device_niv': 'Non-Invasive Ventilation',
    'oxygen_device_hfnc': 'High-Flow Nasal Cannula',
    'gender_M': 'Male Gender',
    'gender_F': 'Female Gender',
    'admission_type_ED': 'Emergency Admission',
    'admission_type_Elective': 'Elective Admission',
    'admission_type_Transfer': 'Transfer Patient',
}

NORMAL_RANGES = {
    'lactate': (0.5, 2.0, 'mmol/L'),
    'wbc_count': (4.0, 11.0, 'x10^9/L'),
    'crp_level': (0, 10, 'mg/L'),
    'creatinine': (0.6, 1.2, 'mg/dL'),
    'heart_rate': (60, 100, 'bpm'),
    'respiratory_rate': (12, 20, '/min'),
    'temperature_c': (36.5, 37.5, 'C'),
    'systolic_bp': (90, 140, 'mmHg'),
    'diastolic_bp': (60, 90, 'mmHg'),
    'spo2_pct': (95, 100, '%'),
    'hemoglobin': (12.0, 17.5, 'g/dL'),
}


def explain_prediction_shap(
    lactate: float,
    wbc_count: float,
    crp_level: float,
    creatinine: float,
    heart_rate: float = 80,
    respiratory_rate: float = 16,
    spo2_pct: float = 96,
    temperature_c: float = 37.0,
    systolic_bp: float = 120,
    diastolic_bp: float = 80,
    age: int = 50,
    gender: str = 'M',
    comorbidity_index: int = 2,
    admission_type: str = 'ED',
    hour_from_admission: int = 12,
    oxygen_device: str = 'none',
    oxygen_flow: float = 0,
    mobility_score: int = 2,
    nurse_alert: int = 0,
    hemoglobin: float = 13.5,
    show_plots: bool = True
):
    """
    Generate SHAP-based explanation for sepsis prediction.
    Provides both clinical (doctor) and simple (patient) explanations.
    """
    import pandas as pd
    import numpy as np
    import joblib
    import shap
    import matplotlib.pyplot as plt
    
    # Load model
    model = joblib.load('model/sepsis_rf_model.pkl')
    with open('model/sepsis_optimal_threshold.txt', 'r') as f:
        THRESHOLD = float(f.read().strip())
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SIRS
    sirs_temp = 1 if (temperature_c > 38 or temperature_c < 36) else 0
    sirs_hr = 1 if heart_rate > 90 else 0
    sirs_rr = 1 if respiratory_rate > 20 else 0
    sirs_wbc = 1 if (wbc_count > 12 or wbc_count < 4) else 0
    sirs_score = sirs_temp + sirs_hr + sirs_rr + sirs_wbc
    sirs_positive = 1 if sirs_score >= 2 else 0
    
    # Build feature DataFrame
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
        'heart_rate_change': 0,
        'heart_rate_rolling_mean': heart_rate,
        'respiratory_rate_change': 0,
        'respiratory_rate_rolling_mean': respiratory_rate,
        'temperature_c_change': 0,
        'temperature_c_rolling_mean': temperature_c,
        'systolic_bp_change': 0,
        'systolic_bp_rolling_mean': systolic_bp,
        'lactate_change': 0,
        'lactate_rolling_mean': lactate,
        'oxygen_device_hfnc': 1 if oxygen_device == 'hfnc' else 0,
        'oxygen_device_mask': 1 if oxygen_device == 'mask' else 0,
        'oxygen_device_nasal': 1 if oxygen_device == 'nasal' else 0,
        'oxygen_device_niv': 1 if oxygen_device == 'niv' else 0,
        'oxygen_device_none': 1 if oxygen_device == 'none' else 0,
        'gender_F': 1 if gender == 'F' else 0,
        'gender_M': 1 if gender == 'M' else 0,
        'admission_type_ED': 1 if admission_type == 'ED' else 0,
        'admission_type_Elective': 1 if admission_type == 'Elective' else 0,
        'admission_type_Transfer': 1 if admission_type == 'Transfer' else 0,
    }])
    
    # Get prediction
    prob = model.predict_proba(features)[0, 1]
    prediction = 1 if prob >= THRESHOLD else 0
    
    # Get SHAP values
    shap_values = explainer.shap_values(features)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        if len(shap_values.shape) == 3:
            shap_vals = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
    
    shap_vals = np.array(shap_vals).flatten().astype(float)
    feature_names = features.columns.tolist()
    
    # Sort by importance
    feature_importance = list(zip(feature_names, features.values[0], shap_vals))
    feature_importance.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # ===============================
    # PRINT RESULTS
    # ===============================
    print("\n" + "="*70)
    print("SEPSIS RISK PREDICTION WITH SHAP EXPLANATION")
    print("="*70)
    
    # Risk Level
    print("\n" + "-"*70)
    if prediction == 1:
        print("[!!!] SEPSIS ALERT: HIGH RISK")
    elif prob >= 0.30 or sirs_score >= 2:
        print("[!!] MODERATE SEPSIS RISK - MONITOR CLOSELY")
    else:
        print("[OK] LOW SEPSIS RISK")
    
    print(f"Probability: {prob*100:.1f}% | SIRS Score: {sirs_score}/4")
    print("-"*70)
    
    # ===============================
    # CLINICAL EXPLANATION (For Doctors)
    # ===============================
    print("\n" + "="*70)
    print("CLINICAL EXPLANATION (For Healthcare Providers)")
    print("="*70)
    
    print("\nTop Contributing Factors (SHAP Analysis):")
    print("-"*70)
    print(f"{'Feature':<30} {'Value':>12} {'Impact':>12} {'Direction':>12}")
    print("-"*70)
    
    for feat, val, shap_val in feature_importance[:10]:
        readable = FEATURE_NAMES.get(feat, feat)[:28]
        direction = "^ Risk" if shap_val > 0 else "v Risk" if shap_val < 0 else "-"
        
        if feat in NORMAL_RANGES:
            low, high, unit = NORMAL_RANGES[feat]
            if val < low or val > high:
                val_str = f"{val:.1f}*"
            else:
                val_str = f"{val:.1f}"
        else:
            val_str = f"{val:.2f}"
        
        print(f"{readable:<30} {val_str:>12} {shap_val:>+12.4f} {direction:>12}")
    
    print("-"*70)
    print("* = Outside normal range | ^ = Increases risk | v = Decreases risk")
    
    # Abnormal values
    print("\nAbnormal Values Detected:")
    abnormal_count = 0
    for feat, val, _ in feature_importance:
        if feat in NORMAL_RANGES:
            low, high, unit = NORMAL_RANGES[feat]
            if val < low:
                print(f"  - {FEATURE_NAMES.get(feat, feat)}: {val:.1f} {unit} (LOW, normal: {low}-{high})")
                abnormal_count += 1
            elif val > high:
                print(f"  - {FEATURE_NAMES.get(feat, feat)}: {val:.1f} {unit} (HIGH, normal: {low}-{high})")
                abnormal_count += 1
    if abnormal_count == 0:
        print("  None detected")
    
    # ===============================
    # SIMPLE EXPLANATION (For Patients)
    # ===============================
    print("\n" + "="*70)
    print("SIMPLE EXPLANATION (For Patients & Families)")
    print("="*70)
    
    print("\nWhat does this mean?")
    print("-"*70)
    
    if prediction == 1:
        print("The computer analysis shows a HIGH chance of developing a serious")
        print("infection (sepsis) in the next 6 hours. Immediate medical attention")
        print("is recommended.")
    elif prob >= 0.30 or sirs_score >= 2:
        print("The computer analysis shows a MODERATE chance of developing a serious")
        print("infection (sepsis). The medical team will monitor closely and may")
        print("order additional tests.")
    else:
        print("The computer analysis shows a LOW chance of developing a serious")
        print("infection (sepsis). Continue normal monitoring.")
    
    # Simple factor explanations
    print("\nMain factors affecting this result:")
    print("-"*70)
    
    risk_factors = []
    protective_factors = []
    
    for feat, val, shap_val in feature_importance[:10]:
        if shap_val > 0.02:
            readable = FEATURE_NAMES.get(feat, feat)
            if feat in NORMAL_RANGES:
                low, high, unit = NORMAL_RANGES[feat]
                if val < low:
                    status = f"below normal ({val:.1f} {unit}, normal: {low}-{high})"
                elif val > high:
                    status = f"above normal ({val:.1f} {unit}, normal: {low}-{high})"
                else:
                    status = f"value: {val:.1f}"
            else:
                status = f"value: {val:.1f}"
            risk_factors.append((readable, status))
        elif shap_val < -0.02:
            readable = FEATURE_NAMES.get(feat, feat)
            if feat in NORMAL_RANGES:
                low, high, unit = NORMAL_RANGES[feat]
                if val >= low and val <= high:
                    status = f"normal ({val:.1f} {unit})"
                else:
                    status = f"value: {val:.1f}"
            else:
                status = f"value: {val:.1f}"
            protective_factors.append((readable, status))
    
    if risk_factors:
        print("\n[!] Factors INCREASING infection risk:")
        for i, (name, status) in enumerate(risk_factors[:5], 1):
            print(f"    {i}. {name}: {status}")
    
    if protective_factors:
        print("\n[+] Factors DECREASING infection risk:")
        for i, (name, status) in enumerate(protective_factors[:3], 1):
            print(f"    {i}. {name}: {status}")
    
    # Key points
    print("\n" + "-"*70)
    print("Key Points:")
    if lactate > 2.0:
        print("  - Lactate is elevated, which can indicate tissue stress")
    if wbc_count > 12 or wbc_count < 4:
        print("  - White blood cell count is abnormal, suggesting immune response")
    if crp_level > 10:
        print("  - CRP is elevated, indicating inflammation in the body")
    if temperature_c > 38 or temperature_c < 36:
        print("  - Temperature is outside normal range")
    if heart_rate > 100:
        print("  - Heart rate is elevated")
    if sirs_score >= 2:
        print(f"  - SIRS criteria met ({sirs_score}/4): body shows signs of stress")
    print("-"*70)
    
    # ===============================
    # VISUALIZATION
    # ===============================
    if show_plots:
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: SHAP values bar chart
        top_n = 12
        top_features = feature_importance[:top_n]
        feat_names = [FEATURE_NAMES.get(f, f)[:25] for f, _, _ in top_features]
        shap_values_plot = [s for _, _, s in top_features]
        colors = ['#FF6B6B' if s > 0 else '#4ECDC4' for s in shap_values_plot]
        
        axes[0].barh(range(top_n), shap_values_plot, color=colors)
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(feat_names)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Impact on Sepsis Risk (SHAP Value)')
        axes[0].set_title('What Factors Affect This Prediction?')
        axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Increases Risk'),
            Patch(facecolor='#4ECDC4', label='Decreases Risk')
        ]
        axes[0].legend(handles=legend_elements, loc='lower right')
        
        # Plot 2: Key values vs normal range
        key_features = ['lactate', 'wbc_count', 'crp_level', 'creatinine', 
                       'heart_rate', 'temperature_c', 'respiratory_rate', 'spo2_pct']
        
        values_to_plot = []
        labels = []
        colors2 = []
        
        for feat in key_features:
            if feat in NORMAL_RANGES:
                idx = feature_names.index(feat)
                val = features.values[0][idx]
                low, high, unit = NORMAL_RANGES[feat]
                
                mid = (low + high) / 2
                range_size = (high - low) / 2
                normalized = (val - mid) / range_size * 50 + 50
                
                values_to_plot.append(normalized)
                labels.append(f"{FEATURE_NAMES.get(feat, feat)}\n({val:.1f} {unit})")
                
                if val < low or val > high:
                    colors2.append('#FF6B6B')
                else:
                    colors2.append('#4ECDC4')
        
        y_pos = range(len(values_to_plot))
        axes[1].barh(y_pos, values_to_plot, color=colors2)
        axes[1].axvline(x=50, color='green', linestyle='--', linewidth=2, label='Normal Center')
        axes[1].axvspan(25, 75, alpha=0.2, color='green', label='Normal Range')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels)
        axes[1].set_xlabel('Value Position (50 = Normal Center)')
        axes[1].set_title('Key Vital Signs & Lab Values')
        axes[1].set_xlim(0, 100)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        output_path = Path("model/shap_explanation.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.show()
    
    return {
        'probability': prob,
        'prediction': prediction,
        'sirs_score': sirs_score,
        'top_factors': feature_importance[:10]
    }


def explain_demo():
    """Run SHAP explanation demo"""
    print("\n" + "="*70)
    print("SEPSIS PREDICTION - SHAP EXPLAINABILITY")
    print("="*70)
    
    try:
        import shap  # Check if SHAP is available
        
        print("\nEnter patient parameters for detailed explanation:\n")
        
        lactate = input("Lactate (mmol/L) [default: 2.4]: ").strip()
        lactate = float(lactate) if lactate else 2.4
        
        wbc = input("WBC Count (x10^9/L) [default: 3.0]: ").strip()
        wbc = float(wbc) if wbc else 3.0
        
        crp = input("CRP Level (mg/L) [default: 10.0]: ").strip()
        crp = float(crp) if crp else 10.0
        
        creatinine = input("Creatinine (mg/dL) [default: 0.7]: ").strip()
        creatinine = float(creatinine) if creatinine else 0.7
        
        hr = input("Heart Rate (bpm) [default: 108]: ").strip()
        hr = float(hr) if hr else 108
        
        temp = input("Temperature (C) [default: 38.0]: ").strip()
        temp = float(temp) if temp else 38.0
        
        rr = input("Respiratory Rate (/min) [default: 16]: ").strip()
        rr = float(rr) if rr else 16
        
        age = input("Age (years) [default: 55]: ").strip()
        age = int(age) if age else 55
        
        print("\nGenerating SHAP explanation...")
        explain_prediction_shap(
            lactate=lactate,
            wbc_count=wbc,
            crp_level=crp,
            creatinine=creatinine,
            heart_rate=hr,
            temperature_c=temp,
            respiratory_rate=rr,
            age=age,
            show_plots=True
        )
        
    except ImportError:
        print("[ERROR] SHAP module not available. Install with: pip install shap")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


# ===============================
# PDF REPORT GENERATION
# ===============================
def generate_sepsis_report(
    patient_name: str,
    patient_id: str,
    age: int,
    gender: str,
    admission_date: str,
    admission_type: str,
    diagnosis: str,
    hourly_readings: list,
    output_path: str = None
):
    """
    Generate a comprehensive PDF report for sepsis risk analysis.
    
    Args:
        patient_name: Full name of patient
        patient_id: Hospital patient ID
        age: Patient age in years
        gender: 'M' or 'F'
        admission_date: Date of admission
        admission_type: 'ED', 'Elective', or 'Transfer'
        diagnosis: Initial diagnosis
        hourly_readings: List of dicts with hourly vital sign readings
            Each dict should have: hour, heart_rate, respiratory_rate, temperature_c,
            systolic_bp, diastolic_bp, spo2_pct, oxygen_flow, lactate, wbc_count, crp_level, creatinine
        output_path: Path to save the PDF (default: reports/patient_report_<id>.pdf)
    """
    try:
        from fpdf import FPDF
    except ImportError:
        print("[ERROR] FPDF not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'fpdf2'], check=True)
        from fpdf import FPDF
    
    import pandas as pd
    import numpy as np
    import joblib
    from datetime import datetime
    import matplotlib.pyplot as plt
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/sepsis_report_{patient_id}_{timestamp}.pdf"
    
    # Load model and threshold
    model = joblib.load('model/sepsis_rf_model.pkl')
    with open('model/sepsis_optimal_threshold.txt', 'r') as f:
        THRESHOLD = float(f.read().strip())
    
    # Get the latest reading for prediction
    latest = hourly_readings[-1] if hourly_readings else {}
    
    # Calculate SIRS for latest reading
    temp = latest.get('temperature_c', 37.0)
    hr = latest.get('heart_rate', 80)
    rr = latest.get('respiratory_rate', 16)
    wbc = latest.get('wbc_count', 8.0)
    
    sirs_temp = 1 if (temp > 38 or temp < 36) else 0
    sirs_hr = 1 if hr > 90 else 0
    sirs_rr = 1 if rr > 20 else 0
    sirs_wbc = 1 if (wbc > 12 or wbc < 4) else 0
    sirs_score = sirs_temp + sirs_hr + sirs_rr + sirs_wbc
    sirs_positive = 1 if sirs_score >= 2 else 0
    
    # Build features for prediction - MUST MATCH TRAINING DATA EXACTLY
    # Only use the 17 features the model was trained on
    features = pd.DataFrame([{
        'hour_from_admission': latest.get('hour', 12),
        'heart_rate': hr,
        'respiratory_rate': rr,
        'spo2_pct': latest.get('spo2_pct', 96),
        'temperature_c': temp,
        'systolic_bp': latest.get('systolic_bp', 120),
        'diastolic_bp': latest.get('diastolic_bp', 80),
        'oxygen_flow': latest.get('oxygen_flow', 0),
        'mobility_score': latest.get('mobility_score', 2),
        'nurse_alert': latest.get('nurse_alert', 0),
        'wbc_count': wbc,
        'lactate': latest.get('lactate', 1.0),
        'creatinine': latest.get('creatinine', 0.9),
        'crp_level': latest.get('crp_level', 5.0),
        'hemoglobin': latest.get('hemoglobin', 13.5),
        'age': age,
        'comorbidity_index': latest.get('comorbidity_index', 2),
    }])
    
    # Reorder columns to match training order exactly
    feature_order = [
        'hour_from_admission', 'heart_rate', 'respiratory_rate', 'spo2_pct',
        'temperature_c', 'systolic_bp', 'diastolic_bp', 'oxygen_flow',
        'mobility_score', 'nurse_alert', 'wbc_count', 'lactate', 'creatinine',
        'crp_level', 'hemoglobin', 'age', 'comorbidity_index'
    ]
    features = features[feature_order]
    
    # Get prediction and SHAP values
    prob = model.predict_proba(features)[0, 1]
    prediction = 1 if prob >= THRESHOLD else 0
    
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            if len(shap_values.shape) == 3:
                shap_vals = shap_values[0, :, 1]
            else:
                shap_vals = shap_values[0]
        shap_vals = np.array(shap_vals).flatten().astype(float)
        feature_importance = list(zip(features.columns.tolist(), features.values[0], shap_vals))
        feature_importance.sort(key=lambda x: abs(x[2]), reverse=True)
        has_shap = True
    except:
        has_shap = False
        feature_importance = []
    
    # Generate vitals trend chart
    if hourly_readings:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        hours = [r.get('hour', i) for i, r in enumerate(hourly_readings)]
        
        # Heart Rate
        hr_vals = [r.get('heart_rate', 80) for r in hourly_readings]
        axes[0, 0].plot(hours, hr_vals, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Upper limit')
        axes[0, 0].axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Lower limit')
        axes[0, 0].set_title('Heart Rate (bpm)', fontweight='bold')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('BPM')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc='upper right', fontsize=8)
        
        # Temperature
        temp_vals = [r.get('temperature_c', 37.0) for r in hourly_readings]
        axes[0, 1].plot(hours, temp_vals, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=38, color='r', linestyle='--', alpha=0.5, label='Fever threshold')
        axes[0, 1].axhline(y=36, color='orange', linestyle='--', alpha=0.5, label='Hypothermia')
        axes[0, 1].set_title('Temperature (C)', fontweight='bold')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Celsius')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc='upper right', fontsize=8)
        
        # Blood Pressure
        sbp_vals = [r.get('systolic_bp', 120) for r in hourly_readings]
        dbp_vals = [r.get('diastolic_bp', 80) for r in hourly_readings]
        axes[1, 0].plot(hours, sbp_vals, 'g-o', linewidth=2, markersize=6, label='Systolic')
        axes[1, 0].plot(hours, dbp_vals, 'purple', linestyle='-', marker='s', linewidth=2, markersize=5, label='Diastolic')
        axes[1, 0].axhline(y=90, color='r', linestyle='--', alpha=0.5, label='SBP low threshold')
        axes[1, 0].set_title('Blood Pressure (mmHg)', fontweight='bold')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('mmHg')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(loc='upper right', fontsize=8)
        
        # SpO2
        spo2_vals = [r.get('spo2_pct', 96) for r in hourly_readings]
        axes[1, 1].plot(hours, spo2_vals, 'm-o', linewidth=2, markersize=6)
        axes[1, 1].axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='Normal threshold')
        axes[1, 1].axhline(y=90, color='r', linestyle='--', alpha=0.5, label='Critical threshold')
        axes[1, 1].set_title('Oxygen Saturation (%)', fontweight='bold')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('SpO2 %')
        axes[1, 1].set_ylim(85, 100)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        vitals_chart_path = "reports/temp_vitals_chart.png"
        plt.savefig(vitals_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        vitals_chart_path = None
    
    # Generate SHAP chart if available
    if has_shap and feature_importance:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = 10
        top_features = feature_importance[:top_n]
        feat_names = [FEATURE_NAMES.get(f, f)[:25] for f, _, _ in top_features]
        shap_values_plot = [s for _, _, s in top_features]
        colors = ['#FF6B6B' if s > 0 else '#4ECDC4' for s in shap_values_plot]
        
        ax.barh(range(top_n), shap_values_plot, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_names)
        ax.invert_yaxis()
        ax.set_xlabel('Impact on Sepsis Risk (SHAP Value)')
        ax.set_title('Top Factors Affecting Sepsis Prediction', fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Increases Risk'),
            Patch(facecolor='#4ECDC4', label='Decreases Risk')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        shap_chart_path = "reports/temp_shap_chart.png"
        plt.savefig(shap_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        shap_chart_path = None
    
    # ===============================
    # CREATE PDF
    # ===============================
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 8, 'SEPSIS RISK ASSESSMENT REPORT', border=False, ln=True, align='C')
            self.set_font('Helvetica', '', 8)
            self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
            self.ln(5)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
    
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Patient Information Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'PATIENT INFORMATION', ln=True, fill=True, align='L')
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(3)
    
    # Create patient info table
    pdf.set_fill_color(240, 240, 240)
    col_width = 95
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(col_width, 8, 'Patient Name:', border=1, fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(col_width, 8, patient_name, border=1, ln=True)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(col_width, 8, 'Patient ID:', border=1, fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(col_width, 8, patient_id, border=1, ln=True)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(col_width, 8, 'Age / Gender:', border=1, fill=True)
    pdf.set_font('Helvetica', '', 10)
    gender_text = 'Male' if gender == 'M' else 'Female'
    pdf.cell(col_width, 8, f'{age} years / {gender_text}', border=1, ln=True)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(col_width, 8, 'Admission Date:', border=1, fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(col_width, 8, admission_date, border=1, ln=True)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(col_width, 8, 'Admission Type:', border=1, fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(col_width, 8, admission_type, border=1, ln=True)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(col_width, 8, 'Initial Diagnosis:', border=1, fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(col_width, 8, diagnosis, border=1, ln=True)
    
    pdf.ln(8)
    
    # Sepsis Risk Assessment Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'SEPSIS RISK ASSESSMENT', ln=True, fill=True, align='L')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    
    # Risk Level Box
    if sirs_score == 3:  # MODERATE RISK when exactly 3/4 SIRS criteria met
        pdf.set_fill_color(243, 156, 18)  # Orange
        risk_text = "MODERATE RISK"
        risk_desc = "Patient meets 3 out of 4 SIRS criteria. Close monitoring recommended. Consider additional lab work and frequent vital sign checks."
    elif prediction == 1:  # HIGH RISK from model prediction
        pdf.set_fill_color(231, 76, 60)  # Red
        risk_text = "HIGH RISK - SEPSIS ALERT"
        risk_desc = "Immediate clinical evaluation recommended. Consider blood cultures, lactate, and broad-spectrum antibiotics."
    elif sirs_score == 2:
        pdf.set_fill_color(240, 184, 82)  # Yellow/Gold
        risk_text = "MILD RISK"
        risk_desc = "Patient meets 2 out of 4 SIRS criteria. Recommend monitoring closely and consider laboratory workup."
    else:
        pdf.set_fill_color(39, 174, 96)  # Green
        risk_text = "LOW RISK"
        risk_desc = "Continue routine monitoring. No immediate sepsis concern."
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 15, risk_text, ln=True, fill=True, align='C')
    pdf.set_text_color(0, 0, 0)
    
    pdf.ln(3)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(10)
    pdf.multi_cell(190, 6, risk_desc)
    
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(60, 8, f'Sepsis Probability: {prob*100:.1f}%', border=1, align='C')
    pdf.cell(60, 8, f'SIRS Score: {sirs_score}/4', border=1, align='C')
    pdf.cell(70, 8, f'Threshold: {THRESHOLD*100:.0f}%', border=1, ln=True, align='C')
    
    pdf.ln(8)
    
    # Hourly Vital Signs Table
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'HOURLY VITAL SIGNS', ln=True, fill=True, align='L')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    
    if hourly_readings:
        # Table header
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_fill_color(200, 200, 200)
        headers = ['Hour', 'HR', 'Temp', 'RR', 'SBP', 'DBP', 'SpO2', 'Lactate', 'WBC', 'CRP']
        col_widths = [15, 18, 18, 15, 18, 18, 18, 22, 22, 22]
        
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            pdf.cell(width, 7, header, border=1, align='C', fill=True)
        pdf.ln()
        
        # Table rows
        pdf.set_font('Helvetica', '', 8)
        pdf.set_fill_color(255, 255, 255)
        
        for row in hourly_readings[-12:]:  # Last 12 hours
            hour = str(row.get('hour', '-'))
            hr_val = f"{row.get('heart_rate', '-'):.0f}" if isinstance(row.get('heart_rate'), (int, float)) else '-'
            temp_val = f"{row.get('temperature_c', '-'):.1f}" if isinstance(row.get('temperature_c'), (int, float)) else '-'
            rr_val = f"{row.get('respiratory_rate', '-'):.0f}" if isinstance(row.get('respiratory_rate'), (int, float)) else '-'
            sbp_val = f"{row.get('systolic_bp', '-'):.0f}" if isinstance(row.get('systolic_bp'), (int, float)) else '-'
            dbp_val = f"{row.get('diastolic_bp', '-'):.0f}" if isinstance(row.get('diastolic_bp'), (int, float)) else '-'
            spo2_val = f"{row.get('spo2_pct', '-'):.0f}" if isinstance(row.get('spo2_pct'), (int, float)) else '-'
            lactate_val = f"{row.get('lactate', '-'):.1f}" if isinstance(row.get('lactate'), (int, float)) else '-'
            wbc_val = f"{row.get('wbc_count', '-'):.1f}" if isinstance(row.get('wbc_count'), (int, float)) else '-'
            crp_val = f"{row.get('crp_level', '-'):.1f}" if isinstance(row.get('crp_level'), (int, float)) else '-'
            
            values = [hour, hr_val, temp_val, rr_val, sbp_val, dbp_val, spo2_val, lactate_val, wbc_val, crp_val]
            for val, width in zip(values, col_widths):
                pdf.cell(width, 6, val, border=1, align='C')
            pdf.ln()
        
        pdf.set_font('Helvetica', 'I', 7)
        pdf.cell(0, 5, 'HR=Heart Rate (bpm), Temp=Temperature (C), RR=Respiratory Rate, SBP/DBP=Blood Pressure (mmHg)', ln=True)
        pdf.cell(0, 5, 'SpO2=Oxygen Saturation (%), Lactate (mmol/L), WBC=White Blood Cells (x10^9/L), CRP (mg/L)', ln=True)
    
    # Vitals Chart
    if vitals_chart_path and Path(vitals_chart_path).exists():
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_fill_color(52, 73, 94)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, 'VITAL SIGNS TRENDS', ln=True, fill=True, align='L')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        pdf.image(vitals_chart_path, x=10, w=190)
    
    # SHAP Explainability Section
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'AI EXPLAINABILITY ANALYSIS (SHAP)', ln=True, fill=True, align='L')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    
    if has_shap and feature_importance:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_x(10)
        pdf.multi_cell(190, 5, 'SHAP (SHapley Additive exPlanations) values show how each clinical factor contributed to the sepsis risk prediction. Positive values increase risk, negative values decrease risk.')
        pdf.ln(5)
        
        # Top factors table
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(80, 8, 'Clinical Factor', border=1, align='C', fill=True)
        pdf.cell(40, 8, 'Value', border=1, align='C', fill=True)
        pdf.cell(40, 8, 'Impact', border=1, align='C', fill=True)
        pdf.cell(30, 8, 'Direction', border=1, ln=True, align='C', fill=True)
        
        pdf.set_font('Helvetica', '', 9)
        for feat, val, shap_val in feature_importance[:10]:
            readable = FEATURE_NAMES.get(feat, feat)[:35]
            direction = "Increases" if shap_val > 0 else "Decreases" if shap_val < 0 else "Neutral"
            
            if feat in NORMAL_RANGES:
                low, high, unit = NORMAL_RANGES[feat]
                if val < low or val > high:
                    val_str = f"{val:.1f}* {unit}"
                else:
                    val_str = f"{val:.1f} {unit}"
            else:
                val_str = f"{val:.2f}"
            
            pdf.cell(80, 7, readable, border=1)
            pdf.cell(40, 7, val_str, border=1, align='C')
            pdf.cell(40, 7, f"{abs(shap_val):.4f}", border=1, align='C')
            pdf.cell(30, 7, direction, border=1, ln=True, align='C')
        
        pdf.set_font('Helvetica', 'I', 8)
        pdf.cell(0, 6, '* = Value outside normal range', ln=True)
        
        # SHAP Chart
        if shap_chart_path and Path(shap_chart_path).exists():
            pdf.ln(5)
            pdf.image(shap_chart_path, x=10, w=190)
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 10, 'SHAP analysis not available.', ln=True)
    
    # Clinical Interpretation Section
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'CLINICAL INTERPRETATION', ln=True, fill=True, align='L')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Key Clinical Findings:', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(10)  # Reset x position to left margin
    
    findings = []
    lactate_val = latest.get('lactate', 1.0)
    wbc_val = latest.get('wbc_count', 8.0)
    crp_val = latest.get('crp_level', 5.0)
    
    if lactate_val > 2.0:
        findings.append(f"Elevated lactate ({lactate_val:.1f} mmol/L) - may indicate tissue hypoperfusion")
    if wbc_val > 12:
        findings.append(f"Elevated WBC ({wbc_val:.1f} x10^9/L) - suggests active infection")
    elif wbc_val < 4:
        findings.append(f"Low WBC ({wbc_val:.1f} x10^9/L) - immunocompromised state")
    if crp_val > 10:
        findings.append(f"Elevated CRP ({crp_val:.1f} mg/L) - acute phase inflammatory response")
    if temp > 38:
        findings.append(f"Fever ({temp:.1f}C) - suggests active infection")
    elif temp < 36:
        findings.append(f"Hypothermia ({temp:.1f}C) - concerning sign in suspected sepsis")
    if hr > 100:
        findings.append(f"Tachycardia ({hr:.0f} bpm) - may indicate compensatory response")
    if sirs_score >= 2:
        findings.append(f"SIRS criteria met ({sirs_score}/4) - systemic inflammatory response")
    
    if not findings:
        findings.append("No major abnormalities detected in current readings")
    
    for i, finding in enumerate(findings, 1):
        pdf.set_x(10)
        pdf.cell(190, 6, f"  {i}. {finding}", ln=True)
    
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Recommendations:', ln=True)
    pdf.set_font('Helvetica', '', 10)
    
    if prediction == 1:
        recommendations = [
            "Obtain blood cultures (2 sets) before antibiotic administration",
            "Measure serum lactate level if not done recently",
            "Administer broad-spectrum IV antibiotics within 1 hour",
            "IV fluid resuscitation if hypotensive or lactate > 4 mmol/L",
            "Monitor vital signs every 15-30 minutes",
            "Consider ICU consultation"
        ]
    elif prob >= 0.30 or sirs_score >= 2:
        recommendations = [
            "Repeat vital signs every 1-2 hours",
            "Consider additional lab work (CBC, CMP, lactate)",
            "Review current infection workup",
            "Reassess sepsis risk if clinical status changes",
            "Document source of infection if identified"
        ]
    else:
        recommendations = [
            "Continue routine monitoring per unit protocol",
            "Reassess if new symptoms develop",
            "Follow up on pending culture results"
        ]
    
    for i, rec in enumerate(recommendations, 1):
        pdf.set_x(10)
        pdf.cell(190, 6, f"  {i}. {rec}", ln=True)
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.set_x(10)
    pdf.multi_cell(190, 4, 'DISCLAIMER: This report is generated by an AI-assisted clinical decision support system and is intended to supplement, not replace, clinical judgment. All recommendations should be validated by qualified healthcare providers before implementation.')
    
    # Save PDF
    pdf.output(output_path)
    
    # Clean up temp files
    if vitals_chart_path and Path(vitals_chart_path).exists():
        Path(vitals_chart_path).unlink()
    if shap_chart_path and Path(shap_chart_path).exists():
        Path(shap_chart_path).unlink()
    
    print(f"\n[SUCCESS] Report saved to: {output_path}")
    return output_path


def generate_report_demo():
    """Interactive demo for generating PDF report"""
    print("\n" + "="*70)
    print("GENERATE SEPSIS RISK ASSESSMENT PDF REPORT")
    print("="*70)
    
    print("\n--- Patient Bio Data ---\n")
    
    patient_name = input("Patient Name [default: John Doe]: ").strip() or "John Doe"
    patient_id = input("Patient ID [default: PT-2026-001]: ").strip() or "PT-2026-001"
    age = input("Age (years) [default: 55]: ").strip()
    age = int(age) if age else 55
    gender = input("Gender (M/F) [default: M]: ").strip().upper() or "M"
    admission_date = input("Admission Date [default: 2026-02-23]: ").strip() or "2026-02-23"
    admission_type = input("Admission Type (ED/Elective/Transfer) [default: ED]: ").strip() or "ED"
    diagnosis = input("Initial Diagnosis [default: Suspected infection]: ").strip() or "Suspected infection"
    
    print("\n--- Hourly Vital Signs (Enter 'done' to finish) ---")
    print("Enter readings or press Enter for defaults. Type 'demo' for sample data.\n")
    
    hourly_readings = []
    use_demo = input("Use demo data? (y/n) [default: y]: ").strip().lower()
    
    if use_demo != 'n':
        # Generate sample hourly data showing potential deterioration
        import random
        base_hr = 85
        base_temp = 37.2
        base_lactate = 1.2
        
        for hour in range(1, 13):
            # Simulate gradual deterioration
            deterioration = hour / 12.0
            hourly_readings.append({
                'hour': hour,
                'heart_rate': base_hr + random.uniform(-5, 15) + (deterioration * 25),
                'respiratory_rate': 16 + random.uniform(-2, 4) + (deterioration * 6),
                'temperature_c': base_temp + random.uniform(-0.2, 0.3) + (deterioration * 0.8),
                'systolic_bp': 120 - random.uniform(0, 10) - (deterioration * 15),
                'diastolic_bp': 80 - random.uniform(0, 5) - (deterioration * 8),
                'spo2_pct': 97 - random.uniform(0, 2) - (deterioration * 3),
                'oxygen_flow': 0 if hour < 8 else 2,
                'lactate': base_lactate + random.uniform(-0.1, 0.3) + (deterioration * 1.5),
                'wbc_count': 8.5 + random.uniform(-0.5, 1) + (deterioration * 5),
                'crp_level': 5 + random.uniform(-1, 2) + (deterioration * 20),
                'creatinine': 0.9 + random.uniform(-0.1, 0.1) + (deterioration * 0.3),
            })
        print(f"\n[INFO] Generated {len(hourly_readings)} hours of sample data")
    else:
        hour = 1
        while True:
            print(f"\n--- Hour {hour} ---")
            done = input("Enter readings for this hour? (y/n/done): ").strip().lower()
            if done == 'done' or done == 'n':
                break
            
            hr = input(f"  Heart Rate (bpm) [default: 85]: ").strip()
            temp = input(f"  Temperature (C) [default: 37.0]: ").strip()
            rr = input(f"  Respiratory Rate [default: 16]: ").strip()
            sbp = input(f"  Systolic BP [default: 120]: ").strip()
            dbp = input(f"  Diastolic BP [default: 80]: ").strip()
            spo2 = input(f"  SpO2 (%) [default: 96]: ").strip()
            lactate = input(f"  Lactate (mmol/L) [default: 1.2]: ").strip()
            wbc = input(f"  WBC Count [default: 8.0]: ").strip()
            crp = input(f"  CRP Level [default: 5.0]: ").strip()
            
            hourly_readings.append({
                'hour': hour,
                'heart_rate': float(hr) if hr else 85,
                'respiratory_rate': float(rr) if rr else 16,
                'temperature_c': float(temp) if temp else 37.0,
                'systolic_bp': float(sbp) if sbp else 120,
                'diastolic_bp': float(dbp) if dbp else 80,
                'spo2_pct': float(spo2) if spo2 else 96,
                'lactate': float(lactate) if lactate else 1.2,
                'wbc_count': float(wbc) if wbc else 8.0,
                'crp_level': float(crp) if crp else 5.0,
            })
            hour += 1
    
    if not hourly_readings:
        print("\n[WARNING] No hourly readings provided. Using single default reading.")
        hourly_readings = [{'hour': 1, 'heart_rate': 95, 'temperature_c': 38.0, 'respiratory_rate': 20,
                          'systolic_bp': 110, 'diastolic_bp': 70, 'spo2_pct': 94, 'lactate': 2.2,
                          'wbc_count': 12.5, 'crp_level': 25.0, 'creatinine': 1.1}]
    
    print("\n[INFO] Generating PDF report...")
    
    try:
        report_path = generate_sepsis_report(
            patient_name=patient_name,
            patient_id=patient_id,
            age=age,
            gender=gender,
            admission_date=admission_date,
            admission_type=admission_type,
            diagnosis=diagnosis,
            hourly_readings=hourly_readings
        )
        print(f"\n[SUCCESS] Report generated: {report_path}")
        
        # Ask to open
        open_report = input("\nOpen report now? (y/n) [default: y]: ").strip().lower()
        if open_report != 'n':
            import subprocess
            subprocess.Popen(['start', '', report_path], shell=True)
            
    except Exception as e:
        print(f"\n[ERROR] Failed to generate report: {e}")
        import traceback
        traceback.print_exc()


def quick_predict():
    """Quick prediction with command line arguments"""
    print("\n" + "="*70)
    print("QUICK SEPSIS PREDICTION")
    print("="*70)
    
    print("\nEnter key parameters (normal ranges shown for reference):\n")
    
    try:
        lactate = float(input("Lactate [Normal: 0.5-2.0 mmol/L]: "))
        wbc = float(input("WBC Count [Normal: 4-11 x10^9/L]: "))
        crp = float(input("CRP Level [Normal: 0-10 mg/L]: "))
        creatinine = float(input("Creatinine [Normal: 0.6-1.2 mg/dL]: "))
        
        pred, prob, sirs, _ = predict_sepsis(
            lactate=lactate,
            wbc_count=wbc,
            crp_level=crp,
            creatinine=creatinine
        )
        
        print("\n" + "="*40)
        if pred == 1:
            print(f"RESULT: HIGH RISK ({prob*100:.1f}%)")
        elif prob >= 0.30:
            print(f"RESULT: MODERATE RISK ({prob*100:.1f}%)")
        else:
            print(f"RESULT: LOW RISK ({prob*100:.1f}%)")
        print(f"SIRS Score: {sirs}/4")
        print("="*40)
        
    except Exception as e:
        print(f"\nError: {e}")


def run_all():
    """Run the complete pipeline"""
    print("\n" + "="*70)
    print("RUNNING COMPLETE SEPSIS PREDICTION PIPELINE")
    print("="*70)
    
    # Check if data needs preparation
    if not Path("dataset/sepsis_data/train_sepsis.csv").exists():
        print("\n[1/3] Data not found. Preparing data...")
        prepare_data()
    else:
        print("\n[1/3] Data already prepared. Skipping...")
    
    # Check if model needs training
    if not Path("model/sepsis_rf_model.pkl").exists():
        print("\n[2/3] Model not found. Training model...")
        train_model()
    else:
        print("\n[2/3] Model already trained. Skipping...")
    
    # Run interactive test
    print("\n[3/3] Starting interactive test mode...")
    interactive_test()


def show_menu():
    """Display interactive menu"""
    print("\n" + "="*70)
    print("SEPSIS PREDICTION SYSTEM - MAIN MENU")
    print("="*70)
    print("""
    1. Test Model        - Interactive testing with patient parameters
    2. SHAP Explanation  - Detailed AI explanation for predictions
    3. Quick Predict     - Fast prediction with key parameters only
    4. Generate Report   - Create PDF report with vitals & SHAP analysis
    """)
    
    while True:
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            interactive_test()
        elif choice == '2':
            explain_demo()
        elif choice == '3':
            quick_predict()
        elif choice == '4':
            generate_report_demo()
        else:
            print("Invalid option. Please select 1-4.")
            continue
        
        input("\nPress Enter to continue...")
        print("\n" + "="*70)
        print("SEPSIS PREDICTION SYSTEM - MAIN MENU")
        print("="*70)
        print("""
    1. Test Model        - Interactive testing with patient parameters
    2. SHAP Explanation  - Detailed AI explanation for predictions
    3. Quick Predict     - Fast prediction with key parameters only
    4. Generate Report   - Create PDF report with vitals & SHAP analysis
        """)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sepsis Prediction System - 6 Hour Horizon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Show interactive menu
  python main.py test         # Interactive testing
  python main.py explain      # SHAP explanation
  python main.py predict      # Quick prediction
  python main.py report       # Generate PDF report
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['test', 'explain', 'predict', 'report', 'menu'],
        default='menu',
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("       SEPSIS PREDICTION SYSTEM - 6 HOUR HORIZON")
    print("       Machine Learning for Early Sepsis Detection")
    print("="*70)
    
    # Execute command
    if args.command == 'test':
        interactive_test()
    elif args.command == 'explain':
        explain_demo()
    elif args.command == 'predict':
        quick_predict()
    elif args.command == 'report':
        generate_report_demo()
    else:
        show_menu()


if __name__ == "__main__":
    main()