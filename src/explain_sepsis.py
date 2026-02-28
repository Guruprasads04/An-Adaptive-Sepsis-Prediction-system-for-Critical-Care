"""
SHAP Explainability for Sepsis Prediction Model
================================================
Provides detailed explanations for predictions:
- Technical view (for doctors/clinicians)
- Simple view (for patients/families)
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# LOAD MODEL AND DATA
# ===============================
print("Loading model and preparing explainer...")
model = joblib.load('model/sepsis_rf_model.pkl')
with open('model/sepsis_optimal_threshold.txt', 'r') as f:
    THRESHOLD = float(f.read().strip())

# Load sample data for background
train_df = pd.read_csv('dataset/sepsis_data/train_sepsis.csv')
X_train = train_df.drop(columns=['sepsis_next_6h'])

# Create SHAP explainer using TreeExplainer without background data
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Feature name mappings for readability
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

# Normal ranges for reference
NORMAL_RANGES = {
    'lactate': (0.5, 2.0, 'mmol/L'),
    'wbc_count': (4.0, 11.0, 'x10^9/L'),
    'crp_level': (0, 10, 'mg/L'),
    'creatinine': (0.6, 1.2, 'mg/dL'),
    'heart_rate': (60, 100, 'bpm'),
    'respiratory_rate': (12, 20, '/min'),
    'temperature_c': (36.5, 37.5, '°C'),
    'systolic_bp': (90, 140, 'mmHg'),
    'diastolic_bp': (60, 90, 'mmHg'),
    'spo2_pct': (95, 100, '%'),
    'hemoglobin': (12.0, 17.5, 'g/dL'),
}

def get_simple_explanation(feature, value, shap_val):
    """Generate simple, patient-friendly explanation"""
    readable_name = FEATURE_NAMES.get(feature, feature)
    
    if shap_val > 0.05:
        impact = "INCREASING"
        impact_word = "higher"
    elif shap_val < -0.05:
        impact = "DECREASING" 
        impact_word = "lower"
    else:
        return None  # Skip negligible impacts
    
    # Check if value is abnormal
    if feature in NORMAL_RANGES:
        low, high, unit = NORMAL_RANGES[feature]
        if value < low:
            status = f"below normal ({value:.1f} {unit}, normal: {low}-{high})"
        elif value > high:
            status = f"above normal ({value:.1f} {unit}, normal: {low}-{high})"
        else:
            status = f"normal ({value:.1f} {unit})"
    else:
        status = f"value: {value:.1f}"
    
    return {
        'feature': readable_name,
        'status': status,
        'impact': impact,
        'impact_word': impact_word,
        'shap_value': shap_val
    }


def explain_prediction(
    # Key parameters
    lactate: float,
    wbc_count: float,
    crp_level: float,
    creatinine: float,
    # Vital signs
    heart_rate: float = 80,
    respiratory_rate: float = 16,
    spo2_pct: float = 96,
    temperature_c: float = 37.0,
    systolic_bp: float = 120,
    diastolic_bp: float = 80,
    # Patient info
    age: int = 50,
    gender: str = 'M',
    comorbidity_index: int = 2,
    admission_type: str = 'ED',
    # Other
    hour_from_admission: int = 12,
    oxygen_device: str = 'none',
    oxygen_flow: float = 0,
    mobility_score: int = 2,
    nurse_alert: int = 0,
    hemoglobin: float = 13.5,
    show_plots: bool = True
):
    """
    Explain a sepsis prediction with SHAP values.
    
    Returns detailed explanation for both clinicians and patients.
    """
    
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
    # For binary classification, get values for positive class
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]  # Class 1 (sepsis)
    else:
        # Handle newer SHAP versions where it returns 3D array
        if len(shap_values.shape) == 3:
            shap_vals = shap_values[0, :, 1]  # First sample, all features, class 1
        elif len(shap_values.shape) == 2:
            shap_vals = shap_values[0]  # First sample
        else:
            shap_vals = shap_values
    
    # Ensure shap_vals is 1D array of floats
    shap_vals = np.array(shap_vals).flatten().astype(float)
    
    # Create SHAP explanation object
    feature_names = features.columns.tolist()
    
    # ===============================
    # PRINT RESULTS
    # ===============================
    
    print("\n" + "="*70)
    print("SEPSIS RISK PREDICTION WITH EXPLANATION")
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
    
    # Sort features by absolute SHAP value
    feature_importance = list(zip(feature_names, features.values[0], shap_vals))
    feature_importance.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nTop Contributing Factors (SHAP Analysis):")
    print("-"*70)
    print(f"{'Feature':<30} {'Value':>12} {'Impact':>12} {'Direction':>12}")
    print("-"*70)
    
    for feat, val, shap_val in feature_importance[:10]:
        readable = FEATURE_NAMES.get(feat, feat)[:28]
        direction = "^ Risk" if shap_val > 0 else "v Risk" if shap_val < 0 else "-"
        
        # Format value
        if feat in NORMAL_RANGES:
            low, high, unit = NORMAL_RANGES[feat]
            if val < low or val > high:
                val_str = f"{val:.1f}*"  # Mark abnormal
            else:
                val_str = f"{val:.1f}"
        else:
            val_str = f"{val:.2f}"
        
        print(f"{readable:<30} {val_str:>12} {shap_val:>+12.4f} {direction:>12}")
    
    print("-"*70)
    print("* = Outside normal range | ^ = Increases risk | v = Decreases risk")
    
    # Abnormal values summary
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
    
    # Risk level in simple terms
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
    
    for feat, val, shap_val in feature_importance[:8]:
        explanation = get_simple_explanation(feat, val, shap_val)
        if explanation:
            if shap_val > 0.02:
                risk_factors.append(explanation)
            elif shap_val < -0.02:
                protective_factors.append(explanation)
    
    if risk_factors:
        print("\n[!] Factors INCREASING infection risk:")
        for i, exp in enumerate(risk_factors[:5], 1):
            print(f"    {i}. {exp['feature']}: {exp['status']}")
    
    if protective_factors:
        print("\n[+] Factors DECREASING infection risk:")
        for i, exp in enumerate(protective_factors[:3], 1):
            print(f"    {i}. {exp['feature']}: {exp['status']}")
    
    # Key recommendations
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
        
        # Create SHAP waterfall plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Top features bar chart (simpler)
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
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Increases Risk'),
            Patch(facecolor='#4ECDC4', label='Decreases Risk')
        ]
        axes[0].legend(handles=legend_elements, loc='lower right')
        
        # Plot 2: Feature values with normal ranges
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
                
                # Normalize to percentage of normal range
                mid = (low + high) / 2
                range_size = (high - low) / 2
                normalized = (val - mid) / range_size * 50 + 50  # 50 = middle
                
                values_to_plot.append(normalized)
                labels.append(f"{FEATURE_NAMES.get(feat, feat)}\n({val:.1f} {unit})")
                
                if val < low or val > high:
                    colors2.append('#FF6B6B')  # Abnormal
                else:
                    colors2.append('#4ECDC4')  # Normal
        
        y_pos = range(len(values_to_plot))
        axes[1].barh(y_pos, values_to_plot, color=colors2)
        axes[1].axvline(x=50, color='green', linestyle='--', linewidth=2, label='Normal Range Center')
        axes[1].axvspan(25, 75, alpha=0.2, color='green', label='Normal Range')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels)
        axes[1].set_xlabel('Value Position (50 = Normal Center)')
        axes[1].set_title('Key Vital Signs & Lab Values')
        axes[1].set_xlim(0, 100)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path("model/shap_explanation.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.show()
    
    return {
        'probability': prob,
        'prediction': prediction,
        'sirs_score': sirs_score,
        'shap_values': dict(zip(feature_names, shap_vals)),
        'top_factors': feature_importance[:10]
    }


# ===============================
# DEMO / TEST
# ===============================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SEPSIS PREDICTION EXPLAINABILITY DEMO")
    print("="*70)
    
    # Test Case: Moderate risk patient
    print("\nAnalyzing patient with elevated markers...")
    
    result = explain_prediction(
        lactate=2.4,
        wbc_count=3.0,       # Low (abnormal)
        crp_level=10.0,
        creatinine=0.7,
        heart_rate=108,      # Elevated
        temperature_c=38.0,  # Fever
        respiratory_rate=16,
        age=55,
        show_plots=True
    )
    
    print("\n" + "="*70)
    print("Want to test with different values?")
    print("="*70)
    print("""
Example usage:
    
    from explain_sepsis import explain_prediction
    
    explain_prediction(
        lactate=3.5,        # Main predictor
        wbc_count=15.0,     # Elevated
        crp_level=50.0,     # High inflammation
        creatinine=1.8,     # Kidney stress
        heart_rate=110,     # Tachycardia
        temperature_c=38.5  # Fever
    )
""")
