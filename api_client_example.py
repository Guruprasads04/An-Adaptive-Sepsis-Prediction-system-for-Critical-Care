"""
Example client code for using the Sepsis Prediction API
"""

import requests
import json
from typing import Dict

# API base URL
API_URL = "http://localhost:8000"

class SepsisPredictionClient:
    """Client for interacting with the Sepsis Prediction API"""
    
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """Check if API is running"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_single(self, patient_data: Dict) -> Dict:
        """Make a prediction for a single patient"""
        response = requests.post(
            f"{self.base_url}/predict",
            json=patient_data,
            params={"model_name": "xgboost"}
        )
        return response.json()
    
    def predict_batch(self, patients: list, model_name: str = "xgboost") -> Dict:
        """Make predictions for multiple patients"""
        response = requests.post(
            f"{self.base_url}/predict-batch",
            json={
                "patients": patients,
                "model_name": model_name
            }
        )
        return response.json()
    
    def get_explanation(self, patient_data: Dict) -> Dict:
        """Get SHAP explanation for a prediction"""
        response = requests.post(
            f"{self.base_url}/explain",
            json=patient_data
        )
        return response.json()
    
    def get_doctor_explanation(self, patient_data: Dict) -> Dict:
        """Get doctor-friendly explanation"""
        response = requests.post(
            f"{self.base_url}/explain-doctor",
            json=patient_data
        )
        return response.json()
    
    def get_patient_explanation(self, patient_data: Dict) -> Dict:
        """Get patient-friendly explanation"""
        response = requests.post(
            f"{self.base_url}/explain-patient",
            json=patient_data
        )
        return response.json()
    
    def calculate_sirs(self, vitals: Dict) -> Dict:
        """Calculate SIRS score"""
        response = requests.post(
            f"{self.base_url}/sirs-score",
            json=vitals
        )
        return response.json()
    
    def validate_vitals(self, vitals: Dict) -> Dict:
        """Validate vital signs"""
        response = requests.post(
            f"{self.base_url}/validate-vitals",
            json=vitals
        )
        return response.json()
    
    def list_models(self) -> Dict:
        """List available models"""
        response = requests.get(f"{self.base_url}/models")
        return response.json()
    
    def get_normal_ranges(self) -> Dict:
        """Get normal value ranges"""
        response = requests.get(f"{self.base_url}/normal-ranges")
        return response.json()
    
    def get_patient_history(self, patient_id: str) -> Dict:
        """Get patient's prediction history"""
        response = requests.get(f"{self.base_url}/patient/{patient_id}/history")
        return response.json()
    
    def get_analytics(self) -> Dict:
        """Get system analytics"""
        response = requests.get(f"{self.base_url}/analytics/summary")
        return response.json()


# Example patient data
PATIENT_DATA = {
    "patient_id": "P001",
    "age": 65,
    "gender": "M",
    "lactate": 2.5,
    "wbc_count": 13.0,
    "crp_level": 15.0,
    "creatinine": 1.1,
    "heart_rate": 105,
    "respiratory_rate": 24,
    "temperature": 38.5,
    "spo2": 94,
    "systolic_bp": 135,
    "diastolic_bp": 85,
    "hemoglobin": 13.5,
    "notes": "Admitted with fever and chills"
}


def main():
    """Example usage of the client"""
    
    # Initialize client
    client = SepsisPredictionClient()
    
    print("=" * 80)
    print("SEPSIS PREDICTION SYSTEM - API CLIENT DEMO")
    print("=" * 80)
    
    # 1. Health Check
    print("\n1. Health Check")
    print("-" * 80)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Available Models: {health['models_available']}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 2. Validate Vitals
    print("\n2. Validate Vital Signs")
    print("-" * 80)
    validation = client.validate_vitals(PATIENT_DATA)
    if validation['is_valid']:
        print("✓ All vitals are valid")
    else:
        print("⚠ Abnormal values detected:")
        for abnormal in validation['abnormal_values']:
            print(f"  - {abnormal['parameter']}: {abnormal['value']} "
                  f"(Normal: {abnormal['normal_range']})")
    
    # 3. Calculate SIRS
    print("\n3. SIRS Score Calculation")
    print("-" * 80)
    sirs = client.calculate_sirs(PATIENT_DATA)
    print(f"SIRS Score: {sirs['sirs_score']}/4")
    print(f"Criteria Met: {', '.join(sirs['criteria_met'])}")
    print(f"SIRS Positive: {'Yes' if sirs['sirs_positive'] else 'No'}")
    print(f"Interpretation: {sirs['interpretation']}")
    
    # 4. Make Prediction
    print("\n4. Sepsis Risk Prediction")
    print("-" * 80)
    prediction = client.predict_single(PATIENT_DATA)
    print(f"Patient ID: {prediction['patient_id']}")
    print(f"Risk Score: {prediction['risk_score']:.1%}")
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"Model Used: {prediction['model_used']}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print(f"Recommendation: {prediction['recommendation']}")
    
    # 5. Get Doctor Explanation
    print("\n5. Doctor-Friendly Explanation")
    print("-" * 80)
    doc_exp = client.get_doctor_explanation(PATIENT_DATA)
    print(doc_exp['clinical_assessment'])
    
    # 6. Get Patient Explanation
    print("\n6. Patient-Friendly Explanation")
    print("-" * 80)
    pat_exp = client.get_patient_explanation(PATIENT_DATA)
    print(pat_exp['simple_explanation'])
    
    # 7. List Models
    print("\n7. Available Models")
    print("-" * 80)
    models = client.list_models()
    print(f"Available Models: {models['available_models']}")
    print(f"Active Model: {models['active_model']}")
    
    # 8. Get Analytics
    print("\n8. System Analytics")
    print("-" * 80)
    analytics = client.get_analytics()
    print(f"Total Predictions: {analytics['system_metrics']['total_predictions']}")
    print(f"Risk Distribution:")
    for risk_level, count in analytics['risk_distribution'].items():
        print(f"  - {risk_level}: {count}")
    
    # 9. Get Normal Ranges
    print("\n9. Normal Value Ranges")
    print("-" * 80)
    ranges = client.get_normal_ranges()
    for param, values in list(ranges['normal_ranges'].items())[:3]:
        print(f"{param}: {values['min']}-{values['max']} {values['unit']}")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
