"""
Unit tests for the Sepsis Prediction API
Run with: pytest test_app.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app import app, calculate_sirs_score, validate_vitals, classify_risk_level, PatientVitals

# Create test client
client = TestClient(app)

# Test data
VALID_VITALS = {
    "patient_id": "TEST001",
    "lactate": 1.5,
    "wbc_count": 8.5,
    "crp_level": 5.0,
    "creatinine": 0.9,
    "heart_rate": 95,
    "respiratory_rate": 18,
    "temperature": 37.0,
    "spo2": 96,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "hemoglobin": 14.0
}

ABNORMAL_VITALS = {
    "patient_id": "TEST002",
    "lactate": 5.0,
    "wbc_count": 15.0,
    "crp_level": 50.0,
    "creatinine": 2.0,
    "heart_rate": 120,
    "respiratory_rate": 30,
    "temperature": 39.5,
    "spo2": 88,
    "systolic_bp": 160,
    "diastolic_bp": 100,
    "hemoglobin": 10.0
}


# ============================================================================
# HEALTH CHECKS
# ============================================================================

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Sepsis Prediction API"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ============================================================================
# SIRS CALCULATION
# ============================================================================

def test_calculate_sirs_positive():
    """Test SIRS calculation with abnormal values"""
    vitals = PatientVitals(**{k:v for k,v in ABNORMAL_VITALS.items() if k != "patient_id"})
    sirs_score, criteria_met, sirs_positive, interpretation = calculate_sirs_score(vitals)
    
    assert sirs_score >= 2
    assert sirs_positive == True
    assert len(criteria_met) > 0


def test_calculate_sirs_negative():
    """Test SIRS calculation with normal values"""
    vitals = PatientVitals(**{k:v for k,v in VALID_VITALS.items() if k != "patient_id"})
    sirs_score, criteria_met, sirs_positive, interpretation = calculate_sirs_score(vitals)
    
    assert sirs_score < 2
    assert sirs_positive == False


def test_sirs_endpoint():
    """Test SIRS endpoint"""
    response = client.post("/sirs-score", json={k:v for k,v in VALID_VITALS.items() if k != "patient_id"})
    assert response.status_code == 200
    result = response.json()
    assert "sirs_score" in result
    assert "criteria_met" in result
    assert "sirs_positive" in result


# ============================================================================
# VALIDATION
# ============================================================================

def test_validate_vitals_valid():
    """Test validation with valid vitals"""
    vitals = PatientVitals(**{k:v for k,v in VALID_VITALS.items() if k != "patient_id"})
    validation = validate_vitals(vitals)
    assert validation.is_valid == True
    assert len(validation.abnormal_values) == 0


def test_validate_vitals_abnormal():
    """Test validation with abnormal vitals"""
    vitals = PatientVitals(**{k:v for k,v in ABNORMAL_VITALS.items() if k != "patient_id"})
    validation = validate_vitals(vitals)
    assert validation.is_valid == False
    assert len(validation.abnormal_values) > 0


def test_validate_vitals_endpoint():
    """Test validation endpoint"""
    response = client.post("/validate-vitals", json={k:v for k,v in VALID_VITALS.items() if k != "patient_id"})
    assert response.status_code == 200
    result = response.json()
    assert "is_valid" in result
    assert "abnormal_values" in result


# ============================================================================
# RISK CLASSIFICATION
# ============================================================================

def test_classify_risk_low():
    """Test risk classification - low"""
    risk_level = classify_risk_level(0.2)
    assert risk_level == "LOW"


def test_classify_risk_moderate():
    """Test risk classification - moderate"""
    risk_level = classify_risk_level(0.5)
    assert risk_level == "MODERATE"


def test_classify_risk_high():
    """Test risk classification - high"""
    risk_level = classify_risk_level(0.75)
    assert risk_level == "HIGH"


def test_classify_risk_critical():
    """Test risk classification - critical"""
    risk_level = classify_risk_level(0.9)
    assert risk_level == "CRITICAL"


# ============================================================================
# PREDICTION
# ============================================================================

def test_predict_valid_patient():
    """Test prediction endpoint with valid data"""
    response = client.post("/predict", json=VALID_VITALS)
    
    # Check status code
    if response.status_code == 503:
        pytest.skip("Model not available")
    
    assert response.status_code == 200
    result = response.json()
    
    assert "risk_score" in result
    assert "risk_level" in result
    assert "sirs_info" in result
    assert "recommendation" in result
    assert 0 <= result["risk_score"] <= 1


def test_predict_abnormal_patient():
    """Test prediction with abnormal vitals"""
    response = client.post("/predict", json=ABNORMAL_VITALS)
    
    if response.status_code == 503:
        pytest.skip("Model not available")
    
    assert response.status_code == 200


# ============================================================================
# EXPLANATIONS
# ============================================================================

def test_explain_doctor():
    """Test doctor explanation endpoint"""
    response = client.post("/explain-doctor", json=VALID_VITALS)
    
    if response.status_code == 503:
        pytest.skip("Model not available")
    
    assert response.status_code == 200
    result = response.json()
    assert "clinical_assessment" in result


def test_explain_patient():
    """Test patient explanation endpoint"""
    response = client.post("/explain-patient", json=VALID_VITALS)
    
    if response.status_code == 503:
        pytest.skip("Model not available")
    
    assert response.status_code == 200
    result = response.json()
    assert "simple_explanation" in result


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def test_list_models():
    """Test list models endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    result = response.json()
    assert "available_models" in result
    assert len(result["available_models"]) > 0


def test_get_normal_ranges():
    """Test normal ranges endpoint"""
    response = client.get("/normal-ranges")
    assert response.status_code == 200
    result = response.json()
    assert "normal_ranges" in result
    assert "sirs_criteria" in result


# ============================================================================
# PATIENT DATA
# ============================================================================

def test_get_patient_history():
    """Test patient history endpoint"""
    response = client.get("/patient/TEST001/history")
    assert response.status_code == 200
    result = response.json()
    assert "patient_id" in result
    assert "predictions" in result


def test_get_current_risk():
    """Test current risk endpoint"""
    response = client.get("/patient/TEST001/current-risk")
    assert response.status_code == 200
    result = response.json()
    assert "current_risk_score" in result
    assert "current_risk_level" in result


# ============================================================================
# ANALYTICS
# ============================================================================

def test_analytics_summary():
    """Test analytics endpoint"""
    response = client.get("/analytics/summary")
    assert response.status_code == 200
    result = response.json()
    assert "system_metrics" in result
    assert "risk_distribution" in result


def test_analytics_time_series():
    """Test time series endpoint"""
    response = client.get("/analytics/time-series/TEST001?hours=24")
    assert response.status_code == 200
    result = response.json()
    assert "data" in result


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

def test_invalid_patient_id():
    """Test with missing patient ID"""
    invalid_data = {k:v for k,v in VALID_VITALS.items() if k != "patient_id"}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_invalid_temperature():
    """Test with invalid temperature"""
    invalid_data = VALID_VITALS.copy()
    invalid_data["temperature"] = 50.0  # Too high
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


def test_invalid_spo2():
    """Test with invalid SPO2"""
    invalid_data = VALID_VITALS.copy()
    invalid_data["spo2"] = 150.0  # Too high
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


# ============================================================================
# BATCH PREDICTIONS
# ============================================================================

def test_predict_batch():
    """Test batch prediction"""
    batch_data = {
        "patients": [VALID_VITALS, ABNORMAL_VITALS],
        "model_name": "xgboost"
    }
    response = client.post("/predict-batch", json=batch_data)
    
    if response.status_code == 503:
        pytest.skip("Model not available")
    
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert result["total"] == 2


# ============================================================================
# THRESHOLDS
# ============================================================================

def test_get_thresholds():
    """Test get thresholds endpoint"""
    response = client.get("/thresholds")
    assert response.status_code == 200
    result = response.json()
    assert "risk_thresholds" in result


def test_update_threshold():
    """Test update threshold endpoint"""
    response = client.post("/thresholds/update?threshold=0.6&category=high")
    assert response.status_code == 200
    result = response.json()
    assert "new_thresholds" in result


# ============================================================================
# FEATURES
# ============================================================================

def test_list_features():
    """Test features endpoint"""
    response = client.get("/features")
    assert response.status_code == 200
    result = response.json()
    assert "features" in result
    assert len(result["features"]) > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
