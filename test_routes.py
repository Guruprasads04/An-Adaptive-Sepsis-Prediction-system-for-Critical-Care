"""Quick test script for all API routes with both models"""
import httpx
import json

BASE = "http://127.0.0.1:8000"
T = 60.0

patient = {
    "patient_id": "P001",
    "lactate": 1.5, "wbc_count": 8.5, "crp_level": 5.0,
    "creatinine": 0.9, "heart_rate": 95, "respiratory_rate": 22,
    "temperature": 38.5, "spo2": 96, "systolic_bp": 120,
    "diastolic_bp": 80, "hemoglobin": 14.0,
    "hour_from_admission": 12, "oxygen_flow": 2.0,
    "mobility_score": 5, "nurse_alert": 0, "comorbidity_index": 1.5,
    "age": 65, "gender": "M", "admission_type": "Emergency", "oxygen_device": "nasal"
}

def test(name, method, url, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        if method == "GET":
            r = httpx.get(url, timeout=T)
        else:
            r = httpx.post(url, timeout=T, **kwargs)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(json.dumps(data, indent=2, default=str))
        return r.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

results = []

# Health
results.append(("GET /health", test("GET /health", "GET", f"{BASE}/health")))

# Predict with XGBoost
results.append(("POST /predict (XGBoost)", test("POST /predict (XGBoost)", "POST", f"{BASE}/predict?model_name=xgboost", json=patient)))

# Predict with Random Forest
results.append(("POST /predict (RF)", test("POST /predict (Random Forest)", "POST", f"{BASE}/predict?model_name=random_forest", json=patient)))

# Compare both models
results.append(("POST /compare", test("POST /compare (both models)", "POST", f"{BASE}/compare", json=patient)))

# Risk stratification (ensemble)
results.append(("POST /risk-stratification", test("POST /risk-stratification (ensemble)", "POST", f"{BASE}/risk-stratification", json=patient)))

# SIRS score
results.append(("POST /sirs-score", test("POST /sirs-score", "POST", f"{BASE}/sirs-score", json=patient)))

# Features for each model
results.append(("GET /features (XGB)", test("GET /features?model_name=xgboost", "GET", f"{BASE}/features?model_name=xgboost")))
results.append(("GET /features (RF)", test("GET /features?model_name=random_forest", "GET", f"{BASE}/features?model_name=random_forest")))

# Models list
results.append(("GET /models", test("GET /models", "GET", f"{BASE}/models")))

# Normal ranges
results.append(("GET /normal-ranges", test("GET /normal-ranges", "GET", f"{BASE}/normal-ranges")))

# Summary
print(f"\n{'='*60}")
print(f"  RESULTS SUMMARY")
print(f"{'='*60}")
passed = sum(1 for _, ok in results if ok)
total = len(results)
for name, ok in results:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
print(f"\n  {passed}/{total} tests passed")
print(f"{'='*60}")
