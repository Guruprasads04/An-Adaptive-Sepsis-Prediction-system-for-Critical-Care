# FastAPI Application Guide

Complete implementation of the Sepsis Prediction System API based on the architectural flows.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Run the API
```bash
# Development mode (with auto-reload)
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Access the API
- **Swagger UI Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## API Endpoints Overview

### Health Check
```
GET  /           - Welcome & status
GET  /health     - Health check
```

### Core Predictions
```
POST /predict                - Single patient prediction
POST /predict-batch          - Batch predictions
POST /risk-stratification    - Risk stratification
```

### Explanability (SHAP)
```
POST /explain           - SHAP explanation
POST /explain-doctor    - Doctor-friendly explanation
POST /explain-patient   - Patient-friendly explanation
```

### SIRS Criteria
```
POST /sirs-score  - Calculate SIRS score
```

### Model Management
```
GET  /models                          - List available models
POST /models/switch/{model_name}      - Switch model
GET  /models/{model_name}/performance - Model performance metrics
```

### Threshold Management
```
GET  /thresholds              - Get current thresholds
POST /thresholds/update       - Update thresholds
```

### Validation
```
POST /validate-vitals  - Validate patient vitals
GET  /normal-ranges    - Get normal value ranges
```

### Features
```
GET /features  - List model features
```

### Analytics
```
GET /analytics/summary                - System statistics
GET /analytics/time-series/{patient_id} - Patient risk trends
```

### Patient Data
```
POST /patient/{patient_id}/predictions      - Log prediction
GET  /patient/{patient_id}/history          - Patient history
GET  /patient/{patient_id}/current-risk     - Current risk status
```

---

## Example Usage

### 1. Single Patient Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
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
    "hemoglobin": 13.5
  }'
```

### 2. SIRS Score Calculation
```bash
curl -X POST "http://localhost:8000/sirs-score" \
  -H "Content-Type: application/json" \
  -d '{
    "lactate": 1.5,
    "wbc_count": 8.5,
    "crp_level": 5.0,
    "creatinine": 0.9,
    "heart_rate": 95,
    "respiratory_rate": 22,
    "temperature": 38.5,
    "spo2": 96,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "hemoglobin": 14.0
  }'
```

### 3. Get Explanation
```bash
curl -X POST "http://localhost:8000/explain-doctor" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
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
    "hemoglobin": 13.5
  }'
```

### 4. Validate Vitals
```bash
curl -X POST "http://localhost:8000/validate-vitals" \
  -H "Content-Type: application/json" \
  -d '{
    "lactate": 1.5,
    "wbc_count": 8.5,
    "crp_level": 5.0,
    "creatinine": 0.9,
    "heart_rate": 95,
    "respiratory_rate": 22,
    "temperature": 38.5,
    "spo2": 96,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "hemoglobin": 14.0
  }'
```

---

## Response Examples

### Successful Prediction
```json
{
  "patient_id": "P001",
  "risk_score": 0.72,
  "probability": 0.72,
  "risk_level": "HIGH",
  "sirs_info": {
    "sirs_score": 3,
    "criteria_met": [
      "Temperature abnormal",
      "Heart rate > 90 bpm",
      "WBC abnormal"
    ],
    "sirs_positive": true,
    "interpretation": "SIRS positive - Sepsis risk elevated"
  },
  "model_used": "xgboost",
  "timestamp": "2026-03-04T10:30:45.123456",
  "confidence": 0.92,
  "recommendation": "IMMEDIATE ACTION REQUIRED. Alert physician immediately. Begin sepsis protocol."
}
```

### Validation Response
```json
{
  "is_valid": false,
  "abnormal_values": [
    {
      "parameter": "wbc_count",
      "value": 13.0,
      "normal_range": "4.0-11.0 x10^9/L",
      "severity": "warning"
    },
    {
      "parameter": "heart_rate",
      "value": 105,
      "normal_range": "60-100 bpm",
      "severity": "warning"
    }
  ],
  "warnings": []
}
```

---

## Environment Setup

### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements_api.txt

# Run API
python app.py
```

### Linux/Mac
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements_api.txt

# Run API
python app.py
```

---

## Docker Deployment

### Build Docker Image
```bash
docker build -t sepsis-prediction-api .
```

### Run Docker Container
```bash
docker run -p 8000:8000 sepsis-prediction-api
```

### Docker Compose
```bash
docker-compose up
```

---

## Configuration

### Model Paths
Update `MODEL_CONFIG` in app.py to match your model locations:
```python
MODEL_CONFIG = {
    "xgboost": {"path": "model/xgb_model.pkl", "threshold": 0.5},
    "random_forest": {"path": "model/rf_model.pkl", "threshold": 0.5},
}
```

### Normal Ranges
Update `NORMAL_RANGES` dictionary for your clinical requirements

### Risk Thresholds
Customize risk classification thresholds:
```python
RISK_THRESHOLDS = {
    "low": 0.3,
    "moderate": 0.5,
    "high": 0.7,
    "critical": 0.85
}
```

---

## Testing

### Run Tests
```bash
pytest test_app.py -v
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load tests
locust -f locustfile.py
```

---

## Logging

All API activity is logged to console. For file logging, update the logging configuration:
```python
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## Features Implemented

✅ **Core Predictions**
- Single patient predictions
- Batch predictions
- Risk stratification

✅ **Explainability**
- SHAP-based explanations
- Doctor-friendly explanations
- Patient-friendly explanations

✅ **SIRS Criteria**
- Automatic SIRS score calculation
- Criteria interpretation

✅ **Model Management**
- Multiple model support (XGBoost, Random Forest)
- Model switching
- Performance metrics

✅ **Input Validation**
- Normal range checking
- Vital sign validation
- Abnormal value detection

✅ **Patient Tracking**
- Prediction logging
- History tracking
- Current risk status

✅ **Analytics**
- System statistics
- Time series analysis
- Risk distribution

✅ **Production Ready**
- Error handling
- Logging
- CORS support
- Swagger documentation
- Input validation with Pydantic

---

## Database Integration (Optional)

For production deployment with database persistence:

```python
# Install database packages
pip install sqlalchemy psycopg2-binary

# Update app.py to include database models
```

---

## Security Considerations

For production deployment, add:

1. **Authentication**
```python
from fastapi.security import HTTPBearer
auth_scheme = HTTPBearer()
```

2. **Rate Limiting**
```python
from slowapi import Limiter
limiter = Limiter(key_func=...)
```

3. **HTTPS**
```bash
uvicorn app:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

---

## Troubleshooting

### Model Not Found
- Ensure model files exist in the `model/` directory
- Check `MODEL_CONFIG` paths

### Port Already in Use
```bash
# Change port
uvicorn app:app --port 8001
```

### Missing Dependencies
```bash
pip install -r requirements_api.txt --upgrade
```

---

## Support

For questions or issues:
1. Check the API documentation at `/docs`
2. Review logs for error messages
3. Validate input data with `/validate-vitals`
4. Check model availability with `/models`

---

**Version**: 1.0.0  
**Last Updated**: March 2026  
**Author**: Sepsis Prediction Team
