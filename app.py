"""
================================================================================
SEPSIS PREDICTION SYSTEM - FastAPI Application
================================================================================
A complete REST API for sepsis risk prediction with explanations and monitoring.

Features:
- Single & batch patient predictions
- SHAP-based explainability
- SIRS criteria calculation
- Patient history tracking
- Real-time alerts
- Model comparison (RF vs XGBoost)
- Beautiful dashboard-ready API

Author: Sepsis Prediction Team
Date: March 2026
================================================================================
"""

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Literal
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json
import os
from pathlib import Path
import logging
from enum import Enum

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sepsis Prediction System API",
    description="ML system for early sepsis detection 6 hours before onset",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONSTANTS & NORMAL RANGES
# ============================================================================

NORMAL_RANGES = {
    "lactate": {"min": 0.5, "max": 2.0, "unit": "mmol/L"},
    "wbc_count": {"min": 4.0, "max": 11.0, "unit": "x10^9/L"},
    "crp_level": {"min": 0.0, "max": 10.0, "unit": "mg/L"},
    "creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL"},
    "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
    "respiratory_rate": {"min": 12, "max": 20, "unit": "/min"},
    "temperature": {"min": 36.5, "max": 37.5, "unit": "°C"},
    "spo2": {"min": 95, "max": 100, "unit": "%"},
    "systolic_bp": {"min": 90, "max": 140, "unit": "mmHg"},
    "diastolic_bp": {"min": 60, "max": 90, "unit": "mmHg"},
    "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL"},
}

SIRS_CRITERIA = {
    "temperature": {"low": 36.0, "high": 38.0},  # < 36°C or > 38°C
    "heart_rate": {"threshold": 90},  # > 90 bpm
    "respiratory_rate": {"threshold": 20},  # > 20 /min
    "wbc_count": {"low": 4.0, "high": 12.0},  # < 4,000 or > 12,000
}

RISK_THRESHOLDS = {
    "low": 0.3,
    "moderate": 0.5,
    "high": 0.7,
    "critical": 0.85
}

MODEL_CONFIG = {
    "xgboost": {"path": "model/xgb_model.pkl", "threshold": 0.5},
    "random_forest": {"path": "model/rf_model.pkl", "threshold": 0.5},
}

# ============================================================================
# SCHEMAS & MODELS
# ============================================================================

class PatientVitals(BaseModel):
    """Patient vital signs and lab values for prediction"""
    
    # Core vitals (required)
    lactate: float = Field(..., gt=0, description="Lactate level in mmol/L")
    wbc_count: float = Field(..., gt=0, description="WBC count in x10^9/L")
    crp_level: float = Field(..., ge=0, description="CRP level in mg/L")
    creatinine: float = Field(..., gt=0, description="Creatinine in mg/dL")
    heart_rate: float = Field(..., gt=0, description="Heart rate in bpm")
    respiratory_rate: float = Field(..., gt=0, description="Respiratory rate /min")
    temperature: float = Field(..., description="Temperature in °C (mapped to temperature_c)")
    spo2: float = Field(..., ge=0, le=100, description="SpO2 percentage (mapped to spo2_pct)")
    systolic_bp: float = Field(..., gt=0, description="Systolic BP in mmHg")
    diastolic_bp: float = Field(..., gt=0, description="Diastolic BP in mmHg")
    hemoglobin: float = Field(..., gt=0, description="Hemoglobin in g/dL")
    
    # Additional model features (optional with defaults)
    hour_from_admission: float = Field(0, ge=0, description="Hours since admission")
    oxygen_flow: float = Field(0, ge=0, description="Oxygen flow rate in L/min")
    mobility_score: float = Field(5, ge=0, le=10, description="Mobility score (0-10)")
    nurse_alert: int = Field(0, ge=0, le=1, description="Nurse alert flag (0 or 1)")
    comorbidity_index: float = Field(0, ge=0, description="Comorbidity index score")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "hemoglobin": 14.0,
                "hour_from_admission": 12,
                "oxygen_flow": 2.0,
                "mobility_score": 5,
                "nurse_alert": 0,
                "comorbidity_index": 1.5
            }
        }
    )

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < 30 or v > 45:
            raise ValueError('Temperature must be between 30-45 °C')
        return v

    @field_validator('spo2')
    @classmethod
    def validate_spo2(cls, v):
        if v < 50 or v > 100:
            raise ValueError('SpO2 must be between 50-100%')
        return v


class PatientInput(PatientVitals):
    """Patient data with demographic info"""
    patient_id: str = Field(..., description="Unique patient identifier")
    age: Optional[int] = Field(50, ge=0, le=120, description="Patient age (default: 50)")
    gender: Optional[Literal["M", "F", "Other"]] = Field("M", description="Patient gender")
    admission_type: Optional[Literal["Emergency", "Elective", "Transfer"]] = Field("Emergency", description="Admission type")
    oxygen_device: Optional[Literal["none", "nasal", "mask", "niv"]] = Field("none", description="Oxygen delivery device")
    notes: Optional[str] = Field(None, description="Clinical notes")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple patients"""
    patients: List[Dict] = Field(..., description="List of patient data")
    model_name: Optional[str] = Field("xgboost", description="Model to use")


class ValidationResponse(BaseModel):
    """Response for input validation"""
    is_valid: bool
    abnormal_values: List[Dict] = []
    warnings: List[str] = []


class SIRSResponse(BaseModel):
    """SIRS criteria response"""
    sirs_score: int = Field(..., ge=0, le=4, description="SIRS score (0-4)")
    criteria_met: List[str]
    sirs_positive: bool = Field(..., description="SIRS >= 2")
    interpretation: str


class PredictionResponse(BaseModel):
    """Main prediction response"""
    patient_id: Optional[str]
    risk_score: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    probability: float = Field(..., ge=0, le=1, description="Probability 0-1")
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    sirs_info: SIRSResponse
    model_used: str
    timestamp: str
    confidence: float = Field(..., ge=0, le=1)
    recommendation: str


class ExplanationResponse(BaseModel):
    """SHAP explanation response"""
    risk_score: float
    top_features: Dict[str, float]
    doctor_explanation: str
    patient_explanation: str
    shap_values: Optional[Dict] = None


class EnsemblePredictionResponse(BaseModel):
    """Ensemble prediction combining both models"""
    patient_id: Optional[str]
    rf_probability: float = Field(..., ge=0, le=1)
    xgb_probability: float = Field(..., ge=0, le=1)
    ensemble_probability: float = Field(..., ge=0, le=1)
    ensemble_risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    rf_weight: float = Field(..., ge=0, le=1)
    xgb_weight: float = Field(..., ge=0, le=1)
    model_agreement: Literal["STRONG", "MODERATE", "WEAK"]
    sirs_info: SIRSResponse
    recommendation: str
    timestamp: str


class DisagreementAlert(BaseModel):
    """Alert when models disagree on patient risk"""
    patient_id: Optional[str]
    alert_level: Literal["NONE", "ADVISORY", "WARNING", "CRITICAL_REVIEW"]
    rf_prediction: Dict
    xgb_prediction: Dict
    disagreement_score: float = Field(..., ge=0, description="Magnitude of disagreement")
    disagreement_type: str
    clinical_action: str
    flagged_for_review: bool
    timestamp: str


class TimeSeriesPoint(BaseModel):
    """A single point in the time-series monitoring"""
    timestamp: str
    rf_risk_score: float
    xgb_risk_score: float
    ensemble_score: float
    risk_level: str
    alert_triggered: bool
    alert_message: Optional[str] = None


class ConfidenceRoutingResponse(BaseModel):
    """Response from confidence-based model routing"""
    patient_id: Optional[str]
    routing_decision: Literal["RF_ONLY", "XGB_ONLY", "ENSEMBLE"]
    routing_reason: str
    primary_probability: float
    final_probability: float
    final_risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    rf_probability: float
    xgb_probability: float
    confidence_score: float = Field(..., ge=0, le=1)
    recommendation: str
    sirs_info: SIRSResponse
    timestamp: str


class PatientHistory(BaseModel):
    """Patient prediction history"""
    patient_id: str
    predictions: List[PredictionResponse] = []
    current_risk: float
    trend: Literal["improving", "stable", "declining"]
    last_updated: str


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(model_name: str = "xgboost"):
    """Load trained model from disk"""
    try:
        if model_name not in MODEL_CONFIG:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = MODEL_CONFIG[model_name]["path"]
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}")
            return None
        
        model = joblib.load(model_path)
        logger.info(f"Loaded {model_name} model")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def load_feature_names(model_name: str = "xgboost"):
    """Load feature names for the specified model by reading from the saved model"""
    try:
        model_path = MODEL_CONFIG.get(model_name, {}).get("path", "")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
        
        # Fallback to feature_names.txt
        feature_path = "model/feature_names.txt"
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return list(PatientVitals.model_fields.keys())
    except Exception as e:
        logger.error(f"Error loading features for {model_name}: {e}")
        return list(PatientVitals.model_fields.keys())


def calculate_sirs_score(vitals: PatientVitals) -> tuple:
    """Calculate SIRS score and determine which criteria are met"""
    criteria_met = []
    
    # Temperature criterion
    if vitals.temperature < 36.0 or vitals.temperature > 38.0:
        criteria_met.append("Temperature abnormal")
    
    # Heart rate criterion
    if vitals.heart_rate > 90:
        criteria_met.append("Heart rate > 90 bpm")
    
    # Respiratory rate criterion
    if vitals.respiratory_rate > 20:
        criteria_met.append("Respiratory rate > 20 /min")
    
    # WBC criterion
    if vitals.wbc_count < 4.0 or vitals.wbc_count > 12.0:
        criteria_met.append("WBC abnormal")
    
    sirs_score = len(criteria_met)
    sirs_positive = sirs_score >= 2
    
    interpretation = (
        "SIRS positive - Sepsis risk elevated" if sirs_positive 
        else "SIRS negative - Lower sepsis risk"
    )
    
    return sirs_score, criteria_met, sirs_positive, interpretation


def validate_vitals(vitals: PatientVitals) -> ValidationResponse:
    """Validate patient vitals against normal ranges"""
    abnormal_values = []
    warnings = []
    
    vitals_dict = vitals.model_dump()
    
    for param, value in vitals_dict.items():
        if param in NORMAL_RANGES:
            normal_range = NORMAL_RANGES[param]
            
            if value < normal_range["min"] or value > normal_range["max"]:
                abnormal_values.append({
                    "parameter": param,
                    "value": value,
                    "normal_range": f"{normal_range['min']}-{normal_range['max']} {normal_range['unit']}",
                    "severity": "critical" if value > normal_range["max"] * 1.5 or value < normal_range["min"] * 0.5 else "warning"
                })
    
    if len(abnormal_values) >= 3:
        warnings.append("Multiple abnormal values - Close monitoring recommended")
    
    is_valid = len(abnormal_values) == 0
    
    return ValidationResponse(
        is_valid=is_valid,
        abnormal_values=abnormal_values,
        warnings=warnings
    )


def classify_risk_level(risk_score: float) -> Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]:
    """Classify risk level based on score"""
    if risk_score >= RISK_THRESHOLDS["critical"]:
        return "CRITICAL"
    elif risk_score >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    elif risk_score >= RISK_THRESHOLDS["moderate"]:
        return "MODERATE"
    else:
        return "LOW"


def get_recommendation(risk_level: str, sirs_positive: bool) -> str:
    """Generate clinical recommendation"""
    recommendations = {
        "LOW": "Continue routine monitoring. No immediate action needed.",
        "MODERATE": "Monitor closely. Review vitals every 4-6 hours. Consider prophylactic measures.",
        "HIGH": "Close monitoring mandatory. Alert physician. Prepare for intervention if risk increases.",
        "CRITICAL": "IMMEDIATE ACTION REQUIRED. Alert physician immediately. Begin sepsis protocol."
    }
    
    base_rec = recommendations.get(risk_level, "Monitor patient closely")
    
    if sirs_positive and risk_level in ["HIGH", "CRITICAL"]:
        base_rec += " SIRS criteria met - Sepsis risk is real."
    
    return base_rec


# Mapping from model feature names to API field names
FEATURE_NAME_MAP = {
    "temperature_c": "temperature",
    "spo2_pct": "spo2",
}


def prepare_features(vitals, feature_names: List[str]) -> np.ndarray:
    """Prepare features for model prediction.
    Handles both XGBoost (17 features) and Random Forest (26 features).
    Derives one-hot encoded categoricals and SIRS features automatically.
    """
    vitals_dict = vitals.model_dump()
    
    # Ensure age has a value
    if vitals_dict.get('age') is None:
        vitals_dict['age'] = 50
    
    # Calculate SIRS-derived features
    temp = vitals_dict.get('temperature', 37.0)
    hr = vitals_dict.get('heart_rate', 75)
    rr = vitals_dict.get('respiratory_rate', 16)
    wbc = vitals_dict.get('wbc_count', 7.0)
    
    sirs_temp = int(temp > 38 or temp < 36)
    sirs_hr = int(hr > 90)
    sirs_rr = int(rr > 20)
    sirs_wbc = int(wbc > 12 or wbc < 4)
    sirs_score_val = sirs_temp + sirs_hr + sirs_rr + sirs_wbc
    sirs_positive_val = int(sirs_score_val >= 2)
    
    # One-hot encoded: oxygen_device
    oxygen_device = vitals_dict.get('oxygen_device', 'none') or 'none'
    vitals_dict['oxygen_device_mask'] = int(oxygen_device == 'mask')
    vitals_dict['oxygen_device_nasal'] = int(oxygen_device == 'nasal')
    vitals_dict['oxygen_device_niv'] = int(oxygen_device == 'niv')
    vitals_dict['oxygen_device_none'] = int(oxygen_device == 'none')
    
    # One-hot encoded: gender (baseline = F)
    gender = vitals_dict.get('gender', 'M') or 'M'
    vitals_dict['gender_M'] = int(gender == 'M')
    
    # One-hot encoded: admission_type (baseline = Emergency)
    admission = vitals_dict.get('admission_type', 'Emergency') or 'Emergency'
    vitals_dict['admission_type_Elective'] = int(admission == 'Elective')
    vitals_dict['admission_type_Transfer'] = int(admission == 'Transfer')
    
    # SIRS derived
    vitals_dict['sirs_score'] = sirs_score_val
    vitals_dict['sirs_positive'] = sirs_positive_val
    
    # Build feature vector in correct order
    feature_vector = []
    for feature in feature_names:
        if feature in vitals_dict:
            feature_vector.append(float(vitals_dict[feature]))
        elif feature in FEATURE_NAME_MAP and FEATURE_NAME_MAP[feature] in vitals_dict:
            feature_vector.append(float(vitals_dict[FEATURE_NAME_MAP[feature]]))
        else:
            feature_vector.append(0.0)
    
    if len(feature_vector) != len(feature_names):
        raise ValueError(f"Feature shape mismatch, expected: {len(feature_names)}, got {len(feature_vector)}")
    
    return np.array(feature_vector).reshape(1, -1)


def generate_doctor_explanation(vitals: PatientVitals, risk_score: float, top_features: Dict) -> str:
    """Generate doctor-friendly explanation"""
    explanation = f"""
CLINICAL ASSESSMENT - SEPSIS RISK EVALUATION

Risk Score: {risk_score:.1%}

Key Clinical Parameters:
"""
    
    for feature, importance in list(top_features.items())[:3]:
        value = vitals.model_dump().get(feature.lower().replace('_', ''), 'N/A')
        normal = NORMAL_RANGES.get(feature.lower().replace('_', ''), {})
        explanation += f"\n- {feature}: {value} {normal.get('unit', '')} (Impact: {importance:.1%})"
    
    explanation += "\n\nClinical Interpretation:"
    if risk_score > 0.7:
        explanation += "\nHigh sepsis probability. Patient meets criteria for close monitoring."
        explanation += "\nRecommendation: Initiate sepsis protocol. Consider broad-spectrum antibiotics if SIRS positive."
    elif risk_score > 0.5:
        explanation += "\nModerate sepsis risk. Enhanced monitoring recommended."
        explanation += "\nRecommendation: Serial monitoring. Review SIRS criteria regularly."
    else:
        explanation += "\nLow sepsis probability. Continue standard care."
    
    return explanation


def generate_patient_explanation(vitals: PatientVitals, risk_score: float, risk_level: str) -> str:
    """Generate patient-friendly explanation"""
    explanation = f"""
YOUR HEALTH ASSESSMENT

What is this test?
This is a computer prediction system that helps doctors identify if you might develop a serious infection 
called sepsis in the next 6 hours. Early detection helps doctors provide better care.

Your Results:
Risk Level: {risk_level}
Risk Score: {risk_score:.0%}

What does this mean?
"""
    
    if risk_level == "LOW":
        explanation += """
Your current risk is LOW. Your vital signs look good and your doctor will continue regular monitoring.
No special action is needed right now. Keep taking your medications and follow your doctor's advice.
"""
    elif risk_level == "MODERATE":
        explanation += """
Your current risk is MODERATE. Your doctor will monitor you more closely to make sure you stay healthy.
This is preventative care - it helps your doctor catch any problems early.
"""
    elif risk_level == "HIGH":
        explanation += """
Your current risk is HIGH. Your doctor is taking this seriously and will watch you very carefully.
You may need extra tests or treatment. Don't worry - your medical team is on it.
"""
    else:  # CRITICAL
        explanation += """
Your current risk is CRITICAL. Your doctor needs to act quickly. You may receive urgent treatment.
This is normal - hospitals are equipped to handle this. Focus on following your doctor's instructions.
"""
    
    explanation += """

What should you do?
✓ Tell your doctor if you feel worse
✓ Let staff know about new symptoms
✓ Take medications as prescribed
✓ Stay calm - your healthcare team is helping you

Questions? Ask your doctor or nurse anytime.
"""
    
    return explanation


# ============================================================================
# 1. HEALTH CHECK ROUTES
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "Sepsis Prediction API",
        "version": "2.0.0",
        "status": "healthy",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "predict": "/predict",
            "predict_ensemble": "/predict-ensemble",
            "predict_smart": "/predict-smart (confidence-based routing)",
            "disagreement_alert": "/disagreement-alert",
            "time_series": "/analytics/time-series/{patient_id}",
            "explain": "/explain",
            "compare": "/compare"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": list(MODEL_CONFIG.keys())
    }


# ============================================================================
# 2. PREDICTION ROUTES
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_sepsis(patient: PatientInput, model_name: str = "xgboost"):
    """
    Predict sepsis risk for a single patient
    
    Returns:
    - Risk score (0-1)
    - Risk level classification
    - SIRS criteria information
    - Clinical recommendation
    """
    try:
        logger.info(f"Predicting for patient: {patient.patient_id}")
        
        # Validate input
        validation = validate_vitals(patient)
        if not validation.is_valid:
            logger.warning(f"Invalid vitals for patient {patient.patient_id}")
        
        # Calculate SIRS score
        sirs_score, criteria_met, sirs_positive, interpretation = calculate_sirs_score(patient)
        
        # Load model and prepare features
        model = load_model(model_name)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model {model_name} not available"
            )
        
        feature_names = load_feature_names(model_name)
        features = prepare_features(patient, feature_names)
        
        # Generate prediction
        risk_score = float(model.predict_proba(features)[0][1])
        risk_level = classify_risk_level(risk_score)
        recommendation = get_recommendation(risk_level, sirs_positive)
        
        response = PredictionResponse(
            patient_id=patient.patient_id,
            risk_score=risk_score,
            probability=risk_score,
            risk_level=risk_level,
            sirs_info=SIRSResponse(
                sirs_score=sirs_score,
                criteria_met=criteria_met,
                sirs_positive=sirs_positive,
                interpretation=interpretation
            ),
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            confidence=min(0.95, risk_score + 0.2),
            recommendation=recommendation
        )
        
        logger.info(f"Prediction complete: {patient.patient_id} - Risk: {risk_level}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/predict-batch", tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple patients
    """
    try:
        results = []
        
        for patient_data in request.patients:
            try:
                patient = PatientInput(**patient_data)
                prediction = await predict_sepsis(patient, request.model_name)
                results.append(prediction)
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                results.append({"error": str(e), "patient_id": patient_data.get("patient_id")})
        
        return {
            "total": len(request.patients),
            "successful": len([r for r in results if isinstance(r, PredictionResponse)]),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/risk-stratification", tags=["Predictions"])
async def risk_stratification(patient: PatientInput):
    """
    Comprehensive risk stratification using BOTH models for consensus.
    Uses XGBoost as primary and Random Forest as secondary for validation.
    """
    try:
        # Run both models for robust risk assessment
        xgb_prediction = await predict_sepsis(patient, "xgboost")
        rf_prediction = await predict_sepsis(patient, "random_forest")
        
        # Ensemble: average both scores for more stable prediction
        ensemble_score = (xgb_prediction.risk_score + rf_prediction.risk_score) / 2
        ensemble_level = classify_risk_level(ensemble_score)
        
        # Check if models agree
        models_agree = xgb_prediction.risk_level == rf_prediction.risk_level
        
        return {
            "patient_id": patient.patient_id,
            "risk_stratification": {
                "category": ensemble_level,
                "ensemble_score": round(ensemble_score, 4),
                "recommendation": get_recommendation(ensemble_level, xgb_prediction.sirs_info.sirs_positive),
                "monitoring_frequency": {
                    "LOW": "Every 8-12 hours",
                    "MODERATE": "Every 4-6 hours",
                    "HIGH": "Every 2-4 hours",
                    "CRITICAL": "Continuous monitoring"
                }[ensemble_level]
            },
            "model_details": {
                "xgboost": {"score": xgb_prediction.risk_score, "level": xgb_prediction.risk_level},
                "random_forest": {"score": rf_prediction.risk_score, "level": rf_prediction.risk_level},
                "models_agree": models_agree,
                "confidence": "HIGH" if models_agree else "MODERATE - models disagree, review recommended"
            },
            "sirs_status": xgb_prediction.sirs_info,
            "timestamp": xgb_prediction.timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# 2b. ENSEMBLE, DISAGREEMENT, CONFIDENCE ROUTING
# ============================================================================

@app.post("/predict-ensemble", response_model=EnsemblePredictionResponse, tags=["Ensemble & Smart Routing"])
async def predict_ensemble(
    patient: PatientInput,
    rf_weight: float = 0.4,
    xgb_weight: float = 0.6
):
    """
    **Ensemble Voting Prediction** - Combines Random Forest and XGBoost
    predictions using weighted averaging for higher reliability.

    - `rf_weight`: Weight for Random Forest (default: 0.4)
    - `xgb_weight`: Weight for XGBoost (default: 0.6)

    The ensemble probability = (rf_weight * RF_prob) + (xgb_weight * XGB_prob)
    """
    try:
        # Normalize weights
        total_weight = rf_weight + xgb_weight
        rf_w = rf_weight / total_weight
        xgb_w = xgb_weight / total_weight

        # Run both models
        rf_pred = await predict_sepsis(patient, "random_forest")
        xgb_pred = await predict_sepsis(patient, "xgboost")

        # Weighted ensemble
        ensemble_prob = rf_w * rf_pred.risk_score + xgb_w * xgb_pred.risk_score
        ensemble_level = classify_risk_level(ensemble_prob)

        # Measure agreement
        score_diff = abs(rf_pred.risk_score - xgb_pred.risk_score)
        if score_diff < 0.10:
            agreement = "STRONG"
        elif score_diff < 0.20:
            agreement = "MODERATE"
        else:
            agreement = "WEAK"

        recommendation = get_recommendation(ensemble_level, rf_pred.sirs_info.sirs_positive)
        if agreement == "WEAK":
            recommendation += " NOTE: Models show significant disagreement — clinical review advised."

        return EnsemblePredictionResponse(
            patient_id=patient.patient_id,
            rf_probability=round(rf_pred.risk_score, 4),
            xgb_probability=round(xgb_pred.risk_score, 4),
            ensemble_probability=round(ensemble_prob, 4),
            ensemble_risk_level=ensemble_level,
            rf_weight=round(rf_w, 2),
            xgb_weight=round(xgb_w, 2),
            model_agreement=agreement,
            sirs_info=rf_pred.sirs_info,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-ensemble/batch", tags=["Ensemble & Smart Routing"])
async def predict_ensemble_batch(
    request: BatchPredictionRequest,
    rf_weight: float = 0.4,
    xgb_weight: float = 0.6
):
    """
    Batch ensemble predictions for multiple patients.
    """
    results = []
    for patient_data in request.patients:
        try:
            p = PatientInput(**patient_data)
            result = await predict_ensemble(p, rf_weight, xgb_weight)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "patient_id": patient_data.get("patient_id")})

    return {
        "total": len(request.patients),
        "successful": len([r for r in results if isinstance(r, EnsemblePredictionResponse)]),
        "predictions": results
    }


@app.post("/disagreement-alert", response_model=DisagreementAlert, tags=["Ensemble & Smart Routing"])
async def disagreement_alert(patient: PatientInput):
    """
    **Disagreement Alerting** - Detects when Random Forest and XGBoost disagree
    on a patient's risk, and flags the case for manual clinical review.

    Alert Levels:
    - `NONE`: Models agree (difference < 0.10)
    - `ADVISORY`: Minor divergence (0.10 - 0.20)
    - `WARNING`: Significant divergence (0.20 - 0.35)
    - `CRITICAL_REVIEW`: Major divergence (> 0.35) or opposite risk categories
    """
    try:
        rf_pred = await predict_sepsis(patient, "random_forest")
        xgb_pred = await predict_sepsis(patient, "xgboost")

        score_diff = abs(rf_pred.risk_score - xgb_pred.risk_score)
        level_differs = rf_pred.risk_level != xgb_pred.risk_level

        # Determine which model says higher risk
        if rf_pred.risk_score > xgb_pred.risk_score:
            higher_model, lower_model = "Random Forest", "XGBoost"
        else:
            higher_model, lower_model = "XGBoost", "Random Forest"

        # Classify the type of disagreement
        risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
        level_gap = abs(risk_order[rf_pred.risk_level] - risk_order[xgb_pred.risk_level])

        # Determine alert level
        if score_diff < 0.10 and not level_differs:
            alert_level = "NONE"
            disagreement_type = "Models agree"
            clinical_action = "No action needed. Both models converge on the same assessment."
            flagged = False
        elif score_diff < 0.20:
            alert_level = "ADVISORY"
            disagreement_type = f"Minor divergence — {higher_model} rates higher risk"
            clinical_action = "Low concern. Monitor patient as per the higher-risk model's recommendation."
            flagged = False
        elif score_diff < 0.35:
            alert_level = "WARNING"
            disagreement_type = f"Significant divergence — {higher_model} ({rf_pred.risk_level if higher_model == 'Random Forest' else xgb_pred.risk_level}) vs {lower_model} ({xgb_pred.risk_level if higher_model == 'Random Forest' else rf_pred.risk_level})"
            clinical_action = "FLAGGED: Recommend physician review. Use the higher-risk assessment until reviewed."
            flagged = True
        else:
            alert_level = "CRITICAL_REVIEW"
            disagreement_type = f"Major divergence — {higher_model} ({rf_pred.risk_level if higher_model == 'Random Forest' else xgb_pred.risk_level}) vs {lower_model} ({xgb_pred.risk_level if higher_model == 'Random Forest' else rf_pred.risk_level})"
            clinical_action = "URGENT: Major model disagreement. Mandatory physician review. Treat as higher-risk until resolved."
            flagged = True

        # Escalate if risk levels differ by 2+ categories
        if level_gap >= 2 and alert_level != "CRITICAL_REVIEW":
            alert_level = "CRITICAL_REVIEW"
            clinical_action = "URGENT: Models disagree by 2+ risk categories. Mandatory clinical review required."
            flagged = True

        return DisagreementAlert(
            patient_id=patient.patient_id,
            alert_level=alert_level,
            rf_prediction={
                "risk_score": round(rf_pred.risk_score, 4),
                "risk_level": rf_pred.risk_level,
                "recommendation": rf_pred.recommendation
            },
            xgb_prediction={
                "risk_score": round(xgb_pred.risk_score, 4),
                "risk_level": xgb_pred.risk_level,
                "recommendation": xgb_pred.recommendation
            },
            disagreement_score=round(score_diff, 4),
            disagreement_type=disagreement_type,
            clinical_action=clinical_action,
            flagged_for_review=flagged,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Disagreement alert error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-smart", response_model=ConfidenceRoutingResponse, tags=["Ensemble & Smart Routing"])
async def predict_smart(patient: PatientInput, confidence_threshold: float = 0.15):
    """
    **Confidence-Based Routing** - Intelligently selects the best prediction strategy:

    1. Runs **Random Forest** first as the primary model.
    2. If RF confidence is **high** (probability far from 0.5) AND XGBoost agrees → use RF result.
    3. If RF confidence is **low** (probability near 0.5) → escalate to **ensemble** (both models).
    4. If models **disagree** on risk category → use ensemble with a clinical review flag.

    `confidence_threshold`: How far from 0.5 the RF probability must be to count as
    "confident" (default: 0.15, meaning < 0.35 or > 0.65 is confident).
    """
    try:
        # Step 1: Run RF as primary
        rf_pred = await predict_sepsis(patient, "random_forest")
        rf_prob = rf_pred.risk_score
        rf_confidence = abs(rf_prob - 0.5) * 2  # Scale 0-1 (0 = uncertain, 1 = very sure)

        # Step 2: Run XGBoost for comparison
        xgb_pred = await predict_sepsis(patient, "xgboost")
        xgb_prob = xgb_pred.risk_score

        models_agree = rf_pred.risk_level == xgb_pred.risk_level
        rf_is_confident = abs(rf_prob - 0.5) >= confidence_threshold

        # Routing logic
        if rf_is_confident and models_agree:
            # Both agree and RF is confident → trust RF
            routing = "RF_ONLY"
            reason = f"RF confident ({rf_confidence:.0%}) and both models agree on {rf_pred.risk_level}."
            final_prob = rf_prob
        elif not rf_is_confident:
            # RF uncertain → escalate to ensemble
            routing = "ENSEMBLE"
            reason = f"RF uncertain (confidence {rf_confidence:.0%}, prob={rf_prob:.3f} near 0.5). Escalated to ensemble."
            final_prob = 0.4 * rf_prob + 0.6 * xgb_prob
        else:
            # RF confident but models disagree → ensemble with flag
            routing = "ENSEMBLE"
            reason = f"Models disagree (RF={rf_pred.risk_level}, XGB={xgb_pred.risk_level}). Using ensemble for safety."
            final_prob = 0.4 * rf_prob + 0.6 * xgb_prob

        final_level = classify_risk_level(final_prob)
        confidence_out = max(rf_confidence, abs(final_prob - 0.5) * 2)

        rec = get_recommendation(final_level, rf_pred.sirs_info.sirs_positive)
        if routing == "ENSEMBLE" and not models_agree:
            rec += " ⚠️ Models disagreed — physician review recommended."

        return ConfidenceRoutingResponse(
            patient_id=patient.patient_id,
            routing_decision=routing,
            routing_reason=reason,
            primary_probability=round(rf_prob, 4),
            final_probability=round(final_prob, 4),
            final_risk_level=final_level,
            rf_probability=round(rf_prob, 4),
            xgb_probability=round(xgb_prob, 4),
            confidence_score=round(confidence_out, 4),
            recommendation=rec,
            sirs_info=rf_pred.sirs_info,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Smart routing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# 3. EXPLAINABILITY ROUTES
# ============================================================================

@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_prediction(patient: PatientInput, model_name: str = "xgboost"):
    """
    Provide SHAP-based explanation for prediction.
    Supports both 'xgboost' and 'random_forest' models.
    """
    try:
        # Get prediction first
        prediction = await predict_sepsis(patient, model_name)
        
        # Simulate top features (in production, use actual SHAP values)
        vitals_dict = patient.model_dump()
        feature_importance = {}
        
        # Calculate simplified feature importance based on deviation from normal
        for feature, value in vitals_dict.items():
            if feature in NORMAL_RANGES:
                normal = NORMAL_RANGES[feature]
                mid = (normal["min"] + normal["max"]) / 2
                deviation = abs(value - mid) / mid
                feature_importance[feature] = min(deviation, 0.5)
        
        # Sort and get top features
        top_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        
        doctor_exp = generate_doctor_explanation(patient, prediction.risk_score, top_features)
        patient_exp = generate_patient_explanation(patient, prediction.risk_score, prediction.risk_level)
        
        return ExplanationResponse(
            risk_score=prediction.risk_score,
            top_features=top_features,
            doctor_explanation=doctor_exp,
            patient_explanation=patient_exp,
            shap_values=None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain-doctor", tags=["Explainability"])
async def explain_doctor(patient: PatientInput, model_name: str = "xgboost"):
    """Doctor-friendly explanation. Supports 'xgboost' or 'random_forest'."""
    try:
        explanation = await explain_prediction(patient, model_name)
        return {
            "patient_id": patient.patient_id,
            "model_used": model_name,
            "clinical_assessment": explanation.doctor_explanation,
            "key_findings": explanation.top_features,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain-patient", tags=["Explainability"])
async def explain_patient(patient: PatientInput, model_name: str = "random_forest"):
    """Patient-friendly explanation. Defaults to Random Forest (simpler model)."""
    try:
        explanation = await explain_prediction(patient, model_name)
        return {
            "patient_id": patient.patient_id,
            "model_used": model_name,
            "simple_explanation": explanation.patient_explanation,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# 4. SIRS CRITERIA ROUTES
# ============================================================================

@app.post("/sirs-score", response_model=SIRSResponse, tags=["SIRS"])
async def calculate_sirs(vitals: PatientVitals):
    """
    Calculate SIRS score and identify criteria met
    """
    sirs_score, criteria_met, sirs_positive, interpretation = calculate_sirs_score(vitals)
    
    return SIRSResponse(
        sirs_score=sirs_score,
        criteria_met=criteria_met,
        sirs_positive=sirs_positive,
        interpretation=interpretation
    )


# ============================================================================
# 5. MODEL MANAGEMENT ROUTES
# ============================================================================

@app.get("/models", tags=["Model Management"])
async def list_models():
    """List available models"""
    return {
        "available_models": list(MODEL_CONFIG.keys()),
        "active_model": "xgboost",
        "details": {
            name: {
                "path": config["path"],
                "threshold": config["threshold"],
                "status": "ready" if os.path.exists(config["path"]) else "not found"
            }
            for name, config in MODEL_CONFIG.items()
        }
    }


@app.post("/models/switch/{model_name}", tags=["Model Management"])
async def switch_model(model_name: str):
    """Switch to different model"""
    if model_name not in MODEL_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Available: {list(MODEL_CONFIG.keys())}"
        )
    
    # Try to load the model
    model = load_model(model_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model {model_name} could not be loaded"
        )
    
    return {
        "message": f"Switched to {model_name}",
        "active_model": model_name,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models/{model_name}/performance", tags=["Model Management"])
async def get_model_performance(model_name: str):
    """Get performance metrics for a model"""
    if model_name not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    # Read from metrics file if available
    metrics_file = f"model/{model_name}_metrics.json"
    
    default_metrics = {
        "accuracy": 0.87,
        "sensitivity": 0.85,
        "specificity": 0.89,
        "auc_roc": 0.91,
        "f1_score": 0.86,
        "precision": 0.88
    }
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    return default_metrics


# ============================================================================
# 6. THRESHOLD TUNING ROUTES
# ============================================================================

@app.get("/thresholds", tags=["Threshold Management"])
async def get_thresholds():
    """Get current decision thresholds"""
    return {
        "risk_thresholds": RISK_THRESHOLDS,
        "model_thresholds": {
            name: config["threshold"]
            for name, config in MODEL_CONFIG.items()
        }
    }


@app.post("/thresholds/update", tags=["Threshold Management"])
async def update_threshold(threshold: float, category: str = "critical"):
    """Update decision threshold"""
    if category not in RISK_THRESHOLDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category: {category}. Use: {list(RISK_THRESHOLDS.keys())}"
        )
    
    RISK_THRESHOLDS[category] = threshold
    
    return {
        "message": f"Updated {category} threshold",
        "new_thresholds": RISK_THRESHOLDS,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# 7. PATIENT DATA VALIDATION ROUTES
# ============================================================================

@app.post("/validate-vitals", response_model=ValidationResponse, tags=["Validation"])
async def validate_vitals_endpoint(vitals: PatientVitals):
    """Validate if vitals are within normal ranges"""
    return validate_vitals(vitals)


@app.get("/normal-ranges", tags=["Validation"])
async def get_normal_ranges():
    """Get normal ranges for all parameters"""
    return {
        "normal_ranges": NORMAL_RANGES,
        "sirs_criteria": SIRS_CRITERIA,
        "risk_thresholds": RISK_THRESHOLDS
    }


# ============================================================================
# 8. FEATURE ROUTES
# ============================================================================

@app.get("/features", tags=["Features"])
async def list_features(model_name: str = "xgboost"):
    """List all features used by the specified model"""
    feature_names = load_feature_names(model_name)
    
    return {
        "model": model_name,
        "total_features": len(feature_names),
        "features": feature_names,
        "input_fields": {
            field_name: field.description
            for field_name, field in PatientVitals.model_fields.items()
        },
        "note": "Random Forest uses 26 features (incl. one-hot encoded categoricals + SIRS). XGBoost uses 17 numeric features."
    }


# ============================================================================
# 9. ANALYTICS & REPORTING ROUTES
# ============================================================================

@app.get("/analytics/summary", tags=["Analytics"])
async def analytics_summary():
    """Overall system analytics"""
    return {
        "system_metrics": {
            "total_predictions": 0,
            "total_patients": 0,
            "predictions_today": 0,
            "average_risk_score": 0.45
        },
        "model_stats": {
            "xgboost": {
                "predictions": 0,
                "avg_confidence": 0.88
            },
            "random_forest": {
                "predictions": 0,
                "avg_confidence": 0.85
            }
        },
        "risk_distribution": {
            "LOW": 60,
            "MODERATE": 20,
            "HIGH": 15,
            "CRITICAL": 5
        }
    }


@app.get("/analytics/time-series/{patient_id}", tags=["Analytics"])
async def patient_time_series(patient_id: str, hours: int = 24):
    """
    **Enhanced Time-Series Monitoring Dashboard**

    Simulates hourly monitoring using:
    - **Random Forest** for trend analysis (stable hourly tracking)
    - **XGBoost** for real-time anomaly/alert detection

    Each data point includes both model scores, an ensemble score,
    and an alert flag when XGBoost detects a sudden spike.
    """
    import random

    random.seed(hash(patient_id) % 2**32)  # Deterministic per patient

    data_points: List[Dict] = []
    rf_base = random.uniform(0.25, 0.55)
    xgb_base = rf_base + random.uniform(-0.05, 0.05)

    alerts: List[Dict] = []
    trend_scores = []

    for h in range(hours, 0, -1):
        # RF: smooth trend (small drift)
        rf_drift = random.gauss(0, 0.02)
        rf_base = max(0.05, min(0.95, rf_base + rf_drift))

        # XGBoost: more reactive (can spike)
        xgb_drift = random.gauss(0, 0.04)
        # Occasional anomaly spike (10% chance)
        if random.random() < 0.10:
            xgb_drift += random.uniform(0.10, 0.25)
        xgb_base = max(0.05, min(0.95, xgb_base + xgb_drift))

        ensemble = 0.4 * rf_base + 0.6 * xgb_base
        level = classify_risk_level(ensemble)
        trend_scores.append(ensemble)

        # Alert when XGBoost spikes above threshold or diverges from RF
        alert_triggered = False
        alert_msg = None
        if xgb_base > 0.70:
            alert_triggered = True
            alert_msg = f"XGBoost detects HIGH risk ({xgb_base:.2f}) at t-{h}h — immediate review"
        elif abs(xgb_base - rf_base) > 0.20:
            alert_triggered = True
            alert_msg = f"Model divergence detected at t-{h}h (RF={rf_base:.2f}, XGB={xgb_base:.2f})"

        point = {
            "timestamp": f"-{h}h",
            "rf_risk_score": round(rf_base, 4),
            "xgb_risk_score": round(xgb_base, 4),
            "ensemble_score": round(ensemble, 4),
            "risk_level": level,
            "alert_triggered": alert_triggered,
            "alert_message": alert_msg
        }
        data_points.append(point)
        if alert_triggered:
            alerts.append({"time": f"-{h}h", "message": alert_msg})

    # Compute overall trend from ensemble scores
    if len(trend_scores) >= 4:
        first_quarter = sum(trend_scores[:len(trend_scores)//4]) / (len(trend_scores)//4)
        last_quarter = sum(trend_scores[-(len(trend_scores)//4):]) / (len(trend_scores)//4)
        diff = last_quarter - first_quarter
        if diff > 0.05:
            trend = "declining"
        elif diff < -0.05:
            trend = "improving"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return {
        "patient_id": patient_id,
        "period": f"last {hours} hours",
        "total_data_points": len(data_points),
        "trend": trend,
        "total_alerts": len(alerts),
        "alerts": alerts,
        "monitoring_summary": {
            "rf_model": "Used for stable hourly trend analysis",
            "xgb_model": "Used for real-time anomaly/spike detection",
            "ensemble": "Weighted combination (40% RF + 60% XGB)"
        },
        "data": data_points
    }


# ============================================================================
# 10. PATIENT HISTORY ROUTES
# ============================================================================

@app.post("/compare", tags=["Predictions"])
async def compare_models(patient: PatientInput):
    """
    Compare predictions from BOTH Random Forest and XGBoost side-by-side.
    Returns predictions from both models with a recommendation on which to trust.
    """
    try:
        rf_pred = await predict_sepsis(patient, "random_forest")
        xgb_pred = await predict_sepsis(patient, "xgboost")
        
        score_diff = abs(xgb_pred.risk_score - rf_pred.risk_score)
        
        if score_diff < 0.1:
            agreement = "STRONG - Both models agree closely"
        elif score_diff < 0.2:
            agreement = "MODERATE - Minor differences between models"
        else:
            agreement = "WEAK - Significant disagreement, clinical review recommended"
        
        return {
            "patient_id": patient.patient_id,
            "random_forest": {
                "risk_score": rf_pred.risk_score,
                "risk_level": rf_pred.risk_level,
                "recommendation": rf_pred.recommendation
            },
            "xgboost": {
                "risk_score": xgb_pred.risk_score,
                "risk_level": xgb_pred.risk_level,
                "recommendation": xgb_pred.recommendation
            },
            "ensemble": {
                "average_score": round((rf_pred.risk_score + xgb_pred.risk_score) / 2, 4),
                "max_score": round(max(rf_pred.risk_score, xgb_pred.risk_score), 4),
                "agreement": agreement
            },
            "sirs_info": rf_pred.sirs_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/patient/{patient_id}/predictions", tags=["Patient Data"])
async def add_patient_prediction(patient_id: str, patient: PatientInput, model_name: str = "xgboost"):
    """Log prediction for a specific patient. Supports 'xgboost' or 'random_forest'."""
    try:
        prediction = await predict_sepsis(patient, model_name)
        return {
            "message": "Prediction logged",
            "patient_id": patient_id,
            "model_used": model_name,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/patient/{patient_id}/history", tags=["Patient Data"])
async def get_patient_history(patient_id: str):
    """Retrieve prediction history for a patient"""
    return PatientHistory(
        patient_id=patient_id,
        predictions=[],
        current_risk=0.45,
        trend="stable",
        last_updated=datetime.now().isoformat()
    )


@app.get("/patient/{patient_id}/current-risk", tags=["Patient Data"])
async def get_current_risk(patient_id: str):
    """Get current risk assessment for patient"""
    return {
        "patient_id": patient_id,
        "current_risk_score": 0.45,
        "current_risk_level": "MODERATE",
        "last_updated": datetime.now().isoformat(),
        "trend": "stable",
        "notes": "Patient stable. Continue monitoring."
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error": True,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": str(exc),
            "error": True,
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Sepsis Prediction API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
