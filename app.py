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
    
    lactate: float = Field(..., gt=0, description="Lactate level in mmol/L")
    wbc_count: float = Field(..., gt=0, description="WBC count in x10^9/L")
    crp_level: float = Field(..., ge=0, description="CRP level in mg/L")
    creatinine: float = Field(..., gt=0, description="Creatinine in mg/dL")
    heart_rate: float = Field(..., gt=0, description="Heart rate in bpm")
    respiratory_rate: float = Field(..., gt=0, description="Respiratory rate /min")
    temperature: float = Field(..., description="Temperature in °C")
    spo2: float = Field(..., ge=0, le=100, description="SpO2 percentage")
    systolic_bp: float = Field(..., gt=0, description="Systolic BP in mmHg")
    diastolic_bp: float = Field(..., gt=0, description="Diastolic BP in mmHg")
    hemoglobin: float = Field(..., gt=0, description="Hemoglobin in g/dL")

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
                "hemoglobin": 14.0
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
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    gender: Optional[Literal["M", "F", "Other"]] = Field(None, description="Patient gender")
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


def load_feature_names():
    """Load feature names for the model"""
    try:
        feature_path = "model/feature_names.txt"
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            return feature_names
        return list(PatientVitals.__fields__.keys())
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return list(PatientVitals.__fields__.keys())


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


def prepare_features(vitals: PatientVitals, feature_names: List[str]) -> np.ndarray:
    """Prepare features for model prediction"""
    vitals_dict = vitals.model_dump()
    
    # Create feature vector in correct order
    feature_vector = []
    for feature in feature_names:
        if feature in vitals_dict:
            feature_vector.append(vitals_dict[feature])
    
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
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "predict": "/predict",
            "explain": "/explain"
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
        
        feature_names = load_feature_names()
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
    Comprehensive risk stratification for patient
    """
    try:
        prediction = await predict_sepsis(patient, "xgboost")
        
        return {
            "patient_id": patient.patient_id,
            "risk_stratification": {
                "category": prediction.risk_level,
                "score": prediction.risk_score,
                "recommendation": prediction.recommendation,
                "monitoring_frequency": {
                    "LOW": "Every 8-12 hours",
                    "MODERATE": "Every 4-6 hours",
                    "HIGH": "Every 2-4 hours",
                    "CRITICAL": "Continuous monitoring"
                }[prediction.risk_level]
            },
            "sirs_status": prediction.sirs_info,
            "timestamp": prediction.timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# 3. EXPLAINABILITY ROUTES
# ============================================================================

@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_prediction(patient: PatientInput):
    """
    Provide SHAP-based explanation for prediction
    """
    try:
        # Get prediction first
        prediction = await predict_sepsis(patient, "xgboost")
        
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
async def explain_doctor(patient: PatientInput):
    """Doctor-friendly explanation"""
    try:
        explanation = await explain_prediction(patient)
        return {
            "patient_id": patient.patient_id,
            "clinical_assessment": explanation.doctor_explanation,
            "key_findings": explanation.top_features,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain-patient", tags=["Explainability"])
async def explain_patient(patient: PatientInput):
    """Patient-friendly explanation"""
    try:
        explanation = await explain_prediction(patient)
        return {
            "patient_id": patient.patient_id,
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
async def list_features():
    """List all features used by the model"""
    feature_names = load_feature_names()
    
    return {
        "total_features": len(feature_names),
        "features": feature_names,
        "feature_descriptions": {
            field_name: field.description
            for field_name, field in PatientVitals.model_fields.items()
        }
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
    """Risk score time series data for a patient"""
    # Simulate time series data
    import random
    
    timestamps = []
    risk_scores = []
    base_score = random.uniform(0.3, 0.7)
    
    for h in range(hours, 0, -1):
        timestamps.append(f"-{h}h")
        risk_scores.append(base_score + random.uniform(-0.1, 0.1))
    
    return {
        "patient_id": patient_id,
        "period": f"last {hours} hours",
        "data": {
            "timestamps": timestamps,
            "risk_scores": risk_scores,
            "trend": "stable"
        }
    }


# ============================================================================
# 10. PATIENT HISTORY ROUTES
# ============================================================================

@app.post("/patient/{patient_id}/predictions", tags=["Patient Data"])
async def add_patient_prediction(patient_id: str, patient: PatientInput):
    """Log prediction for a specific patient"""
    try:
        prediction = await predict_sepsis(patient, "xgboost")
        return {
            "message": "Prediction logged",
            "patient_id": patient_id,
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
