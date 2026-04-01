"""
🚀 DiabetAI API Service v8.0
Backend: FastAPI + XGBoost (Physics-Informed)
Update: Aligned with v8 training signature (17 Features)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import os
from datetime import datetime
import math

app = FastAPI(title="DiabetAI Clinical API v8", version="8.0")

# ============================================================
# CONFIGURATION
# ============================================================
# Model Yolu (v8'e güncellendi)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "xgboost_clinical_v8.json")
model = None

# v8 Özellik Listesi (Sıralama EĞİTİM ile birebir aynı olmalı)
V8_FEATURES = [
    'CGM',               # 0
    'carbs',             # 1
    'iob',               # 2
    'est_tdd',           # 3
    'est_isf',           # 4
    'est_icr',           # 5
    'cgm_delta_5min',    # 6
    'cgm_delta_15min',   # 7
    'cgm_velocity',      # 8
    'basal',             # 9
    'basal_delta_5min',  # 10
    'time_sin',          # 11
    'time_cos',          # 12
    'is_weekend',        # 13
    'age',               # 14
    'weight',            # 15
    'gender_encoded'     # 16
]

# ============================================================
# DATA MODELS
# ============================================================
class PatientState(BaseModel):
    glucose: float
    trend: float = 0.0
    iob: float = 0.0
    basal_rate: float = 1.0

class TherapySettings(BaseModel):
    isf: float = 30.0
    icr: float = 10.0
    target_bg: float = 100.0

class MealInfo(BaseModel):
    carbs: float = 0.0

class UserProfile(BaseModel):
    age: int = 30
    weight: float = 70.0
    gender: str = "Male"

class PredictionRequest(BaseModel):
    patient: PatientState
    therapy: TherapySettings
    meal: MealInfo
    profile: UserProfile

# ============================================================
# STARTUP EVENT
# ============================================================
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.Booster()
            model.load_model(MODEL_PATH)
            print(f"✅ v8 Model Loaded Successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Model Load Failed: {e}")
    else:
        print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")

# ============================================================
# FEATURE ENGINEERING (v8 Logic)
# ============================================================
def prepare_v8_features(req: PredictionRequest):
    # Time Features
    now = datetime.now()
    hour = now.hour + now.minute / 60
    time_sin = math.sin(2 * math.pi * hour / 24)
    time_cos = math.cos(2 * math.pi * hour / 24)
    is_weekend = 1.0 if now.weekday() >= 5 else 0.0

    # Robust TDD Estimation (Same logic as app.py)
    # TDD = (Basal * 24) + (Daily Carbs / ICR)
    # Günlük karbonhidrat bilinmiyorsa: Şu anki öğün * 3.5 veya min 150g varsayılır.
    est_daily_carbs = max(150.0, req.meal.carbs * 3.5) if req.meal.carbs > 0 else 150.0
    est_bolus_daily = est_daily_carbs / req.therapy.icr if req.therapy.icr > 0 else 15.0
    est_tdd = (req.patient.basal_rate * 24.0) + est_bolus_daily
    est_tdd = max(est_tdd, 15.0)

    # ISF/ICR
    est_isf = req.therapy.isf
    est_icr = req.therapy.icr

    # Deltas
    # API'ye genelde anlık velocity gelir. Bunu 5 ve 15 dk'ya genişletiyoruz.
    cgm_delta_5min = req.patient.trend
    cgm_delta_15min = req.patient.trend * 3.0
    cgm_velocity = cgm_delta_15min / 15.0

    basal_delta_5min = 0.0 # Anlık değişim bilinmiyor varsayalım

    # Encoding
    gender_encoded = 1.0 if req.profile.gender == "Female" else 0.0

    # BUILD VECTOR (17 Features - EXACT ORDER)
    features = [
        req.patient.glucose,     # 0: CGM
        req.meal.carbs,          # 1: carbs
        req.patient.iob,         # 2: iob
        est_tdd,                 # 3: est_tdd
        est_isf,                 # 4: est_isf
        est_icr,                 # 5: est_icr
        cgm_delta_5min,          # 6: cgm_delta_5min
        cgm_delta_15min,         # 7: cgm_delta_15min
        cgm_velocity,            # 8: cgm_velocity
        req.patient.basal_rate,  # 9: basal
        basal_delta_5min,        # 10: basal_delta_5min
        time_sin,                # 11: time_sin
        time_cos,                # 12: time_cos
        is_weekend,              # 13: is_weekend
        float(req.profile.age),  # 14: age
        float(req.profile.weight),# 15: weight
        gender_encoded           # 16: gender_encoded
    ]

    return np.array([features], dtype=np.float32)

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def health_check():
    return {"status": "active", "model": "v8.0 (Physics-Informed)"}

@app.post("/predict/bolus")
def predict_bolus(req: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Prepare Features
        features = prepare_v8_features(req)
        dmatrix = xgb.DMatrix(features, feature_names=V8_FEATURES)

        # 2. Predict
        ai_dose = float(model.predict(dmatrix)[0])
        ai_dose = max(0.0, ai_dose)

        # 3. Math Safety Check (Hybrid Logic inside API)
        # Matematiksel doz hesabı
        meal_bolus = req.meal.carbs / req.therapy.icr if req.therapy.icr > 0 else 0
        correction = 0
        if req.patient.glucose > req.therapy.target_bg:
            correction = (req.patient.glucose - req.therapy.target_bg) / req.therapy.isf
        math_dose = max(0.0, meal_bolus + correction - req.patient.iob)

        # Safety Fallback
        final_dose = ai_dose
        source = "AI (v8)"

        # Eğer AI çok düşük verirse (ve karbonhidrat varsa) matematiğe dön
        if req.meal.carbs > 10 and ai_dose < 0.5:
             final_dose = math_dose * 0.9
             source = "Safety Fallback (Math)"

        # Eğer AI matematikten çok saparsa (%150) ortalamasını al (Clamp)
        if ai_dose > (math_dose * 1.5) and math_dose > 1.0:
            final_dose = (ai_dose + math_dose) / 2
            source = "AI+Math Consensus"

        return {
            "recommended_bolus": round(final_dose, 2),
            "source": source,
            "ai_raw": round(ai_dose, 2),
            "math_ref": round(math_dose, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)