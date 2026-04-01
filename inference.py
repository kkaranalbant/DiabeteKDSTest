"""
🧠 DiabetAI v10 - Production Inference Engine
Clinical Safety Guardrails + Physics Features

Rules:
1. Meal Cap: AI capped at Math * 1.30
2. Correction Trust: AI trusted for pure corrections
3. Hypo Shield: BG < 70 → 0U
"""

import os
import numpy as np
import xgboost as xgb
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, List


# ============================================================
# CONFIGURATION
# ============================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "xgboost_clinical_v10.json")
TARGET_BG = 100.0

# v10 Feature List (20 features - NO is_synthetic)
V10_FEATURES = [
    'CGM', 'carbs', 'iob',
    'phys_meal_bolus', 'phys_corr_bolus', 'phys_net_need',
    'est_tdd', 'est_isf', 'est_icr',
    'cgm_delta_5min', 'cgm_delta_15min', 'cgm_velocity',
    'basal', 'basal_delta_5min',
    'time_sin', 'time_cos', 'is_weekend',
    'age', 'weight', 'gender_encoded'
]


@dataclass
class InferenceResult:
    """Result from the inference engine."""
    final_dose: float
    ai_raw_dose: float
    math_dose: float
    source: str
    guardrail_log: List[str]
    debug: Dict


class InferenceEngine:
    """
    Production inference engine for DiabetAI v10.
    Loads the model and applies clinical safety guardrails.
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the XGBoost model."""
        try:
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            self.model = None
    
    # ============================================================
    # FEATURE ENGINEERING (Exact v10 training logic)
    # ============================================================
    def create_features(self, cgm: float, carbs: float, iob: float,
                        basal: float, velocity: float, isf: float, icr: float,
                        age: float, weight: float, gender: str) -> Tuple[np.ndarray, Dict]:
        """
        Create the 20-feature vector for v10 model.
        Applies physics features (phys_meal_bolus, phys_corr_bolus, phys_net_need).
        """
        # Time features
        now = datetime.now()
        hour = now.hour + now.minute / 60
        time_sin = float(np.sin(2 * np.pi * hour / 24))
        time_cos = float(np.cos(2 * np.pi * hour / 24))
        is_weekend = 1.0 if now.weekday() >= 5 else 0.0
        
        # TDD Estimation
        est_daily_carbs = max(150.0, float(carbs) * 3.5) if carbs > 0 else 150.0
        est_bolus_daily = est_daily_carbs / float(icr) if icr > 0 else 15.0
        est_tdd = (float(basal) * 24.0) + est_bolus_daily
        est_tdd = max(est_tdd, 15.0)
        
        # ISF/ICR
        est_isf = float(isf) if isf > 0 else (1800.0 / est_tdd)
        est_icr = float(icr) if icr > 0 else (500.0 / est_tdd)
        
        # CGM Deltas
        cgm_delta_5min = float(velocity)
        cgm_delta_15min = float(velocity) * 3.0
        cgm_velocity = cgm_delta_15min / 15.0
        
        # Basal Delta
        basal_delta_5min = 0.0
        
        # Gender
        gender_encoded = 1.0 if gender == "Female" else 0.0
        
        # ========== PHYSICS FEATURES ==========
        phys_meal_bolus = float(carbs) / max(1.0, est_icr) if carbs > 0 else 0.0
        phys_corr_bolus = max(0.0, (float(cgm) - TARGET_BG) / max(1.0, est_isf))
        phys_net_need = max(0.0, phys_meal_bolus + phys_corr_bolus - float(iob))
        
        # Build feature array (20 items)
        features = np.array([[
            float(cgm), float(carbs), float(iob),
            phys_meal_bolus, phys_corr_bolus, phys_net_need,
            float(est_tdd), float(est_isf), float(est_icr),
            float(cgm_delta_5min), float(cgm_delta_15min), float(cgm_velocity),
            float(basal), float(basal_delta_5min),
            time_sin, time_cos, is_weekend,
            float(age), float(weight), gender_encoded
        ]], dtype=np.float32)
        
        debug = {
            'est_tdd': est_tdd,
            'est_isf': est_isf,
            'est_icr': est_icr,
            'phys_meal_bolus': phys_meal_bolus,
            'phys_corr_bolus': phys_corr_bolus,
            'phys_net_need': phys_net_need
        }
        
        return features, debug
    
    # ============================================================
    # CLINICAL MATH DOSE
    # ============================================================
    def calculate_math_dose(self, cgm: float, carbs: float, iob: float,
                            isf: float, icr: float, target: float = 100.0) -> float:
        """Standard clinical calculation."""
        meal = float(carbs) / float(icr) if icr > 0 and carbs > 0 else 0.0
        correction = max(0.0, (float(cgm) - target) / float(isf)) if cgm > target and isf > 0 else 0.0
        return max(0.0, meal + correction - float(iob))
    
    # ============================================================
    # SAFETY GUARDRAILS (The Braking System)
    # ============================================================
    def apply_guardrails(self, ai_raw: float, math_dose: float,
                         cgm: float, carbs: float) -> Tuple[float, str, List[str]]:
        """
        Apply clinical safety guardrails.
        
        Rules:
        1. Hypo Shield: BG < 70 → 0U
        2. Meal Cap: Carbs > 0 → MIN(AI, Math * 1.30)
        3. Correction Trust: Carbs == 0 → Trust AI
        """
        logs = []
        final_dose = ai_raw
        source = "AI (v10)"
        
        # RULE 1: Hypo Shield (Highest Priority)
        if cgm < 70:
            final_dose = 0.0
            source = "HYPO SHIELD"
            logs.append(f"🛡️ HYPO SHIELD: BG {cgm} < 70 → 0U")
            return final_dose, source, logs
        
        # RULE 2: Meal Cap (AI too aggressive on meals)
        if carbs > 0:
            meal_cap = math_dose * 1.30
            if ai_raw > meal_cap:
                final_dose = meal_cap
                source = "MEAL CAP (Math*1.3)"
                logs.append(f"🧢 MEAL CAP: AI {ai_raw:.2f}U > Math*1.3 ({meal_cap:.2f}U) → Capped")
            else:
                logs.append(f"✅ Meal OK: AI {ai_raw:.2f}U within limit")
        
        # RULE 3: Correction Trust (AI proved itself)
        elif carbs == 0 and cgm > 100:
            source = "AI CORRECTION (Trusted)"
            logs.append(f"✅ Correction: Trusting AI {ai_raw:.2f}U")
        
        # Final safety clamp
        final_dose = max(0.0, min(final_dose, 15.0))
        
        if not logs:
            logs.append("✅ No guardrail triggered")
        
        return round(final_dose, 2), source, logs
    
    # ============================================================
    # MAIN PREDICTION
    # ============================================================
    def predict(self, cgm: float, carbs: float, iob: float,
                basal: float, velocity: float, isf: float, icr: float,
                age: float = 30, weight: float = 70, gender: str = "Male") -> InferenceResult:
        """
        Main prediction pipeline with safety guardrails.
        """
        # 1. Math Dose (Reference)
        math_dose = self.calculate_math_dose(cgm, carbs, iob, isf, icr)
        
        # 2. Feature Engineering
        features, debug = self.create_features(
            cgm, carbs, iob, basal, velocity, isf, icr, age, weight, gender
        )
        
        # 3. AI Prediction
        ai_raw = 0.0
        if self.model is not None:
            dmatrix = xgb.DMatrix(features, feature_names=V10_FEATURES)
            ai_raw = max(0.0, float(self.model.predict(dmatrix)[0]))
        
        # 4. Apply Guardrails
        final_dose, source, logs = self.apply_guardrails(ai_raw, math_dose, cgm, carbs)
        
        return InferenceResult(
            final_dose=final_dose,
            ai_raw_dose=round(ai_raw, 2),
            math_dose=round(math_dose, 2),
            source=source,
            guardrail_log=logs,
            debug=debug
        )


# ============================================================
# STRESS TEST
# ============================================================
def run_stress_test():
    """Run the 4-scenario stress test."""
    print("="*70)
    print("🧪 v10 INFERENCE ENGINE - 4-SCENARIO STRESS TEST")
    print("="*70)
    
    engine = InferenceEngine()
    
    if engine.model is None:
        print("❌ Model not loaded. Stress test aborted.")
        return
    
    scenarios = [
        {"name": "A: Meal (120, 50g)", "cgm": 120, "carbs": 50, "iob": 0, "velocity": 0,
         "target": (4.5, 7.5)},  # Should be capped now!
        {"name": "B: Correction (250, 0g)", "cgm": 250, "carbs": 0, "iob": 0, "velocity": 2,
         "target": (4.0, 6.0)},  # AI trusted
        {"name": "C: Meal+Correction", "cgm": 180, "carbs": 40, "iob": 1, "velocity": 1,
         "target": (4.0, 8.0)},
        {"name": "D: Hypo Risk (60)", "cgm": 60, "carbs": 0, "iob": 0, "velocity": -3,
         "target": (0.0, 0.0)},  # Must be 0!
    ]
    
    print(f"\n{'Scenario':<25} {'Math':>8} {'AI Raw':>8} {'Final':>8} {'Source':<25} {'Status':>8}")
    print("-"*90)
    
    all_pass = True
    
    for s in scenarios:
        result = engine.predict(
            cgm=s['cgm'], carbs=s['carbs'], iob=s['iob'],
            basal=1.0, velocity=s['velocity'],
            isf=30, icr=10
        )
        
        low, high = s['target']
        in_range = low <= result.final_dose <= high
        status = "✅ PASS" if in_range else "❌ FAIL"
        
        if not in_range:
            all_pass = False
        
        print(f"{s['name']:<25} {result.math_dose:>8.2f} {result.ai_raw_dose:>8.2f} "
              f"{result.final_dose:>8.2f} {result.source:<25} {status:>8}")
    
    print("-"*90)
    
    if all_pass:
        print("\n🎉 ALL SCENARIOS PASSED!")
        print("✅ Scenario A: Meal cap working (no more 8.95U madness)")
        print("✅ Scenario B: Correction trust working")
        print("✅ Scenario D: Hypo shield working")
    else:
        print("\n⚠️ Some scenarios failed")


if __name__ == "__main__":
    run_stress_test()
