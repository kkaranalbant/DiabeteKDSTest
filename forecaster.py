"""
🔬 GlucoseForecaster — Hybrid Ensemble Prediction Engine
Predicts blood glucose 30 minutes into the future using:
  Model A (Absolute): Stability-focused prediction
  Model B (Physics Residual): Safety-focused, biology-aware prediction
  
Final = w_a × Pred_A + w_b × (Linear_Forecast + Pred_B_Residual)
"""

import numpy as np
import xgboost as xgb
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import os


# ============================================================
# CONFIGURATION
# ============================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_A = os.path.join(_BASE_DIR, "models", "ensemble_model_a_absolute.json")
DEFAULT_MODEL_B = os.path.join(_BASE_DIR, "models", "ensemble_model_b_residual.json")

# Ensemble weights (from Optuna sweep — v5 notebook)
W_A = 0.60  # Absolute model weight
W_B = 0.40  # Residual model weight

# Feature order expected by the glucose forecasting models
FORECAST_FEATURES = [
    "CGM", "IOB", "COB", "Vel_5min", "Vel_15min", "Acc_15min",
    "Roll_Mean_60", "Roll_Std_60", "Hour_Sin", "Hour_Cos",
    "Insulin_Sensitivity_Risk",
]


@dataclass
class ForecastResult:
    """Result from the glucose forecaster."""
    predicted_glucose: float       # Final ensemble prediction (mg/dL)
    pred_a: float                  # Model A raw prediction
    pred_b: float                  # Model B reconstructed prediction
    linear_forecast: float         # Physics-only baseline
    confidence_band: float         # ± MAE (estimated uncertainty)
    risk_level: str                # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    risk_color: str                # For UI display


class GlucoseForecaster:
    """
    Hybrid Ensemble Glucose Forecaster.
    
    Uses two complementary XGBoost models:
      - Model A: Predicts absolute glucose (stable, but persistence-biased)
      - Model B: Predicts physics residual (dynamic, captures biology)
    
    Final prediction blends both for accuracy + safety.
    """

    def __init__(
        self,
        model_a_path: str = DEFAULT_MODEL_A,
        model_b_path: str = DEFAULT_MODEL_B,
        w_a: float = W_A,
        w_b: float = W_B,
    ):
        self.w_a = w_a
        self.w_b = w_b
        self.model_a = None
        self.model_b = None
        self._load_models(model_a_path, model_b_path)

    def _load_models(self, path_a: str, path_b: str):
        """Load both XGBoost ensemble models."""
        for label, path in [("A (Absolute)", path_a), ("B (Residual)", path_b)]:
            if not os.path.exists(path):
                print(f"⚠️  Model {label} not found: {path}")
                continue

        try:
            self.model_a = xgb.XGBRegressor()
            self.model_a.load_model(path_a)
            print(f"✅ Model A loaded: {os.path.basename(path_a)}")
        except Exception as e:
            print(f"❌ Model A load failed: {e}")

        try:
            self.model_b = xgb.XGBRegressor()
            self.model_b.load_model(path_b)
            print(f"✅ Model B loaded: {os.path.basename(path_b)}")
        except Exception as e:
            print(f"❌ Model B load failed: {e}")

    @property
    def is_ready(self) -> bool:
        return self.model_a is not None and self.model_b is not None

    # ============================================================
    # FEATURE ENGINEERING (Single-Point Inference)
    # ============================================================
    def _build_features(
        self,
        cgm: float,
        iob: float,
        cob: float,
        trend: float,  # mg/dL per 5 min
    ) -> np.ndarray:
        """
        Build the 11-feature vector from scalar inputs.

        Since we don't have full patient history in real-time app usage,
        we approximate rolling/kinetic features from the available inputs.
        """
        now = datetime.now()
        hour = now.hour + now.minute / 60.0

        # Kinetic approximations
        vel_5min = float(trend)
        vel_15min = float(trend) * 3.0  # Assume stable trend
        acc_15min = 0.0                 # Assume constant velocity

        # Rolling stats (best guess without history)
        roll_mean_60 = float(cgm)       # Current CGM as proxy
        roll_std_60 = 0.0               # Assume stability

        # Circadian features
        hour_sin = float(np.sin(2 * np.pi * hour / 24))
        hour_cos = float(np.cos(2 * np.pi * hour / 24))

        # Interaction feature
        insulin_sensitivity_risk = float(iob) * vel_15min

        features = np.array([[
            float(cgm),
            float(iob),
            float(cob),
            vel_5min,
            vel_15min,
            acc_15min,
            roll_mean_60,
            roll_std_60,
            hour_sin,
            hour_cos,
            insulin_sensitivity_risk,
        ]], dtype=np.float32)

        return features

    # ============================================================
    # RISK ASSESSMENT
    # ============================================================
    @staticmethod
    def _assess_risk(predicted: float, current: float) -> tuple:
        """Determine clinical risk level from predicted glucose."""
        if predicted < 54:
            return "CRITICAL", "#FF0000"   # Severe hypo
        elif predicted < 70:
            return "HIGH", "#FF4444"       # Hypoglycemia
        elif predicted > 300:
            return "HIGH", "#FF8800"       # Severe hyper
        elif predicted > 250:
            return "MEDIUM", "#FFAA00"     # Hyperglycemia
        elif predicted > 180:
            return "MEDIUM", "#FFD700"     # Above target
        elif 70 <= predicted <= 180:
            return "LOW", "#00CC66"        # In range
        else:
            return "MEDIUM", "#FFAA00"

    # ============================================================
    # MAIN PREDICTION
    # ============================================================
    def predict(
        self,
        cgm: float,
        iob: float = 0.0,
        cob: float = 0.0,
        trend: float = 0.0,
    ) -> Optional[ForecastResult]:
        """
        Predict glucose 30 minutes into the future.

        Args:
            cgm:   Current blood glucose (mg/dL)
            iob:   Insulin On Board (units)
            cob:   Carbs On Board (grams)
            trend: CGM velocity (mg/dL per 5 minutes)

        Returns:
            ForecastResult with ensemble prediction and metadata.
        """
        if not self.is_ready:
            return None

        # 1. Build features
        features = self._build_features(cgm, iob, cob, trend)

        # 2. Physics baseline
        vel_15min = trend * 3.0
        linear_forecast = cgm + (vel_15min * 2.0)

        # 3. Model A: Absolute prediction
        pred_a = float(self.model_a.predict(features)[0])
        pred_a = np.clip(pred_a, 20, 500)

        # 4. Model B: Residual prediction → reconstruct
        pred_b_residual = float(self.model_b.predict(features)[0])
        pred_b = linear_forecast + pred_b_residual
        pred_b = np.clip(pred_b, 20, 500)

        # 5. Ensemble blend
        predicted = self.w_a * pred_a + self.w_b * pred_b
        predicted = np.clip(predicted, 20, 500)

        # 6. Risk assessment
        risk_level, risk_color = self._assess_risk(predicted, cgm)

        # 7. Confidence band (approximate ± MAE from training)
        confidence = 12.0  # Conservative estimate

        return ForecastResult(
            predicted_glucose=round(float(predicted), 1),
            pred_a=round(float(pred_a), 1),
            pred_b=round(float(pred_b), 1),
            linear_forecast=round(float(linear_forecast), 1),
            confidence_band=confidence,
            risk_level=risk_level,
            risk_color=risk_color,
        )


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 GlucoseForecaster — Standalone Test")
    print("=" * 60)

    forecaster = GlucoseForecaster()

    if not forecaster.is_ready:
        print("⚠️  Models not loaded. Ensure ensemble models exist in models/")
        print(f"   Expected: {DEFAULT_MODEL_A}")
        print(f"   Expected: {DEFAULT_MODEL_B}")
    else:
        scenarios = [
            {"name": "Normal Stable",   "cgm": 120, "iob": 0.5, "cob": 0,  "trend": 0},
            {"name": "Rising Fast",     "cgm": 150, "iob": 0,   "cob": 40, "trend": 3},
            {"name": "Falling (Hypo)",  "cgm": 85,  "iob": 2.0, "cob": 0,  "trend": -3},
            {"name": "High Correction", "cgm": 280, "iob": 1.0, "cob": 0,  "trend": 1},
        ]

        print(f"\n{'Scenario':<20} {'CGM':>5} {'Trend':>6} | {'Pred':>6} {'Linear':>7} "
              f"{'A':>6} {'B':>6} | {'Risk':<10}")
        print("-" * 80)

        for s in scenarios:
            r = forecaster.predict(**{k: v for k, v in s.items() if k != "name"})
            if r:
                print(f"{s['name']:<20} {s['cgm']:>5} {s['trend']:>+6.1f} | "
                      f"{r.predicted_glucose:>6.1f} {r.linear_forecast:>7.1f} "
                      f"{r.pred_a:>6.1f} {r.pred_b:>6.1f} | {r.risk_level:<10}")
