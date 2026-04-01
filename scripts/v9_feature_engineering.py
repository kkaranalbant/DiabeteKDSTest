"""
🧬 DiabetAI v9 - Physics Feature Engineering
Copy this section into app.py after training v9 model in Colab.
"""

# ============================================================
# v9 CONFIGURATION (20 Features with Physics)
# ============================================================
MODEL_PATH_V9 = r"D:\Diabete\models\xgboost_clinical_v9.json"

# Physics target glucose
TARGET_BG = 110.0

# v9 Feature List (20 features)
V9_FEATURES = [
    # Core Clinical
    'CGM',              # 0
    'carbs',            # 1
    'iob',              # 2
    
    # Physics Features (The Trojan Horse)
    'phys_meal_bolus',  # 3  = carbs / ICR
    'phys_corr_bolus',  # 4  = (CGM - 110) / ISF
    'phys_net_need',    # 5  = meal + corr - iob
    
    # Therapy Settings
    'est_tdd',          # 6
    'est_isf',          # 7
    'est_icr',          # 8
    
    # Trends
    'cgm_delta_5min',   # 9
    'cgm_delta_15min',  # 10
    'cgm_velocity',     # 11
    
    # Basal
    'basal',            # 12
    'basal_delta_5min', # 13
    
    # Time
    'time_sin',         # 14
    'time_cos',         # 15
    'is_weekend',       # 16
    
    # Demographics
    'age',              # 17
    'weight',           # 18
    'gender_encoded'    # 19
]


def create_v9_features(cgm, carbs, iob, basal, velocity, isf, icr,
                       age, weight, gender):
    """
    Create EXACT 20-feature vector matching v9 training signature.
    Includes pre-calculated physics features.
    """
    # Time features
    now = datetime.now()
    hour = now.hour + now.minute / 60
    time_sin = float(np.sin(2 * np.pi * hour / 24))
    time_cos = float(np.cos(2 * np.pi * hour / 24))
    is_weekend = 1.0 if now.weekday() >= 5 else 0.0

    # Robust TDD Estimation
    est_daily_carbs = max(150.0, float(carbs) * 3.5) if carbs > 0 else 150.0
    est_bolus_daily = est_daily_carbs / float(icr) if icr > 0 else 15.0
    est_tdd = (float(basal) * 24.0) + est_bolus_daily
    est_tdd = max(est_tdd, 15.0)

    # ISF/ICR (use provided or estimate)
    est_isf = float(isf) if isf > 0 else (1800.0 / est_tdd)
    est_icr = float(icr) if icr > 0 else (500.0 / est_tdd)

    # CGM Deltas
    cgm_delta_5min = float(velocity)
    cgm_delta_15min = float(velocity) * 3.0
    cgm_velocity = cgm_delta_15min / 15.0

    # Basal Delta
    basal_delta_5min = 0.0

    # Gender Encoding
    gender_encoded = 1.0 if gender == "Female" else 0.0

    # ========== PHYSICS FEATURES (The Trojan Horse) ==========
    # These are pre-calculated clinical formulas
    
    # Meal Bolus: carbs / ICR
    phys_meal_bolus = float(carbs) / max(1.0, est_icr) if carbs > 0 else 0.0
    
    # Correction Bolus: (CGM - target) / ISF (only positive)
    phys_corr_bolus = max(0.0, (float(cgm) - TARGET_BG) / max(1.0, est_isf))
    
    # Net Need: meal + correction - IOB
    phys_net_need = max(0.0, phys_meal_bolus + phys_corr_bolus - float(iob))

    # ========== BUILD FEATURE ARRAY (20 items) ==========
    features = np.array([[
        float(cgm),           # 0
        float(carbs),         # 1
        float(iob),           # 2
        phys_meal_bolus,      # 3  (Physics)
        phys_corr_bolus,      # 4  (Physics)
        phys_net_need,        # 5  (Physics)
        float(est_tdd),       # 6
        float(est_isf),       # 7
        float(est_icr),       # 8
        float(cgm_delta_5min),   # 9
        float(cgm_delta_15min),  # 10
        float(cgm_velocity),     # 11
        float(basal),            # 12
        float(basal_delta_5min), # 13
        time_sin,             # 14
        time_cos,             # 15
        is_weekend,           # 16
        float(age),           # 17
        float(weight),        # 18
        gender_encoded        # 19
    ]], dtype=np.float32)

    debug_info = {
        'est_tdd': est_tdd,
        'est_isf': est_isf,
        'est_icr': est_icr,
        'phys_meal_bolus': phys_meal_bolus,
        'phys_corr_bolus': phys_corr_bolus,
        'phys_net_need': phys_net_need,
        'features': features[0].tolist()
    }

    return features, debug_info


# ============================================================
# USAGE IN MAIN APP
# ============================================================
"""
# In main(), replace create_v8_features with:

features, debug_data = create_v9_features(
    cgm, carbs, iob, basal, velocity, isf, icr, age, weight, gender
)
dmatrix = xgb.DMatrix(features, feature_names=V9_FEATURES)
ai_dose = float(model.predict(dmatrix)[0])

# Debug console will now show physics features:
# - phys_meal_bolus: Carbs / ICR
# - phys_corr_bolus: (CGM - 110) / ISF
# - phys_net_need: Total need after IOB subtraction
"""
