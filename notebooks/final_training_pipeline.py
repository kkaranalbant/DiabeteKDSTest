"""
🚀 Final Model Training Pipeline
================================
Insulin Dosage Prediction using XGBoost (GPU Accelerated)

This script performs:
1. External Test Data Preprocessing - Transforms raw test data to match training schema
2. Full Training & Evaluation - Trains on 100% of augmented training data

Author: Senior Data Scientist / ML Engineer
"""

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os
import gc
import math

# ==============================================================================
# 🔧 CONFIGURATION
# ==============================================================================

# Google Colab-style paths (modify for local use if needed)
# For Colab: "/content/drive/MyDrive/DiabetesKDS/data/..."
# For Local Windows: "d:/Diabete/data/..."

USE_COLAB = True  # Set to False for local Windows execution

if USE_COLAB:
    BASE_PATH = "/content/drive/MyDrive/DiabetesKDS/data"
    RAW_TEST_FILE = f"{BASE_PATH}/raw/metabonet_public_test.parquet"
    PROCESSED_TEST_FILE = f"{BASE_PATH}/processed/test_ready_for_xgboost.parquet"
    TRAIN_FILE = f"{BASE_PATH}/processed/train_augmented_fixed.parquet"
    MODEL_OUTPUT = f"{BASE_PATH}/processed/xgboost_final_model.json"
else:
    BASE_PATH = "d:/Diabete/data"
    RAW_TEST_FILE = f"{BASE_PATH}/metabonet_public_test.parquet"  # In root data folder
    PROCESSED_TEST_FILE = f"{BASE_PATH}/processed/test_ready_for_xgboost.parquet"
    TRAIN_FILE = f"{BASE_PATH}/processed/train_augmented_fixed.parquet"
    MODEL_OUTPUT = f"{BASE_PATH}/processed/xgboost_final_model.json"

# Feature definitions
FEATURES = ['CGM', 'carbs', 'time_sin', 'time_cos', 'is_weekend', 
            'cgm_delta_5min', 'cgm_delta_15min', 'weight', 'basal', 
            'age', 'gender', 'insulin_type_bolus']
TARGET = 'bolus'

# Encoding mappings
GENDER_MAP = {'Male': 0, 'Female': 1}
INSULIN_TYPE_MAP = {'Humalog': 3, 'Novolog': 6}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'device': 'cuda',
    'tree_method': 'hist',
    'max_depth': 12,
    'min_child_weight': 5,
    'gamma': 0.1,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 50,
    'seed': 42,
}

NUM_ROUNDS = 500
RANDOM_SEED = 42


# ==============================================================================
# 🟢 STEP 1: EXTERNAL TEST DATA PREPROCESSING
# ==============================================================================

def preprocess_test_data():
    """
    Preprocess raw test data to match the training data schema.
    
    Transformations:
    - Date conversion to datetime
    - Cyclic time features (sin/cos)
    - Weekend flag
    - Lag features (cgm_delta_5min, cgm_delta_15min)
    - Categorical encoding (gender, insulin_type_bolus)
    - Missing value imputation
    """
    print("=" * 70)
    print("🟢 STEP 1: EXTERNAL TEST DATA PREPROCESSING")
    print("=" * 70)
    
    # 1. Load Raw Data
    print(f"\n📂 Loading raw test data from: {RAW_TEST_FILE}")
    start_time = time.time()
    
    df = pl.read_parquet(RAW_TEST_FILE)
    
    load_time = time.time() - start_time
    print(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"⏱️  Load time: {load_time:.2f} seconds")
    print(f"📊 Columns: {df.columns}")
    
    # 2. Apply Feature Engineering
    print("\n🔧 Applying feature engineering...")
    
    # 2a. Date conversion to datetime
    if 'date' in df.columns:
        df = df.with_columns(
            pl.col('date').cast(pl.Datetime)
        )
        print("   ✓ Date converted to datetime")
    
    # 2b. Cyclic Time Features (from hour)
    if 'date' in df.columns:
        df = df.with_columns([
            (2 * math.pi * pl.col('date').dt.hour() / 24).sin().alias('time_sin'),
            (2 * math.pi * pl.col('date').dt.hour() / 24).cos().alias('time_cos'),
        ])
        print("   ✓ Created time_sin and time_cos features")
    
    # 2c. Weekend Flag
    if 'date' in df.columns:
        df = df.with_columns(
            (pl.col('date').dt.weekday() >= 5).cast(pl.Int32).alias('is_weekend')
        )
        print("   ✓ Created is_weekend feature")
    
    # 2d. Lag Features (sort by id and date first)
    if 'id' in df.columns and 'date' in df.columns and 'CGM' in df.columns:
        df = df.sort(['id', 'date'])
        
        df = df.with_columns([
            (pl.col('CGM') - pl.col('CGM').shift(1).over('id')).alias('cgm_delta_5min'),
            (pl.col('CGM') - pl.col('CGM').shift(3).over('id')).alias('cgm_delta_15min'),
        ])
        
        # Fill nulls in lag features with 0
        df = df.with_columns([
            pl.col('cgm_delta_5min').fill_null(0.0),
            pl.col('cgm_delta_15min').fill_null(0.0),
        ])
        print("   ✓ Created cgm_delta_5min and cgm_delta_15min features")
    
    # 2e. Encoding - Gender
    if 'gender' in df.columns:
        # Create mapping expression
        df = df.with_columns(
            pl.col('gender')
            .map_elements(lambda x: GENDER_MAP.get(x, -1) if x is not None else -1, return_dtype=pl.Int32)
            .alias('gender')
        )
        print("   ✓ Encoded gender column")
    else:
        # Create gender column with -1 if not exists
        df = df.with_columns(pl.lit(-1).cast(pl.Int32).alias('gender'))
        print("   ✓ Created gender column (filled with -1)")
    
    # 2f. Encoding - Insulin Type Bolus
    if 'insulin_type_bolus' in df.columns:
        df = df.with_columns(
            pl.col('insulin_type_bolus')
            .map_elements(lambda x: INSULIN_TYPE_MAP.get(x, -1) if x is not None else -1, return_dtype=pl.Int32)
            .alias('insulin_type_bolus')
        )
        print("   ✓ Encoded insulin_type_bolus column")
    else:
        df = df.with_columns(pl.lit(-1).cast(pl.Int32).alias('insulin_type_bolus'))
        print("   ✓ Created insulin_type_bolus column (filled with -1)")
    
    # 2g. Imputation for weight, basal, age
    for col in ['weight', 'basal', 'age']:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).fill_null(-1.0).cast(pl.Float32)
            )
        else:
            df = df.with_columns(pl.lit(-1.0).cast(pl.Float32).alias(col))
    print("   ✓ Imputed missing values in weight, basal, age with -1.0")
    
    # 3. Selection - Select only model features (and bolus if available)
    print("\n📋 Selecting model features...")
    
    available_features = [f for f in FEATURES if f in df.columns]
    selection_cols = available_features.copy()
    
    has_target = TARGET in df.columns
    if has_target:
        selection_cols.append(TARGET)
        print(f"   ✓ Target column '{TARGET}' found in test data")
    else:
        print(f"   ⚠️ Target column '{TARGET}' NOT found in test data")
    
    df_selected = df.select(selection_cols)
    
    # Cast feature columns to Float32 for consistency
    for col in available_features:
        df_selected = df_selected.with_columns(
            pl.col(col).cast(pl.Float32)
        )
    
    print(f"   ✓ Selected {len(selection_cols)} columns: {selection_cols}")
    
    # 4. Save Processed Data
    print(f"\n💾 Saving processed test data to: {PROCESSED_TEST_FILE}")
    df_selected.write_parquet(PROCESSED_TEST_FILE)
    
    file_size_mb = os.path.getsize(PROCESSED_TEST_FILE) / (1024**2)
    print(f"✅ Saved: {len(df_selected):,} rows × {len(df_selected.columns)} columns")
    print(f"📦 File size: {file_size_mb:.2f} MB")
    
    # Cleanup
    del df
    gc.collect()
    
    return df_selected, has_target


# ==============================================================================
# 🔴 STEP 2: FULL TRAINING & EVALUATION
# ==============================================================================

def train_and_evaluate(df_test=None, has_target=True):
    """
    Train XGBoost on 100% of training data and evaluate on external test set.
    
    Args:
        df_test: Preprocessed test dataframe (if None, loads from file)
        has_target: Whether the test set contains the target column
    """
    print("\n" + "=" * 70)
    print("🔴 STEP 2: FULL TRAINING & EVALUATION")
    print("=" * 70)
    
    # 1. Load Training Data (100% - NO split!)
    print(f"\n📂 Loading training data from: {TRAIN_FILE}")
    start_time = time.time()
    
    df_train = pl.read_parquet(TRAIN_FILE)
    
    load_time = time.time() - start_time
    print(f"✅ Loaded: {len(df_train):,} rows × {len(df_train.columns)} columns")
    print(f"⏱️  Load time: {load_time:.2f} seconds")
    print(f"📊 Training columns: {df_train.columns}")
    
    # Check feature availability in training data
    train_features = [f for f in FEATURES if f in df_train.columns]
    print(f"\n📊 Features for training: {train_features}")
    
    # Prepare X_train and y_train (100% of training data!)
    X_train = df_train.select(train_features).to_numpy().astype(np.float32)
    y_train = df_train.select(TARGET).to_numpy().ravel().astype(np.float32)
    
    print(f"\n🏋️ Training set (100% - NO SPLIT!):")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    
    # Show class distribution
    n_positive = np.sum(y_train > 0)
    n_negative = np.sum(y_train == 0)
    print(f"\n📊 Class Distribution:")
    print(f"   Positive (bolus > 0): {n_positive:,} ({n_positive/len(y_train)*100:.1f}%)")
    print(f"   Negative (bolus = 0): {n_negative:,} ({n_negative/len(y_train)*100:.1f}%)")
    
    # Free memory
    del df_train
    gc.collect()
    
    # 2. Load Test Data (if not passed)
    if df_test is None:
        print(f"\n📂 Loading processed test data from: {PROCESSED_TEST_FILE}")
        df_test = pl.read_parquet(PROCESSED_TEST_FILE)
        has_target = TARGET in df_test.columns
    
    # Use same features that exist in both train and test
    test_features = [f for f in train_features if f in df_test.columns]
    X_test = df_test.select(test_features).to_numpy().astype(np.float32)
    
    if has_target:
        y_test = df_test.select(TARGET).to_numpy().ravel().astype(np.float32)
    
    print(f"\n📊 Test set:")
    print(f"   X_test shape: {X_test.shape}")
    if has_target:
        print(f"   y_test shape: {y_test.shape}")
    
    del df_test
    gc.collect()
    
    # 3. Train XGBoost
    print("\n" + "-" * 70)
    print("🚀 TRAINING XGBoost (GPU Accelerated)")
    print("-" * 70)
    
    print("\n⚙️ XGBoost Parameters:")
    for k, v in XGBOOST_PARAMS.items():
        print(f"   {k}: {v}")
    print(f"   num_boost_round: {NUM_ROUNDS}")
    
    # Create DMatrix
    print("\n📦 Creating DMatrix...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=test_features)
    dtest = xgb.DMatrix(X_test, feature_names=test_features)
    
    print(f"   ✓ Train DMatrix: {dtrain.num_row():,} rows")
    print(f"   ✓ Test DMatrix: {dtest.num_row():,} rows")
    
    # Train
    print("\n🏋️ Training started...")
    train_start = time.time()
    
    # Note: No evals for early stopping since we're doing final training
    # But we can still monitor with verbose
    evals = [(dtrain, 'train')]
    
    model = xgb.train(
        params=XGBOOST_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_ROUNDS,
        evals=evals,
        verbose_eval=100  # Print every 100 rounds
    )
    
    train_time = time.time() - train_start
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED!")
    print(f"⏱️  Training time: {train_time:.2f} seconds ({train_time/60:.1f} minutes)")
    print("=" * 70)
    
    # Save Model
    print(f"\n💾 Saving model to: {MODEL_OUTPUT}")
    model.save_model(MODEL_OUTPUT)
    model_size_mb = os.path.getsize(MODEL_OUTPUT) / (1024**2)
    print(f"✅ Model saved: {model_size_mb:.2f} MB")
    
    # 4. Evaluation
    print("\n" + "-" * 70)
    print("📊 MODEL EVALUATION")
    print("-" * 70)
    
    if has_target:
        # Generate predictions
        y_pred = model.predict(dtest)
        
        # Clip negative predictions (insulin can't be negative)
        y_pred = np.maximum(y_pred, 0)
        
        print(f"\n📊 Prediction Statistics:")
        print(f"   Min: {y_pred.min():.4f}")
        print(f"   Max: {y_pred.max():.4f}")
        print(f"   Mean: {y_pred.mean():.4f}")
        print(f"   Std: {y_pred.std():.4f}")
        
        # Global MAE & RMSE
        global_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        global_mae = mean_absolute_error(y_test, y_pred)
        
        print("\n" + "-" * 50)
        print("📈 GLOBAL METRICS (All Test Data)")
        print("-" * 50)
        print(f"   RMSE: {global_rmse:.4f}")
        print(f"   MAE:  {global_mae:.4f}")
        print(f"   Test samples: {len(y_test):,}")
        
        # Positive-Only MAE & RMSE
        positive_mask = y_test > 0
        y_test_positive = y_test[positive_mask]
        y_pred_positive = y_pred[positive_mask]
        
        if len(y_test_positive) > 0:
            positive_rmse = np.sqrt(mean_squared_error(y_test_positive, y_pred_positive))
            positive_mae = mean_absolute_error(y_test_positive, y_pred_positive)
            
            print("\n" + "-" * 50)
            print("⭐ POSITIVE-ONLY METRICS (bolus > 0)")
            print("-" * 50)
            print(f"   RMSE: {positive_rmse:.4f}")
            print(f"   MAE:  {positive_mae:.4f}  ← KEY METRIC!")
            print(f"   Positive samples: {len(y_test_positive):,}")
            print(f"   Mean actual bolus: {y_test_positive.mean():.4f}")
            print(f"   Mean predicted bolus: {y_pred_positive.mean():.4f}")
        else:
            print("\n⚠️ No positive samples in test set for positive-only metrics")
            positive_mae = None
            positive_rmse = None
        
        # Summary Report
        print("\n" + "=" * 70)
        print("📋 SUMMARY REPORT")
        print("=" * 70)
        print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  GLOBAL METRICS (All Test Data: {len(y_test):,} samples)
├─────────────────────────────────────────────────────────────────┤
│  RMSE:  {global_rmse:>10.4f}
│  MAE:   {global_mae:>10.4f}
└─────────────────────────────────────────────────────────────────┘
""")
        
        if positive_mae is not None:
            print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  ⭐ POSITIVE-ONLY METRICS (bolus > 0: {len(y_test_positive):,} samples)
├─────────────────────────────────────────────────────────────────┤
│  RMSE:  {positive_rmse:>10.4f}
│  MAE:   {positive_mae:>10.4f}  ← KEY METRIC
└─────────────────────────────────────────────────────────────────┘
""")
        
        print(f"🏋️ Training Time: {train_time:.1f} seconds")
        print(f"📦 Model Size: {model_size_mb:.2f} MB")
        
    else:
        # No labels - just generate predictions
        print("\n⚠️ Labels not found, generated predictions only.")
        
        y_pred = model.predict(dtest)
        y_pred = np.maximum(y_pred, 0)
        
        print(f"\n📊 Prediction Statistics:")
        print(f"   Total predictions: {len(y_pred):,}")
        print(f"   Min: {y_pred.min():.4f}")
        print(f"   Max: {y_pred.max():.4f}")
        print(f"   Mean: {y_pred.mean():.4f}")
        print(f"   Std: {y_pred.std():.4f}")
        
        # Save predictions
        pred_output = MODEL_OUTPUT.replace('.json', '_predictions.npy')
        np.save(pred_output, y_pred)
        print(f"\n💾 Predictions saved to: {pred_output}")
    
    # Feature Importance
    print("\n" + "-" * 70)
    print("📊 FEATURE IMPORTANCE (Gain)")
    print("-" * 70)
    
    importance = model.get_score(importance_type='gain')
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    max_score = max(importance.values()) if importance else 1
    for i, (feat, score) in enumerate(importance_sorted, 1):
        bar = '█' * int(score / max_score * 30)
        print(f"   {i}. {feat:20s} {score:>12.2f} {bar}")
    
    print("\n" + "=" * 70)
    print("🎉 FINAL MODEL TRAINING PIPELINE COMPLETED!")
    print("=" * 70)
    
    return model


# ==============================================================================
# 🚀 MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          🚀 FINAL MODEL TRAINING PIPELINE - Insulin Dosage Prediction         ║
║                                                                               ║
║  Step 1: External Test Data Preprocessing                                     ║
║  Step 2: Full Training & Evaluation                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n📍 Configuration:")
    print(f"   Mode: {'Google Colab' if USE_COLAB else 'Local Windows'}")
    print(f"   Raw Test: {RAW_TEST_FILE}")
    print(f"   Processed Test: {PROCESSED_TEST_FILE}")
    print(f"   Training Data: {TRAIN_FILE}")
    print(f"   Model Output: {MODEL_OUTPUT}")
    print(f"   Features: {len(FEATURES)} features")
    
    # GPU Check
    print("\n🔍 Checking GPU availability...")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract GPU name from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'Tesla' in line:
                    print(f"   ✅ GPU Found: {line.strip()}")
                    break
            else:
                print("   ✅ GPU Available (nvidia-smi successful)")
        else:
            print("   ⚠️ nvidia-smi failed, GPU may not be available")
    except Exception as e:
        print(f"   ⚠️ Could not check GPU: {e}")
    
    # Execute Pipeline
    start_total = time.time()
    
    # Step 1: Preprocess Test Data
    df_test, has_target = preprocess_test_data()
    
    # Step 2: Train and Evaluate
    model = train_and_evaluate(df_test, has_target)
    
    total_time = time.time() - start_total
    
    print(f"\n⏱️ Total Pipeline Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("\n✅ Pipeline execution completed successfully!")
