"""
🧹 Finalize Training Data - Toxic Filter & Sanitization
Creates "Gold Standard" dataset for XGBoost training

Filters:
- CGM: 39-400 mg/dL (physiological range)
- IOB: < 30 Units (biological maximum)
- Bolus: < 25 Units (single dose maximum)
- Carbs: < 200g (meal maximum)
- Removes NULL warm-up rows
"""

import polars as pl
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
# Adjust paths based on environment
try:
    # Colab
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = "/content/drive/MyDrive/DiabetesKDS/data"
except:
    # Local
    BASE_PATH = r"D:\Diabete\data"

INPUT_FILE = f"{BASE_PATH}/processed/master_train_v7_clean_core.parquet"
OUTPUT_FILE = f"{BASE_PATH}/processed/train_ready_for_xgboost_v7.parquet"

# ============================================================
# TOXIC DATA THRESHOLDS (Clinical Safety Rules)
# ============================================================
THRESHOLDS = {
    'CGM_MIN': 39,      # Sensor noise below this
    'CGM_MAX': 400,     # Sensor max
    'IOB_MAX': 30,      # Max biological IOB
    'BOLUS_MAX': 25,    # Max single dose
    'CARBS_MAX': 200,   # Max single meal
}

print("="*70)
print("🧹 FINALIZE TRAINING DATA - TOXIC FILTER")
print("="*70)
print(f"\n📁 Input:  {INPUT_FILE}")
print(f"📁 Output: {OUTPUT_FILE}")
print(f"\n🎯 Thresholds:")
for k, v in THRESHOLDS.items():
    print(f"   {k}: {v}")

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "="*70)
print("📂 STEP 1: LOADING DATA")
print("="*70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pl.read_parquet(INPUT_FILE)
original_count = len(df)
original_cols = df.columns

print(f"\n✅ Loaded: {original_count:,} rows, {len(original_cols)} columns")

# Pre-filter statistics
print("\n📊 Pre-Filter Statistics:")
stats_cols = ['CGM', 'iob', 'bolus', 'carbs']
for col in stats_cols:
    if col in df.columns:
        vals = df[col].drop_nulls()
        print(f"   {col}: min={vals.min():.2f}, max={vals.max():.2f}, nulls={df[col].null_count():,}")

# ============================================================
# APPLY TOXIC FILTERS
# ============================================================
print("\n" + "="*70)
print("🧹 STEP 2: APPLYING TOXIC FILTERS")
print("="*70)

filter_stats = {}

# Filter 1: CGM Range
print(f"\n1️⃣ CGM Filter ({THRESHOLDS['CGM_MIN']} <= CGM <= {THRESHOLDS['CGM_MAX']})...")
before = len(df)
df = df.filter(
    (pl.col('CGM') >= THRESHOLDS['CGM_MIN']) &
    (pl.col('CGM') <= THRESHOLDS['CGM_MAX'])
)
filter_stats['CGM Range'] = before - len(df)
print(f"   Removed: {filter_stats['CGM Range']:,} rows")

# Filter 2: IOB Maximum
print(f"\n2️⃣ IOB Filter (IOB < {THRESHOLDS['IOB_MAX']})...")
before = len(df)
if 'iob' in df.columns:
    df = df.filter(
        (pl.col('iob') < THRESHOLDS['IOB_MAX']) | pl.col('iob').is_null()
    )
filter_stats['IOB Range'] = before - len(df)
print(f"   Removed: {filter_stats['IOB Range']:,} rows")

# Filter 3: Bolus Maximum
print(f"\n3️⃣ Bolus Filter (bolus < {THRESHOLDS['BOLUS_MAX']})...")
before = len(df)
if 'bolus' in df.columns:
    df = df.filter(
        (pl.col('bolus') < THRESHOLDS['BOLUS_MAX']) | pl.col('bolus').is_null()
    )
filter_stats['Bolus Range'] = before - len(df)
print(f"   Removed: {filter_stats['Bolus Range']:,} rows")

# Filter 4: Carbs Maximum
print(f"\n4️⃣ Carbs Filter (carbs < {THRESHOLDS['CARBS_MAX']})...")
before = len(df)
if 'carbs' in df.columns:
    df = df.filter(
        (pl.col('carbs') < THRESHOLDS['CARBS_MAX']) | pl.col('carbs').is_null()
    )
filter_stats['Carbs Range'] = before - len(df)
print(f"   Removed: {filter_stats['Carbs Range']:,} rows")

# ============================================================
# HANDLE NULLS (Warm-up Period)
# ============================================================
print("\n" + "="*70)
print("🧹 STEP 3: REMOVING WARM-UP NULLS")
print("="*70)

# Drop rows with NULL clinical parameters (from rolling window warm-up)
null_cols = ['est_isf', 'est_icr', 'est_tdd', 'cgm_delta_15min']
before = len(df)

for col in null_cols:
    if col in df.columns:
        pre = len(df)
        df = df.filter(pl.col(col).is_not_null())
        dropped = pre - len(df)
        if dropped > 0:
            print(f"   Dropped {dropped:,} NULL rows from '{col}'")

filter_stats['Warm-up NULLs'] = before - len(df)
print(f"   Total warm-up removed: {filter_stats['Warm-up NULLs']:,} rows")

# ============================================================
# FINAL CHECKS
# ============================================================
print("\n" + "="*70)
print("🔍 STEP 4: FINAL CHECKS")
print("="*70)

# Check for negative values
print("\n1️⃣ Checking for negative values...")
for col in ['carbs', 'basal', 'bolus']:
    if col in df.columns:
        neg_count = df.filter(pl.col(col) < 0).height
        if neg_count > 0:
            print(f"   ⚠️ Found {neg_count} negative values in '{col}' - removing...")
            df = df.filter(pl.col(col) >= 0)
        else:
            print(f"   ✅ '{col}' has no negative values")

# Check for Unknown gender
print("\n2️⃣ Checking for Unknown gender...")
if 'gender_encoded' in df.columns:
    unknown_count = df.filter(pl.col('gender_encoded') == -1).height
    if unknown_count > 0:
        print(f"   ⚠️ Found {unknown_count} Unknown gender rows - removing...")
        df = df.filter(pl.col('gender_encoded') != -1)
    else:
        print(f"   ✅ No Unknown gender values")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("📊 CLEANUP SUMMARY")
print("="*70)

final_count = len(df)
total_removed = original_count - final_count
removal_pct = (total_removed / original_count) * 100

print(f"\n{'Filter':<25} {'Rows Removed':>15}")
print("-"*45)
for filter_name, count in filter_stats.items():
    print(f"{filter_name:<25} {count:>15,}")
print("-"*45)
print(f"{'TOTAL':<25} {total_removed:>15,}")

print(f"\n📈 Data Retention:")
print(f"   Original:  {original_count:,} rows")
print(f"   Final:     {final_count:,} rows")
print(f"   Removed:   {total_removed:,} rows ({removal_pct:.2f}%)")
print(f"   Retained:  {100 - removal_pct:.2f}%")

# Post-filter statistics
print("\n📊 Post-Filter Statistics:")
for col in stats_cols:
    if col in df.columns:
        vals = df[col].drop_nulls()
        print(f"   {col}: min={vals.min():.2f}, max={vals.max():.2f}")

# ============================================================
# SAVE OUTPUT
# ============================================================
print("\n" + "="*70)
print("💾 SAVING OUTPUT")
print("="*70)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.write_parquet(OUTPUT_FILE, compression='zstd')

if os.path.exists(OUTPUT_FILE):
    size_gb = os.path.getsize(OUTPUT_FILE) / (1024**3)
    print(f"\n✅ Saved: {OUTPUT_FILE}")
    print(f"   Size: {size_gb:.2f} GB")
    print(f"   Rows: {final_count:,}")
    print(f"   Columns: {len(df.columns)}")

# Final verification
print("\n📋 Final Schema:")
for col, dtype in df.schema.items():
    print(f"   {col}: {dtype}")

print("\n" + "="*70)
print("🎉 GOLD STANDARD DATASET READY FOR XGBOOST!")
print("="*70)
