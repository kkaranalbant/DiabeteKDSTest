"""
🧪 Finalize Test Data - Real World Evaluation
Creates sanitized test dataset preserving natural class imbalance

⚠️ CRITICAL: Unlike training set:
- NO downsampling (keep all zeros)
- NO shuffling (preserve time-series order)
- Only remove sensor errors/impossible values
"""

import polars as pl
import os

# ============================================================
# CONFIGURATION
# ============================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = "/content/drive/MyDrive/DiabetesKDS/data"
except:
    BASE_PATH = r"D:\Diabete\data"

INPUT_FILE = f"{BASE_PATH}/processed/master_test_v7_clean_core.parquet"
OUTPUT_FILE = f"{BASE_PATH}/processed/test_ready_real_world_v7.parquet"

THRESHOLDS = {
    'CGM_MIN': 39, 'CGM_MAX': 400,
    'IOB_MAX': 30, 'BOLUS_MAX': 25, 'CARBS_MAX': 250
}

print("="*70)
print("🧪 FINALIZE TEST DATA - REAL WORLD")
print("="*70)
print("\n⚠️ REAL-WORLD MODE:")
print("   ❌ NO downsampling")
print("   ❌ NO shuffling")
print("   ✅ Safety filters only")
print("   ✅ Preserve time-series order")

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "="*70)
print("📂 STEP 1: LOAD DATA")
print("="*70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pl.read_parquet(INPUT_FILE)
original = len(df)
print(f"\n✅ Loaded: {original:,} rows")

# Pre-filter distribution
action_pre = df.filter(pl.col('bolus') > 0).height
zero_pre = df.filter(pl.col('bolus') == 0).height
ratio_pre = zero_pre / max(1, action_pre)

print(f"\n📊 Pre-Filter Distribution:")
print(f"   🎯 Action (bolus > 0): {action_pre:,} ({action_pre/original*100:.2f}%)")
print(f"   ⚪ Zeros (bolus = 0):  {zero_pre:,} ({zero_pre/original*100:.2f}%)")
print(f"   📈 Real-world ratio:  1:{ratio_pre:.1f}")

# ============================================================
# APPLY SAFETY FILTERS ONLY
# ============================================================
print("\n" + "="*70)
print("🧹 STEP 2: SAFETY FILTERS (Sanitization Only)")
print("="*70)

# CGM Range
print(f"\n1️⃣ CGM Filter ({THRESHOLDS['CGM_MIN']} <= CGM <= {THRESHOLDS['CGM_MAX']})...")
before = len(df)
df = df.filter(
    (pl.col('CGM') >= THRESHOLDS['CGM_MIN']) &
    (pl.col('CGM') <= THRESHOLDS['CGM_MAX'])
)
print(f"   Removed: {before - len(df):,} rows")

# IOB Max
print(f"\n2️⃣ IOB Filter (IOB < {THRESHOLDS['IOB_MAX']})...")
before = len(df)
if 'iob' in df.columns:
    df = df.filter(
        (pl.col('iob') < THRESHOLDS['IOB_MAX']) | pl.col('iob').is_null()
    )
print(f"   Removed: {before - len(df):,} rows")

# Bolus Max
print(f"\n3️⃣ Bolus Filter (bolus < {THRESHOLDS['BOLUS_MAX']})...")
before = len(df)
df = df.filter(
    (pl.col('bolus') < THRESHOLDS['BOLUS_MAX']) | pl.col('bolus').is_null()
)
print(f"   Removed: {before - len(df):,} rows")

# Carbs Max
print(f"\n4️⃣ Carbs Filter (carbs < {THRESHOLDS['CARBS_MAX']})...")
before = len(df)
df = df.filter(
    (pl.col('carbs') < THRESHOLDS['CARBS_MAX']) | pl.col('carbs').is_null()
)
print(f"   Removed: {before - len(df):,} rows")

# Null warm-up periods
print("\n5️⃣ Removing NULL warm-up rows...")
before = len(df)
null_cols = ['est_isf', 'est_icr', 'est_tdd', 'cgm_delta_5min', 'cgm_delta_15min']
for col in null_cols:
    if col in df.columns:
        df = df.filter(pl.col(col).is_not_null())
print(f"   Removed: {before - len(df):,} rows")

filtered = len(df)
print(f"\n✅ After filters: {filtered:,} rows ({filtered/original*100:.1f}% retained)")

# ============================================================
# PRESERVE REALITY (NO BALANCE, NO SHUFFLE)
# ============================================================
print("\n" + "="*70)
print("🌍 STEP 3: PRESERVE REAL-WORLD ORDER")
print("="*70)

# Sort by patient and time (preserve order)
print("\n📐 Sorting by ['id', 'date']...")
df = df.sort(['id', 'date'])
print("   ✅ Time-series order preserved")

print("\n⚠️ NOT downsampling - keeping all zeros")
print("⚠️ NOT shuffling - preserving temporal order")

# ============================================================
# FINAL DISTRIBUTION
# ============================================================
print("\n" + "="*70)
print("📊 FINAL REAL-WORLD DISTRIBUTION")
print("="*70)

action_final = df.filter(pl.col('bolus') > 0).height
zero_final = df.filter(pl.col('bolus') == 0).height
total = len(df)
ratio_final = zero_final / max(1, action_final)

print(f"\n   🎯 Action (bolus > 0): {action_final:,} ({action_final/total*100:.2f}%)")
print(f"   ⚪ Zeros (bolus = 0):  {zero_final:,} ({zero_final/total*100:.2f}%)")
print(f"   📈 Imbalance ratio:   1:{ratio_final:.1f}")

if ratio_final > 20:
    print(f"\n   ✅ Ratio 1:{ratio_final:.0f} confirms REAL-WORLD distribution!")
else:
    print(f"\n   ⚠️ Ratio seems balanced - verify this is correct")

# ============================================================
# SAVE OUTPUT
# ============================================================
print("\n" + "="*70)
print("💾 STEP 4: SAVE OUTPUT")
print("="*70)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.write_parquet(OUTPUT_FILE, compression='zstd')

if os.path.exists(OUTPUT_FILE):
    size_gb = os.path.getsize(OUTPUT_FILE) / (1024**3)
    print(f"\n✅ Saved: {OUTPUT_FILE}")
    print(f"   Size: {size_gb:.2f} GB")
    print(f"   Rows: {total:,}")

# Summary
print("\n" + "="*70)
print("📋 SUMMARY")
print("="*70)
print(f"   Original:   {original:,} rows")
print(f"   Filtered:   {filtered:,} rows")
print(f"   Removed:    {original - filtered:,} ({(original-filtered)/original*100:.1f}%)")
print(f"\n   Action/Zero: {action_final:,} / {zero_final:,}")
print(f"   Ratio:       1:{ratio_final:.0f}")

print("\n🎉 REAL-WORLD TEST SET READY FOR EVALUATION!")
