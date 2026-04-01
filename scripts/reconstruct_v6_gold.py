"""
🔬 Reconstruct Balanced Master Dataset v6
==========================================
Create a "World" that teaches the model logic, not noise.

Strategy:
1. Clean both Train + Test (DROP toxic rows)
2. Create balanced pools (50% Action, 50% Safe)
3. Special focus on 70-110 mg/dL range

Output:
- master_train_v6_balanced.parquet
- master_val_v6.parquet

Usage (Google Colab):
  !python reconstruct_v6_gold.py
"""

import polars as pl
import os
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_PATH = "/content/drive/MyDrive/DiabetesKDS/data"
RAW_PATH = f"{BASE_PATH}/raw"
OUTPUT_PATH = f"{BASE_PATH}/processed"

FILES = {
    "train": f"{RAW_PATH}/metabonet_public_train.parquet",
    "test": f"{RAW_PATH}/metabonet_public_test.parquet",
    "output_train": f"{OUTPUT_PATH}/master_train_v6_balanced.parquet",
    "output_val": f"{OUTPUT_PATH}/master_val_v6.parquet"
}

# Target sizes
TARGET_ACTION_POOL = 23_000_000  # 23M action rows
VAL_SIZE = 1_000_000  # 1M validation rows

print("=" * 70)
print("🔬 RECONSTRUCT BALANCED MASTER DATASET v6")
print("=" * 70)

# ==============================================================================
# STEP 1: CLEAN & SCAN
# ==============================================================================

print("\n📂 STEP 1: Loading and Cleaning Data...")
print("-" * 50)

def get_toxic_mask():
    """Define toxic scenarios to DROP."""
    return (
        (pl.col("CGM") < 110) & 
        (pl.col("carbs") == 0) & 
        (pl.col("bolus") > 0)
    )

# Columns to keep
COLS = ['CGM', 'bolus', 'carbs', 'basal', 'age', 'weight', 'time_sin', 'time_cos', 
        'cgm_delta_5min', 'cgm_delta_15min', 'is_weekend', 'gender_encoded', 'insulin_type_encoded']

def scan_and_clean(path, label):
    """Scan parquet, select columns, clean toxic rows."""
    print(f"\n   📂 Scanning {label}...")
    
    lf = pl.scan_parquet(path)
    
    # Get available columns
    available = [c for c in COLS if c in lf.columns]
    print(f"      Available columns: {len(available)}")
    
    # Select and clean
    lf_clean = (
        lf
        .select(available)
        .filter(~get_toxic_mask())  # DROP toxic rows
    )
    
    return lf_clean, available

# Scan both files
print("\n   Scanning train...")
lf_train, train_cols = scan_and_clean(FILES["train"], "TRAIN")

print("\n   Scanning test...")
lf_test, test_cols = scan_and_clean(FILES["test"], "TEST")

# Use common columns
common_cols = list(set(train_cols) & set(test_cols))
print(f"\n   Common columns: {common_cols}")

# ==============================================================================
# STEP 2: CREATE BALANCED POOLS (STREAMING)
# ==============================================================================

print("\n📊 STEP 2: Creating Balanced Pools...")
print("-" * 50)

# Concatenate both lazy frames
print("   Concatenating train + test (lazy)...")
lf_all = pl.concat([
    lf_train.select(common_cols),
    lf_test.select(common_cols)
])

# Count total rows (streaming)
print("   Counting total rows (streaming)...")
total_count = lf_all.select(pl.count()).collect(streaming=True).item()
print(f"   Total clean rows: {total_count:,}")

# Pool Action: bolus > 0
print("\n   🎯 Creating Pool_Action (bolus > 0)...")
pool_action = (
    lf_all
    .filter(pl.col("bolus") > 0)
    .collect(streaming=True)
)
n_action = len(pool_action)
print(f"      Pool_Action size: {n_action:,}")

# Pool Safe: bolus == 0, balanced sampling
print("\n   🛡️ Creating Pool_Safe (bolus == 0)...")

# Get all safe rows
pool_safe_all = (
    lf_all
    .filter(pl.col("bolus") == 0)
    .collect(streaming=True)
)
n_safe_all = len(pool_safe_all)
print(f"      Total safe rows: {n_safe_all:,}")

# Target: same size as action pool
target_safe = min(n_action, n_safe_all)

# Split safe pool by CGM range
print(f"      Balancing by CGM range...")

# Low CGM safe rows (CGM < 120) - 50% of safe pool
pool_safe_low = pool_safe_all.filter(pl.col("CGM") < 120)
n_safe_low = len(pool_safe_low)

# High CGM safe rows (CGM >= 120) - 50% of safe pool
pool_safe_high = pool_safe_all.filter(pl.col("CGM") >= 120)
n_safe_high = len(pool_safe_high)

print(f"      Safe Low CGM (<120): {n_safe_low:,}")
print(f"      Safe High CGM (≥120): {n_safe_high:,}")

# Sample 50% from each range
target_low = target_safe // 2
target_high = target_safe - target_low

# Ensure we don't exceed available
sample_low = min(target_low, n_safe_low)
sample_high = min(target_high, n_safe_high)

print(f"      Sampling: {sample_low:,} low + {sample_high:,} high")

# Sample from each pool
if sample_low > 0:
    pool_safe_low_sampled = pool_safe_low.sample(n=sample_low, seed=42)
else:
    pool_safe_low_sampled = pool_safe_low

if sample_high > 0:
    pool_safe_high_sampled = pool_safe_high.sample(n=sample_high, seed=42)
else:
    pool_safe_high_sampled = pool_safe_high

# Combine safe pools
pool_safe = pl.concat([pool_safe_low_sampled, pool_safe_high_sampled])
n_safe = len(pool_safe)
print(f"      Final Pool_Safe size: {n_safe:,}")

# ==============================================================================
# STEP 3: MERGE & SHUFFLE
# ==============================================================================

print("\n🔀 STEP 3: Merging and Shuffling...")
print("-" * 50)

# Concatenate action + safe
print("   Concatenating pools...")
df_master = pl.concat([pool_action, pool_safe])
print(f"   Combined size: {len(df_master):,}")

# Shuffle
print("   Shuffling...")
df_master = df_master.sample(fraction=1.0, seed=42, shuffle=True)

# Cast to Float32
print("   Casting to Float32...")
float_cols = [c for c in df_master.columns if c not in ['is_weekend']]
df_master = df_master.cast({c: pl.Float32 for c in float_cols if c in df_master.columns})

# ==============================================================================
# STEP 4: SPLIT & EXPORT
# ==============================================================================

print("\n💾 STEP 4: Splitting and Exporting...")
print("-" * 50)

# Split validation
val_size = min(VAL_SIZE, len(df_master) // 10)
print(f"   Splitting validation set: {val_size:,} rows")

df_val = df_master.head(val_size)
df_train = df_master.slice(val_size)

print(f"   Train size: {len(df_train):,}")
print(f"   Val size: {len(df_val):,}")

# Export
print(f"\n   Saving train to: {FILES['output_train']}")
df_train.write_parquet(FILES['output_train'])

print(f"   Saving val to: {FILES['output_val']}")
df_val.write_parquet(FILES['output_val'])

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("📋 RECONSTRUCTION SUMMARY")
print("=" * 70)

# Balance check
train_action = len(df_train.filter(pl.col("bolus") > 0))
train_safe = len(df_train.filter(pl.col("bolus") == 0))

print(f"""
INPUT:
   Original Train + Test: {total_count:,} clean rows (after toxic removal)

OUTPUT:
   Master Train: {len(df_train):,} rows
   Master Val:   {len(df_val):,} rows

BALANCE CHECK (Train):
   Action (bolus > 0): {train_action:,} ({train_action/len(df_train)*100:.1f}%)
   Safe (bolus == 0):  {train_safe:,} ({train_safe/len(df_train)*100:.1f}%)

CGM RANGE IN SAFE POOL:
   Low CGM (<120):  {sample_low:,} ({sample_low/(sample_low+sample_high)*100:.1f}%)
   High CGM (≥120): {sample_high:,} ({sample_high/(sample_low+sample_high)*100:.1f}%)

FILES CREATED:
   ✅ {FILES['output_train']}
   ✅ {FILES['output_val']}
""")

print("=" * 70)
print("🎉 RECONSTRUCTION COMPLETE!")
print("=" * 70)
