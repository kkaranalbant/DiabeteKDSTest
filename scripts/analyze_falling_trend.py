"""
🔬 Training Data Audit: Falling Glucose Scenarios
==================================================
Investigate if model learned from "toxic" training examples
where insulin was given during low/falling glucose without food.

Usage:
    python analyze_falling_trend.py
"""

import polars as pl
import os

# ============================================================
# CONFIGURATION
# ============================================================

# Local path (Windows)
DATA_PATH = r"d:\Diabete\data\processed\train_augmented_fixed.parquet"

# Colab path (uncomment for Google Colab)
# DATA_PATH = "/content/drive/MyDrive/DiabetesKDS/data/processed/train_augmented_fixed.parquet"

print("=" * 70)
print("🔬 TRAINING DATA AUDIT: Falling Glucose Scenarios")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    print(f"❌ File not found: {DATA_PATH}")
    print("   Try updating DATA_PATH for your environment.")
    exit(1)

print(f"\n📂 Loading: {DATA_PATH}")
df = pl.read_parquet(DATA_PATH)
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Check available columns
print(f"\n📋 Columns: {df.columns}")

# ============================================================
# DEFINE DANGER SCENARIO
# ============================================================
print("\n" + "=" * 70)
print("🎯 DEFINING 'DANGER SCENARIO' FILTER")
print("=" * 70)

# Check if required columns exist
required_cols = ['CGM', 'cgm_delta_15min', 'carbs', 'bolus']
missing = [c for c in required_cols if c not in df.columns]

if missing:
    print(f"⚠️ Missing columns: {missing}")
    print(f"   Available columns: {df.columns}")
    
    # Try alternative column names
    if 'cgm_delta_15min' not in df.columns:
        # Calculate it if we have lagged values
        print("   Attempting to calculate cgm_delta_15min...")
        if 'CGM' in df.columns:
            df = df.with_columns(
                (pl.col('CGM').diff().fill_null(0)).alias('cgm_delta_15min')
            )

# Apply filters
print("""
Danger Scenario Definition:
  - CGM < 110 mg/dL (Low/Normal glucose)
  - cgm_delta_15min < -5 (Falling trend)
  - carbs == 0 (No food intake)
""")

try:
    scenario_df = df.filter(
        (pl.col("CGM") < 110) &
        (pl.col("cgm_delta_15min") < -5) &
        (pl.col("carbs") == 0)
    )
    
    total_rows = len(df)
    scenario_rows = len(scenario_df)
    pct = scenario_rows / total_rows * 100 if total_rows > 0 else 0
    
    print(f"🔍 SCENARIO ROWS: {scenario_rows:,} / {total_rows:,} ({pct:.2f}%)")

except Exception as e:
    print(f"❌ Filter error: {e}")
    print("   Trying with available columns...")
    scenario_df = df.filter(pl.col("bolus") > 0)

# ============================================================
# ANALYZE BOLUS BEHAVIOR
# ============================================================
print("\n" + "=" * 70)
print("📊 BOLUS ANALYSIS IN DANGER SCENARIO")
print("=" * 70)

if len(scenario_df) > 0:
    zero_bolus = scenario_df.filter(pl.col("bolus") == 0)
    positive_bolus = scenario_df.filter(pl.col("bolus") > 0)
    
    n_zero = len(zero_bolus)
    n_pos = len(positive_bolus)
    n_total = len(scenario_df)
    
    print(f"\n✅ Safe Rows (bolus=0):  {n_zero:,} ({n_zero/n_total*100:.1f}%)")
    print(f"⚠️ Toxic Rows (bolus>0): {n_pos:,} ({n_pos/n_total*100:.1f}%)")
    
    if n_pos > 0:
        print("\n" + "=" * 70)
        print("⚠️ TOXIC DATA SAMPLES (Model learned from these!)")
        print("=" * 70)
        
        # Get statistics
        max_bolus = positive_bolus['bolus'].max()
        mean_bolus = positive_bolus['bolus'].mean()
        
        print(f"\n   Max Bolus in falling trend:  {max_bolus:.2f} U")
        print(f"   Mean Bolus in falling trend: {mean_bolus:.4f} U")
        
        # Show top 10 worst offenders
        cols_to_show = ['CGM', 'cgm_delta_15min', 'carbs', 'bolus']
        extra_cols = ['age', 'weight', 'basal']
        for col in extra_cols:
            if col in positive_bolus.columns:
                cols_to_show.append(col)
        
        print(f"\n📋 TOP 10 HIGHEST BOLUS IN FALLING TREND:")
        print("-" * 60)
        
        top_toxic = positive_bolus.sort("bolus", descending=True).head(10)
        available_cols = [c for c in cols_to_show if c in top_toxic.columns]
        print(top_toxic.select(available_cols))
        
        # Distribution
        print(f"\n📊 BOLUS DISTRIBUTION IN TOXIC SUBSET:")
        print("-" * 40)
        bolus_counts = positive_bolus.group_by(
            pl.col("bolus").round(0).cast(pl.Int32).alias("bolus_rounded")
        ).agg(pl.count().alias("count")).sort("bolus_rounded")
        print(bolus_counts.head(15))
        
    else:
        print("\n✅ GOOD NEWS: No toxic examples found!")
        print("   All falling glucose + no food scenarios have bolus=0")
        
else:
    print("ℹ️ No rows matched the danger scenario filter.")

# ============================================================
# ADDITIONAL ANALYSIS: Overall Class Distribution
# ============================================================
print("\n" + "=" * 70)
print("📊 OVERALL DATASET CLASS DISTRIBUTION")
print("=" * 70)

n_zero_total = len(df.filter(pl.col("bolus") == 0))
n_pos_total = len(df.filter(pl.col("bolus") > 0))
ratio = n_zero_total / n_pos_total if n_pos_total > 0 else 0

print(f"   Zero bolus:     {n_zero_total:,} ({n_zero_total/len(df)*100:.1f}%)")
print(f"   Positive bolus: {n_pos_total:,} ({n_pos_total/len(df)*100:.1f}%)")
print(f"   Imbalance Ratio: {ratio:.2f}")

# ============================================================
# RECOMMENDATION
# ============================================================
print("\n" + "=" * 70)
print("💡 RECOMMENDATION")
print("=" * 70)

if n_pos > 0:
    toxicity_rate = n_pos / n_total * 100
    if toxicity_rate > 10:
        print(f"""
⚠️ HIGH TOXICITY DETECTED ({toxicity_rate:.1f}%)

The model learned from {n_pos:,} examples where insulin was given
during LOW/FALLING glucose WITHOUT food. This is dangerous!

Options to Fix:
1. DATA CLEANING: Remove these rows from training data
2. RULE OVERRIDE: Add hard rule in API: IF cgm_delta < -5 AND CGM < 100 → dose=0
3. RETRAIN: Use cleaned dataset

Suggested Cleaning Code:
    clean_df = df.filter(
        ~((pl.col("CGM") < 110) & 
          (pl.col("cgm_delta_15min") < -5) & 
          (pl.col("carbs") == 0) & 
          (pl.col("bolus") > 0))
    )
""")
    else:
        print(f"   Low toxicity rate ({toxicity_rate:.1f}%). Consider adding safety rule.")
else:
    print("   ✅ Dataset looks clean for this scenario.")

print("\n" + "=" * 70)
print("🏁 ANALYSIS COMPLETE")
print("=" * 70)
