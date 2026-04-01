"""
🔬 Comprehensive Toxicity Audit for Training Data
==================================================
Identify ALL potential "Toxic Data Patterns" where insulin was
administered incorrectly based on clinical logic.

Usage:
    python audit_all_toxicity.py
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
print("🔬 COMPREHENSIVE TOXICITY AUDIT")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    print(f"❌ File not found: {DATA_PATH}")
    exit(1)

print(f"\n📂 Loading: {DATA_PATH}")
df = pl.read_parquet(DATA_PATH)
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# DEFINE DANGER SCENARIOS
# ============================================================

print("\n" + "=" * 70)
print("🎯 DEFINING 4 DANGER SCENARIOS")
print("=" * 70)

scenarios = {
    "1. Falling & Low (CGM<110, Delta<-5, Carbs=0)": 
        (pl.col("CGM") < 110) & 
        (pl.col("cgm_delta_15min") < -5) & 
        (pl.col("carbs") == 0),
    
    "2. Hypo Hazard (CGM<70, Carbs=0)": 
        (pl.col("CGM") < 70) & 
        (pl.col("carbs") == 0),
    
    "3. Rapid Drop (Delta<-15, Carbs=0)": 
        (pl.col("cgm_delta_15min") < -15) & 
        (pl.col("carbs") == 0),
    
    "4. Stable Low (70<=CGM<100, |Delta|<3, Carbs=0)": 
        (pl.col("CGM") >= 70) & 
        (pl.col("CGM") < 100) & 
        (pl.col("cgm_delta_15min").abs() < 3) & 
        (pl.col("carbs") == 0)
}

# ============================================================
# AUDIT EACH SCENARIO
# ============================================================

total_toxic_count = 0
audit_results = []

for name, condition in scenarios.items():
    print(f"\n{'=' * 70}")
    print(f"📊 ANALYZING: {name}")
    print("=" * 70)
    
    try:
        subset = df.filter(condition)
        n_total = len(subset)
        
        if n_total == 0:
            print("   ℹ️ No rows match this condition.")
            audit_results.append({
                "scenario": name,
                "total": 0,
                "toxic": 0,
                "rate": 0
            })
            continue

        # Identify Toxic Rows (Bolus > 0)
        toxic_subset = subset.filter(pl.col("bolus") > 0)
        n_toxic = len(toxic_subset)
        rate = n_toxic / n_total * 100
        
        print(f"   Total Scenario Rows: {n_total:,}")
        print(f"   ✅ Safe Rows (Bolus=0): {n_total - n_toxic:,}")
        print(f"   ⚠️ TOXIC Rows (Bolus>0): {n_toxic:,} ({rate:.1f}%)")
        
        audit_results.append({
            "scenario": name,
            "total": n_total,
            "toxic": n_toxic,
            "rate": rate
        })
        
        if n_toxic > 0:
            max_bolus = toxic_subset['bolus'].max()
            mean_bolus = toxic_subset['bolus'].mean()
            
            print(f"\n   📈 Statistics:")
            print(f"      Max Bolus:  {max_bolus:.2f} U")
            print(f"      Mean Bolus: {mean_bolus:.4f} U")
            
            # Show sample toxic rows
            cols_to_show = ['CGM', 'cgm_delta_15min', 'carbs', 'bolus']
            available = [c for c in cols_to_show if c in toxic_subset.columns]
            
            print(f"\n   📋 Sample Toxic Rows (Top 5 by Bolus):")
            print("-" * 50)
            sample = toxic_subset.sort("bolus", descending=True).head(5)
            print(sample.select(available))
            
            total_toxic_count += n_toxic
        else:
            print("\n   ✅ No toxic rows in this scenario!")
            
    except Exception as e:
        print(f"   ❌ Error analyzing scenario: {e}")

# ============================================================
# SUMMARY REPORT
# ============================================================

print("\n" + "=" * 70)
print("📊 TOXICITY AUDIT SUMMARY")
print("=" * 70)

print("\n{:<50} {:>10} {:>10} {:>10}".format("Scenario", "Total", "Toxic", "Rate%"))
print("-" * 82)

for result in audit_results:
    print("{:<50} {:>10,} {:>10,} {:>9.1f}%".format(
        result["scenario"][:48],
        result["total"],
        result["toxic"],
        result["rate"]
    ))

print("-" * 82)
print(f"\n🚨 GRAND TOTAL TOXIC ROWS: {total_toxic_count:,}")
print(f"   (Out of {len(df):,} total rows = {total_toxic_count/len(df)*100:.3f}%)")

# ============================================================
# RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS")
print("=" * 70)

if total_toxic_count > 0:
    print(f"""
⚠️ TOXIC DATA DETECTED: {total_toxic_count:,} rows

These rows teach the model to recommend insulin in dangerous situations.

OPTIONS:

1. DATA CLEANING (Remove toxic rows before retraining):
   
   clean_df = df.filter(
       ~(
           # Remove Falling & Low toxic
           ((pl.col("CGM") < 110) & (pl.col("cgm_delta_15min") < -5) & 
            (pl.col("carbs") == 0) & (pl.col("bolus") > 0)) |
           
           # Remove Hypo Hazard toxic
           ((pl.col("CGM") < 70) & (pl.col("carbs") == 0) & 
            (pl.col("bolus") > 0)) |
           
           # Remove Rapid Drop toxic
           ((pl.col("cgm_delta_15min") < -15) & (pl.col("carbs") == 0) & 
            (pl.col("bolus") > 0)) |
           
           # Remove Stable Low toxic
           ((pl.col("CGM") >= 70) & (pl.col("CGM") < 100) & 
            (pl.col("cgm_delta_15min").abs() < 3) & 
            (pl.col("carbs") == 0) & (pl.col("bolus") > 0))
       )
   )
   
   clean_df.write_parquet("train_cleaned.parquet")

2. API SAFETY RULES (Faster fix, add to main.py):
   
   # In /predict/bolus endpoint, BEFORE Stage 1:
   if request.CGM < 70:
       return 0.0  # Hypo protection
   
   cgm_delta = request.CGM - request.prev_cgm_15 if request.prev_cgm_15 else 0
   if cgm_delta < -15 and request.carbs == 0:
       return 0.0  # Rapid drop protection

3. BOTH (Recommended): Clean data AND add safety rules.
""")
else:
    print("✅ No toxic data patterns detected. Training data looks clean!")

print("\n" + "=" * 70)
print("🏁 AUDIT COMPLETE")
print("=" * 70)
