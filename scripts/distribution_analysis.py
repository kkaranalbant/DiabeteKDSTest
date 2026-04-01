"""
📊 Train vs Validation Distribution Analysis
=============================================
Analyze distribution differences between training and validation sets.

Usage:
    python distribution_analysis.py
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Local paths
TRAIN_PATH = r"D:\Diabete\data\processed\train_augmented_fixed.parquet"
VAL_PATH = r"d:\Diabete\data\processed\test_ready_for_xgboost.parquet"

# Colab paths (uncomment for Google Colab)
# TRAIN_PATH = "/content/drive/MyDrive/DiabetesKDS/data/processed/train_pure_v5.parquet"
# VAL_PATH = "/content/drive/MyDrive/DiabetesKDS/data/processed/test_ready_for_xgboost.parquet"

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

print("=" * 60)
print("📊 TRAIN vs VALIDATION DISTRIBUTION ANALYSIS")
print("=" * 60)

if not os.path.exists(TRAIN_PATH):
    print(f"❌ Train file not found: {TRAIN_PATH}")
    exit(1)
if not os.path.exists(VAL_PATH):
    print(f"❌ Val file not found: {VAL_PATH}")
    exit(1)

df_train = pl.read_parquet(TRAIN_PATH)
df_val = pl.read_parquet(VAL_PATH)

print(f"\n📂 Veri Boyutları: Train={len(df_train):,}, Val={len(df_val):,}")

# ==============================================================================
# 2. TARGET (BOLUS) DISTRIBUTION ANALYSIS
# ==============================================================================

def analyze_target(df, name):
    zeros = df.filter(pl.col("bolus") == 0).shape[0]
    positives = df.filter(pl.col("bolus") > 0).shape[0]
    print(f"\n🎯 {name} Hedef Dağılımı:")
    print(f"   Sıfır Değerler: {zeros:,} ({zeros/len(df)*100:.1f}%)")
    print(f"   Pozitif Dozlar: {positives:,} ({positives/len(df)*100:.1f}%)")
    print(f"   Ortalama Doz:  {df['bolus'].mean():.4f}")
    print(f"   Maksimum Doz:  {df['bolus'].max():.2f}")
    return zeros, positives

print("\n" + "=" * 60)
print("📋 HEDEF DEĞİŞKEN ANALİZİ")
print("=" * 60)

train_zeros, train_pos = analyze_target(df_train, "TRAIN")
val_zeros, val_pos = analyze_target(df_val, "VALIDATION")

# ==============================================================================
# 3. FEATURE MISMATCH ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("📉 ÖZELLİK KARŞILAŞTIRMASI (Mean Values)")
print("=" * 60)

features = ['CGM', 'carbs', 'cgm_delta_15min', 'age']
available_features = [f for f in features if f in df_train.columns and f in df_val.columns]

print(f"\n{'Feature':15} | {'Train':>10} | {'Val':>10} | {'Diff %':>10}")
print("-" * 52)

for f in available_features:
    t_mean = df_train[f].mean()
    v_mean = df_val[f].mean()
    diff = abs(t_mean - v_mean) / t_mean * 100 if t_mean != 0 else 0
    flag = "⚠️" if diff > 20 else ""
    print(f"{f:15} | {t_mean:>10.2f} | {v_mean:>10.2f} | {diff:>9.1f}% {flag}")

# ==============================================================================
# 4. CORRELATION ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("🔗 KORELASYON KARŞILAŞTIRMASI (Feature vs Bolus)")
print("=" * 60)

print(f"\n{'Feature':15} | {'Train Corr':>12} | {'Val Corr':>12} | {'Diff':>10}")
print("-" * 56)

for f in available_features:
    try:
        t_corr = np.corrcoef(df_train[f].to_numpy(), df_train['bolus'].to_numpy())[0, 1]
        v_corr = np.corrcoef(df_val[f].to_numpy(), df_val['bolus'].to_numpy())[0, 1]
        diff = abs(t_corr - v_corr)
        flag = "⚠️" if diff > 0.1 else ""
        print(f"{f:15} | {t_corr:>12.3f} | {v_corr:>12.3f} | {diff:>9.3f} {flag}")
    except Exception as e:
        print(f"{f:15} | Error: {e}")

# ==============================================================================
# 5. VISUAL ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("📈 GÖRSEL ANALİZ OLUŞTURULUYOR...")
print("=" * 60)

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Bolus Distribution KDE
ax1 = axes[0, 0]
sns.kdeplot(df_train['bolus'].to_numpy(), label='Train', fill=True, ax=ax1)
sns.kdeplot(df_val['bolus'].to_numpy(), label='Val', fill=True, ax=ax1)
ax1.set_title("Bolus Dağılım Farkı (KDE)")
ax1.set_xlabel("Bolus (U)")
ax1.legend()
ax1.set_xlim(0, 15)

# 2. CGM Distribution Boxplot
ax2 = axes[0, 1]
box_data = [df_train['CGM'].to_numpy(), df_val['CGM'].to_numpy()]
ax2.boxplot(box_data, labels=['Train CGM', 'Val CGM'])
ax2.set_title("Glikoz Ölçek Farkı")
ax2.set_ylabel("CGM (mg/dL)")

# 3. Carbs Distribution
ax3 = axes[1, 0]
if 'carbs' in df_train.columns:
    ax3.hist(df_train['carbs'].to_numpy(), bins=30, alpha=0.5, label='Train', color='blue')
    ax3.hist(df_val['carbs'].to_numpy(), bins=30, alpha=0.5, label='Val', color='red')
    ax3.set_title("Karbonhidrat Dağılımı")
    ax3.set_xlabel("Carbs (g)")
    ax3.legend()

# 4. Zero vs Positive Pie Charts
ax4 = axes[1, 1]
train_ratio = train_pos / (train_zeros + train_pos) * 100
val_ratio = val_pos / (val_zeros + val_pos) * 100
x = ['Train', 'Val']
positive_pcts = [train_ratio, val_ratio]
zero_pcts = [100 - train_ratio, 100 - val_ratio]
width = 0.35
ax4.bar(x, positive_pcts, width, label='Positive Bolus', color='coral')
ax4.bar(x, zero_pcts, width, bottom=positive_pcts, label='Zero Bolus', color='steelblue')
ax4.set_ylabel('%')
ax4.set_title('Bolus Sınıf Dağılımı')
ax4.legend()

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=150)
plt.show()
print("✅ Görsel kaydedildi: distribution_analysis.png")

# ==============================================================================
# 6. SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("📋 ÖZET")
print("=" * 60)

train_zero_ratio = train_zeros / len(df_train) * 100
val_zero_ratio = val_zeros / len(df_val) * 100

print(f"""
HEDEF DAĞILIM:
   Train Zero Ratio: {train_zero_ratio:.1f}%
   Val Zero Ratio:   {val_zero_ratio:.1f}%
   Ratio Difference: {abs(train_zero_ratio - val_zero_ratio):.1f}%

ORTALAMA BOLUS:
   Train: {df_train['bolus'].mean():.4f}
   Val:   {df_val['bolus'].mean():.4f}

⚠️ OVERFITTING NEDENİ:
""")

if val_zero_ratio > train_zero_ratio + 5:
    print("   → Val'da daha fazla sıfır var - Model sıfır tahmin ederek düşük MAE alıyor")
if df_val['bolus'].mean() < df_train['bolus'].mean() * 0.8:
    print("   → Val ortalama bolus daha düşük - Iteration 0'da bile iyi MAE")

print("\n" + "=" * 60)
