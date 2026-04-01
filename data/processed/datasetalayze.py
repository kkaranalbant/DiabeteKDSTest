# ===========================================================
# 1. KÜTÜPHANELER
# ===========================================================
# Eğer yüklü değilse terminalden şunu çalıştır:
# pip install polars plotly pandas pyarrow
import os
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===========================================================
# 2. YEREL DOSYA YOLLARI (WINDOWS)
# ===========================================================
# r"..." kullanarak Windows path hatalarını önlüyoruz
TRAIN_PATH = r"D:\Diabete\data\raw\metabonet_public_train.parquet"
TEST_PATH = r"D:\Diabete\data\processed\test_ready_for_xgboost.parquet"

print(f"📂 Train Hedef: {TRAIN_PATH}")
print(f"📂 Test Hedef:  {TEST_PATH}")


# ===========================================================
# 3. ANALİZ MOTORU
# ===========================================================

def analyze_dataset(file_path, name):
    print(f"\n{'=' * 60}")
    print(f"🕵️‍♂️ ANALİZ EDİLİYOR: {name}")
    print(f"{'=' * 60}")

    if not os.path.exists(file_path):
        print(f"❌ HATA: Dosya bulunamadı! Lütfen yolu kontrol et: {file_path}")
        return None

    # Load Data (Polars ile yerel diskten okuma)
    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f"❌ Okuma Hatası: {e}")
        return None

    print(f"✅ Boyut: {df.shape[0]:,} satır | {df.shape[1]} sütun")
    print(f"📋 Sütunlar: {df.columns}")

    # Hedef Değişken Kontrolü (Bolus var mı?)
    target_col = 'bolus'
    if target_col not in df.columns:
        print(f"⚠️ UYARI: '{target_col}' sütunu bulunamadı! İnsülin verisi bu dosyada yok.")
        return df

    # --- KRİTİK BÖLÜM: ZERO-INFLATION ANALİZİ ---
    total_rows = len(df)
    zeros = df.filter(pl.col(target_col) == 0).height
    positives = df.filter(pl.col(target_col) > 0).height

    # Sıfıra bölme hatasını önle
    if total_rows > 0:
        zero_ratio = (zeros / total_rows) * 100
        pos_ratio = (positives / total_rows) * 100
    else:
        zero_ratio, pos_ratio = 0, 0

    ratio_text = f"{zeros / positives:.1f}" if positives > 0 else "Sonsuz"

    print(f"\n⚖️ SINIF DENGESİ (Class Imbalance):")
    print(f"   🔴 SIFIR Dozlar (No Bolus):  {zeros:,} ({zero_ratio:.2f}%)")
    print(f"   🟢 POZİTİF Dozlar (Bolus):   {positives:,} ({pos_ratio:.2f}%)")
    print(f"   👉 Dengesizlik Oranı: 1 Pozitif örneğe karşılık {ratio_text} tane Sıfır var!")

    # --- İSTATİSTİKSEL ANALİZ (Sadece Pozitifler) ---
    if positives > 0:
        print(f"\n📊 İSTATİSTİKLER (Sadece Pozitif Dozlar İçin):")
        pos_df = df.filter(pl.col(target_col) > 0)
        stats = pos_df.select([
            pl.col(target_col).min().alias('Min'),
            pl.col(target_col).mean().alias('Ortalama'),
            pl.col(target_col).median().alias('Medyan'),
            pl.col(target_col).max().alias('Max'),
            pl.col(target_col).std().alias('Std Sapma')
        ])
        print(stats)
    else:
        print("\n⚠️ Verisetinde hiç pozitif bolus değeri yok!")

    # --- KORELASYON KONTROLÜ (Carbs vs Bolus) ---
    if 'carbs' in df.columns and positives > 0:
        corr = df.select(pl.corr("carbs", "bolus")).item()
        print(f"\n🔗 Carbs vs Bolus Korelasyonu: {corr:.4f}")
        if corr < 0.5:
            print("   ⚠️ UYARI: Karbonhidrat ile İnsülin arasındaki ilişki zayıf!")

    return df


# ===========================================================
# 4. ÇALIŞTIRMA VE GÖRSELLEŞTİRME
# ===========================================================

print("⏳ Veriler işleniyor...")

# Train Analizi
df_train = analyze_dataset(TRAIN_PATH, "EĞİTİM SETİ (TRAIN)")

# Test Analizi
df_test = analyze_dataset(TEST_PATH, "TEST SETİ")

# Görselleştirme (Eğer Train yüklendiyse)
if df_train is not None and 'bolus' in df_train.columns:
    print("\n🎨 Grafikler oluşturuluyor (İlk 50.000 satır örneklem)...")

    # Polars -> Pandas (Sadece grafik için küçük bir parça)
    sample = df_train[:50000].to_pandas()

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Bolus Dağılımı (Logaritmik)", "Carbs vs Bolus İlişkisi"))

    # 1. Histogram
    fig.add_trace(go.Histogram(x=sample['bolus'], name='Bolus', nbinsx=30), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1, title="Kayıt Sayısı (Log)")

    # 2. Scatter (Sadece pozitifler)
    pos_sample = sample[sample['bolus'] > 0]
    if not pos_sample.empty:
        fig.add_trace(go.Scatter(
            x=pos_sample['carbs'],
            y=pos_sample['bolus'],
            mode='markers',
            marker=dict(opacity=0.5, size=5, color='orange'),
            name='Bolus > 0'
        ), row=1, col=2)
        fig.update_xaxes(title="Karbonhidrat (g)", row=1, col=2)
        fig.update_yaxes(title="İnsülin (Ünite)", row=1, col=2)

    fig.update_layout(title_text="Veri Seti Dengesi ve İlişkisi", showlegend=False)
    fig.show()