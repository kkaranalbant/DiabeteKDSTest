# 🩸 DiabetAI - Klinik Karar Destek Sistemi (Test Sürümü)

Diyabet hastaları için **kan şekeri tahmini** ve **insülin dozaj önerisi** yapan yapay zeka destekli karar destek sistemi.

> ⚠️ **Bu sistem bir KARAR DESTEK aracıdır. Otomatik insülin verme sistemi DEĞİLDİR.**
> Önerilen dozlar her zaman sağlık profesyoneli tarafından doğrulanmalıdır.

---

## 🏗️ Mimari: Two-Stage Pipeline

```
📥 Hasta Girdileri (CGM, Carbs, IOB, Trend, Yaş, Kilo...)
       ↓
🧠 Stage 1: Glucose Forecaster (Hybrid Ensemble XGBoost)
       ↓ → 30 dakika sonraki kan şekeri tahmini
🧠 Stage 2: Insulin Dosage Engine (Physics-Informed XGBoost v10)
       ↓ → Ham insülin dozu önerisi
🛡️ Safety Guardrails (Hypo Shield, Meal Cap, Correction Trust)
       ↓
✅ Final Güvenli Doz Önerisi
```

## 🚀 Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI_ADI/DiabeteKDSTest.git
cd DiabeteKDSTest

# 2. Sanal ortam oluştur (önerilen)
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

## 🧪 Modeli Test Etme

### 1. Komut Satırı (Hızlı Test)

```bash
# Tek hasta bilgisi ile test
python test_model.py --cgm 180 --carbs 40 --iob 1.0 --trend 2 --isf 30 --icr 10

# İnteraktif mod (adım adım sorular)
python test_model.py
```

### 2. Toplu Test (Batch - CSV)

Hastaneden gelen verileri CSV formatına dönüştürüp toplu test yapabilirsiniz:

```bash
python test_model.py --batch hastalar.csv --output sonuclar.csv
```

**CSV Formatı** (`hastalar.csv`):
```csv
hasta_id,cgm,carbs,iob,trend,basal,isf,icr,age,weight,gender,gercek_doz,gercek_glukoz_30dk
H001,180,40,1.0,2.0,1.0,30,10,45,75,Male,5.5,210
H002,90,0,0.5,-1.0,0.8,35,12,60,68,Female,0,85
```

| Sütun | Açıklama | Birim |
|-------|----------|-------|
| `hasta_id` | Benzersiz hasta tanımlayıcı | - |
| `cgm` | Anlık kan şekeri ölçümü | mg/dL |
| `carbs` | Alınan karbonhidrat miktarı | gram |
| `iob` | Aktif insülin (Insulin on Board) | Ünite |
| `trend` | Kan şekeri değişim hızı (5 dk'lık) | mg/dL/5dk |
| `basal` | Bazal insülin oranı | U/saat |
| `isf` | İnsülin duyarlılık faktörü | mg/dL/U |
| `icr` | İnsülin/karbonhidrat oranı | g/U |
| `age` | Yaş | yıl |
| `weight` | Kilo | kg |
| `gender` | Cinsiyet | Male/Female |
| `gercek_doz` | Gerçekte verilen insülin dozu (karşılaştırma için) | Ünite |
| `gercek_glukoz_30dk` | 30 dk sonra ölçülen gerçek glukoz (karşılaştırma için) | mg/dL |

> 💡 `gercek_doz` ve `gercek_glukoz_30dk` sütunları opsiyoneldir. Doldurulursa model doğruluk metrikleri hesaplanır.

### 3. Streamlit Web Arayüzü

```bash
streamlit run app.py
```

Tarayıcıda interaktif bir arayüz açılır. Sol panelden hasta bilgilerini girip sonuçları görsel olarak takip edebilirsiniz.

## 📊 Çıktılar

### Sonuç CSV (`sonuclar.csv`)
Batch test sonucunda her hasta için şu bilgiler üretilir:

- **`tahmin_glukoz_30dk`**: AI modelin 30 dk sonrası için kan şekeri tahmini
- **`final_doz`**: Güvenlik kontrolleri uygulanmış insülin doz önerisi
- **`matematik_doz`**: Klinik formül ile hesaplanan doz (referans)
- **`doz_farki`**: Gerçek doz - Model dozu (doğruluk ölçümü)
- **`glukoz_farki`**: Gerçek glukoz - Tahmin (doğruluk ölçümü)

### Özet İstatistikler
Batch test sonunda konsola otomatik olarak yazdırılır:
- **MAE (Mean Absolute Error)**: Ortalama mutlak hata
- **Ortalama Sapma**: Modelin sistematik sapması (bias)
- **Max Sapma**: En büyük hata

## 🛡️ Güvenlik Kuralları

| Kural | Koşul | Aksiyon |
|-------|-------|---------|
| **Hypo Shield** | CGM < 70 mg/dL | Doz = 0 (acil durum) |
| **Meal Cap** | Karbonhidrat > 0 | AI dozu ≤ Matematik × 1.30 |
| **Correction Trust** | Carbs = 0, CGM > 100 | AI dozuna güven |
| **Max Limit** | Her durumda | Doz ≤ 15U |

## 📂 Proje Yapısı

```
DiabeteKDSTest/
├── README.md                 # Bu dosya
├── requirements.txt          # Python bağımlılıkları
├── .gitignore                # Git dışında tutulan dosyalar
├── test_model.py             # 🧪 Ekip test aracı (CLI + Batch)
├── ornek_hastalar.csv        # Örnek test CSV dosyası
├── app.py                    # Streamlit web arayüzü (v11)
├── inference.py              # İnsülin doz tahmini motoru (v10)
├── forecaster.py             # Kan şekeri tahmin motoru (Ensemble)
├── main.py                   # FastAPI backend (v8)
├── dashboard.py              # Ek dashboard arayüzü
├── config.py                 # Yapılandırma dosyası
├── models/
│   ├── xgboost_clinical_v10.json       # İnsülin dozaj modeli
│   ├── ensemble_model_a_absolute.json  # Glukoz tahmin - Model A
│   ├── ensemble_model_b_residual.json  # Glukoz tahmin - Model B
│   ├── model_config_v10.json           # Model konfigürasyonu
│   └── ensemble_config_v5.json         # Ensemble konfigürasyonu
├── scripts/                  # Yardımcı script'ler
├── notebooks/                # Colab eğitim notebook'ları
└── docs/                     # Ek dokümantasyon
```

## 🔬 Model Bilgileri

| Model | Görevi | Özellik Sayısı |
|-------|--------|----------------|
| **XGBoost v10** | İnsülin doz tahmini | 20 (Physics-Informed) |
| **Ensemble A** | Glukoz tahmini (Stabilizer) | 11 |
| **Ensemble B** | Glukoz tahmini (Physics Residual) | 11 |

## 🛠️ Teknoloji

- **ML Framework**: XGBoost 2.0
- **Web UI**: Streamlit
- **API**: FastAPI
- **Data**: NumPy, Pandas
- **Visualization**: Plotly

---

**Sürüm**: v11.0 (Test Release)
**Son Güncelleme**: 2026-04-01
