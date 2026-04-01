# 🩸 Diabetes Neural Decision Support System

## Proje Özeti
Diyabet hastaları için kan şekeri tahmini ve insülin dozaj önerisi yapan yapay zeka destekli karar destek sistemi.

## Mimari

### Dual-Model Cascade Pipeline
```
📥 Hasta Girdileri
       ↓
🧠 Model 1: Glucose Predictor (XGBoost)
       ↓ → 30 dakika sonraki kan şekeri tahmini
🛡️ Safety Layer A (Sanity Check)
       ↓
💉 Model 2: Insulin Recommender (XGBoost)
       ↓ → Ham insülin dozu önerisi
🛡️ Safety Layer B (Physiological Guardrails)
       ↓
✅ Final Güvenli Doz
```

### Güvenlik Katmanları
| Katman | Fonksiyon |
|--------|-----------|
| **Safety A** | Tahmin değerlerini makul aralıklara sınırlar |
| **Safety B** | IOB braking, hipoglisemi koruması, düşüş trendi algılama |

## Model Bilgileri

| Model | Dosya | Özellikler |
|-------|-------|------------|
| Glucose Predictor | `xgboost_glucose_model_v2.json` | CGM, velocity, IOB, demographic features |
| Insulin Recommender | `xgboost_final_model.json` | Predicted CGM, carbs, time features |

## Teknoloji Stack
- **Frontend**: Streamlit
- **ML Framework**: XGBoost
- **Data Processing**: Polars, NumPy
- **Visualization**: Plotly

## Çalıştırma
```bash
streamlit run app.py
```

## Proje Yapısı
```
D:\Diabete\
├── app.py                    # Ana uygulama (self-contained)
├── config.py                 # Yapılandırma
├── requirements.txt          # Bağımlılıklar
├── models/
│   ├── xgboost_final_model.json
│   └── xgboost_glucose_model_v2.json
├── data/
│   ├── processed/            # İşlenmiş veri
│   └── raw/                  # Ham veri
├── notebooks/                # Colab training notebooks
└── docs/                     # Dokümantasyon
```

---
**Son Güncelleme**: 2026-02-06
