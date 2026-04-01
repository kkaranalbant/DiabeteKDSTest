# Model Eğitim Kılavuzu

## Genel Bakış
Modeller **Google Colab** üzerinde GPU destekli olarak eğitilmektedir.

## Eğitim Pipeline'ı

### 1. Veri Hazırlama
```
notebooks/00_merge_parquet_colab.ipynb    → Parquet dosyalarını birleştir
notebooks/01_preprocessing_colab.ipynb    → Veri temizleme & feature engineering
```

### 2. Model Eğitimi
```
notebooks/02_training_colab.ipynb         → XGBoost model eğitimi
notebooks/final_training_pipeline.ipynb   → Final model pipeline
```

### 3. Transfer Learning (Opsiyonel)
```
notebooks/Transfer_Learning_Personalization.ipynb → Kişiselleştirme
```

## XGBoost Konfigürasyonu

```python
xgb_params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',  # GPU
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'early_stopping_rounds': 50
}
```

## Özellikler (Features)

### Model 1 - Glucose Predictor
- `CGM`, `bolus`, `velocity`, `acceleration`
- `rolling_mean`, `rolling_std`, `active_insulin`
- `time_sin`, `time_cos`
- `age`, `weight`, `gender_encoded`

### Model 2 - Insulin Recommender
- `CGM`, `carbs`
- `time_sin`, `time_cos`, `is_weekend`
- `cgm_delta_5min`, `cgm_delta_15min`
- `basal`, `age`, `weight`
- `gender_encoded`, `insulin_type_encoded`

## Model Kaydetme ve Yükleme

```python
# Eğitim sonrası (Colab'da)
model.save_model('xgboost_final_model.json')

# app.py'de yükleme
booster = xgb.Booster()
booster.load_model('models/xgboost_final_model.json')
```

## Performans Metrikleri
- **RMSE**: Hedef < 20 mg/dL
- **MAE**: Hedef < 15 mg/dL
- **Zone A Accuracy**: Hedef > 80%

---
**Son Güncelleme**: 2026-02-06
