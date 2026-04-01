# Güvenlik Protokolleri

## Güvenlik Katmanları

### Safety Layer A: Prediction Sanity Check
Tahmin değerlerinin fizyolojik olarak anlamlı olmasını sağlar.

| Kontrol | Aksiyon |
|---------|---------|
| Tahmin 40 mg/dL altı | 40'a clamp |
| Tahmin 400 mg/dL üstü | 400'e clamp |
| 50+ mg/dL ani sıçrama | Uyarı göster |

### Safety Layer B: Physiological Guardrails

```python
# Kural 0: Phantom Bolus Engelleme
if current_glucose < 110 and carbs == 0 and dose > 0.5:
    if velocity < 2.0:
        return 0.0  # Doz engelle

# Kural 1: Hızlı Düşüş Freni
if current_glucose < 110 and velocity < -2.0:
    return 0.0  # Doz durdur

# Kural 2: Kritik Düşük
if current_glucose < 70:
    return 0.0  # ACİL DURUM

# Kural 3: Gelecek Hipoglisemi Riski
if predicted_glucose < 75:
    return 0.0  # Risk var

# Kural 4: IOB Braking
if iob > 0.5:
    dose = dose - (iob * 0.5)  # Stacking önleme
```

## Klinik Uyarılar

> ⚠️ **UYARI**: Bu sistem bir KARAR DESTEK aracıdır. Otomatik insülin verme sistemi DEĞİLDİR.

- Önerilen dozlar her zaman sağlık profesyoneli tarafından onaylanmalıdır
- Sistem, gerçek klinik tedavinin yerini almaz
- Hipoglisemi riski durumunda sistem otomatik olarak doz önerisini durdurur

## Güvenlik Limitleri

| Parametre | Min | Max |
|-----------|-----|-----|
| CGM | 40 mg/dL | 400 mg/dL |
| Max Doz | 0 U | Kullanıcı limiti |
| IOB Brake Threshold | 0.5 U | - |
| Hypo Threshold | 75 mg/dL | - |
| Target Range | 70-180 mg/dL | - |

---
**Son Güncelleme**: 2026-02-06
