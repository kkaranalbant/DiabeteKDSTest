"""
🧪 DiabetAI Model Test Aracı
==============================
Ekip üyelerinin modeli gerçek hasta verileri ile test etmesi için CLI aracı.

Kullanım:
  # Tek hasta testi (interaktif)
  python test_model.py

  # Tek hasta testi (parametreli)
  python test_model.py --cgm 180 --carbs 40 --iob 1.0 --trend 2 --isf 30 --icr 10

  # Toplu test (CSV'den oku → sonuçları CSV'ye yaz)
  python test_model.py --batch hastalar.csv --output sonuclar.csv

Batch CSV Formatı (hastalar.csv):
  hasta_id, cgm, carbs, iob, trend, basal, isf, icr, age, weight, gender, gercek_doz, gercek_glukoz_30dk
  H001, 180, 40, 1.0, 2.0, 1.0, 30, 10, 45, 75, Male, 5.5, 210
  H002, 90, 0, 0.5, -1.0, 0.8, 35, 12, 60, 68, Female, 0, 85
"""

import os
import sys
import argparse
import csv
from datetime import datetime

# Proje kök dizinini ayarla
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from inference import InferenceEngine
from forecaster import GlucoseForecaster


def load_models():
    """Model yükle ve durumu göster."""
    print("=" * 70)
    print("🧪 DiabetAI Model Test Aracı")
    print("=" * 70)

    # Dosage Engine (v10)
    engine = InferenceEngine()
    if engine.model is None:
        print("❌ İnsülin dozaj modeli yüklenemedi!")
        print(f"   Beklenen: models/xgboost_clinical_v10.json")
        return None, None

    # Glucose Forecaster (Ensemble)
    model_a = os.path.join(BASE_DIR, "models", "ensemble_model_a_absolute.json")
    model_b = os.path.join(BASE_DIR, "models", "ensemble_model_b_residual.json")

    forecaster = None
    try:
        forecaster = GlucoseForecaster(model_a_path=model_a, model_b_path=model_b)
        if not forecaster.is_ready:
            print("⚠️  Glukoz tahmin modelleri bulunamadı. Lineer tahmin kullanılacak.")
            forecaster = None
    except Exception as e:
        print(f"⚠️  Forecaster yüklenemedi: {e}")

    return engine, forecaster


def predict_single(engine, forecaster, cgm, carbs, iob, trend, basal,
                   isf, icr, age, weight, gender):
    """Tek hasta için tahmin yap."""

    # Stage 1: Glukoz Tahmini (30 dk)
    predicted_glucose = cgm + (trend * 6)  # Fallback: lineer
    forecast = None

    if forecaster and forecaster.is_ready:
        forecast = forecaster.predict(cgm=cgm, iob=iob, cob=float(carbs), trend=trend)
        if forecast:
            predicted_glucose = forecast.predicted_glucose

    # Stage 2: İnsülin Doz Önerisi
    result = engine.predict(
        cgm=cgm, carbs=carbs, iob=iob,
        basal=basal, velocity=trend,
        isf=isf, icr=icr,
        age=age, weight=weight, gender=gender
    )

    return {
        "predicted_glucose_30m": round(predicted_glucose, 1),
        "forecast_method": "AI Ensemble" if forecast else "Lineer Ekstrapolasyon",
        "risk_level": forecast.risk_level if forecast else "N/A",
        "ai_raw_dose": result.ai_raw_dose,
        "math_dose": result.math_dose,
        "final_dose": result.final_dose,
        "source": result.source,
        "guardrails": " | ".join(result.guardrail_log),
    }


def run_interactive(engine, forecaster):
    """İnteraktif mod: Kullanıcıdan girdi al."""
    print("\n📝 Hasta bilgilerini girin (varsayılan değerler parantez içinde):\n")

    cgm = float(input("  🩸 Anlık Kan Şekeri (mg/dL) [150]: ").strip() or "150")
    carbs = float(input("  🍽️  Karbonhidrat (g) [0]: ").strip() or "0")
    iob = float(input("  💉 Aktif İnsülin / IOB (U) [0]: ").strip() or "0")
    trend = float(input("  📈 Trend (mg/dL / 5dk) [0]: ").strip() or "0")
    basal = float(input("  💧 Bazal Oran (U/sa) [1.0]: ").strip() or "1.0")
    isf = float(input("  ⚙️  ISF - İnsülin Duyarlılık Faktörü [30]: ").strip() or "30")
    icr = float(input("  ⚙️  ICR - İnsülin/Karb Oranı [10]: ").strip() or "10")
    age = float(input("  👤 Yaş [30]: ").strip() or "30")
    weight = float(input("  ⚖️  Kilo (kg) [70]: ").strip() or "70")
    gender = input("  🚻 Cinsiyet (Male/Female) [Male]: ").strip() or "Male"

    result = predict_single(engine, forecaster, cgm, carbs, iob, trend,
                            basal, isf, icr, age, weight, gender)

    print("\n" + "=" * 70)
    print("📊 TAHMİN SONUÇLARI")
    print("=" * 70)
    print(f"  🔮 30 dk Sonraki Glukoz Tahmini : {result['predicted_glucose_30m']} mg/dL")
    print(f"     Tahmin Yöntemi               : {result['forecast_method']}")
    print(f"     Risk Seviyesi                : {result['risk_level']}")
    print(f"  💉 AI Ham Doz                    : {result['ai_raw_dose']:.2f} U")
    print(f"  📐 Matematik Doz                 : {result['math_dose']:.2f} U")
    print(f"  ✅ Final Önerilen Doz            : {result['final_dose']:.2f} U")
    print(f"     Kaynak                       : {result['source']}")
    print(f"  🛡️  Güvenlik Kontrolleri         : {result['guardrails']}")
    print("=" * 70)

    # Gerçek değer karşılaştırma
    print("\n📋 Gerçek değerlerle karşılaştırma (opsiyonel):")
    actual_dose = input("  Gerçekte verilen doz (U) [boş bırak = atla]: ").strip()
    actual_glucose = input("  30 dk sonra gerçek glukoz (mg/dL) [boş bırak = atla]: ").strip()

    if actual_dose:
        diff = float(actual_dose) - result['final_dose']
        print(f"\n  📊 Doz Farkı: {diff:+.2f} U (Gerçek: {actual_dose}U vs Model: {result['final_dose']:.2f}U)")

    if actual_glucose:
        diff_g = float(actual_glucose) - result['predicted_glucose_30m']
        print(f"  📊 Glukoz Farkı: {diff_g:+.1f} mg/dL (Gerçek: {actual_glucose} vs Tahmin: {result['predicted_glucose_30m']})")

    print()


def run_batch(engine, forecaster, input_file, output_file):
    """Toplu test: CSV'den oku, sonuçları CSV'ye yaz."""
    print(f"\n📂 Batch modu: {input_file} → {output_file}")

    if not os.path.exists(input_file):
        print(f"❌ Dosya bulunamadı: {input_file}")
        return

    rows = []
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            rows.append(row)

    if not rows:
        print("❌ CSV dosyası boş!")
        return

    print(f"✅ {len(rows)} hasta kaydı okundu.\n")

    results = []
    header_printed = False

    for i, row in enumerate(rows):
        hasta_id = row.get('hasta_id', f'H{i+1:03d}')

        try:
            cgm = float(row.get('cgm', 150))
            carbs = float(row.get('carbs', 0))
            iob = float(row.get('iob', 0))
            trend = float(row.get('trend', 0))
            basal = float(row.get('basal', 1.0))
            isf = float(row.get('isf', 30))
            icr = float(row.get('icr', 10))
            age = float(row.get('age', 30))
            weight = float(row.get('weight', 70))
            gender = row.get('gender', 'Male').strip()
            gercek_doz = row.get('gercek_doz', '')
            gercek_glukoz = row.get('gercek_glukoz_30dk', '')
        except ValueError as e:
            print(f"  ⚠️  {hasta_id}: Veri hatası - {e}")
            continue

        pred = predict_single(engine, forecaster, cgm, carbs, iob, trend,
                              basal, isf, icr, age, weight, gender)

        # Karşılaştırma metrikleri
        doz_farki = ""
        glukoz_farki = ""
        if gercek_doz and gercek_doz.strip():
            doz_farki = f"{float(gercek_doz) - pred['final_dose']:+.2f}"
        if gercek_glukoz and gercek_glukoz.strip():
            glukoz_farki = f"{float(gercek_glukoz) - pred['predicted_glucose_30m']:+.1f}"

        result_row = {
            'hasta_id': hasta_id,
            'cgm': cgm,
            'carbs': carbs,
            'iob': iob,
            'trend': trend,
            'tahmin_glukoz_30dk': pred['predicted_glucose_30m'],
            'tahmin_yontemi': pred['forecast_method'],
            'risk_seviyesi': pred['risk_level'],
            'ai_ham_doz': pred['ai_raw_dose'],
            'matematik_doz': pred['math_dose'],
            'final_doz': pred['final_dose'],
            'kaynak': pred['source'],
            'gercek_doz': gercek_doz,
            'gercek_glukoz_30dk': gercek_glukoz,
            'doz_farki': doz_farki,
            'glukoz_farki': glukoz_farki,
            'guvenlik': pred['guardrails'],
        }
        results.append(result_row)

        # Konsol çıktısı
        if not header_printed:
            print(f"  {'ID':<8} {'CGM':>5} {'Carbs':>6} | {'Tahmin':>7} {'Risk':<8} | "
                  f"{'Final':>6} {'Kaynak':<20} | {'DozFark':>8} {'GluFark':>8}")
            print("  " + "-" * 95)
            header_printed = True

        print(f"  {hasta_id:<8} {cgm:>5.0f} {carbs:>6.0f} | "
              f"{pred['predicted_glucose_30m']:>7.1f} {pred['risk_level']:<8} | "
              f"{pred['final_dose']:>6.2f} {pred['source']:<20} | "
              f"{doz_farki:>8} {glukoz_farki:>8}")

    # CSV'ye yaz
    if results:
        fieldnames = results[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✅ Sonuçlar kaydedildi: {output_file}")
        print(f"   Toplam: {len(results)} hasta | Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Özet istatistikler
        doz_farklari = [float(r['doz_farki']) for r in results if r['doz_farki']]
        glukoz_farklari = [float(r['glukoz_farki']) for r in results if r['glukoz_farki']]

        if doz_farklari:
            import numpy as np
            print(f"\n📊 DOZ KARŞILAŞTIRMA ÖZETİ:")
            print(f"   Ortalama Fark (MAE): {np.mean(np.abs(doz_farklari)):.2f} U")
            print(f"   Ortalama Sapma:      {np.mean(doz_farklari):+.2f} U")
            print(f"   Max Sapma:           {max(doz_farklari, key=abs):+.2f} U")

        if glukoz_farklari:
            import numpy as np
            print(f"\n📊 GLUKOZ TAHMİN KARŞILAŞTIRMA ÖZETİ:")
            print(f"   Ortalama Fark (MAE): {np.mean(np.abs(glukoz_farklari)):.1f} mg/dL")
            print(f"   Ortalama Sapma:      {np.mean(glukoz_farklari):+.1f} mg/dL")
            print(f"   Max Sapma:           {max(glukoz_farklari, key=abs):+.1f} mg/dL")


def run_single_cli(engine, forecaster, args):
    """CLI parametreleri ile tek hasta testi."""
    result = predict_single(
        engine, forecaster,
        cgm=args.cgm, carbs=args.carbs, iob=args.iob,
        trend=args.trend, basal=args.basal,
        isf=args.isf, icr=args.icr,
        age=args.age, weight=args.weight, gender=args.gender
    )

    print(f"\n📊 TAHMİN SONUÇLARI")
    print(f"{'='*50}")
    print(f"  Girdi  → CGM: {args.cgm} | Carbs: {args.carbs}g | IOB: {args.iob}U | Trend: {args.trend}")
    print(f"  🔮 Glukoz (30dk): {result['predicted_glucose_30m']} mg/dL ({result['forecast_method']})")
    print(f"  💉 Final Doz:     {result['final_dose']:.2f} U ({result['source']})")
    print(f"  📐 Matematik Doz: {result['math_dose']:.2f} U")
    print(f"  🛡️  Güvenlik:     {result['guardrails']}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="DiabetAI Model Test Aracı - Kan Şekeri Tahmini & İnsülin Doz Önerisi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python test_model.py                                              # İnteraktif mod
  python test_model.py --cgm 180 --carbs 40 --trend 2              # Tek hasta
  python test_model.py --batch hastalar.csv --output sonuclar.csv   # Toplu test
        """
    )

    # Batch modu
    parser.add_argument('--batch', type=str, help='Toplu test için CSV dosya yolu')
    parser.add_argument('--output', type=str, default='sonuclar.csv',
                        help='Sonuç CSV dosya yolu (varsayılan: sonuclar.csv)')

    # Tek hasta parametreleri
    parser.add_argument('--cgm', type=float, help='Anlık kan şekeri (mg/dL)')
    parser.add_argument('--carbs', type=float, default=0, help='Karbonhidrat (g)')
    parser.add_argument('--iob', type=float, default=0, help='Aktif insülin (U)')
    parser.add_argument('--trend', type=float, default=0, help='Trend (mg/dL/5dk)')
    parser.add_argument('--basal', type=float, default=1.0, help='Bazal oran (U/sa)')
    parser.add_argument('--isf', type=float, default=30, help='İnsülin duyarlılık faktörü')
    parser.add_argument('--icr', type=float, default=10, help='İnsülin/karb oranı')
    parser.add_argument('--age', type=float, default=30, help='Yaş')
    parser.add_argument('--weight', type=float, default=70, help='Kilo (kg)')
    parser.add_argument('--gender', type=str, default='Male', choices=['Male', 'Female'],
                        help='Cinsiyet')

    args = parser.parse_args()

    # Modelleri yükle
    engine, forecaster = load_models()
    if engine is None:
        sys.exit(1)

    # Mod seçimi
    if args.batch:
        run_batch(engine, forecaster, args.batch, args.output)
    elif args.cgm is not None:
        run_single_cli(engine, forecaster, args)
    else:
        run_interactive(engine, forecaster)


if __name__ == "__main__":
    main()
