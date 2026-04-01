"""
Proje Konfigürasyon Dosyası
Tüm dosya yollarını merkezi olarak yönetir
"""

import os

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Kaynak kod dizinleri
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_MODULE_DIR = os.path.join(SRC_DIR, 'data')
MODELS_MODULE_DIR = os.path.join(SRC_DIR, 'models')
RECOMMENDER_MODULE_DIR = os.path.join(SRC_DIR, 'recommender')
VALIDATION_MODULE_DIR = os.path.join(SRC_DIR, 'validation')

# Veri dizinleri
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model dizini
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Çıktı dizinleri
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')

# Demo dizini
DEMOS_DIR = os.path.join(PROJECT_ROOT, 'demos')

# Dokümantasyon dizini
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')

# Harici veri kaynakları (projeye göre ayarlayın)
XML_DATA_DIR = os.environ.get("XML_DATA_DIR", os.path.join(DATA_DIR, "external"))
CSV_DATA_DIR = os.environ.get("CSV_DATA_DIR", os.path.join(DATA_DIR, "external"))

# ============================================================
# Dosya Yolları
# ============================================================

# Ham veri dosyaları
TRAINING_DATA_FULL = os.path.join(RAW_DATA_DIR, 'diyabet_model_egitim_veriseti_FULL.csv')
TRAINING_DATA_EXPANDED = os.path.join(RAW_DATA_DIR, 'diyabet_model_egitim_veriseti_EXPANDED.csv')

# İşlenmiş veri dosyaları
PATIENT_PARAMS_FILE = os.path.join(PROCESSED_DATA_DIR, 'patient_insulin_parameters.csv')
PATIENT_PARAMS_WITH_NEW_FILE = os.path.join(PROCESSED_DATA_DIR, 'patient_insulin_parameters_with_new.csv')
CALIBRATION_FACTORS_FILE = os.path.join(PROCESSED_DATA_DIR, 'patient_calibration_factors.csv')
VALIDATION_RESULTS_FILE = os.path.join(PROCESSED_DATA_DIR, 'insulin_validation_results.csv')

# Model dosyaları
GLUCOSE_MODEL_FILE = os.path.join(MODELS_DIR, 'best_glucose_model.pkl')
FEATURE_SCALER_FILE = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
CLUSTER_MODEL_FILE = os.path.join(MODELS_DIR, 'cluster_model.pkl')
CLUSTER_SCALER_FILE = os.path.join(MODELS_DIR, 'cluster_scaler.pkl')
CLUSTER_PARAMS_FILE = os.path.join(MODELS_DIR, 'cluster_params.pkl')
NEW_PATIENT_ESTIMATOR_FILE = os.path.join(MODELS_DIR, 'new_patient_estimator.pkl')

# Grafik dosyaları
MODEL_COMPARISON_FIG = os.path.join(FIGURES_DIR, 'all_models_comparison.png')
MODEL_DASHBOARD_FIG = os.path.join(FIGURES_DIR, 'model_performance_dashboard.png')
ERROR_ANALYSIS_FIG = os.path.join(FIGURES_DIR, 'model_error_analysis.png')
INSULIN_IMPROVEMENT_FIG = os.path.join(FIGURES_DIR, 'insulin_improvement_analysis.png')
INSULIN_VALIDATION_FIG = os.path.join(FIGURES_DIR, 'insulin_validation_dashboard.png')
PATIENT_PARAMS_FIG = os.path.join(FIGURES_DIR, 'patient_parameters_analysis.png')
GENERALIZATION_FIG = os.path.join(FIGURES_DIR, 'generalization_system.png')

# Rapor dosyaları
TRAINING_REPORT_FILE = os.path.join(REPORTS_DIR, 'model_training_report.txt')
VALIDATION_REPORT_FILE = os.path.join(REPORTS_DIR, 'insulin_validation_report.txt')


def get_path(relative_path):
    """Proje kök dizinine göre mutlak yol döndürür"""
    return os.path.join(PROJECT_ROOT, relative_path)


def ensure_dir(path):
    """Dizin yoksa oluşturur"""
    os.makedirs(path, exist_ok=True)
    return path
