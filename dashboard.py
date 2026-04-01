"""
🏥 DiabetAI - Insulin Decision Support Dashboard (Simplified)
=============================================================
Streamlit UI for Two-Stage Insulin Pipeline

Usage:
  streamlit run dashboard.py
"""

import streamlit as st
import requests
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DiabetAI - Insulin Support",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FORCE LIGHT THEME CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #F0F4F8 !important;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #000000 !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #000000 !important;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5276 !important;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #E74C3C;
        margin-bottom: 2rem;
        background: white;
    }
    .stNumberInput input, .stTextInput input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    .stSelectbox > div > div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stMetricValue"] {
        color: #E74C3C !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
    }
    .stButton > button {
        background-color: #E74C3C !important;
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background-color: #C0392B !important;
    }
    .dose-card {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(231,76,60,0.3);
    }
    .dose-value {
        font-size: 5rem;
        font-weight: 800;
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .dose-label {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .zero-dose-card {
        background: linear-gradient(135deg, #27AE60 0%, #1E8449 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(39,174,96,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# API CONFIGURATION
# ============================================================

API_BASE = "http://localhost:8000"

def call_api(endpoint: str, payload: dict) -> dict:
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "API offline - Run: uvicorn main:app --port 8000"}
    except requests.exceptions.Timeout:
        return {"error": "API timeout"}
    except Exception as e:
        return {"error": str(e)}

def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE}/", timeout=3)
        return response.status_code == 200
    except:
        return False

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">💉 DiabetAI - Insulin Decision Support</div>', unsafe_allow_html=True)

api_online = check_api_health()
if api_online:
    st.success("✅ API Connected | Classifier Threshold: 0.30")
else:
    st.error("⚠️ API Offline - Run: `uvicorn main:app --port 8000`")

# ============================================================
# SIDEBAR: PATIENT PARAMETERS
# ============================================================

st.sidebar.markdown("## 👤 Patient Parameters")

cgm = st.sidebar.number_input(
    "🩸 Current Glucose (mg/dL)", 
    min_value=40, max_value=500, value=150, step=5
)

# =============================================
# GLUCOSE TREND (with Session State)
# =============================================
st.sidebar.markdown("### 📈 Glucose Trend")

# Initialize session state with FLAT trend (no auto-reset)
if "prev_15" not in st.session_state:
    st.session_state.prev_15 = 150  # Default starting value
if "prev_5" not in st.session_state:
    st.session_state.prev_5 = 150

# Quick Trend Preset Buttons
st.sidebar.caption("Quick Presets:")
col_t1, col_t2, col_t3 = st.sidebar.columns(3)

if col_t1.button("↘️ Drop", use_container_width=True):
    st.session_state.prev_15 = min(500, cgm + 20)
    st.session_state.prev_5 = min(500, cgm + 10)
    st.rerun()

if col_t2.button("➡️ Flat", use_container_width=True):
    st.session_state.prev_15 = cgm
    st.session_state.prev_5 = cgm
    st.rerun()

if col_t3.button("↗️ Rise", use_container_width=True):
    st.session_state.prev_15 = max(40, cgm - 20)
    st.session_state.prev_5 = max(40, cgm - 10)
    st.rerun()

# Sliders (user can manually adjust)
prev_cgm_15 = st.sidebar.slider(
    "CGM 15 min ago", 
    min_value=40, max_value=500, 
    value=int(st.session_state.prev_15),
    key="slider_prev_15"
)
prev_cgm_5 = st.sidebar.slider(
    "CGM 5 min ago", 
    min_value=40, max_value=500, 
    value=int(st.session_state.prev_5),
    key="slider_prev_5"
)

# Update session state from slider changes
st.session_state.prev_15 = prev_cgm_15
st.session_state.prev_5 = prev_cgm_5

# Show trend direction
delta_15 = cgm - prev_cgm_15
delta_5 = cgm - prev_cgm_5
if delta_15 > 5:
    trend_icon = "📈 Rising"
elif delta_15 < -5:
    trend_icon = "📉 Falling"
else:
    trend_icon = "➡️ Stable"
st.sidebar.caption(f"Trend: {trend_icon} ({delta_15:+.0f} mg/dL in 15 min)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🍽️ Meal & Insulin")

carbs = st.sidebar.number_input("🥗 Carbs (grams)", 0, 200, 0, step=5)
basal = st.sidebar.number_input("💧 Basal Rate (U/hr)", 0.0, 5.0, 0.8, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Demographics")

col_age, col_weight = st.sidebar.columns(2)
age = col_age.number_input("Age", 1, 100, 35)
weight = col_weight.number_input("Weight", 30, 200, 70)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
insulin_type = st.sidebar.selectbox("Insulin Type", ["Humalog", "Novolog", "Fiasp", "Apidra", "Regular"])

st.sidebar.markdown("---")
analyze_button = st.sidebar.button("💉 CALCULATE DOSE", use_container_width=True, type="primary")

# ============================================================
# MAIN AREA: RESULTS
# ============================================================

if analyze_button:
    if not api_online:
        st.error("❌ API is offline!")
    else:
        payload = {
            "CGM": cgm,
            "carbs": carbs,
            "basal": basal,
            "age": age,
            "weight": weight,
            "gender": gender,
            "insulin_type": insulin_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prev_cgm_5": prev_cgm_5,
            "prev_cgm_15": prev_cgm_15
        }
        
        with st.spinner("🔄 Analyzing..."):
            result = call_api("/predict/bolus", payload)
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            dose = result.get("recommended_dose", 0)
            stage1_prob = result.get("stage1_probability", 0)
            safety_alert = result.get("safety_alert", "None")
            stage1_triggered = result.get("stage1_triggered", False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if dose > 0:
                    st.markdown(f"""
                    <div class="dose-card">
                        <div class="dose-value">{dose:.1f} U</div>
                        <div class="dose-label">Recommended Bolus</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="zero-dose-card">
                        <div class="dose-value">0 U</div>
                        <div class="dose-label">No Insulin Needed</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 Classifier Score")
                st.metric("Probability", f"{stage1_prob*100:.1f}%")
                st.progress(min(stage1_prob, 1.0))
                
                if stage1_triggered:
                    st.success("✅ Stage 1 Triggered")
                else:
                    st.info("⏸️ Below Threshold (30%)")
            
            st.markdown("---")
            if safety_alert != "None":
                if "HYPO" in safety_alert:
                    st.error(f"🛑 **SAFETY OVERRIDE:** {safety_alert}")
                elif "HYPER" in safety_alert:
                    st.warning(f"⚡ **SAFETY CORRECTION:** {safety_alert}")
                elif "MAX" in safety_alert:
                    st.warning(f"🔒 **DOSE LIMIT:** {safety_alert}")
            else:
                st.success("✅ No safety overrides applied")
            
            with st.expander("📊 Pipeline Details"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Raw Dose", f"{result.get('raw_dose', 0):.4f} U" if result.get('raw_dose') else "N/A")
                col_b.metric("Pipeline Stage", result.get('pipeline_stage', 'N/A'))
                col_c.metric("Current CGM", f"{cgm} mg/dL")

            st.markdown("### 📋 Input Summary")
            input_cols = st.columns(5)
            input_cols[0].metric("CGM", f"{cgm} mg/dL")
            input_cols[1].metric("Carbs", f"{carbs} g")
            input_cols[2].metric("Trend (15m)", f"{delta_15:+.0f}")
            input_cols[3].metric("Basal", f"{basal} U/hr")
            input_cols[4].metric("Bolus", f"{dose:.1f} U")

else:
    st.info("👈 Enter patient parameters and click **CALCULATE DOSE**")
    
    st.markdown("### 🔬 Two-Stage Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Stage 1: Classifier
        - Detects if insulin needed
        - **Threshold: 30%**
        """)
    
    with col2:
        st.markdown("""
        #### Stage 2: Regressor
        - Calculates exact dose
        - Safety net applied
        """)
    
    st.markdown("---")
    st.markdown("""
    #### 🛡️ Safety Rules
    - **Hypo:** CGM < 70 → Dose = 0
    - **Hyper:** CGM > 250 + Carbs → Min 1U
    - **Max:** Capped at 15U
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666666 !important;'>"
    "💉 DiabetAI v2.1 | Insulin-Only Mode"
    "</div>",
    unsafe_allow_html=True
)
