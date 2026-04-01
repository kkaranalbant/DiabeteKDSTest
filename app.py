"""
🩸 DiabetAI Clinical Decision Support v11.0
Two-Stage Pipeline:
  Stage 1: Glucose Forecasting (Hybrid Ensemble)
  Stage 2: Insulin Dosage (Physics-Informed XGBoost + Guardrails)
"""

import streamlit as st
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add parent dir for modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import InferenceEngine, V10_FEATURES
from forecaster import GlucoseForecaster

# ============================================================
# CONFIGURATION
# ============================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSAGE_MODEL = os.path.join(_BASE_DIR, "models", "xgboost_clinical_v10.json")
ENSEMBLE_A   = os.path.join(_BASE_DIR, "models", "ensemble_model_a_absolute.json")
ENSEMBLE_B   = os.path.join(_BASE_DIR, "models", "ensemble_model_b_residual.json")

CGM_MIN, CGM_MAX = 40, 400
TARGET_LOW, TARGET_HIGH = 70, 180
HYPO_THRESHOLD = 70

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DiabetAI v11.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .math-fallback {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safety-override {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 20px; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe-zone {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px; border-radius: 15px; color: white; text-align: center;
    }
    .forecast-card {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 20px; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .cascade-step {
        background: #f0f2f6; padding: 10px; border-radius: 8px;
        margin: 5px 0; border-left: 4px solid #667eea;
    }
    .risk-low { border-left: 4px solid #00CC66; }
    .risk-medium { border-left: 4px solid #FFD700; }
    .risk-high { border-left: 4px solid #FF4444; }
    .risk-critical { border-left: 4px solid #FF0000; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'last_glucose' not in st.session_state:
    st.session_state.last_glucose = 150
if 'velocity_val' not in st.session_state:
    st.session_state.velocity_val = 0.0


def update_velocity():
    current = st.session_state.glucose_input
    previous = st.session_state.last_glucose
    velocity = np.clip(current - previous, -20, 20)
    st.session_state.velocity_val = float(velocity)
    st.session_state.last_glucose = current


# ============================================================
# LOAD ENGINES
# ============================================================
@st.cache_resource
def load_engines():
    """Load both the dosage engine and glucose forecaster."""
    engine = None
    forecaster = None

    try:
        engine = InferenceEngine(model_path=DOSAGE_MODEL)
    except Exception as e:
        print(f"⚠️ Dosage engine error: {e}")

    try:
        forecaster = GlucoseForecaster(
            model_a_path=ENSEMBLE_A,
            model_b_path=ENSEMBLE_B,
        )
    except Exception as e:
        print(f"⚠️ Forecaster error: {e}")

    return engine, forecaster


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("🧠 DiabetAI Clinical v11.0")
    st.markdown("*Two-Stage Pipeline: AI Glucose Forecast → Intelligent Dosing*")

    engine, forecaster = load_engines()

    # Status bar
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if engine and engine.model:
            st.success("✅ Stage 2: Dosage Engine Active")
        else:
            st.error("❌ Dosage Model Not Found")
    with col_s2:
        if forecaster and forecaster.is_ready:
            st.success("✅ Stage 1: Glucose Forecaster Active")
        else:
            st.warning("⚠️ Forecaster unavailable — using linear extrapolation")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📊 Patient State")
        cgm = st.slider("Glucose (mg/dL)", 40, 400, st.session_state.last_glucose,
                         key='glucose_input', on_change=update_velocity)
        velocity = st.number_input("Trend (mg/dL/5min)", -20.0, 20.0,
                                   st.session_state.velocity_val, 0.5)

        st.header("🍽️ Meal & Therapy")
        carbs = st.number_input("Carbs (g)", 0, 200, 0, 5)
        iob = st.number_input("Active Insulin (IOB)", 0.0, 20.0, 0.0, 0.1)
        cob = st.number_input("Carbs On Board (COB)", 0.0, 200.0, float(carbs), 5.0)
        basal = st.number_input("Basal Rate (U/hr)", 0.0, 5.0, 1.0, 0.1)

        st.header("⚙️ Clinical Params")
        isf = st.number_input("ISF (mg/dL/U)", 10, 150, 30)
        icr = st.number_input("ICR (g/U)", 1, 50, 10)

        st.header("👤 Demographics")
        age = st.number_input("Age", 1, 100, 30)
        weight = st.number_input("Weight (kg)", 20, 200, 70)
        gender = st.selectbox("Gender", ["Male", "Female"])

    # ============================================================
    # STAGE 1: GLUCOSE FORECAST
    # ============================================================
    st.markdown("---")
    st.subheader("🔬 Stage 1 — AI Glucose Forecast (+30 min)")

    forecast = None
    predicted_glucose = cgm + (velocity * 6)  # Fallback: linear

    if forecaster and forecaster.is_ready:
        forecast = forecaster.predict(
            cgm=cgm, iob=iob, cob=cob, trend=velocity
        )
        if forecast:
            predicted_glucose = forecast.predicted_glucose

    # Forecast display
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        delta = predicted_glucose - cgm
        delta_arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        st.metric("🎯 Predicted (30m)", f"{predicted_glucose:.0f} mg/dL",
                  f"{delta:+.1f} {delta_arrow}")
    with fc2:
        st.metric("📍 Current CGM", f"{cgm} mg/dL")
    with fc3:
        if forecast:
            st.metric("📐 Physics Baseline", f"{forecast.linear_forecast:.0f} mg/dL")
        else:
            st.metric("📐 Physics Baseline", f"{cgm + velocity * 6:.0f} mg/dL")
    with fc4:
        if forecast:
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}
            st.metric("⚠️ Risk Level", f"{risk_emoji.get(forecast.risk_level, '❓')} {forecast.risk_level}")
        else:
            st.metric("⚠️ Risk Level", "N/A (no model)")

    # Ensemble detail (if available)
    if forecast:
        with st.expander("🔬 Ensemble Breakdown"):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.write(f"**Model A (Stabilizer):** {forecast.pred_a:.1f} mg/dL")
                st.write(f"Weight: {forecaster.w_a:.0%}")
            with ec2:
                st.write(f"**Model B (Navigator):** {forecast.pred_b:.1f} mg/dL")
                st.write(f"Weight: {forecaster.w_b:.0%}")
            with ec3:
                st.write(f"**Linear Forecast:** {forecast.linear_forecast:.1f} mg/dL")
                st.write(f"**Final Blend:** {forecast.predicted_glucose:.1f} mg/dL")

    # ============================================================
    # STAGE 2: DOSAGE RECOMMENDATION
    # ============================================================
    st.markdown("---")
    st.subheader("💉 Stage 2 — Insulin Dosage Recommendation")

    if engine and engine.model:
        result = engine.predict(
            cgm=cgm, carbs=carbs, iob=iob,
            basal=basal, velocity=velocity,
            isf=isf, icr=icr,
            age=age, weight=weight, gender=gender
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Decision Pipeline**")
            st.markdown(f"<div class='cascade-step'>📥 <b>Input:</b> {cgm} mg/dL | {carbs}g Carbs</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='cascade-step'>🔬 <b>AI Forecast:</b> {predicted_glucose:.0f} mg/dL in 30m</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='cascade-step'>📐 <b>Clinical Math:</b> {result.math_dose:.2f} U</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='cascade-step'>🧠 <b>AI Raw:</b> {result.ai_raw_dose:.2f} U</div>",
                        unsafe_allow_html=True)

        with col2:
            st.markdown("**Final Recommendation**")
            if "HYPO" in result.source:
                style = "safety-override"
            elif "CAP" in result.source:
                style = "math-fallback"
            else:
                style = "metric-card"

            st.markdown(f"""
            <div class='{style}'>
                <h2>{result.final_dose:.2f} Units</h2>
                <p>Source: {result.source}</p>
            </div>
            """, unsafe_allow_html=True)

        # Guardrail Log
        st.markdown("---")
        st.subheader("🛡️ Guardrail Log")
        for log in result.guardrail_log:
            if "HYPO" in log:
                st.error(log)
            elif "CAP" in log:
                st.warning(log)
            else:
                st.success(log)
    else:
        st.error("❌ Dosage model not loaded.")
        result = None

    # ============================================================
    # TRAJECTORY CHART (Enhanced with AI Forecast)
    # ============================================================
    st.markdown("---")
    st.subheader("📈 Glucose Trajectory (30 min)")

    now = datetime.now()
    future = now + timedelta(minutes=30)

    fig = go.Figure()

    # Target zone
    fig.add_hrect(y0=TARGET_LOW, y1=TARGET_HIGH,
                  fillcolor="rgba(0,204,102,0.1)", line_width=0,
                  annotation_text="Target Zone", annotation_position="top left")
    fig.add_hrect(y0=0, y1=HYPO_THRESHOLD,
                  fillcolor="rgba(255,0,0,0.08)", line_width=0,
                  annotation_text="Hypo Zone")

    # Linear extrapolation (dashed, faint)
    linear_pred = cgm + (velocity * 6)
    fig.add_trace(go.Scatter(
        x=[now, future], y=[cgm, linear_pred],
        mode='lines', line=dict(color='rgba(150,150,150,0.5)', width=2, dash='dash'),
        name='Linear Extrapolation',
    ))

    # AI Ensemble forecast (bold)
    forecast_color = "#FF6B6B"
    if forecast:
        forecast_color = forecast.risk_color

    fig.add_trace(go.Scatter(
        x=[now, future], y=[cgm, predicted_glucose],
        mode='lines+markers',
        line=dict(color=forecast_color, width=3),
        marker=dict(size=12, symbol=['circle', 'diamond']),
        name='AI Ensemble Forecast',
    ))

    # Confidence band
    if forecast:
        cb = forecast.confidence_band
        fig.add_trace(go.Scatter(
            x=[future, future], y=[predicted_glucose - cb, predicted_glucose + cb],
            mode='lines', line=dict(color=forecast_color, width=8),
            opacity=0.3, name=f'±{cb:.0f} mg/dL',
            showlegend=True,
        ))

    # Clinical thresholds
    fig.add_hline(y=70, line_dash='dot', line_color='red', line_width=1)
    fig.add_hline(y=180, line_dash='dot', line_color='orange', line_width=1)

    y_max = max(300, predicted_glucose + 60, linear_pred + 40)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        yaxis=dict(range=[30, y_max]),
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # DEBUG CONSOLE
    # ============================================================
    with st.expander("🔍 Architect Debug Console"):
        col_db1, col_db2, col_db3 = st.columns(3)

        with col_db1:
            st.write("**🔬 Forecast Debug:**")
            if forecast:
                st.write(f"- Model A (abs): {forecast.pred_a:.1f}")
                st.write(f"- Model B (res): {forecast.pred_b:.1f}")
                st.write(f"- Linear:        {forecast.linear_forecast:.1f}")
                st.write(f"- Ensemble:      {forecast.predicted_glucose:.1f}")
                st.write(f"- Weights: A={forecaster.w_a:.2f}, B={forecaster.w_b:.2f}")
            else:
                st.write("Forecaster not active")

        with col_db2:
            if result:
                st.write("**📊 Physics Features:**")
                st.write(f"- phys_meal_bolus: {result.debug.get('phys_meal_bolus', 0):.2f}")
                st.write(f"- phys_corr_bolus: {result.debug.get('phys_corr_bolus', 0):.2f}")
                st.write(f"- phys_net_need: {result.debug.get('phys_net_need', 0):.2f}")

        with col_db3:
            if result:
                st.write("**🎯 Dose Comparison:**")
                st.write(f"- Math: {result.math_dose:.2f}U")
                st.write(f"- AI Raw: {result.ai_raw_dose:.2f}U")
                st.write(f"- Final: {result.final_dose:.2f}U")

        st.markdown("---")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.write("**📋 Dosage Features (v10, 20):**")
            st.write(V10_FEATURES)
        with col_f2:
            st.write("**📋 Forecast Features (11):**")
            from forecaster import FORECAST_FEATURES
            st.write(FORECAST_FEATURES)

    # Footer
    st.caption("⚠️ Decision support only. Always verify with healthcare provider. v11.0")


if __name__ == "__main__":
    main()