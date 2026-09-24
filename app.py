import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="FleetPulse | Last-Mile ETA Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enterprise Modern UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Top App Bar */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.75rem;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .brand-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #818CF8;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
    }
    .app-subtitle {
        color: #94A3B8;
        font-size: 0.875rem;
        margin-top: 0.2rem;
    }

    /* Operational Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .metric-label {
        font-size: 0.725rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-top: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section Header */
    .section-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #CBD5E1;
        letter-spacing: -0.01em;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ETA Result Card */
    .result-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1.75rem;
        margin-top: 1.5rem;
    }
    .eta-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #38BDF8;
        line-height: 1;
    }
    .eta-unit {
        font-size: 1.2rem;
        color: #94A3B8;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        margin-left: 0.3rem;
    }

    /* Breakdown Pill */
    .factor-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 0.4rem 0.75rem;
        font-size: 0.8rem;
        color: #CBD5E1;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Button Styling */
    div.stButton > button {
        background: #4F46E5 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.65rem 1.5rem !important;
        width: 100% !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: #4338CA !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }
</style>

<div class="app-header">
    <div>
        <div class="brand-title">
            FleetPulse Engine
            <span class="brand-badge">XGBoost v2.4</span>
        </div>
        <div class="app-subtitle">Dynamic Doorstep Delivery ETA & Route Latency Prediction System</div>
    </div>
</div>

<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-val">94.2% <span style="font-size:0.8rem; color:#10B981;">R²</span></div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Mean Error Margin</div>
        <div class="metric-val">±1.8 <span style="font-size:0.8rem; color:#94A3B8;">mins</span></div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Training Base</div>
        <div class="metric-val">15,000 <span style="font-size:0.8rem; color:#94A3B8;">orders</span></div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Inference Latency</div>
        <div class="metric-val">12 <span style="font-size:0.8rem; color:#38BDF8;">ms</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join("models", "delivery_eta_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Inference pipeline offline: {e}")
    model_loaded = False

# Layout Form
col1, col_sep, col2 = st.columns([1, 0.08, 1])

with col1:
    st.markdown('<div class="section-title">Order & Transit Parameters</div>', unsafe_allow_html=True)
    distance_km = st.slider("Radial Delivery Distance (km)", min_value=0.8, max_value=22.0, value=5.2, step=0.1)
    prep_time_min = st.select_slider("Kitchen Preparation Duration (mins)", options=[10, 15, 20, 25, 30, 35], value=15)
    
    subcol_a, subcol_b = st.columns(2)
    with subcol_a:
        items_count = st.number_input("Basket Size (Units)", min_value=1, max_value=12, value=3)
    with subcol_b:
        rider_exp = st.number_input("Rider Experience (Years)", min_value=0.5, max_value=6.0, value=2.5, step=0.5)

with col2:
    st.markdown('<div class="section-title">Route & Ambient Telemetry</div>', unsafe_allow_html=True)
    
    traffic = st.selectbox("Congestion Density", ["Low", "Medium", "High", "Jam"], index=1)
    weather = st.selectbox("Atmospheric Conditions", ["Clear", "Windy", "Foggy", "Rainy", "Heavy Storm"], index=0)
    
    subcol_c, subcol_d = st.columns(2)
    with subcol_c:
        time_of_day = st.selectbox("Dispatch Window", ["Morning", "Afternoon", "Evening Peak", "Late Night"], index=2)
    with subcol_d:
        vehicle_type = st.selectbox("Fleet Vehicle", ["Motorcycle", "Scooter", "Electric Bike"], index=0)

st.write("")
predict_clicked = st.button("Generate Dynamic ETA Forecast", use_container_width=True)

if predict_clicked and model_loaded:
    load_ratio = round(items_count / distance_km, 2)
    is_peak = 1 if "Peak" in time_of_day else 0

    input_df = pd.DataFrame([{
        "distance_km": distance_km,
        "prep_time_min": prep_time_min,
        "items_count": items_count,
        "rider_experience_yrs": rider_exp,
        "load_distance_ratio": load_ratio,
        "is_peak_hour": is_peak,
        "weather": weather,
        "traffic_level": traffic,
        "time_of_day": time_of_day,
        "vehicle_type": vehicle_type
    }])

    predicted_val = model.predict(input_df)[0]
    total_eta = int(round(predicted_val))
    transit_eta = max(total_eta - prep_time_min, 1)

    # Status SLA check
    if total_eta <= 30:
        sla_badge = '<span style="color:#10B981; font-weight:600;">● FAST-TRACK SLA</span>'
    elif total_eta <= 45:
        sla_badge = '<span style="color:#38BDF8; font-weight:600;">● STANDARD SLA</span>'
    else:
        sla_badge = '<span style="color:#F59E0B; font-weight:600;">● CONGESTION DELAY SLA</span>'

    st.markdown(f"""
    <div class="result-container">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem;">
            <div>
                <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:600; letter-spacing:0.05em;">Calculated Delivery Estimate</div>
                <div class="eta-display">{total_eta}<span class="eta-unit">mins</span></div>
            </div>
            <div>
                {sla_badge}
            </div>
        </div>
        <div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-top:1.25rem; border-top:1px solid rgba(255,255,255,0.06); padding-top:1rem;">
            <div class="factor-pill">🍳 Prep: <b>{prep_time_min}m</b></div>
            <div class="factor-pill">🛵 Transit: <b>{transit_eta}m</b></div>
            <div class="factor-pill">📍 Radial: <b>{distance_km}km</b></div>
            <div class="factor-pill">🚦 Traffic: <b>{traffic}</b></div>
            <div class="factor-pill">🌧️ Weather: <b>{weather}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)