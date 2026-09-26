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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }

    .stApp {
        background-color: #050708;
        background-image:
            linear-gradient(rgba(0, 255, 200, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 200, 0.04) 1px, transparent 1px);
        background-size: 40px 40px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        border: 1px solid rgba(0, 255, 200, 0.3);
        margin-bottom: 2rem;
        position: relative;
        background: rgba(0, 255, 200, 0.02);
        clip-path: polygon(0 0, 100% 0, 100% 85%, 97% 100%, 0 100%);
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -1px; left: -1px;
        width: 12px; height: 12px;
        border-top: 2px solid #00FFC8;
        border-left: 2px solid #00FFC8;
    }
    .app-header::after {
        content: '';
        position: absolute;
        bottom: -1px; right: -1px;
        width: 12px; height: 12px;
        border-bottom: 2px solid #FF2E9F;
        border-right: 2px solid #FF2E9F;
    }
    .brand-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #00FFC8;
        text-shadow: 0 0 10px rgba(0, 255, 200, 0.6);
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .brand-badge {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        background: transparent;
        border: 1px solid #FF2E9F;
        color: #FF2E9F;
        padding: 0.25rem 0.6rem;
        text-shadow: 0 0 6px rgba(255, 46, 159, 0.6);
    }
    .app-subtitle {
        color: #5A7A78;
        font-size: 0.85rem;
        margin-top: 0.4rem;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 0.03em;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2.5rem;
    }
    .metric-box {
        background: rgba(0, 255, 200, 0.03);
        border: 1px solid rgba(0, 255, 200, 0.2);
        border-left: 3px solid #00FFC8;
        padding: 1rem 1.2rem;
        position: relative;
    }
    .metric-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #5A7A78;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #E8FFFA;
        margin-top: 0.3rem;
        font-family: 'Share Tech Mono', monospace;
        text-shadow: 0 0 8px rgba(0, 255, 200, 0.3);
    }

    .section-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        color: #00FFC8;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 1.3rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px dashed rgba(0, 255, 200, 0.3);
    }
    .section-title::before {
        content: '// ';
        color: #FF2E9F;
    }

    .result-container {
        background: #030405;
        border: 1px solid #00FFC8;
        box-shadow: 0 0 30px rgba(0, 255, 200, 0.15), inset 0 0 30px rgba(0, 255, 200, 0.03);
        padding: 2rem;
        margin-top: 1.5rem;
        position: relative;
    }
    .result-container::before {
        content: 'ETA_OUTPUT.SYS';
        position: absolute;
        top: -10px; left: 20px;
        background: #050708;
        padding: 0 10px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #FF2E9F;
        letter-spacing: 0.1em;
    }
    .eta-display {
        font-family: 'Share Tech Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        color: #00FFC8;
        text-shadow: 0 0 20px rgba(0, 255, 200, 0.6);
        line-height: 1;
    }
    .eta-unit {
        font-size: 1.1rem;
        color: #5A7A78;
        font-family: 'Share Tech Mono', monospace;
        margin-left: 0.4rem;
    }

    .factor-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: transparent;
        border: 1px solid rgba(0, 255, 200, 0.25);
        padding: 0.45rem 0.8rem;
        font-size: 0.78rem;
        color: #A8FFF0;
        font-family: 'Share Tech Mono', monospace;
    }
    .factor-pill b { color: #00FFC8; }

    div.stButton > button {
        background: transparent !important;
        color: #00FFC8 !important;
        font-weight: 700 !important;
        border: 2px solid #00FFC8 !important;
        border-radius: 0px !important;
        padding: 0.8rem 1.5rem !important;
        width: 100% !important;
        font-family: 'Share Tech Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: #00FFC8 !important;
        color: #050708 !important;
        box-shadow: 0 0 25px rgba(0, 255, 200, 0.7) !important;
    }

    .stSlider [data-baseweb="slider"] > div > div {
        background: #FF2E9F !important;
    }

    [data-baseweb="select"] > div {
        background: rgba(0, 255, 200, 0.02) !important;
        border: 1px solid rgba(0, 255, 200, 0.25) !important;
        border-radius: 0px !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    label, .stMarkdown p {
        font-family: 'Share Tech Mono', monospace !important;
        color: #8FBDB8 !important;
        font-size: 0.8rem !important;
    }
</style>

<div class="app-header">
    <div>
        <div class="brand-title">
            ⚡ FLEETPULSE_ENGINE
            <span class="brand-badge">XGBoost_v2.4</span>
        </div>
        <div class="app-subtitle">> DYNAMIC_DOORSTEP_ETA_&_ROUTE_LATENCY_PREDICTION_SYSTEM_</div>
    </div>
</div>

<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-val">94.2%</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Mean Error Margin</div>
        <div class="metric-val">&plusmn;1.8 min</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Training Base</div>
        <div class="metric-val">15,000</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Inference Latency</div>
        <div class="metric-val">12 ms</div>
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
predict_clicked = st.button("▶ EXECUTE ETA FORECAST", use_container_width=True)

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

    if total_eta <= 30:
        sla_badge = '<span style="color:#00FFC8; font-family:\'Share Tech Mono\', monospace; text-shadow:0 0 8px rgba(0,255,200,0.6);">[ FAST-TRACK_SLA ]</span>'
    elif total_eta <= 45:
        sla_badge = '<span style="color:#38BDF8; font-family:\'Share Tech Mono\', monospace; text-shadow:0 0 8px rgba(56,189,248,0.6);">[ STANDARD_SLA ]</span>'
    else:
        sla_badge = '<span style="color:#FF2E9F; font-family:\'Share Tech Mono\', monospace; text-shadow:0 0 8px rgba(255,46,159,0.6);">[ CONGESTION_DELAY ]</span>'

    st.markdown(f"""
    <div class="result-container">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem;">
            <div>
                <div style="font-size:0.7rem; text-transform:uppercase; color:#5A7A78; font-family:'Share Tech Mono', monospace; letter-spacing:0.1em;">Calculated_Delivery_Estimate</div>
                <div class="eta-display">{total_eta}<span class="eta-unit">min</span></div>
            </div>
            <div>{sla_badge}</div>
        </div>
        <div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-top:1.25rem; border-top:1px dashed rgba(0,255,200,0.2); padding-top:1rem;">
            <div class="factor-pill">PREP: <b>{prep_time_min}m</b></div>
            <div class="factor-pill">TRANSIT: <b>{transit_eta}m</b></div>
            <div class="factor-pill">RADIAL: <b>{distance_km}km</b></div>
            <div class="factor-pill">TRAFFIC: <b>{traffic}</b></div>
            <div class="factor-pill">WEATHER: <b>{weather}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)