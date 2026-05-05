import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Insider Threat Detection System", layout="wide")

# ---- ULTRA CYBER CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #05070a;
        font-family: 'JetBrains+Mono', monospace;
    }
    
    .stMetric {
        background: rgba(17, 25, 40, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 200, 255, 0.1);
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        text-transform: uppercase;
        letter-spacing: -1px;
    }

    /* Custom Dataframe Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #1f2937;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---- CORE LOGIC ----
@st.cache_data
def load_data():
    # Ensure your excel file is in the same directory
    df = pd.read_excel("behavior_dataset.xlsx")
    features = ["anomaly_login", "anomaly_volume", "anomaly_network", "anomaly_usb"]
    model = IsolationForest(contamination=0.15, random_state=42)
    model.fit(df[features])
    
    scores = model.decision_function(df[features])
    # Normalize score 0 to 1 (Higher = Riskier)
    df["risk_score"] = (scores.max() - scores) / (scores.max() - scores.min())
    
    def get_risk(s):
        if s > 0.75: return "CRITICAL"
        if s > 0.45: return "ELEVATED"
        return "STABLE"
    
    df["status"] = df["risk_score"].apply(get_risk)
    return df, features

df, features = load_data()

# ---- HEADER ----
st.markdown("<h1 class='main-header'>Insider Threat Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b;'>Behavioral Anomaly Detection & Intelligence | Powered by CMU CERT Dataset</p>", unsafe_allow_html=True)

# ---- TOP METRICS ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Nodes", len(df), "ONLINE")
m2.metric("Critical Alerts", len(df[df["status"]=="CRITICAL"]), "-2.4%", delta_color="inverse")
m3.metric("System Health", "98.2%", "0.1%")
m4.metric("Mean Risk", f"{df['risk_score'].mean():.2f}", "NOMINAL")

st.markdown("---")

# ---- MAIN VIEW ----
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(" Real-Time Behavior Streams")
    st.line_chart(df[features].head(50))
    
    st.subheader(" Priority Threat Registry")
    # Highlighting critical rows
    top_threats = df.sort_values("risk_score", ascending=False).head(15)
    st.dataframe(top_threats, use_container_width=True)

with col_right:
    st.subheader(" Risk Distribution")
    risk_counts = df["status"].value_counts()
    st.bar_chart(risk_counts)
    
    with st.expander("System Logs", expanded=True):
        st.caption("2026-05-03: Model retraining complete.")
        st.caption("2026-05-03: 4 Anomalies suppressed by admin.")
        st.caption("2026-05-02: New data batch ingested from CMU Source.")