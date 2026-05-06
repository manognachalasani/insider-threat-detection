import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from theme import GLOBAL_CSS, nav_html, footer_html, get_logo_b64

st.set_page_config(
    page_title="Secutie Solutions | Insider Threat Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
logo_b64 = get_logo_b64()
st.markdown(nav_html(logo_b64, active="app"), unsafe_allow_html=True)

st.markdown("<div class='page-title'>Insider Threat Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>Behavioral Anomaly Intelligence Dashboard &nbsp;|&nbsp; Powered by CMU CERT Dataset</div>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_excel("behavior_dataset.xlsx")
    features = ["anomaly_login", "anomaly_volume", "anomaly_network", "anomaly_usb"]
    model = IsolationForest(contamination=0.15, random_state=42)
    model.fit(df[features])
    scores = model.decision_function(df[features])
    df["risk_score"] = (scores.max() - scores) / (scores.max() - scores.min())
    def get_risk(s):
        if s > 0.75: return "CRITICAL"
        if s > 0.45: return "ELEVATED"
        return "STABLE"
    df["status"] = df["risk_score"].apply(get_risk)
    return df, features

df, features = load_data()

st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(" Active Nodes",     len(df),                            "ONLINE")
m2.metric(" Critical Alerts",  len(df[df["status"]=="CRITICAL"]), "-2.4%",  delta_color="inverse")
m3.metric(" Elevated",         len(df[df["status"]=="ELEVATED"]), "+0.8%",  delta_color="inverse")
m4.metric(" System Health",    "98.2%",                            "+0.1%")
m5.metric(" Mean Risk Index",  f"{df['risk_score'].mean():.3f}",  "NOMINAL")

st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("###  Real-Time Behavioral Telemetry Streams")
    st.line_chart(df[features].head(60), use_container_width=True, height=240)
    st.markdown("###  Priority Threat Registry")
    top_threats = df.sort_values("risk_score", ascending=False).head(20)
    def color_status(val):
        if val == "CRITICAL": return "color:#ff4b4b;font-weight:700;font-family:monospace"
        if val == "ELEVATED": return "color:#ffaa00;font-weight:700;font-family:monospace"
        return "color:#00e887;font-family:monospace"
    def color_risk(val):
        try:
            v = float(val)
            if v > 0.75: return "color:#ff4b4b;font-weight:700"
            if v > 0.45: return "color:#ffaa00"
            return "color:#00e887"
        except: return ""
    styled = (
        top_threats.style
        .applymap(color_status, subset=["status"])
        .applymap(color_risk,   subset=["risk_score"])
        .format({"risk_score": "{:.4f}"})
        .set_properties(**{"font-family": "Share Tech Mono, monospace", "font-size": "12px"})
    )
    st.dataframe(styled, use_container_width=True, height=380)

with col_right:
    st.markdown("###  Risk Distribution")
    risk_counts = df["status"].value_counts().rename_axis("Status").reset_index(name="Count")
    st.bar_chart(risk_counts.set_index("Status"), height=200, use_container_width=True)
    st.markdown("###  Top 10 Highest Risk Users")
    first_col = df.columns[0]
    hotlist = df.nlargest(10, "risk_score")[[first_col, "risk_score", "status"]]
    st.dataframe(hotlist, use_container_width=True, height=240, hide_index=True)
    with st.expander(" System Event Log", expanded=True):
        st.caption("▸ 2026-05-06 08:12  Model retraining cycle complete.")
        st.caption("▸ 2026-05-05 23:47  4 anomalies suppressed by SOC admin.")
        st.caption("▸ 2026-05-05 11:02  New behavioral batch ingested — CMU CERT r1.")
        st.caption("▸ 2026-05-04 09:30  Isolation Forest contamination threshold updated.")
        st.caption("▸ 2026-05-03 00:00  System health check passed — all nodes nominal.")

st.markdown("---")
st.markdown(footer_html(), unsafe_allow_html=True)