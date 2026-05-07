import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from theme import GLOBAL_CSS, nav_html, footer_html, get_logo_b64

st.set_page_config(page_title="Threat Simulator | Secutie Solutions", layout="centered", initial_sidebar_state="collapsed")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
logo_b64 = get_logo_b64()
st.markdown(nav_html(logo_b64, active="1_Try_Model"), unsafe_allow_html=True)

st.markdown("<div class='page-title'>Threat Vector Simulator</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>Adjust behavioral parameters and run real-time AI-powered threat analysis</div>", unsafe_allow_html=True)
st.info(" **Simulation Mode Active** — Adjust sliders to simulate a user's behavioral profile. The Isolation Forest model evaluates anomaly scores and the logic engine applies contextual threat intelligence.")

@st.cache_resource
def load_model():

    df_train = pd.read_excel("behavior_dataset.xlsx")

    features = [
        "anomaly_login_isolationforest",
        "anomaly_volume_isolationforest",
        "anomaly_network_isolationforest",
        "anomaly_usb_isolationforest"
    ]

    # Keep only valid columns
    features = [f for f in features if f in df_train.columns]

    # Fill missing values
    df_train[features] = df_train[features].fillna(0)

    model = IsolationForest(
        contamination=0.15,
        random_state=42
    )

    model.fit(df_train[features])

    return model, df_train, features

model, df_train, features = load_model()


st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='section-block'><h3 style='margin-top:0;margin-bottom:14px;'> Authentication Vectors</h3>", unsafe_allow_html=True)
    login   = st.slider("Login Anomaly Level",  0.0, 1.0, 0.1, 0.01)
    network = st.slider("Network Divergence",    0.0, 1.0, 0.2, 0.01)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='section-block'><h3 style='margin-top:0;margin-bottom:14px;'> Data Movement Vectors</h3>", unsafe_allow_html=True)
    volume = st.slider("Exfiltration Volume",  0.0, 1.0, 0.1, 0.01)
    usb    = st.slider("Hardware Interaction", 0.0, 1.0, 0.0, 0.01)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("###  Contextual Intelligence Modifiers")
col_t1, col_t2 = st.columns(2)
with col_t1:
    role_flag = st.toggle(" Out-of-Role Resource Access")
with col_t2:
    time_flag = st.toggle(" Off-Hours Activity Peak")

st.markdown("<br>", unsafe_allow_html=True)
if st.button(" RUN SECURITY ANALYSIS", use_container_width=True):
    input_data   = np.array([[login, volume, network, usb]])
    raw_score    = model.decision_function(input_data)
    train_scores = model.decision_function(df_train[features])
    norm_score   = (train_scores.max() - raw_score[0]) / (train_scores.max() - train_scores.min())
    final_score  = norm_score
    reasons      = []

    if login   > 0.7: reasons.append(" Multiple failed logins or unusual authentication source detected.")
    if volume  > 0.7: reasons.append(" Data access volume exceeds normal 24-hour baseline.")
    if network > 0.7: reasons.append(" Active connection to unverified or blacklisted external IP.")
    if usb     > 0.8: reasons.append(" Large-scale data transfer to unauthorized USB/removable device.")
    if role_flag:
        final_score += 0.15
        reasons.append(" Privilege Escalation: Accessing files outside job scope.")
    if time_flag:
        final_score += 0.10
        reasons.append(" Temporal Anomaly: High activity during org-inactive hours.")

    final_score = min(final_score, 1.0)
    if final_score > 0.75:   css_class, lvl, badge = "status-high", " CRITICAL THREAT",      "#ff4b4b"
    elif final_score > 0.45: css_class, lvl, badge = "status-med",  " SUSPICIOUS / ELEVATED", "#ffaa00"
    else:
        css_class, lvl, badge = "status-low", " SECURE / NOMINAL", "#00e887"
        if not reasons: reasons.append(" All behavioral vectors consistent with baseline profiles.")

    st.markdown(f"""
    <div class="result-box {css_class}">
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;letter-spacing:3px;opacity:0.55;margin-bottom:8px;">
        ANALYSIS COMPLETE — SECUTIE ENGINE
      </div>
      <h3 style="font-family:'Orbitron',monospace;font-size:1.05rem;margin:0 0 12px 0;letter-spacing:2px;">{lvl}</h3>
      <h1 style="font-family:'Orbitron',monospace;font-size:3rem;margin:0;letter-spacing:3px;">RISK: {final_score:.3f}</h1>
      <div style="width:100%;height:6px;background:rgba(255,255,255,0.07);border-radius:3px;margin-top:16px;overflow:hidden;">
        <div style="width:{final_score*100:.1f}%;height:100%;background:{badge};box-shadow:0 0 10px {badge};border-radius:3px;"></div>
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;opacity:0.45;margin-top:8px;letter-spacing:2px;">
        ISOLATION FOREST + CONTEXT LOGIC ENGINE
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("###  Security Reasoning Log")
    st.markdown(
        "<div class='section-block'>" +
        "".join(f"<p style='font-family:Share Tech Mono,monospace;font-size:0.8rem;color:#c8d8e8;margin:6px 0;'>{r}</p>" for r in reasons) +
        "</div>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(footer_html(), unsafe_allow_html=True)