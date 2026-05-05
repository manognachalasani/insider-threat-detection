import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Threat Simulator", layout="centered")

# Custom UI for the Simulator
st.markdown("""
<style>
    .result-box {
        padding: 30px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        border: 2px solid;
    }
    .status-high { background-color: #440000; color: #ff4b4b; border-color: #ff4b4b; }
    .status-med { background-color: #332200; color: #ffa500; border-color: #ffa500; }
    .status-low { background-color: #002211; color: #00ff88; border-color: #00ff88; }
</style>
""", unsafe_allow_html=True)

st.title(" Threat Vector Simulator")
st.info("Adjust the sliders to simulate user behavior and observe the AI reasoning.")

# Load training context
df_train = pd.read_excel("behavior_dataset.xlsx")
features = ["anomaly_login", "anomaly_volume", "anomaly_network", "anomaly_usb"]
model = IsolationForest(contamination=0.15, random_state=42)
model.fit(df_train[features])

with st.container():
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###  Authentication")
        login = st.slider("Login Anomaly Level", 0.0, 1.0, 0.1)
        network = st.slider("Network Divergence", 0.0, 1.0, 0.2)
    with c2:
        st.markdown("###  Data Movement")
        volume = st.slider("Exfiltration Volume", 0.0, 1.0, 0.1)
        usb = st.slider("Hardware Interaction", 0.0, 1.0, 0.0)

    st.markdown("###  Contextual Modifiers")
    role_flag = st.toggle("Out-of-Role Resource Access")
    time_flag = st.toggle("Off-Hours Activity Peak")

if st.button("RUN SECURITY ANALYSIS", use_container_width=True):
    # ML Prediction
    input_data = np.array([[login, volume, network, usb]])
    raw_score = model.decision_function(input_data)
    
    # Normalization (Based on training data bounds)
    train_scores = model.decision_function(df_train[features])
    norm_score = (train_scores.max() - raw_score[0]) / (train_scores.max() - train_scores.min())
    
    # Adding Contextual Weights
    final_score = norm_score
    reasons = []

    # Logic Engine
    if login > 0.7: reasons.append("⚠️ Multiple failed logins or unusual auth source.")
    if volume > 0.7: reasons.append("⚠️ Data access volume exceeds normal 24h baseline.")
    if network > 0.7: reasons.append("⚠️ Connection detected to unverified external IP.")
    if usb > 0.8: reasons.append("⚠️ Massive data dump to unauthorized USB device.")
    
    if role_flag: 
        final_score += 0.15
        reasons.append(" Privilege Escalation: Accessing files outside job scope.")
    if time_flag:
        final_score += 0.10
        reasons.append(" Temporal Anomaly: Activity during inactive organization hours.")

    final_score = min(final_score, 1.0)
    
    # Determine Style
    if final_score > 0.75:
        css_class = "status-high"
        lvl = "CRITICAL THREAT"
    elif final_score > 0.45:
        css_class = "status-med"
        lvl = "SUSPICIOUS"
    else:
        css_class = "status-low"
        lvl = "SECURE / NORMAL"
        if not reasons: reasons.append(" Behavior consistent with baseline profiles.")

    # Output UI
    st.markdown(f"""
    <div class="result-box {css_class}">
        <h3>{lvl}</h3>
        <h1>Risk Score: {final_score:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Security Reasoning Log")
    for r in reasons:
        st.write(r)