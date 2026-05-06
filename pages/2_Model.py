import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
from theme import GLOBAL_CSS, nav_html, footer_html, get_logo_b64

st.set_page_config(page_title="Methodology | Secutie Solutions", layout="wide", initial_sidebar_state="collapsed")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
logo_b64 = get_logo_b64()
st.markdown(nav_html(logo_b64, active="2_Model"), unsafe_allow_html=True)

st.markdown("<div class='page-title'>Model & Risk Methodology</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>How the Secutie Solutions scores, classifies, and escalates insider threats</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.markdown("###  Detection Architecture")
    st.markdown("""
    <div class='section-block'>
    <p style='font-family:Rajdhani,sans-serif;color:#8aa8c0;font-size:1rem;line-height:1.7;'>
    The Secutie Solutions combines <strong style='color:#00c8ff;'>unsupervised machine learning</strong>
    with a <strong style='color:#00c8ff;'>rule-based context logic layer</strong> to produce explainable,
    actionable risk scores — not just black-box predictions.<br><br>
    Unlike signature-based systems, Isolation Forest detects <em>behavioral deviation</em> from an
    established baseline, catching novel threats with no prior pattern.
    </p></div>
    """, unsafe_allow_html=True)

    st.markdown("###  Behavioral Risk Factors")
    for icon_name, desc in [
        (" Login Anomaly",    "Unusual login timing, geography, device ID, or authentication failure patterns."),
        (" File Access",      "Abnormal volume of files read, written, or copied relative to role baseline."),
        (" System Access",    "Accessing infrastructure, databases, or repos outside defined role permissions."),
        (" Network Activity", "Unusual data transfer volume, non-standard ports, or unknown external endpoints."),
    ]:
        st.markdown(f"<div class='info-card'><h4>{icon_name}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

with col2:
    st.markdown("###  Weighted Risk Formula")
    st.markdown("""
    <div class='section-block'>
    <p style='font-family:"Share Tech Mono",monospace;color:#4facfe;font-size:0.8rem;line-height:2.1;'>
    Base Risk Score =<br>
    &nbsp;&nbsp;0.16 &times; Login Anomaly<br>
    + 0.28 &times; File Access<br>
    + 0.28 &times; System Access<br>
    + 0.28 &times; Network Activity<br><br>
    <span style='color:rgba(0,200,255,0.4);font-size:0.68rem;'>
    Higher weights on data-centric vectors reflect greater exfiltration risk.
    </span></p></div>
    """, unsafe_allow_html=True)

    st.markdown("###  Contextual Risk Adjustments")
    st.markdown("""
    <div class='section-block'>
    <div class='info-card'>
        <h4> Role Deviation (+0.15)</h4>
        <p>User accesses systems or files outside their defined job role. Indicates potential privilege misuse or lateral movement.</p>
    </div>
    <div class='info-card'>
        <h4> Temporal Deviation (+0.10)</h4>
        <p>Significant activity during org-inactive hours. Often correlates with covert data staging operations.</p>
    </div></div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("###  Risk Classification Matrix")
levels = [
    ("0.00 – 0.20", "NORMAL",   "#00e887", "No action. Behavior fully consistent with baseline."),
    ("0.20 – 0.40", "LOW",      "#7cffb2", "Passive monitoring. 7-day rolling log review."),
    ("0.40 – 0.60", "MEDIUM",   "#ffdd57", "SOC review triggered. Analyst notified within 4 hrs."),
    ("0.60 – 0.80", "HIGH",     "#ff8c00", "Automated alert. Supervisor + HR notified immediately."),
    ("0.80 – 1.00", "CRITICAL", "#ff4b4b", "Immediate incident response. Session may terminate."),
]
for col, (rng, label, color, action) in zip(st.columns(5), levels):
    with col:
        st.markdown(f"""
        <div style='background:rgba(0,0,0,0.3);border:1px solid {color}40;border-top:3px solid {color};
                    border-radius:8px;padding:16px;text-align:center;'>
          <div style='font-family:"Orbitron",monospace;font-size:0.58rem;color:{color};letter-spacing:2px;margin-bottom:8px;'>{rng}</div>
          <div style='font-family:"Orbitron",monospace;font-size:0.85rem;font-weight:900;color:{color};
                      text-shadow:0 0 12px {color}80;margin-bottom:10px;'>{label}</div>
          <div style='font-family:Rajdhani,sans-serif;font-size:0.78rem;color:#6a8aa0;line-height:1.4;'>{action}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("###  Why This Approach Works")
for col, (title, body) in zip(st.columns(3), [
    (" ML-Driven Detection",    "Isolation Forest isolates anomalies in high-dimensional behavioral space — no labeled data or prior signatures required."),
    (" Rule-Based Context Layer","Security domain rules augment raw ML scores with interpretable justifications, reducing false positives."),
    (" Explainability First",    "Every risk score includes a reasoning log — SOC analysts see exactly which behaviors triggered the alert."),
]):
    with col:
        st.markdown(f"""
        <div class='team-card'>
          <h3 style='margin-top:0;font-size:0.85rem;'>{title}</h3>
          <p style='font-family:Rajdhani,sans-serif;color:#8aa8c0;font-size:0.9rem;line-height:1.6;margin:0;'>{body}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(footer_html(), unsafe_allow_html=True)