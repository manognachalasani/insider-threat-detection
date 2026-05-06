import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from theme import GLOBAL_CSS, nav_html, footer_html, get_logo_b64

st.set_page_config(page_title="Dataset Intelligence | Secutie Solutions", layout="wide", initial_sidebar_state="collapsed")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
logo_b64 = get_logo_b64()
st.markdown(nav_html(logo_b64, active="3_Dataset"), unsafe_allow_html=True)

st.markdown("<div class='page-title'>Intelligence Source</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>Dataset architecture, feature engineering, and raw telemetry explorer</div>", unsafe_allow_html=True)

st.markdown("###  Dataset Architecture")
d1, d2 = st.columns([3, 2], gap="large")

with d1:
    st.markdown("""
    <div class='section-block'>
    <h3 style='margin-top:0;'>CERT Insider Threat Dataset (r1)</h3>
    <p style='font-family:Rajdhani,sans-serif;color:#8aa8c0;font-size:1rem;line-height:1.8;'>
    Sourced from <strong style='color:#00c8ff;'>Carnegie Mellon University CERT Division</strong> —
    the gold-standard benchmark for insider threat research. Raw enterprise logs including
    <em>login activity, web access logs,</em> and <em>device usage logs</em> were processed to engineer
    behavioral features such as login irregularity, after-hours access, sensitive data usage, and USB activity.<br><br>
    The final dataset contains <strong style='color:#00c8ff;'>1,000 user profiles</strong> and
    <strong style='color:#00c8ff;'>20+ behavioral features</strong> used for anomaly detection and insider threat analysis.
    </p>
    <div style='font-family:"Share Tech Mono",monospace;font-size:0.7rem;color:rgba(0,200,255,0.45);
                border-top:1px solid rgba(0,200,255,0.13);padding-top:12px;margin-top:12px;font-style:italic;'>
    "We transformed millions of raw enterprise log events into structured behavioral
    intelligence features suitable for UEBA and anomaly detection."
    </div></div>
    """, unsafe_allow_html=True)

with d2:
    stats = [
        (" User Profiles",       "1,000"),
        (" Behavioral Features", "20+"),
        (" Source",              "CMU CERT r1"),
        (" Data Type",           "Synthetic Enterprise"),
        (" Representation",      "UEBA-ready"),
        (" Timeframe",           "Multi-month logs"),
    ]
    rows = "".join(f"""
    <div style='display:flex;justify-content:space-between;align-items:center;
                padding:8px 0;border-bottom:1px solid rgba(0,200,255,0.07);'>
      <span style='font-family:"Share Tech Mono",monospace;font-size:0.7rem;color:rgba(0,200,255,0.45);'>{l}</span>
      <span style='font-family:"Orbitron",monospace;font-size:0.7rem;color:#00c8ff;font-weight:700;'>{v}</span>
    </div>""" for l, v in stats)
    st.markdown(f"<div class='section-block' style='padding:16px 20px;'>{rows}</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("###  Primary Threat Vector Features")
f1, f2 = st.columns(2)
f3, f4 = st.columns(2)
for col, title, key, desc in [
    (f1, " Feature 1: Login Anomaly",   "anomaly_login",   "Analyzes geolocation shifts, device fingerprints, login timestamps, and auth failure rates."),
    (f2, " Feature 2: Volume Anomaly",  "anomaly_volume",  "Monitors read/write byte ratio against a 30-day rolling average. Detects bulk data access and exfiltration events."),
    (f3, " Feature 3: Network Anomaly", "anomaly_network", "Tracks packets to non-standard ports, unknown external IPs, and unauthorized cloud storage endpoints."),
    (f4, " Feature 4: USB Anomaly",     "anomaly_usb",     "Detects removable hardware with vendor IDs associated with bulk data transfer. Flags high-volume writes to external devices."),
]:
    with col:
        st.markdown(f"""
        <div class='info-card' style='margin-bottom:0;'>
          <h4>{title}</h4>
          <p style='margin-bottom:6px;'>{desc}</p>
          <span style='font-family:"Share Tech Mono",monospace;font-size:0.62rem;color:rgba(0,200,255,0.38);letter-spacing:1px;'>KEY: {key}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("###  Raw Telemetry Data Explorer")

try:
    df = pd.read_excel("behavior_dataset.xlsx")
    col_search, col_filter = st.columns([2, 1])
    with col_filter:
        if "status" in df.columns:
            status_filter = st.selectbox("Filter by Status", ["ALL"] + list(df["status"].unique()))
            if status_filter != "ALL":
                df = df[df["status"] == status_filter]
    with col_search:
        st.markdown(
            f"<p style='font-family:Share Tech Mono,monospace;font-size:0.68rem;"
            f"color:rgba(0,200,255,0.38);padding-top:28px;'>Showing {len(df):,} records</p>",
            unsafe_allow_html=True
        )
    st.dataframe(df, use_container_width=True, height=420)
except FileNotFoundError:
    st.markdown("""
    <div class='section-block' style='text-align:center;padding:40px;'>
      <p style='font-family:"Share Tech Mono",monospace;color:rgba(0,200,255,0.4);font-size:0.85rem;'>
       behavior_dataset.xlsx not found. Place it in the project root directory.
      </p></div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(footer_html(), unsafe_allow_html=True)