import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Intelligence", layout="wide")

st.title(" Intelligence Source")

st.markdown("""
### Dataset Architecture
Our model is trained on a synthetic representation of the **CERT Insider Threat Dataset**. 
It focuses on four primary vectors of exfiltration and sabotage.
""")

col1, col2 = st.columns(2)

with col1:
    st.info("**Feature 1: Login Anomaly**\n\nAnalyzes geolocations, device IDs, and login timestamps.")
    st.info("**Feature 2: Volume Anomaly**\n\nMonitors the 'Read/Write' byte ratio compared to a 30-day rolling average.")

with col2:
    st.info("**Feature 3: Network Anomaly**\n\nTracks data packets sent to non-standard ports or external cloud storage.")
    st.info("**Feature 4: USB Anomaly**\n\nDetects the mounting of hardware with specific vendor IDs known for data theft.")

st.divider()
df = pd.read_excel("behavior_dataset.xlsx")
st.subheader("Raw Telemetry Data")
st.dataframe(df, use_container_width=True)