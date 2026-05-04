import streamlit as st

st.title("Model & Risk Methodology")

st.markdown("### How Risk is Calculated")

st.write("""
The system evaluates user behavior across multiple dimensions and assigns a risk score based on both anomaly detection and predefined security logic.
""")

st.markdown("---")

st.subheader("Behavioral Risk Factors")

st.write("""
Each user is analyzed using four key indicators:

• Login Anomaly → unusual login timing  
• File Access → abnormal volume of data accessed  
• System Access → accessing resources outside role  
• Network Activity → unusual data transfer or connections  

Each of these contributes differently to risk.
""")

st.markdown("---")

st.subheader("Weighted Risk Formula")

st.write("""
We use a weighted formula:

Base Risk Score =
0.16 × Login +
0.28 × File Access +
0.28 × System Access +
0.28 × Network

This ensures high-impact behaviors (like data access) contribute more than low-risk ones.
""")

st.markdown("---")

st.subheader("Context-Based Risk Adjustment")

st.write("""
The model is enhanced with contextual logic:

• Role Deviation  
If a user accesses resources outside their role → risk increases significantly  

• Behavior Deviation  
If user behavior suddenly changes → moderate risk increase  

This helps reduce false positives and improves detection accuracy.
""")

st.markdown("---")

st.subheader("Risk Levels & Actions")

st.write("""
0.0 – 0.2 → Normal → No action  
0.2 – 0.4 → Low → Monitor  
0.4 – 0.6 → Medium → Review  
0.6 – 0.8 → High → Alert  
0.8 – 1.0 → Critical → Immediate response  
""")

st.markdown("---")

st.subheader("Why This Works")

st.write("""
Instead of relying only on ML, the system combines:

• Statistical anomaly detection  
• Security rules  
• Context-aware logic  

This makes the system both accurate and explainable.
""")