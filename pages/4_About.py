import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
from theme import GLOBAL_CSS, nav_html, footer_html, get_logo_b64

st.set_page_config(page_title="About | Secutie Solutions", layout="centered", initial_sidebar_state="collapsed")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
logo_b64 = get_logo_b64()
st.markdown(nav_html(logo_b64, active="4_About"), unsafe_allow_html=True)

st.markdown("<div class='page-title'>About & Team</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>The mission, the team, and the intelligence behind Secutie</div>", unsafe_allow_html=True)

if logo_b64:
    st.markdown(f"""
    <div style='text-align:center;padding:32px 0 24px 0;'>
      <img src="data:image/png;base64,{logo_b64}"
           style="height:120px;filter:drop-shadow(0 0 28px rgba(0,200,255,0.55));">
      <div style='font-family:"Orbitron",monospace;font-size:1.4rem;font-weight:900;
                  color:#00c8ff;letter-spacing:4px;text-transform:uppercase;
                  margin-top:14px;'>Secutie Solutions</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### The Mission")
st.markdown("""
<div class='section-block'>
<p style='font-family:Rajdhani,sans-serif;color:#8aa8c0;font-size:1.05rem;line-height:1.9;'>
Insider threats account for a majority of enterprise data breaches — 
yet most organizations rely on rule-based systems that only detect known attacks.<br><br>

Secutie focuses on detecting behavioral deviations instead of predefined threats.<br><br>

By combining <strong style='color:#00c8ff;'>Isolation Forest anomaly detection</strong> with a
<strong style='color:#00c8ff;'>context-aware intelligence layer</strong>, the system identifies
unusual patterns in user behavior that traditional tools fail to capture.
</p></div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Development Team")

team = [
    ("Aishika Mareddy", "UI/UX & Integration Lead",
     "Designed the dashboard and handled system integration.",
     "aishikamareddy@gmail.com", ""),

    ("Manogna Chalasani", "Risk Intelligence Engineer",
     "Developed risk scoring and contextual logic.",
     "manognachalasani@gmail.com", ""),

    ("Joshitha", "ML Model Architect",
     "Implemented anomaly detection models.",
     None, ""),

    ("Shriyans", "Security Logic & Analysis",
     "Worked on threat detection logic and analysis.",
     None, ""),
]

c1, c2 = st.columns(2)

for i, (name, role, scope, email, icon) in enumerate(team):
    email_html = (
        f"<p style='font-family:Share Tech Mono,monospace;font-size:0.65rem;color:rgba(0,200,255,0.5);'>{email}</p>"
    ) if email else ""

    with (c1 if i % 2 == 0 else c2):
        st.markdown(f"""
        <div class='team-card'>
          <div style='font-family:"Orbitron";color:#00c8ff;font-size:0.9rem;font-weight:700;'>{name}</div>
          <div style='font-family:"Share Tech Mono";font-size:0.65rem;color:rgba(0,200,255,0.4);'>{role}</div>
          <p style='font-family:Rajdhani;color:#8aa8c0;font-size:0.9rem;'>{scope}</p>
          {email_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Contact")

st.markdown("""
<div class='section-block'>
<p style='color:#8aa8c0;'>For inquiries or collaboration:</p>

<p style='font-family:Share Tech Mono;color:#00c8ff;'>aishikamareddy@gmail.com</p>
<p style='font-family:Share Tech Mono;color:#00c8ff;'>manognachalasani@gmail.com</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(footer_html(), unsafe_allow_html=True)