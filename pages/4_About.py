import streamlit as st

st.set_page_config(page_title="Team & Mission", layout="centered")

st.title(" About ")

st.markdown("""
### The Mission
Insider threats represent **60% of data breaches**. Our goal was to create a system that 
doesn't just look for "bad actions," but looks for **"different behavior."** 

By utilizing Isolation Forests, we can detect outliers in high-dimensional data that 
standard rule-based firewalls would miss.
""")

st.subheader("The Development Team")

# Team cards with a bit of style
t1, t2 = st.columns(2)
with t1:
    st.markdown("""
    **Model & Logic**
    - Joshitha
    - Shriyans
    
    **Risk Intelligence**
    - Manogna Chalasani
    """)

with t2:
    st.markdown("""
    **UI/UX & Integration**
    - Aishika Mareddy
    """)

st.divider()

st.subheader(" Contact Intelligence Team")
st.write("For technical inquiries or system access:")

st.markdown("""
- **Aishika Mareddy**: [aishikamareddy@gmail.com](mailto:aishikamareddy@gmail.com)
- **Manogna Chalasani**: [manognachalasani@gmail.com](mailto:manognachalasani@gmail.com)
""")

st.caption("Sentinel II Framework v2.0.4 | © 2026 Cyber Intel Division")