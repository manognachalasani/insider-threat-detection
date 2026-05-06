import base64
import pathlib

TEAM_NAMES = "Aishika · Manogna · Shriyans · Joshitha"

def get_logo_b64() -> str:
    for candidate in [
        pathlib.Path("secutie_solutions.png"),
        pathlib.Path(__file__).parent / "secutie_solutions.png",
        pathlib.Path(__file__).parent.parent / "secutie_solutions.png",
    ]:
        if candidate.exists():
            return base64.b64encode(candidate.read_bytes()).decode()
    return ""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #040810 !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: #c8d8e8 !important;
}

/* ✅ SIDEBAR ENABLED */
[data-testid="stSidebar"] {
    display: block !important;
    background-color: #040810 !important;
    border-right: 1px solid rgba(0,200,255,0.2);
}

#MainMenu { visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent !important; }

.page-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
    color: #00c8ff !important;
    letter-spacing: 2px;
    margin-top: 20px;
}

.page-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: rgba(0,200,255,0.4);
    margin-bottom: 20px;
}

.stButton > button {
    background: rgba(0,200,255,0.1) !important;
    border: 1px solid rgba(0,200,255,0.4) !important;
    color: #00c8ff !important;
}

.cyber-footer {
    text-align: center;
    font-size: 0.6rem;
    color: rgba(0,200,255,0.3);
    margin-top: 40px;
}
</style>
"""

def nav_html(logo_b64: str, active: str = "") -> str:
    return f"""
    <div style="padding:10px;border-bottom:1px solid rgba(0,200,255,0.2);">
        <h3 style="color:#00c8ff;margin:0;">Secutie Solutions</h3>
    </div>
    """

def footer_html() -> str:
    return f"<div class='cyber-footer'>© 2026 Secutie Solutions · {TEAM_NAMES}</div>"