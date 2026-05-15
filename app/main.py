import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REF_DIR = ROOT / "ref"
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))

import base64
import streamlit as st
from app.components.styles import GLOBAL_CSS

_logo_path = ROOT / "logo.png"
_logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode() if _logo_path.exists() else ""

st.set_page_config(
    page_title="Network IDS Dashboard",
    page_icon=str(_logo_path) if _logo_path.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Top Navigation Bar ──────────────────────────────────────────────
_logo_html = (
    f'<img src="data:image/png;base64,{_logo_b64}" style="width:2.2rem; height:2.2rem; border-radius:8px; object-fit:cover;">'
    if _logo_b64 else
    ''
)
st.markdown(f"""
<div class="top-navbar" style="display: flex; align-items: center; justify-content: space-between; padding: 0.8rem 1.5rem; background: rgba(15,23,42,0.8); border-bottom: 1px solid rgba(148,163,184,0.2); border-radius: 12px; margin-bottom: 1rem; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);">
    <div style="display: flex; align-items: center; gap: 0.8rem;">
        {_logo_html}
        <h1 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.025em;">Traffic Monitoring & Alerting</h1>
    </div>
</div>
""", unsafe_allow_html=True)

_, col_nav, _ = st.columns([1, 4, 1])
with col_nav:
    page = st.radio(
        "Navigation",
        ["Monitor", "Benchmark"],
        horizontal=True,
        label_visibility="collapsed",
        key="nav_page",
    )

st.markdown("---")

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Monitor":
    from app.pages.monitoring import render
    render()
else:
    from app.pages.benchmarking import render
    render()
