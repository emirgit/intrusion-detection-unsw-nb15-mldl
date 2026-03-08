import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REF_DIR = ROOT / "ref"
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))

import streamlit as st
from app.components.styles import GLOBAL_CSS

st.set_page_config(
    page_title="Network IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Top Navigation Bar ──────────────────────────────────────────────
st.markdown("""
<div class="top-navbar" style="display: flex; align-items: center; justify-content: space-between; padding: 0.8rem 1.5rem; background: rgba(15,23,42,0.8); border-bottom: 1px solid rgba(148,163,184,0.2); border-radius: 12px; margin-bottom: 1rem; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);">
    <div style="display: flex; align-items: center; gap: 0.8rem;">
        <div style="background: linear-gradient(135deg, #3b82f6, #6366f1); padding: 0.4rem 0.6rem; border-radius: 8px; font-weight: bold; color: white; display:flex; align-items:center; justify-content:center;">
            <span style="font-size: 1.1rem; line-height: 1;">🛡️</span>
        </div>
        <h1 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.025em;">Traffic Monitoring & Alerting</h1>
    </div>
</div>
""", unsafe_allow_html=True)

_, col_nav, _ = st.columns([2, 1, 2])
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
