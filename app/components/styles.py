"""Shared CSS styles injected into every page."""

GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&display=swap');

    /* ── Base ─────────────────────────────────────────────────────────── */
    .stApp { 
        background: #0b1120; 
        color: #f8fafc; 
        font-family: 'Public Sans', sans-serif;
    }
    .main .block-container {
        padding-top: 1rem !important;
        max-width: 1200px;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Hide default page navigation in sidebar */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #94a3b8 !important;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.8rem;
    }
    
    /* Input and Select styling in sidebar */
    .stSelectbox > div > div {
        background-color: rgba(59, 130, 246, 0.05) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 0.75rem !important;
        color: #f8fafc !important;
    }
    
    .stSlider > div > div > div {
        background-color: rgba(59, 130, 246, 0.2) !important;
    }

    /* ── Glassmorphic Radio Pills (navbar + inline radios) ────────── */
    [data-testid="stRadio"] > div[role="radiogroup"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        padding: 0.25rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        gap: 0.15rem !important;
        justify-content: center;
    }
    /* Hide radio circles */
    [data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    /* Radio pill labels */
    [data-testid="stRadio"] [role="radiogroup"] label {
        padding: 0.5rem 1.8rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: rgba(248, 250, 252, 0.5) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        margin: 0 !important;
    }
    /* Active pill */
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(99,102,241,0.25)) !important;
        color: #f8fafc !important;
        box-shadow: 0 4px 15px rgba(59,130,246,0.15) !important;
    }
    /* Hover pill */
    [data-testid="stRadio"] [role="radiogroup"] label:hover {
        color: rgba(248, 250, 252, 0.8) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }

    /* ── Metric cards ────────────────────────────────────────────── */
    div[data-testid="metric-container"] {
        background: rgba(30,41,59,0.85);
        border: 1px solid rgba(148,163,184,0.3);
        border-radius: 0.75rem; padding: 1.25rem;
        box-shadow: 0 10px 30px rgba(15,23,42,0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(15,23,42,0.45);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important; font-size: 0.85rem !important; font-weight: 600 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f8fafc !important; font-size: 1.75rem !important; font-weight: 700 !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────── */
    .stButton button {
        background: linear-gradient(120deg,#2563eb 0%,#1d4ed8 100%);
        color: #f8fafc; border: none; border-radius: 0.75rem;
        padding: 0.6rem 1.2rem; font-weight: 600;
        transition: opacity 0.2s ease;
    }
    .stButton button:hover {
        opacity: 0.9;
        color: #f8fafc;
    }

    /* ── Traffic feed ────────────────────────────────────────────── */
    .traffic-feed {
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 0.75rem; overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .traffic-row {
        display: grid;
        grid-template-columns: 90px 110px 110px 80px 70px 90px 70px;
        gap: 0.4rem; padding: 0.75rem 1rem; align-items: center;
        font-size: 0.85rem;
        transition: background-color 0.15s ease;
    }
    .traffic-row + .traffic-row { border-top: 1px solid rgba(148,163,184,0.12); }
    .traffic-row.header {
        text-transform: uppercase; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.05em; color: #94a3b8;
        background: rgba(59, 130, 246, 0.05);
    }
    .traffic-row:not(.header):hover { background: rgba(59, 130, 246, 0.05); }
    .traffic-row.attack { background: rgba(248,113,113,0.08); border-left: 3px solid #f87171; }
    .traffic-row.normal { background: rgba(34,197,94,0.05); border-left: 3px solid #34d399; }
    .traffic-cell { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: monospace; }
    .status-attack { color: #f87171; font-weight: 700; display: flex; align-items: center; gap: 4px; }
    .status-normal { color: #34d399; font-weight: 700; display: flex; align-items: center; gap: 4px; }
    .status-attack::before { content: "●"; font-size: 0.6rem; animation: pulse 2s infinite; }
    .status-normal::before { content: "●"; font-size: 0.6rem; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* ── Alert cards ─────────────────────────────────────────────── */
    .alert-card {
        padding: 1rem; border-radius: 0.75rem;
        background: rgba(15,23,42,0.9); margin-bottom: 0.75rem;
        border-left: 4px solid #38bdf8; color: #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .alert-card.critical { border-color: #ef4444; background: rgba(239, 68, 68, 0.05); }
    .alert-card.high { border-color: #f97316; background: rgba(249, 115, 22, 0.05); }
    .alert-card.medium { border-color: #fbbf24; background: rgba(251, 191, 36, 0.05); }
    .alert-card.low { border-color: #22c55e; background: rgba(34, 197, 94, 0.05); }
    .alert-title { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.25rem; }
    .alert-meta { font-size: 0.75rem; color: #94a3b8; margin-top: 0.5rem; display: flex; justify-content: space-between;}

    /* ── Attack detail panel ─────────────────────────────────────── */
    .attack-detail {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 0.75rem; padding: 1.5rem; margin: 0.8rem 0;
        font-family: monospace;
    }
    .attack-header { font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; border-bottom: 1px solid rgba(148, 163, 184, 0.2); padding-bottom: 0.5rem;}
    .attack-info-row {
        margin: 0.5rem 0;
        display: flex;
    }
    .attack-label { color: #f87171; font-weight: 600; font-size: 0.85rem; width: 150px; flex-shrink: 0; }
    .attack-value { color: #f8fafc; font-size: 0.85rem; }

    /* ── Glass card utility ──────────────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 0.75rem;
        padding: 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }

    /* ── Hide streamlit branding ─────────────────────────────────── */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    /* ── Subheaders ──────────────────────────────────────────────── */
    h3, h4 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin-bottom: 1rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
"""
