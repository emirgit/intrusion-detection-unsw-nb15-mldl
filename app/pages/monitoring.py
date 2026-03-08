"""Page — Monitoring and Alerting Dashboard.

No business logic here — only calls to SimulationEngine / WrapperEngine
and rendering via component helpers.
"""

from collections import deque
import time

import streamlit as st
import pandas as pd

from app.components.rendering import (
    render_traffic_feed,
    render_alert_feed,
    render_attack_detail,
    fmt_ts,
)
from src.config import (
    RECENT_PACKETS_COUNT,
    DEFAULT_SPEED,
    MIN_SPEED,
    MAX_SPEED,
    SPEED_STEP,
)
from src.simulation_engine import SimulationEngine
from src.wrapper_engine import WrapperEngine


def render():
    """Render the monitoring & alerting page."""
    # ── session state init ────────────────────────────────────────────────────
    if "sim_engine" not in st.session_state:
        st.session_state.sim_engine = SimulationEngine()
        st.session_state.sim_engine.load()

    if "wrapper" not in st.session_state:
        st.session_state.wrapper = WrapperEngine()

    if "packet_history" not in st.session_state:
        st.session_state.packet_history = deque(maxlen=500)

    if "latest_attack" not in st.session_state:
        st.session_state.latest_attack = None

    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    if "speed" not in st.session_state:
        st.session_state.speed = DEFAULT_SPEED

    if "custom_dataset" not in st.session_state:
        st.session_state.custom_dataset = None
        st.session_state.custom_dataset_name = None

    sim: SimulationEngine = st.session_state.sim_engine
    wrapper: WrapperEngine = st.session_state.wrapper

    # ── Sidebar Configuration ──────────────────────────────────────────────────
    st.sidebar.markdown('<h3>⚙️ Configuration</h3>', unsafe_allow_html=True)
    
    # Model Selection
    available = wrapper.model_repo.get_available_models()
    active = wrapper.model_repo.get_active_model()
    active_idx = available.index(active) if active in available else 0

    st.sidebar.markdown('<label style="font-size:0.85rem; font-weight:600; color:#94a3b8; display:block; margin-bottom:0.2rem;">Model Selection</label>', unsafe_allow_html=True)
    selected = st.sidebar.selectbox(
        "Active detection model",
        options=available,
        format_func=lambda m: m.replace("_", " ").title(),
        index=active_idx,
        key="model_select",
        label_visibility="collapsed"
    )
    if selected != active:
        wrapper.model_repo.set_active_model(selected)
        st.rerun()

    st.sidebar.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    
    # Simulation Controls
    st.sidebar.markdown('<label style="font-size:0.85rem; font-weight:600; color:#94a3b8; display:block; margin-bottom:0.2rem;">Simulation Controls</label>', unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)

    with c1:
        if st.button("▶ Start", use_container_width=True, disabled=st.session_state.is_running):
            if st.session_state.custom_dataset is not None:
                st.session_state.sim_engine = SimulationEngine(
                    data_frame=st.session_state.custom_dataset
                )
                st.session_state.sim_engine.load()
                sim = st.session_state.sim_engine
            sim.start(shuffle=True)
            st.session_state.is_running = True
            st.rerun()

    with c2:
        if st.button("■ Stop", use_container_width=True, disabled=not st.session_state.is_running):
            sim.stop()
            st.session_state.is_running = False
            st.rerun()

    st.sidebar.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    
    # Simulation Speed
    st.sidebar.markdown('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2rem;"><label style="font-size:0.85rem; font-weight:600; color:#94a3b8;">Simulation Speed</label><span style="font-size:0.75rem; font-family:monospace; background:rgba(59,130,246,0.1); padding:2px 6px; border-radius:4px; color:#60a5fa;">sec/pkt</span></div>', unsafe_allow_html=True)
    st.session_state.speed = st.sidebar.slider(
        "Speed",
        min_value=MIN_SPEED,
        max_value=MAX_SPEED,
        value=st.session_state.speed,
        step=SPEED_STEP,
        key="speed_slider",
        label_visibility="collapsed"
    )

    st.sidebar.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
    
    # Custom Dataset
    with st.sidebar.expander("Upload Custom Dataset", expanded=False):
        uploaded = st.file_uploader("CSV or Parquet file", type=["csv", "parquet"])
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".parquet"):
                    df = pd.read_parquet(uploaded)
                else:
                    df = pd.read_csv(uploaded)
                st.session_state.custom_dataset = df
                st.session_state.custom_dataset_name = uploaded.name
                st.success(f"Loaded {len(df):,} rows.")
            except Exception as exc:
                st.error(f"Failed to load: {exc}")
        if st.session_state.custom_dataset is not None:
            st.caption(f"Active: **{st.session_state.custom_dataset_name}**")
            if st.button("Remove dataset"):
                st.session_state.custom_dataset = None
                st.session_state.custom_dataset_name = None
                st.rerun()

    st.sidebar.markdown('<div style="margin-top: auto; border-top: 1px solid rgba(148,163,184,0.1); padding-top: 1.5rem; margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
    
    # Model Status Card
    if st.session_state.is_running:
        st.sidebar.markdown(
            '<div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 0.75rem; padding: 1rem; color: #d1d5db; font-size: 0.85rem;"><div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; color: #10b981; font-weight: bold;"><span style="display:inline-block; width:8px; height:8px; background:#10b981; border-radius:50%; box-shadow: 0 0 8px #10b981;"></span> Model Status: ACTIVE</div>Running live inference at {:.2f}s per packet.</div>'.format(st.session_state.speed),
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 0.75rem; padding: 1rem; color: #d1d5db; font-size: 0.85rem;"><div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; color: #ef4444; font-weight: bold;"><span style="display:inline-block; width:8px; height:8px; background:#ef4444; border-radius:50%;"></span> Model Status: INACTIVE</div>Simulation stopped. Press Start to begin monitoring.</div>',
            unsafe_allow_html=True
        )


    # ── process one packet per rerun ──────────────────────────────────────────
    if st.session_state.is_running and sim.is_running:
        pkt = sim.next_packet()
        if pkt is not None:
            result = wrapper.process_packet(pkt)
            st.session_state.packet_history.append(result)
            if result["prediction"] == "attack" and result.get("attack_type"):
                st.session_state.latest_attack = result


    # ── Main Content Area ──────────────────────────────────────────────────────
    
    # Live counters
    stats = wrapper.get_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Packets", f"{stats['packets_processed']:,}")
    m2.metric("Attacks Detected", f"{stats['attacks_detected']:,}")
    m3.metric("Attack Ratio", f"{stats['attack_ratio']:.3f}%")
    m4.metric("Active Model", (stats["active_model"] or "—").replace("_", " ").title())

    st.markdown("<br>", unsafe_allow_html=True)

    # Traffic Feed
    st.markdown("<h3>📡 Live Traffic Feed</h3>", unsafe_allow_html=True)
    recent = list(st.session_state.packet_history)[-RECENT_PACKETS_COUNT:]
    render_traffic_feed(recent)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alerts and Details
    col_alerts, col_detail = st.columns(2)

    with col_alerts:
        st.markdown('<h3>🚨 Automated Alert Feed</h3>', unsafe_allow_html=True)
        alerts = wrapper.alert_system.get_recent_alerts(10)
        render_alert_feed(alerts)

    with col_detail:
        st.markdown('<h3>🔍 Packet Feature Inspection</h3>', unsafe_allow_html=True)
        if st.session_state.latest_attack:
            render_attack_detail(st.session_state.latest_attack)
        else:
            st.info("No attacks detected yet. Attack details will appear here.")

    # ── auto-rerun while running ──────────────────────────────────────────────
    if st.session_state.is_running:
        time.sleep(st.session_state.speed)
        st.rerun()
