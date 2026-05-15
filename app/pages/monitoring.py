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

# ── Fusion helpers ────────────────────────────────────────────────────────────

# Estimated accuracies for ML models (no offline metrics JSON for these).
# Updated automatically if benchmarking results exist in session state.
_DEFAULT_ML_ACCURACY: dict = {
    "random_forest_binary": 0.911,
    "random_forest_multi": 0.893,
    "decision_tree_binary": 0.872,
    "decision_tree_multi": 0.854,
    "knn_binary": 0.871,
    "knn_multi": 0.849,
    "lsvm_binary": 0.836,
    "lsvm_multi": 0.820,
    "logistic_regressor_binary": 0.803,
    "logistic_regressor_multi": 0.786,
    "linear_regressor_binary": 0.752,
    "linear_regressor_multi": 0.741,
}


def _get_accuracy(wrapper: WrapperEngine, model_id: str) -> float:
    """Return accuracy for a model from offline metrics, bench results, or defaults."""
    offline = wrapper.model_repo.get_offline_metrics()
    if model_id in offline and "accuracy" in offline[model_id]:
        return float(offline[model_id]["accuracy"])
    bench = st.session_state.get("bench_results")
    if bench:
        for row in bench["summary"]:
            if row.get("Model ID") == model_id:
                return float(row["Accuracy"])
    return _DEFAULT_ML_ACCURACY.get(model_id, 0.80)


def _compute_fusion(wrapper: WrapperEngine, pkt: dict, ml_id: str, dl_id: str) -> dict:
    """Run ML + DL on a raw packet dict and return fused result."""
    raw_df = pd.DataFrame([pkt])

    ml_feats = wrapper.processor.transform(raw_df, ml_id)
    dl_feats = wrapper.processor.transform(raw_df, dl_id)

    ml_res = wrapper.model_repo.predict(ml_id, ml_feats)
    dl_res = wrapper.model_repo.predict(dl_id, dl_feats)

    acc_ml = _get_accuracy(wrapper, ml_id)
    acc_dl = _get_accuracy(wrapper, dl_id)
    total_acc = acc_ml + acc_dl
    w_ml = acc_ml / total_acc
    w_dl = acc_dl / total_acc

    fusion_prob = round(w_dl * dl_res["probability"] + w_ml * ml_res["probability"], 4)
    fusion_label = "attack" if fusion_prob >= 0.5 else "normal"

    return {
        "ml_model": ml_id,
        "ml_label": ml_res["predicted_label"],
        "ml_prob": ml_res["probability"],
        "ml_accuracy": acc_ml,
        "ml_weight": w_ml,
        "dl_model": dl_id,
        "dl_label": dl_res["predicted_label"],
        "dl_prob": dl_res["probability"],
        "dl_accuracy": acc_dl,
        "dl_weight": w_dl,
        "fusion_prob": fusion_prob,
        "fusion_label": fusion_label,
    }


def _render_fusion_panel(fusion: dict) -> None:
    """Render the three-column ML | DL | Fusion result card."""
    col_ml, col_dl, col_fuse = st.columns(3)

    def _label_badge(label: str) -> str:
        if label == "attack":
            return '<span style="background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.4);border-radius:6px;padding:2px 10px;font-weight:700;font-size:0.9rem;">ATTACK</span>'
        return '<span style="background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.35);border-radius:6px;padding:2px 10px;font-weight:700;font-size:0.9rem;">NORMAL</span>'

    with col_ml:
        st.markdown(
            f"""
            <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.3);
                        border-radius:0.75rem;padding:1.1rem 1rem;height:100%;">
              <div style="font-size:0.72rem;font-weight:700;color:#60a5fa;letter-spacing:.08em;
                          text-transform:uppercase;margin-bottom:0.5rem;">ML Model</div>
              <div style="font-size:0.82rem;color:#94a3b8;margin-bottom:0.7rem;">
                {fusion['ml_model'].replace('_',' ').title()}
              </div>
              {_label_badge(fusion['ml_label'])}
              <div style="margin-top:0.75rem;font-size:0.8rem;color:#cbd5e1;">
                Prob: <b style="color:#f1f5f9;">{fusion['ml_prob']:.2%}</b>
              </div>
              <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.25rem;">
                Accuracy: {fusion['ml_accuracy']:.1%} &nbsp;|&nbsp; Weight: {fusion['ml_weight']:.3f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_dl:
        st.markdown(
            f"""
            <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.3);
                        border-radius:0.75rem;padding:1.1rem 1rem;height:100%;">
              <div style="font-size:0.72rem;font-weight:700;color:#a78bfa;letter-spacing:.08em;
                          text-transform:uppercase;margin-bottom:0.5rem;">DL Model</div>
              <div style="font-size:0.82rem;color:#94a3b8;margin-bottom:0.7rem;">
                {fusion['dl_model'].replace('_',' ').title()}
              </div>
              {_label_badge(fusion['dl_label'])}
              <div style="margin-top:0.75rem;font-size:0.8rem;color:#cbd5e1;">
                Prob: <b style="color:#f1f5f9;">{fusion['dl_prob']:.2%}</b>
              </div>
              <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.25rem;">
                Accuracy: {fusion['dl_accuracy']:.1%} &nbsp;|&nbsp; Weight: {fusion['dl_weight']:.3f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_fuse:
        fuse_color = "#f87171" if fusion["fusion_label"] == "attack" else "#34d399"
        fuse_bg = "rgba(239,68,68,0.08)" if fusion["fusion_label"] == "attack" else "rgba(16,185,129,0.08)"
        fuse_border = "rgba(239,68,68,0.3)" if fusion["fusion_label"] == "attack" else "rgba(16,185,129,0.3)"
        st.markdown(
            f"""
            <div style="background:{fuse_bg};border:1px solid {fuse_border};
                        border-radius:0.75rem;padding:1.1rem 1rem;height:100%;">
              <div style="font-size:0.72rem;font-weight:700;color:{fuse_color};letter-spacing:.08em;
                          text-transform:uppercase;margin-bottom:0.5rem;">Fusion Output</div>
              <div style="font-size:0.82rem;color:#94a3b8;margin-bottom:0.7rem;">
                Weighted Probability Fusion
              </div>
              {_label_badge(fusion['fusion_label'])}
              <div style="margin-top:0.75rem;font-size:0.8rem;color:#cbd5e1;">
                Fused Prob: <b style="color:#f1f5f9;">{fusion['fusion_prob']:.2%}</b>
              </div>
              <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.25rem;">
                w<sub>DL</sub>·p<sub>DL</sub> + w<sub>ML</sub>·p<sub>ML</sub>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Page ──────────────────────────────────────────────────────────────────────

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

    if "speed_slider" not in st.session_state:
        st.session_state.speed_slider = DEFAULT_SPEED

    if "custom_dataset" not in st.session_state:
        st.session_state.custom_dataset = None
        st.session_state.custom_dataset_name = None

    if "fusion_enabled" not in st.session_state:
        st.session_state.fusion_enabled = False

    if "latest_fusion" not in st.session_state:
        st.session_state.latest_fusion = None

    sim: SimulationEngine = st.session_state.sim_engine
    wrapper: WrapperEngine = st.session_state.wrapper

    # ── Sidebar Configuration ─────────────────────────────────────────────────
    st.sidebar.markdown('<h3>Configuration</h3>', unsafe_allow_html=True)

    # Active model selection (single model for normal mode)
    available = wrapper.model_repo.get_available_models()
    active = wrapper.model_repo.get_active_model()
    active_idx = available.index(active) if active in available else 0

    if not st.session_state.get("fusion_toggle", st.session_state.fusion_enabled):
        st.sidebar.markdown(
            '<label style="font-size:0.85rem; font-weight:600; color:#94a3b8; display:block; margin-bottom:0.2rem;">Model Selection</label>',
            unsafe_allow_html=True,
        )
        selected = st.sidebar.selectbox(
            "Active detection model",
            options=available,
            format_func=lambda m: m.replace("_", " ").title(),
            index=active_idx,
            key="model_select",
            label_visibility="collapsed",
        )
        if selected != active:
            wrapper.model_repo.set_active_model(selected)
            st.rerun()

    st.sidebar.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # ── Integrated ML + DL Fusion ─────────────────────────────────────────────
    st.sidebar.markdown(
        '<label style="font-size:0.85rem; font-weight:600; color:#94a3b8; display:block; margin-bottom:0.4rem;">Integrated ML + DL Mode</label>',
        unsafe_allow_html=True,
    )

    ml_models = [m for m in available if wrapper.model_repo.get_model_type(m) == "sklearn"]
    dl_models = [m for m in available if wrapper.model_repo.get_model_type(m) == "pytorch"]

    fusion_available = bool(ml_models and dl_models)
    if not fusion_available:
        st.sidebar.caption("Both ML and DL models must be loaded to enable fusion.")

    fusion_enabled = st.sidebar.toggle(
        "Enable Fusion",
        value=st.session_state.fusion_enabled,
        key="fusion_toggle",
        disabled=not fusion_available,
    )
    st.session_state.fusion_enabled = fusion_enabled and fusion_available

    if st.session_state.fusion_enabled:
        prev_ml = st.session_state.get("fusion_ml_model", ml_models[0])
        ml_idx = ml_models.index(prev_ml) if prev_ml in ml_models else 0
        fusion_ml = st.sidebar.selectbox(
            "ML Model",
            options=ml_models,
            format_func=lambda m: m.replace("_", " ").title(),
            index=ml_idx,
            key="fusion_ml_select",
        )
        st.session_state.fusion_ml_model = fusion_ml

        prev_dl = st.session_state.get("fusion_dl_model", dl_models[0])
        dl_idx = dl_models.index(prev_dl) if prev_dl in dl_models else 0
        fusion_dl = st.sidebar.selectbox(
            "DL Model",
            options=dl_models,
            format_func=lambda m: m.replace("_", " ").title(),
            index=dl_idx,
            key="fusion_dl_select",
        )
        st.session_state.fusion_dl_model = fusion_dl

        acc_ml = _get_accuracy(wrapper, fusion_ml)
        acc_dl = _get_accuracy(wrapper, fusion_dl)
        total = acc_ml + acc_dl
        st.sidebar.markdown(
            f"""
            <div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.25);
                        border-radius:0.6rem;padding:0.65rem 0.8rem;margin-top:0.5rem;font-size:0.78rem;color:#94a3b8;">
              <b style="color:#a5b4fc;">Fusion Weights</b><br>
              DL weight: <b style="color:#f1f5f9;">{acc_dl/total:.3f}</b>
              &nbsp;(acc {acc_dl:.1%})<br>
              ML weight: <b style="color:#f1f5f9;">{acc_ml/total:.3f}</b>
              &nbsp;(acc {acc_ml:.1%})
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # Simulation Controls
    st.sidebar.markdown(
        '<label style="font-size:0.85rem; font-weight:600; color:#94a3b8; display:block; margin-bottom:0.2rem;">Simulation Controls</label>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.sidebar.columns(2)

    with c1:
        if st.button("Start", use_container_width=True, disabled=st.session_state.is_running):
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
        if st.button("Stop", use_container_width=True, disabled=not st.session_state.is_running):
            sim.stop()
            st.session_state.is_running = False
            st.rerun()

    st.sidebar.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # Simulation Speed
    st.sidebar.markdown(
        '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2rem;">'
        '<label style="font-size:0.85rem; font-weight:600; color:#94a3b8;">Simulation Speed</label>'
        '<span style="font-size:0.75rem; font-family:monospace; background:rgba(59,130,246,0.1); '
        'padding:2px 6px; border-radius:4px; color:#60a5fa;">sec/pkt</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.slider(
        "Speed",
        min_value=MIN_SPEED,
        max_value=MAX_SPEED,
        step=SPEED_STEP,
        key="speed_slider",
        label_visibility="collapsed",
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

    st.sidebar.markdown(
        '<div style="margin-top: auto; border-top: 1px solid rgba(148,163,184,0.1); '
        'padding-top: 1.5rem; margin-bottom: 1rem;"></div>',
        unsafe_allow_html=True,
    )

    # Model Status Card
    if st.session_state.is_running:
        st.sidebar.markdown(
            '<div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); '
            'border-radius: 0.75rem; padding: 1rem; color: #d1d5db; font-size: 0.85rem;">'
            '<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; '
            'color: #10b981; font-weight: bold;">'
            '<span style="display:inline-block; width:8px; height:8px; background:#10b981; '
            'border-radius:50%; box-shadow: 0 0 8px #10b981;"></span> Model Status: ACTIVE</div>'
            f'Running live inference at {st.session_state.speed_slider:.2f}s per packet.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); '
            'border-radius: 0.75rem; padding: 1rem; color: #d1d5db; font-size: 0.85rem;">'
            '<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; '
            'color: #ef4444; font-weight: bold;">'
            '<span style="display:inline-block; width:8px; height:8px; background:#ef4444; '
            'border-radius:50%;"></span> Model Status: INACTIVE</div>'
            'Simulation stopped. Press Start to begin monitoring.</div>',
            unsafe_allow_html=True,
        )

    # ── process one packet per rerun ──────────────────────────────────────────
    if st.session_state.is_running and sim.is_running:
        pkt = sim.next_packet()
        if pkt is not None:
            result = wrapper.process_packet(pkt)

            if st.session_state.fusion_enabled:
                ml_id = st.session_state.get("fusion_ml_model")
                dl_id = st.session_state.get("fusion_dl_model")
                if ml_id and dl_id:
                    try:
                        st.session_state.latest_fusion = _compute_fusion(wrapper, pkt, ml_id, dl_id)
                    except Exception:
                        pass

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
    if st.session_state.fusion_enabled:
        ml_id = st.session_state.get("fusion_ml_model", "")
        dl_id = st.session_state.get("fusion_dl_model", "")
        active_model_label = f"Fusion ({ml_id.replace('_',' ').title()} + {dl_id.replace('_',' ').title()})" if ml_id and dl_id else "Fusion"
    else:
        active_model_label = (stats["active_model"] or "—").replace("_", " ").title()
    m4.metric("Active Model", active_model_label)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Integrated ML + DL Fusion Panel ───────────────────────────────────────
    if st.session_state.fusion_enabled:
        st.markdown("<h3>Integrated ML + DL Analysis</h3>", unsafe_allow_html=True)
        if st.session_state.latest_fusion:
            _render_fusion_panel(st.session_state.latest_fusion)
        else:
            st.info("Start the simulation to see ML + DL fusion predictions per packet.")
        st.markdown("<br>", unsafe_allow_html=True)

    # Traffic Feed
    st.markdown("<h3>Live Traffic Feed</h3>", unsafe_allow_html=True)
    recent = list(st.session_state.packet_history)[-RECENT_PACKETS_COUNT:]
    render_traffic_feed(recent)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alerts and Details
    col_alerts, col_detail = st.columns(2)

    with col_alerts:
        st.markdown('<h3>Automated Alert Feed</h3>', unsafe_allow_html=True)
        alerts = wrapper.alert_system.get_recent_alerts(10)
        render_alert_feed(alerts)

    with col_detail:
        st.markdown('<h3>Packet Feature Inspection</h3>', unsafe_allow_html=True)
        if st.session_state.latest_attack:
            render_attack_detail(st.session_state.latest_attack)
        else:
            st.info("No attacks detected yet. Attack details will appear here.")

    # ── auto-rerun while running ──────────────────────────────────────────────
    if st.session_state.is_running:
        time.sleep(st.session_state.get("speed_slider", DEFAULT_SPEED))
        st.rerun()
