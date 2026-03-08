"""Page — Benchmarking Interface.

Offline model evaluation, confusion matrices, ROC curves, and comparison
tables. No business logic — delegates to WrapperEngine / ModelRepository.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config import X_TEST_PATH, Y_TEST_PATH
from src.wrapper_engine import WrapperEngine
from src.packet_processor import PacketProcessor

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(26,31,58,0.5)",
    plot_bgcolor="rgba(26,31,58,0.8)",
)


def _progressive_step(total: int) -> tuple[int, int]:
    """Return (step, default_value) scaled to dataset size."""
    if total >= 50_000:
        return 1000, min(5000, total)
    if total >= 10_000:
        return 500, min(2000, total)
    if total >= 5_000:
        return 100, min(1000, total)
    if total >= 1_000:
        return 50, min(500, total)
    if total >= 500:
        return 10, min(100, total)
    if total >= 100:
        return 5, min(50, total)
    return 1, total


@st.cache_data
def load_builtin_test():
    X = pd.read_csv(X_TEST_PATH)
    y = pd.read_csv(Y_TEST_PATH).iloc[:, 0].values
    return X, y


def _run_evaluation(wrapper, features, labels, sample_size):
    """Run evaluation across all models and return results dict."""
    rng = np.random.default_rng(42)
    total_rows = len(labels)
    indices = np.sort(rng.choice(total_rows, size=min(sample_size, total_rows), replace=False))

    model_ids = wrapper.model_repo.get_available_models()
    acc = {m: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "probs": [], "preds": []}
           for m in model_ids}

    progress = st.progress(0)
    for i, idx in enumerate(indices):
        row = features.iloc[idx:idx + 1]
        true = int(labels[idx])
        all_preds = wrapper.model_repo.predict_all(row)
        for mid, res in all_preds.items():
            pred = res["prediction"]
            prob = res["probability"]
            acc[mid]["probs"].append(prob)
            acc[mid]["preds"].append(pred)
            if true == 1 and pred == 1:
                acc[mid]["tp"] += 1
            elif true == 0 and pred == 0:
                acc[mid]["tn"] += 1
            elif true == 0 and pred == 1:
                acc[mid]["fp"] += 1
            else:
                acc[mid]["fn"] += 1
        if (i + 1) % max(1, len(indices) // 20) == 0:
            progress.progress((i + 1) / len(indices))
    progress.progress(1.0)

    # ensemble (majority voting)
    all_pred_arrays = {m: np.array(acc[m]["preds"]) for m in model_ids}
    vote_sum = sum(all_pred_arrays.values())
    ensemble_preds = (vote_sum > len(model_ids) / 2).astype(int)
    ensemble_probs = np.mean([np.array(acc[m]["probs"]) for m in model_ids], axis=0)
    sampled_labels = labels[indices]

    cm_e = confusion_matrix(sampled_labels, ensemble_preds, labels=[0, 1])
    tn_e, fp_e, fn_e, tp_e = cm_e.ravel() if cm_e.size == 4 else (0, 0, 0, 0)
    acc["ensemble"] = {
        "tp": int(tp_e), "tn": int(tn_e), "fp": int(fp_e), "fn": int(fn_e),
        "probs": ensemble_probs.tolist(),
        "preds": ensemble_preds.tolist(),
    }

    # build summary table
    summary_rows = []
    for mid, s in acc.items():
        total = s["tp"] + s["tn"] + s["fp"] + s["fn"]
        if total == 0:
            continue
        accuracy = (s["tp"] + s["tn"]) / total
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0
        rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr = s["fp"] / (s["fp"] + s["tn"]) if (s["fp"] + s["tn"]) else 0
        avg_prob = np.mean(s["probs"]) if s["probs"] else 0
        summary_rows.append({
            "Model": mid.replace("_", " ").title(),
            "Model ID": mid,
            "Accuracy": accuracy,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "FPR": fpr,
            "Avg Attack Prob": avg_prob,
        })

    return {
        "summary": summary_rows,
        "acc": acc,
        "labels": sampled_labels,
        "sample_size": len(indices),
    }


def _render_metrics_tab(wrapper):
    """Render the Metrics sub-tab."""
    # ── static offline metrics ────────────────────────────────────────────
    st.markdown("#### Static Model Metrics (from training phase)")
    offline = wrapper.model_repo.get_offline_metrics()
    if offline:
        rows = []
        for mid, m in offline.items():
            rows.append({
                "Model": mid.replace("_", " ").title(),
                "Accuracy": m.get("accuracy", 0),
                "Precision": m.get("precision", 0),
                "Recall": m.get("recall", 0),
                "F1 Score": m.get("f1_score", 0),
                "ROC-AUC": m.get("roc_auc", 0),
            })
        offline_df = pd.DataFrame(rows)
        styled = offline_df.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}",
            "Recall": "{:.2%}", "F1 Score": "{:.2%}", "ROC-AUC": "{:.4f}",
        }).highlight_max(subset=["Accuracy", "F1 Score", "ROC-AUC"], color="#14532d")
        st.dataframe(styled, use_container_width=True, height=250)
    else:
        st.info("No offline metrics file found. They will appear once models are trained.")

    st.markdown("---")

    # ── on-demand evaluation results ──────────────────────────────────────
    if "bench_results" in st.session_state:
        res = st.session_state.bench_results
        summary_df = pd.DataFrame(res["summary"])
        display_df = summary_df.drop(columns=["Model ID"], errors="ignore")

        st.markdown("#### On-Demand Evaluation Results")
        st.caption(f"{res['sample_size']:,} samples evaluated &mdash; {res.get('dataset_name', 'built-in test set')}")

        styled = display_df.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
            "F1 Score": "{:.2%}", "FPR": "{:.2%}", "Avg Attack Prob": "{:.2%}",
        }).highlight_max(
            subset=["Accuracy", "F1 Score"], color="#14532d",
        ).highlight_min(
            subset=["FPR"], color="#14532d",
        )
        st.dataframe(styled, use_container_width=True, height=350)
    else:
        st.info("Run an evaluation above to see on-demand metrics.")


def _render_comparison_tab():
    """Render the Comparison sub-tab."""
    if "bench_results" not in st.session_state:
        st.info("Run an evaluation first to unlock model comparison charts.")
        return

    res = st.session_state.bench_results
    model_keys = list(res["acc"].keys())

    # ── Confusion matrix ──────────────────────────────────────────────────
    st.markdown("#### Confusion Matrix")
    cm_model = st.selectbox(
        "Select model",
        options=model_keys,
        format_func=lambda m: m.replace("_", " ").title(),
        key="cm_model",
    )

    s = res["acc"][cm_model]
    cm = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Normal", "Attack"], y=["Normal", "Attack"],
        colorscale="RdYlGn_r", text=cm, texttemplate="%{text}",
        textfont={"size": 18}, showscale=False,
    ))
    fig.update_layout(
        title=f"{cm_model.replace('_', ' ').title()} — Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Actual",
        height=400, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── ROC curves ────────────────────────────────────────────────────────
    st.markdown("#### ROC Curves")
    colors = ["#00d4ff", "#ff00ea", "#00ff88", "#fbbf24", "#f59e0b", "#8b5cf6"]
    fig_roc = go.Figure()
    for i, mid in enumerate(model_keys):
        probs = np.array(res["acc"][mid]["probs"])
        if len(np.unique(probs)) < 2:
            continue
        try:
            fpr_c, tpr_c, _ = roc_curve(res["labels"], probs)
            auc_val = auc(fpr_c, tpr_c)
        except Exception:
            continue
        display = mid.replace("_", " ").title()
        if mid == "ensemble":
            display = "Ensemble (Majority Voting)"
        fig_roc.add_trace(go.Scatter(
            x=fpr_c, y=tpr_c, mode="lines",
            name=f"{display} (AUC={auc_val:.3f})",
            line=dict(color=colors[i % len(colors)], width=2 if mid != "ensemble" else 3),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color="gray", dash="dash", width=1),
    ))
    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=500, legend=dict(x=0.55, y=0.1), **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── Attack probability distribution ───────────────────────────────────
    st.markdown("#### Average Attack Probability Distribution")
    prob_data = [{"Model": r["Model"], "Avg Attack Prob": r["Avg Attack Prob"]}
                 for r in res["summary"]]
    prob_df = pd.DataFrame(prob_data)
    fig_prob = px.bar(
        prob_df, x="Model", y="Avg Attack Prob",
        color="Avg Attack Prob", color_continuous_scale="RdYlGn_r",
        text="Avg Attack Prob",
    )
    fig_prob.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig_prob.update_layout(
        yaxis_title="Avg Attack Probability",
        yaxis_range=[0, min(1.0, prob_df["Avg Attack Prob"].max() + 0.15)],
        height=420,
        coloraxis_colorbar=dict(
            title="Risk", tickvals=[0, 0.5, 1],
            ticktext=["Safe", "Uncertain", "Attack"],
        ),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_prob, use_container_width=True)


def render():
    """Render the benchmarking page."""
    # ── session state ─────────────────────────────────────────────────────────
    if "bench_wrapper" not in st.session_state:
        st.session_state.bench_wrapper = WrapperEngine()

    wrapper: WrapperEngine = st.session_state.bench_wrapper

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="color:#f8fafc;">Benchmarking Interface</h2>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        # ── dataset source + evaluation trigger ───────────────────────────────────
        st.markdown("### Dataset & Evaluation")

        source = st.selectbox(
            "Dataset Source",
            ["Built-in test set", "Upload CSV / Parquet"],
            label_visibility="collapsed",
        )

        features, labels = None, None
        dataset_name = "UNSW-NB15 built-in test split"

        if source == "Built-in test set":
            features, labels = load_builtin_test()
        else:
            uploaded = st.file_uploader(
                "Upload dataset (must contain a `label` column)",
                type=["csv", "parquet"],
            )
            if uploaded is not None:
                try:
                    if uploaded.name.lower().endswith(".parquet"):
                        custom = pd.read_parquet(uploaded)
                    else:
                        custom = pd.read_csv(uploaded)
                    if "label" not in custom.columns:
                        st.error("Uploaded file must contain a `label` column.")
                    else:
                        processor = PacketProcessor()
                        labels = custom["label"].values
                        features = processor.transform(custom)
                        dataset_name = uploaded.name
                        st.success(f"Loaded {len(custom):,} rows from {uploaded.name}.")
                except Exception as exc:
                    st.error(f"Failed to load: {exc}")

        if features is not None and labels is not None:
            total_rows = len(labels)
            step, default_val = _progressive_step(total_rows)

            sample_size = st.slider(
                "Sample size",
                min_value=step,
                max_value=total_rows,
                value=default_val,
                step=step,
            )
            
            run_eval = st.button("Run Evaluation", use_container_width=True)

            if run_eval:
                with st.spinner(f"Evaluating on {sample_size:,} rows from {dataset_name}..."):
                    results = _run_evaluation(wrapper, features, labels, sample_size)
                    results["dataset_name"] = dataset_name
                    st.session_state.bench_results = results

    # ── Sub-tabs: Metrics | Comparison ────────────────────────────────────────
    tab_metrics, tab_comparison = st.tabs(["Metrics", "Comparison"])

    with tab_metrics:
        _render_metrics_tab(wrapper)

    with tab_comparison:
        _render_comparison_tab()
