"""Reusable UI rendering helpers for the Streamlit dashboard."""

from datetime import datetime
from typing import Dict, List

import streamlit as st

# ── Attack explanations ───────────────────────────────────────────────────────
ATTACK_EXPLANATIONS = {
    "Backdoor": {
        "what": "Backdoor attack — unauthorized remote access attempt",
        "danger": "Attacker tries to keep persistent access",
        "indicators": "Unusual port usage, remote shells",
        "action": "Block IP, hunt for persistence, forensic scan",
    },
    "Exploits": {
        "what": "Exploit attempt — abusing software vulnerability",
        "danger": "Can escalate privileges or drop payloads",
        "indicators": "Malformed packets, buffer overflow patterns",
        "action": "Patch target, isolate asset, review logs",
    },
    "DoS": {
        "what": "Denial of Service / Flood traffic",
        "danger": "Impacts availability, saturates resources",
        "indicators": "Spike in packets, repeated connection attempts",
        "action": "Enable rate limiting, block offending IPs",
    },
    "Reconnaissance": {
        "what": "Recon / scanning",
        "danger": "Mapping network before attack",
        "indicators": "Port sweeps, service enumeration",
        "action": "Monitor closely, consider blocking source",
    },
    "Shellcode": {
        "what": "Shellcode injection attempt",
        "danger": "Could lead to full compromise",
        "indicators": "Executable payloads in data field",
        "action": "Isolate host, run malware analysis",
    },
    "Worms": {
        "what": "Self-propagating malware",
        "danger": "Spreads laterally at speed",
        "indicators": "Outbound scanning to many hosts",
        "action": "Quarantine, block egress until cleaned",
    },
    "Fuzzers": {
        "what": "Fuzzing / malformed traffic tests",
        "danger": "Looking for crashes to exploit later",
        "indicators": "Random payloads, boundary tests",
        "action": "Activate rate limits, log payloads, patch targets",
    },
    "Analysis": {
        "what": "Traffic analysis / sniffing",
        "danger": "Metadata leakage, recon",
        "indicators": "Passive taps, unusual sniffing tools",
        "action": "Enforce encryption, monitor NICs",
    },
    "Generic": {
        "what": "Generic malicious signal",
        "danger": "Unclassified threat pattern",
        "indicators": "Abnormal protocol mix, heuristics triggered",
        "action": "Investigate session, capture PCAP",
    },
}


def fmt_ts(val, fmt: str = "%H:%M:%S") -> str:
    if isinstance(val, datetime):
        return val.strftime(fmt)
    if val is None:
        return "N/A"
    return str(val)


def fmt_pct(val) -> str:
    if isinstance(val, (int, float)):
        return f"{val:.1%}"
    return "N/A"


def fmt_dur(val) -> str:
    try:
        return f"{float(val):.3f}s"
    except (TypeError, ValueError):
        return "N/A"


# ── Live traffic row HTML ─────────────────────────────────────────────────────

def build_packet_row_html(pkt: Dict) -> str:
    is_attack = pkt.get("prediction") == "attack"
    row_cls = "attack" if is_attack else "normal"
    status_cls = "status-attack" if is_attack else "status-normal"
    label = (pkt.get("attack_type") or "ATTACK").upper() if is_attack else "NORMAL"

    return (
        f'<div class="traffic-row {row_cls}">'
        f'<div class="traffic-cell">{fmt_ts(pkt.get("timestamp"))}</div>'
        f'<div class="traffic-cell">{pkt.get("srcip", "N/A")}</div>'
        f'<div class="traffic-cell">{pkt.get("dstip", "N/A")}</div>'
        f'<div class="traffic-cell">{pkt.get("service", "-")}</div>'
        f'<div class="traffic-cell">{fmt_dur(pkt.get("dur"))}</div>'
        f'<div class="traffic-cell {status_cls}">{label}</div>'
        f'<div class="traffic-cell">{fmt_pct(pkt.get("probability"))}</div>'
        f'</div>'
    )


def render_traffic_feed(packets: List[Dict]) -> None:
    if not packets:
        st.info("Waiting for packets...")
        return
    header = (
        '<div class="traffic-row header">'
        '<div class="traffic-cell">Time</div>'
        '<div class="traffic-cell">Source IP</div>'
        '<div class="traffic-cell">Dest IP</div>'
        '<div class="traffic-cell">Service</div>'
        '<div class="traffic-cell">Duration</div>'
        '<div class="traffic-cell">Status</div>'
        '<div class="traffic-cell">Conf.</div>'
        '</div>'
    )
    rows = "".join(build_packet_row_html(p) for p in reversed(packets))
    st.markdown(f'<div class="traffic-feed">{header}{rows}</div>', unsafe_allow_html=True)


# ── Alert card HTML ───────────────────────────────────────────────────────────

def build_alert_card_html(alert: Dict) -> str:
    sev = alert.get("severity", "low")
    return (
        f'<div class="alert-card {sev}">'
        f'<div class="alert-title">[{sev.upper()}] {alert.get("attack_category", "Unknown")}</div>'
        f'<div>{alert.get("description", "")}</div>'
        f'<div class="alert-meta">Confidence: {fmt_pct(alert.get("confidence"))} | '
        f'{fmt_ts(alert.get("timestamp"))}</div>'
        f'</div>'
    )


def render_alert_feed(alerts: List[Dict]) -> None:
    if not alerts:
        st.success("No threats detected yet.")
        return
    cards = "".join(build_alert_card_html(a) for a in alerts)
    st.markdown(cards, unsafe_allow_html=True)


# ── Latest attack detail panel ────────────────────────────────────────────────

def render_attack_detail(pkt: Dict) -> None:
    atype = pkt.get("attack_type") or "Generic"
    info = ATTACK_EXPLANATIONS.get(atype, ATTACK_EXPLANATIONS["Generic"])

    html = (
        '<div class="attack-detail">'
        f'<div class="attack-header">ATTACK DETECTED: {atype.upper()}</div>'
        f'<div class="attack-info-row"><div class="attack-label">WHEN</div>'
        f'<div class="attack-value">{fmt_ts(pkt.get("timestamp"), "%Y-%m-%d %H:%M:%S")}</div></div>'
        f'<div class="attack-info-row"><div class="attack-label">WHAT</div>'
        f'<div class="attack-value">{info["what"]}</div></div>'
        f'<div class="attack-info-row"><div class="attack-label">WHY DANGEROUS</div>'
        f'<div class="attack-value">{info["danger"]}</div></div>'
        f'<div class="attack-info-row"><div class="attack-label">SOURCE &amp; TARGET</div>'
        f'<div class="attack-value">From: <span style="color:#ef4444;">{pkt.get("srcip","N/A")}</span>'
        f' &rarr; To: <span style="color:#fbbf24;">{pkt.get("dstip","N/A")}</span>'
        f' | Service: <span style="color:#38bdf8;">{pkt.get("service","-")}</span>'
        f' | Duration: {fmt_dur(pkt.get("dur"))}</div></div>'
        f'<div class="attack-info-row"><div class="attack-label">MODEL CONFIDENCE</div>'
        f'<div class="attack-value">{fmt_pct(pkt.get("probability"))}</div></div>'
        f'<div class="attack-info-row"><div class="attack-label">RECOMMENDED ACTION</div>'
        f'<div class="attack-value" style="color:#fca5a5;font-weight:800;">{info["action"]}</div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
