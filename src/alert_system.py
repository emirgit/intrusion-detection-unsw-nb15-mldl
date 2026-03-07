"""AlertSystem — generates and stores severity-tagged alerts for detected attacks."""

from datetime import datetime
from typing import Dict, List, Optional

from src.config import (
    SEVERITY_CRITICAL_THRESHOLD,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_MEDIUM_THRESHOLD,
    MAX_ALERTS,
)


class AlertSeverity:
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Maps attack category -> base severity (overridden by confidence thresholds)
_CATEGORY_SEVERITY = {
    "Backdoor": AlertSeverity.CRITICAL,
    "Shellcode": AlertSeverity.CRITICAL,
    "Exploits": AlertSeverity.HIGH,
    "DoS": AlertSeverity.HIGH,
    "Worms": AlertSeverity.HIGH,
    "Reconnaissance": AlertSeverity.MEDIUM,
    "Fuzzers": AlertSeverity.MEDIUM,
    "Analysis": AlertSeverity.LOW,
    "Generic": AlertSeverity.LOW,
}

_ALERT_MESSAGES = {
    "Backdoor": "Backdoor attack detected — unauthorized remote access attempt",
    "Shellcode": "Shellcode injection attempt detected",
    "Exploits": "Exploit attempt — software vulnerability abuse",
    "DoS": "Denial of Service flood traffic detected",
    "Worms": "Self-propagating worm activity detected",
    "Reconnaissance": "Reconnaissance / port scanning detected",
    "Fuzzers": "Fuzzing / malformed traffic attack",
    "Analysis": "Traffic analysis / sniffing activity",
    "Generic": "Generic malicious activity detected",
}


class AlertSystem:
    """Generates, stores, and queries security alerts."""

    def __init__(self) -> None:
        self._alerts: List[Dict] = []
        self._counts = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 0,
            AlertSeverity.MEDIUM: 0,
            AlertSeverity.LOW: 0,
        }

    # ── public API ────────────────────────────────────────────────────────

    def create_alert(
        self,
        attack_category: str,
        confidence: float,
        source_ip: str,
        dest_ip: str,
        service: str,
        extra: Optional[Dict] = None,
    ) -> Dict:
        severity = self._resolve_severity(confidence, attack_category)
        description = _ALERT_MESSAGES.get(attack_category, _ALERT_MESSAGES["Generic"])

        alert = {
            "id": len(self._alerts) + 1,
            "timestamp": datetime.now(),
            "severity": severity,
            "attack_category": attack_category,
            "confidence": round(confidence, 4),
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "service": service,
            "description": f"[{severity.upper()}] {description} | {source_ip} -> {dest_ip}:{service}",
            "extra": extra or {},
        }

        self._alerts.append(alert)
        self._counts[severity] = self._counts.get(severity, 0) + 1

        if len(self._alerts) > MAX_ALERTS:
            removed = self._alerts.pop(0)
            self._counts[removed["severity"]] -= 1

        return alert

    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        return list(reversed(self._alerts[-n:]))

    def get_summary(self) -> Dict:
        return {
            "total": len(self._alerts),
            **self._counts,
        }

    def clear(self) -> None:
        self._alerts.clear()
        for key in self._counts:
            self._counts[key] = 0

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_severity(confidence: float, category: str) -> str:
        if confidence >= SEVERITY_CRITICAL_THRESHOLD:
            return AlertSeverity.CRITICAL
        if confidence >= SEVERITY_HIGH_THRESHOLD:
            return AlertSeverity.HIGH
        if confidence >= SEVERITY_MEDIUM_THRESHOLD:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW
