from __future__ import annotations


def evaluate_kill_switch(
    vol_pct: float,
    mdd_pct: float,
    event_risk_score: float,
    vol_threshold: float = 4.0,
    mdd_threshold: float = -15.0,
    event_threshold: float = 0.75,
) -> dict:
    reasons = []

    if vol_pct >= vol_threshold:
        reasons.append(f"변동성 {vol_pct:.2f}% >= {vol_threshold:.2f}%")
    if mdd_pct <= mdd_threshold:
        reasons.append(f"MDD {mdd_pct:.2f}% <= {mdd_threshold:.2f}%")
    if event_risk_score >= event_threshold:
        reasons.append(f"이벤트 리스크 {event_risk_score:.2f} >= {event_threshold:.2f}")

    active = len(reasons) > 0
    risk_mode = "OFF" if not active else "RISK_OFF"
    action = "정상 운용" if not active else "신규 진입 차단 + 기존 포지션 축소"

    return {
        "active": active,
        "risk_mode": risk_mode,
        "action": action,
        "reasons": reasons,
    }
