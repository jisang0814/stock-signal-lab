from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EVENT_PATH = DATA_DIR / "macro_events.csv"


def _ensure_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENT_PATH.exists():
        pd.DataFrame(columns=["date", "name", "market", "severity"]).to_csv(EVENT_PATH, index=False)


def load_macro_events() -> pd.DataFrame:
    _ensure_file()
    df = pd.read_csv(EVENT_PATH)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(1).clip(1, 3)
    df["market"] = df["market"].fillna("ALL").astype(str).str.upper()
    return df.dropna(subset=["date"])


def add_macro_event(event_date: date, name: str, market: str, severity: int) -> None:
    _ensure_file()
    row = {
        "date": event_date.isoformat(),
        "name": name.strip() or "Macro Event",
        "market": (market or "ALL").upper(),
        "severity": int(max(1, min(3, severity))),
    }
    pd.DataFrame([row]).to_csv(EVENT_PATH, mode="a", index=False, header=False)


def overwrite_macro_events(df: pd.DataFrame) -> None:
    _ensure_file()
    cols = ["date", "name", "market", "severity"]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols]
    out.to_csv(EVENT_PATH, index=False)


def export_macro_events_csv_bytes() -> bytes:
    _ensure_file()
    return EVENT_PATH.read_bytes()


def compute_macro_risk(market: str, ref_date: date | None = None, max_window_days: int = 7) -> dict:
    df = load_macro_events()
    if df.empty:
        return {
            "risk_score": 0.0,
            "risk_label": "없음",
            "days_to_event": None,
            "note": "등록된 거시 이벤트가 없습니다.",
        }

    today = ref_date or datetime.utcnow().date()
    mkt = (market or "US").upper()
    candidates = df[(df["market"].isin(["ALL", mkt])) & (df["date"] >= today)].copy()

    if candidates.empty:
        return {
            "risk_score": 0.0,
            "risk_label": "없음",
            "days_to_event": None,
            "note": "가까운 거시 이벤트가 없습니다.",
        }

    candidates["days_to_event"] = candidates["date"].apply(lambda d: (d - today).days)
    near = candidates[candidates["days_to_event"] <= max_window_days]

    if near.empty:
        best = candidates.sort_values("days_to_event").iloc[0]
        return {
            "risk_score": 0.0,
            "risk_label": "없음",
            "days_to_event": float(best["days_to_event"]),
            "note": f"다음 이벤트: {best['name']} ({int(best['days_to_event'])}일 후)",
        }

    best = near.sort_values(["severity", "days_to_event"], ascending=[False, True]).iloc[0]
    days = float(best["days_to_event"])
    sev = float(best["severity"])

    # severity(1~3)와 남은 일수(0~7)를 이용해 0~1 리스크로 변환
    sev_score = sev / 3.0
    time_score = max(0.0, (max_window_days - days) / max_window_days)
    risk_score = min(1.0, 0.55 * sev_score + 0.45 * time_score)

    if risk_score >= 0.75:
        label = "높음"
    elif risk_score >= 0.45:
        label = "중간"
    elif risk_score > 0:
        label = "낮음"
    else:
        label = "없음"

    return {
        "risk_score": float(risk_score),
        "risk_label": label,
        "days_to_event": days,
        "note": f"{best['name']}까지 약 {int(days)}일",
    }
