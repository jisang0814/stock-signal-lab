from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "signal_log.csv"


def _ensure_log_file():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        pd.DataFrame(
            columns=[
                "ts",
                "symbol",
                "ticker",
                "market",
                "profile",
                "action",
                "score",
                "confidence",
                "regime",
                "price",
                "stop",
                "tp1",
                "rr_tp1",
                "event_risk",
                "event_days",
            ]
        ).to_csv(LOG_PATH, index=False)


def append_signal_log(row: dict):
    _ensure_log_file()
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        **row,
    }
    pd.DataFrame([payload]).to_csv(LOG_PATH, mode="a", index=False, header=False)


def load_signal_log(limit: int = 1000) -> pd.DataFrame:
    _ensure_log_file()
    df = pd.read_csv(LOG_PATH)
    if len(df) > limit:
        return df.tail(limit).reset_index(drop=True)
    return df


def summarize_logs(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "avg_score": 0.0,
            "avg_confidence": 0.0,
        }

    return {
        "rows": int(len(df)),
        "buy_signals": int((df["action"] == "추가매수").sum()),
        "sell_signals": int((df["action"] == "매도").sum()),
        "avg_score": float(pd.to_numeric(df["score"], errors="coerce").fillna(0).mean()),
        "avg_confidence": float(pd.to_numeric(df["confidence"], errors="coerce").fillna(0).mean()),
    }
