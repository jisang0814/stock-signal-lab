from __future__ import annotations

import pandas as pd


def detect_signal_drift(
    perf_df: pd.DataFrame,
    recent_days: int = 30,
    baseline_days: int = 180,
    min_samples: int = 20,
) -> dict:
    if perf_df.empty:
        return {"status": "no_data", "message": "데이터 없음"}

    df = perf_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    if df.empty:
        return {"status": "no_data", "message": "유효 시계열 없음"}

    now = pd.Timestamp.now()
    recent = df[df["ts"] >= (now - pd.Timedelta(days=recent_days))]
    baseline = df[(df["ts"] >= (now - pd.Timedelta(days=baseline_days))) & (df["ts"] < (now - pd.Timedelta(days=recent_days)))]

    if len(recent) < min_samples or len(baseline) < min_samples:
        return {
            "status": "insufficient",
            "recent_samples": int(len(recent)),
            "baseline_samples": int(len(baseline)),
            "message": "표본 부족",
        }

    r = pd.to_numeric(recent["return_net_pct"], errors="coerce").dropna()
    b = pd.to_numeric(baseline["return_net_pct"], errors="coerce").dropna()
    if len(r) < min_samples or len(b) < min_samples:
        return {
            "status": "insufficient",
            "recent_samples": int(len(r)),
            "baseline_samples": int(len(b)),
            "message": "수익률 표본 부족",
        }

    r_mean = float(r.mean())
    b_mean = float(b.mean())
    delta = r_mean - b_mean

    r_win = float((r > 0).mean() * 100)
    b_win = float((b > 0).mean() * 100)
    win_delta = r_win - b_win

    drift = (delta <= -1.0) or (win_delta <= -8.0)
    status = "drift" if drift else "stable"

    return {
        "status": status,
        "recent_mean": r_mean,
        "baseline_mean": b_mean,
        "mean_delta": delta,
        "recent_win": r_win,
        "baseline_win": b_win,
        "win_delta": win_delta,
        "recent_samples": int(len(r)),
        "baseline_samples": int(len(b)),
    }
