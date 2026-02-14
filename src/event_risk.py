from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def get_earnings_event_risk(ticker: str) -> dict:
    now = datetime.now(timezone.utc)

    try:
        tk = yf.Ticker(ticker)
        edf = tk.get_earnings_dates(limit=8)
        if edf is None or len(edf) == 0:
            return {
                "event_type": "earnings",
                "days_to_event": None,
                "risk_score": 0.0,
                "risk_label": "없음",
                "note": "예정 실적일 데이터를 찾지 못했습니다.",
            }

        idx = pd.to_datetime(edf.index)
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")

        future_dates = [d.to_pydatetime() for d in idx if d.to_pydatetime() >= now]
        if not future_dates:
            return {
                "event_type": "earnings",
                "days_to_event": None,
                "risk_score": 0.0,
                "risk_label": "없음",
                "note": "가까운 미래 실적일이 없습니다.",
            }

        target = min(future_dates)
        days = (target - now).days

        if days <= 3:
            risk_score = 1.0
            label = "높음"
        elif days <= 7:
            risk_score = 0.6
            label = "중간"
        elif days <= 14:
            risk_score = 0.3
            label = "낮음"
        else:
            risk_score = 0.0
            label = "없음"

        return {
            "event_type": "earnings",
            "days_to_event": float(days),
            "risk_score": risk_score,
            "risk_label": label,
            "note": f"다음 실적 발표까지 약 {days}일",
        }
    except Exception as exc:
        return {
            "event_type": "earnings",
            "days_to_event": None,
            "risk_score": 0.0,
            "risk_label": "알 수 없음",
            "note": f"이벤트 조회 실패: {exc}",
        }
