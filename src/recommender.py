from __future__ import annotations

from typing import Iterable

from .data_providers import detect_market, fetch_price_history
from .fundamentals import compute_hybrid_score, get_fundamental_score
from .indicators import add_indicators
from .signals import evaluate_signal


def scan_candidates(symbols: Iterable[str], period: str = "6mo", interval: str = "1d") -> list[dict]:
    rows = []

    for symbol in symbols:
        market = detect_market(symbol)
        try:
            resolved, df = fetch_price_history(symbol, market, period=period, interval=interval)
            ind_df = add_indicators(df)
            latest = ind_df.iloc[-1]
            signal = evaluate_signal(latest)

            fundamental_score = get_fundamental_score(symbol=symbol, market=market, default_value=50.0)
            hybrid = compute_hybrid_score(
                fundamental_score=fundamental_score,
                technical_score=signal.score,
                confidence=signal.confidence,
            )

            rows.append(
                {
                    "symbol": symbol,
                    "ticker": resolved,
                    "market": market,
                    "price": float(latest["close"]),
                    "action": signal.action,
                    "score": signal.score,
                    "confidence": signal.confidence,
                    "fundamental_score": hybrid.fundamental_score,
                    "hybrid_score": hybrid.hybrid_score,
                    "hybrid_label": hybrid.label,
                    "rsi14": round(signal.reasons["rsi14"], 2),
                    "momentum20_pct": round(signal.reasons["momentum20_pct"], 2),
                }
            )
        except Exception:
            continue

    rows.sort(key=lambda x: (x.get("hybrid_score", x["score"]), x["score"], x["confidence"]), reverse=True)
    return rows
