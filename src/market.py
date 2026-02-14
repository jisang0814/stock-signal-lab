from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf

from .data_providers import detect_market, fetch_price_history, symbol_with_name
from .fundamentals import compute_hybrid_score, get_fundamental_score
from .indicators import add_indicators
from .signals import evaluate_signal


@dataclass
class SymbolSnapshot:
    symbol: str
    ticker: str
    market: str
    price: float
    change_pct: float
    volume_value: float
    market_cap: float
    signal: str
    score: float
    confidence: float
    rsi14: float
    fundamental_score: float
    hybrid_score: float
    hybrid_label: str


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _ticker_info(ticker: str) -> tuple[float, float]:
    try:
        info = yf.Ticker(ticker).fast_info
        mcap = _safe_float(getattr(info, "market_cap", 0.0), 0.0)
        last_price = _safe_float(getattr(info, "last_price", 0.0), 0.0)
        return mcap, last_price
    except Exception:
        return 0.0, 0.0


def build_snapshots(symbols: Iterable[str], period: str = "6mo", interval: str = "1d") -> list[SymbolSnapshot]:
    rows: list[SymbolSnapshot] = []

    for symbol in symbols:
        market = detect_market(symbol)
        try:
            ticker, df = fetch_price_history(symbol, market, period=period, interval=interval)
            if len(df) < 40:
                continue

            ind_df = add_indicators(df)
            latest = ind_df.iloc[-1]
            prev_close = float(ind_df.iloc[-2]["close"]) if len(ind_df) >= 2 else float(latest["close"])
            close = float(latest["close"])
            change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0.0

            signal = evaluate_signal(latest)
            mcap, last_price = _ticker_info(ticker)
            price = close if close > 0 else last_price
            volume_value = _safe_float(latest["volume"]) * price

            fundamental_score = get_fundamental_score(symbol=symbol, market=market, default_value=50.0)
            hybrid = compute_hybrid_score(
                fundamental_score=fundamental_score,
                technical_score=signal.score,
                confidence=signal.confidence,
            )

            rows.append(
                SymbolSnapshot(
                    symbol=symbol,
                    ticker=ticker,
                    market=market,
                    price=price,
                    change_pct=change_pct,
                    volume_value=volume_value,
                    market_cap=mcap,
                    signal=signal.action,
                    score=signal.score,
                    confidence=signal.confidence,
                    rsi14=float(latest["rsi14"]),
                    fundamental_score=hybrid.fundamental_score,
                    hybrid_score=hybrid.hybrid_score,
                    hybrid_label=hybrid.label,
                )
            )
        except Exception:
            continue

    return rows


def snapshots_to_df(rows: list[SymbolSnapshot]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def generate_ai_comment(row: pd.Series) -> str:
    signal = row.get("signal", "보유")
    score = _safe_float(row.get("score", 50.0))
    hybrid_score = _safe_float(row.get("hybrid_score", score))
    confidence = _safe_float(row.get("confidence", 50.0))
    change = _safe_float(row.get("change_pct", 0.0))
    rsi = _safe_float(row.get("rsi14", 50.0))

    if signal == "추가매수":
        stance = "추세/모멘텀이 살아 있어 분할 추가매수 우위"
    elif signal == "매도":
        stance = "리스크 관리 관점에서 비중 축소 우위"
    else:
        stance = "뚜렷한 우위가 없어 보유 관망이 합리적"

    return (
        f"{row.get('symbol_name', symbol_with_name(str(row.get('symbol', '')), str(row.get('market', 'US'))))}은(는) 기술 점수 {score:.1f}/100, 하이브리드 점수 {hybrid_score:.1f}/100, 신뢰도 {confidence:.1f}%입니다. "
        f"일일 변동률 {change:+.2f}%, RSI {rsi:.1f} 기준으로 {stance}."
    )
