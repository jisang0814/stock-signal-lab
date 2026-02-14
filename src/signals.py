from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .presets import get_profile_config


@dataclass
class SignalResult:
    action: str
    score: float
    confidence: float
    regime: str
    reasons: dict[str, float]
    trade_plan: dict[str, float]
    components: dict[str, float]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _detect_regime(close: float, sma20: float, sma60: float, slope20: float) -> str:
    if close > sma20 > sma60 and slope20 >= 0:
        return "bull"
    if close < sma20 < sma60 and slope20 <= 0:
        return "bear"
    return "sideways"


def recommend_position_size(
    account_size: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> dict[str, float]:
    account = max(0.0, account_size)
    risk_budget = account * (max(0.0, risk_pct) / 100.0)
    per_share_risk = max(0.0, entry_price - stop_price)

    if account <= 0 or risk_budget <= 0 or per_share_risk <= 0:
        return {"risk_budget": risk_budget, "qty": 0.0, "position_value": 0.0}

    qty = risk_budget / per_share_risk
    position_value = qty * entry_price
    return {
        "risk_budget": risk_budget,
        "qty": qty,
        "position_value": position_value,
    }


def evaluate_signal(row, profile: str = "balanced", event_risk_score: float = 0.0) -> SignalResult:
    close = float(row.get("close", 0))
    sma20 = _safe_float(row.get("sma20", close), close)
    sma60 = _safe_float(row.get("sma60", close), close)
    rsi = _safe_float(row.get("rsi14", 50), 50)
    macd_hist = _safe_float(row.get("macd_hist", 0), 0)
    macd_hist_z = _safe_float(row.get("macd_hist_z60", 0), 0)
    momentum20 = _safe_float(row.get("momentum20", 0), 0)
    vol20 = _safe_float(row.get("volatility20", 0), 0)
    atr14 = _safe_float(row.get("atr14", close * 0.02), close * 0.02)
    atr14_pct = _safe_float(row.get("atr14_pct", 0.02), 0.02)
    slope20 = _safe_float(row.get("sma20_slope5", 0), 0)
    slope60 = _safe_float(row.get("sma60_slope5", 0), 0)
    dist_sma20 = _safe_float(row.get("dist_sma20_pct", 0), 0)

    regime = _detect_regime(close, sma20, sma60, slope20)

    trend = 0.0
    if close > sma20 > sma60:
        trend += 22
    elif close > sma20 and sma20 >= sma60:
        trend += 16
    elif close > sma20:
        trend += 10
    elif close < sma20 < sma60:
        trend -= 16
    else:
        trend -= 8

    trend += _clamp(slope20 * 700, -8, 8)
    trend += _clamp(slope60 * 450, -5, 5)

    momentum = 0.0
    momentum += _clamp(momentum20 * 100, -10, 10)
    momentum += _clamp(macd_hist_z * 4, -8, 8)
    if 45 <= rsi <= 62:
        momentum += 9
    elif 35 <= rsi < 45:
        momentum += 6
    elif 62 < rsi <= 72:
        momentum += 3
    elif rsi > 78:
        momentum -= 10
    elif rsi < 28:
        momentum -= 3

    quality = 0.0
    if vol20 <= 0.018:
        quality += 7
    elif vol20 <= 0.03:
        quality += 3
    else:
        quality -= _clamp((vol20 - 0.03) * 350, 0, 10)

    if atr14_pct > 0.055:
        quality -= 8
    elif atr14_pct < 0.02:
        quality += 4

    # 너무 이격이 큰 구간은 신규 진입 품질 하향
    quality -= _clamp(abs(dist_sma20) * 250, 0, 8)

    profile_cfg = get_profile_config(profile)

    regime_weights = {
        "bull": {"trend": 1.1, "momentum": 1.0, "quality": 0.9},
        "sideways": {"trend": 0.8, "momentum": 0.9, "quality": 1.2},
        "bear": {"trend": 1.0, "momentum": 0.8, "quality": 1.3},
    }
    w = regime_weights[regime]
    w["trend"] += profile_cfg["trend_boost"]
    w["momentum"] += profile_cfg["momentum_boost"]
    w["quality"] += profile_cfg["quality_boost"]

    weighted_trend = trend * w["trend"]
    weighted_momentum = momentum * w["momentum"]
    weighted_quality = quality * w["quality"]

    event_penalty = _clamp(event_risk_score, 0, 1) * 8.0
    raw_score = 50.0 + weighted_trend + weighted_momentum + weighted_quality - event_penalty
    score = _clamp(raw_score, 0, 100)

    bullish_context = close > sma20 and macd_hist > 0 and slope20 >= 0
    overheated = rsi >= 78 and macd_hist < 0
    weak_context = close < sma60 and momentum20 < 0 and macd_hist <= 0

    entry_threshold = profile_cfg["entry_threshold"]
    sell_threshold = profile_cfg["sell_threshold"]

    if score >= entry_threshold and bullish_context and not overheated:
        action = "추가매수"
    elif score <= sell_threshold or weak_context or overheated:
        action = "매도"
    else:
        action = "보유"

    confidence = _clamp(abs(score - 50) * 1.7 + abs(weighted_trend) * 0.7, 0, 100)

    base_stop_mult = {"bull": 1.6, "sideways": 1.9, "bear": 2.2}[regime]
    if atr14_pct > 0.055:
        base_stop_mult += 0.4
    elif atr14_pct < 0.02:
        base_stop_mult -= 0.2
    stop_mult = _clamp(base_stop_mult + profile_cfg["stop_mult_adj"], 1.2, 3.0)

    tp1_mult = stop_mult * 1.35
    tp2_mult = stop_mult * 2.1

    stop_price = max(0.0, close - (stop_mult * atr14))
    take_profit_1 = close + (tp1_mult * atr14)
    take_profit_2 = close + (tp2_mult * atr14)
    rr = 0.0
    if close > stop_price:
        rr = (take_profit_1 - close) / (close - stop_price)

    reasons = {
        "close": close,
        "sma20": sma20,
        "sma60": sma60,
        "rsi14": rsi,
        "macd_hist": macd_hist,
        "macd_hist_z60": macd_hist_z,
        "momentum20_pct": momentum20 * 100,
        "volatility20_pct": vol20 * 100,
        "atr14": atr14,
        "atr14_pct": atr14_pct * 100,
        "sma20_slope5_pct": slope20 * 100,
        "sma60_slope5_pct": slope60 * 100,
        "dist_sma20_pct": dist_sma20 * 100,
        "event_penalty": event_penalty,
        "regime_bull": 1.0 if regime == "bull" else 0.0,
        "regime_sideways": 1.0 if regime == "sideways" else 0.0,
        "regime_bear": 1.0 if regime == "bear" else 0.0,
    }

    trade_plan = {
        "entry": close,
        "stop": stop_price,
        "tp1": take_profit_1,
        "tp2": take_profit_2,
        "rr_tp1": rr,
        "stop_mult_atr": stop_mult,
        "profile": 1.0,
    }

    components = {
        "trend": weighted_trend,
        "momentum": weighted_momentum,
        "quality": weighted_quality,
    }

    return SignalResult(
        action=action,
        score=round(score, 1),
        confidence=round(confidence, 1),
        regime=regime,
        reasons=reasons,
        trade_plan=trade_plan,
        components=components,
    )


def backtest_signal_quality(
    df,
    lookahead: int = 10,
    profile: str = "balanced",
    event_risk_score: float = 0.0,
) -> dict[str, float]:
    if len(df) < max(80, lookahead + 20):
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "expectancy_pct": 0.0,
        }

    returns = []
    wins = 0

    for i in range(60, len(df) - lookahead):
        row = df.iloc[i]
        signal = evaluate_signal(row, profile=profile, event_risk_score=event_risk_score)
        if signal.action == "보유":
            continue

        entry = _safe_float(row.get("close", 0), 0)
        future = _safe_float(df.iloc[i + lookahead].get("close", 0), 0)
        if entry <= 0 or future <= 0:
            continue

        if signal.action == "추가매수":
            ret = (future / entry - 1.0) * 100
        else:  # 매도 신호는 하락 예측으로 간주
            ret = (entry / future - 1.0) * 100

        returns.append(ret)
        if ret > 0:
            wins += 1

    trades = len(returns)
    if trades == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "expectancy_pct": 0.0,
        }

    win_rate = (wins / trades) * 100.0
    avg_return = sum(returns) / trades
    pos = [r for r in returns if r > 0]
    neg = [r for r in returns if r <= 0]
    avg_win = sum(pos) / len(pos) if pos else 0.0
    avg_loss = sum(neg) / len(neg) if neg else 0.0
    expectancy = (win_rate / 100.0) * avg_win + (1 - win_rate / 100.0) * avg_loss

    return {
        "trades": float(trades),
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "expectancy_pct": expectancy,
    }


def walk_forward_signal_quality(
    df,
    lookahead: int = 10,
    window: int = 120,
    step: int = 20,
    profile: str = "balanced",
    event_risk_score: float = 0.0,
) -> dict[str, float]:
    if len(df) < (window + lookahead + step):
        return {
            "folds": 0.0,
            "trades_avg": 0.0,
            "win_rate_avg": 0.0,
            "win_rate_std": 0.0,
            "expectancy_avg": 0.0,
        }

    fold_metrics: list[dict[str, float]] = []
    start = window
    while start + step + lookahead < len(df):
        fold_df = df.iloc[start - window : start + step].copy()
        m = backtest_signal_quality(
            fold_df,
            lookahead=lookahead,
            profile=profile,
            event_risk_score=event_risk_score,
        )
        if m["trades"] > 0:
            fold_metrics.append(m)
        start += step

    if not fold_metrics:
        return {
            "folds": 0.0,
            "trades_avg": 0.0,
            "win_rate_avg": 0.0,
            "win_rate_std": 0.0,
            "expectancy_avg": 0.0,
        }

    win_rates = [m["win_rate"] for m in fold_metrics]
    trades = [m["trades"] for m in fold_metrics]
    expectancies = [m["expectancy_pct"] for m in fold_metrics]

    win_rate_avg = sum(win_rates) / len(win_rates)
    variance = sum((x - win_rate_avg) ** 2 for x in win_rates) / len(win_rates)
    win_rate_std = variance ** 0.5

    return {
        "folds": float(len(fold_metrics)),
        "trades_avg": float(sum(trades) / len(trades)),
        "win_rate_avg": float(win_rate_avg),
        "win_rate_std": float(win_rate_std),
        "expectancy_avg": float(sum(expectancies) / len(expectancies)),
    }
