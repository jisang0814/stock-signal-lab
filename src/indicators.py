from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    high = out["high"]
    low = out["low"]
    close = out["close"]
    out["sma20"] = close.rolling(20).mean()
    out["sma60"] = close.rolling(60).mean()
    out["sma20_slope5"] = out["sma20"].pct_change(5)
    out["sma60_slope5"] = out["sma60"].pct_change(5)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["macd_hist_z60"] = (
        (out["macd_hist"] - out["macd_hist"].rolling(60).mean())
        / out["macd_hist"].rolling(60).std().replace(0, np.nan)
    ).fillna(0)

    out["rsi14"] = _rsi(close, 14)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr14_pct"] = (out["atr14"] / close.replace(0, np.nan)).fillna(0)

    out["ret1"] = close.pct_change()
    out["volatility20"] = out["ret1"].rolling(20).std()
    out["momentum20"] = close.pct_change(20)
    out["dist_sma20_pct"] = ((close / out["sma20"]) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
    out["dist_sma60_pct"] = ((close / out["sma60"]) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)

    return out
