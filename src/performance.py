from __future__ import annotations

import math

import pandas as pd


def _max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    cummax = close.cummax()
    dd = (close / cummax) - 1.0
    return float(dd.min() * 100)


def summarize_price_performance(df: pd.DataFrame, periods_per_year: int = 252) -> dict[str, float]:
    if df.empty or len(df) < 3:
        return {
            "return_pct": 0.0,
            "ann_return_pct": 0.0,
            "ann_vol_pct": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown_pct": 0.0,
        }

    close = df["close"].dropna()
    ret = close.pct_change().dropna()
    total_ret = (close.iloc[-1] / close.iloc[0] - 1.0) * 100

    n = len(ret)
    years = max(n / periods_per_year, 1e-9)
    ann_ret = ((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1.0) * 100
    ann_vol = float(ret.std() * math.sqrt(periods_per_year) * 100) if len(ret) > 1 else 0.0
    sharpe_like = ann_ret / ann_vol if ann_vol > 0 else 0.0

    return {
        "return_pct": float(total_ret),
        "ann_return_pct": float(ann_ret),
        "ann_vol_pct": float(ann_vol),
        "sharpe_like": float(sharpe_like),
        "max_drawdown_pct": float(_max_drawdown(close)),
    }


def compare_with_benchmark(asset_df: pd.DataFrame, bench_df: pd.DataFrame) -> dict[str, float]:
    a = summarize_price_performance(asset_df)
    b = summarize_price_performance(bench_df)

    return {
        "asset_return_pct": a["return_pct"],
        "bench_return_pct": b["return_pct"],
        "alpha_pct": a["return_pct"] - b["return_pct"],
        "asset_mdd_pct": a["max_drawdown_pct"],
        "bench_mdd_pct": b["max_drawdown_pct"],
        "mdd_gap_pct": a["max_drawdown_pct"] - b["max_drawdown_pct"],
        "asset_sharpe_like": a["sharpe_like"],
        "bench_sharpe_like": b["sharpe_like"],
        "sharpe_edge": a["sharpe_like"] - b["sharpe_like"],
    }
