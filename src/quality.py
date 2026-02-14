from __future__ import annotations

import math

import pandas as pd


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def summarize_calibration(
    outcomes_df: pd.DataFrame,
    confidence_col: str = "confidence",
    return_col: str = "aligned_return_pct",
    bins: int = 5,
    min_samples_per_bin: int = 5,
) -> dict:
    if outcomes_df.empty:
        return {
            "by_bin": pd.DataFrame(),
            "summary": {"samples": 0, "win_rate": 0.0, "avg_return": 0.0, "expectancy": 0.0, "brier": 0.0},
        }

    df = outcomes_df.copy()
    df[confidence_col] = pd.to_numeric(df[confidence_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna(subset=[confidence_col, return_col])
    if df.empty:
        return {
            "by_bin": pd.DataFrame(),
            "summary": {"samples": 0, "win_rate": 0.0, "avg_return": 0.0, "expectancy": 0.0, "brier": 0.0},
        }

    df["pred_p"] = (df[confidence_col] / 100.0).clip(0.0, 1.0)
    df["win"] = (df[return_col] > 0).astype(int)

    edges = [i * (100.0 / bins) for i in range(bins + 1)]
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(bins)]
    df["conf_bin"] = pd.cut(df[confidence_col].clip(0, 100), bins=edges, labels=labels, include_lowest=True, right=True)

    by_bin = (
        df.groupby("conf_bin", observed=False)
        .agg(
            samples=("win", "count"),
            pred_p=("pred_p", "mean"),
            win_rate=("win", "mean"),
            avg_return=(return_col, "mean"),
        )
        .reset_index()
    )
    by_bin = by_bin[by_bin["samples"] >= int(min_samples_per_bin)].copy()
    by_bin["win_rate_pct"] = by_bin["win_rate"] * 100.0
    by_bin["pred_p_pct"] = by_bin["pred_p"] * 100.0

    pos = df[df[return_col] > 0][return_col]
    neg = df[df[return_col] <= 0][return_col]
    avg_win = float(pos.mean()) if not pos.empty else 0.0
    avg_loss = float(neg.mean()) if not neg.empty else 0.0
    win_rate = float(df["win"].mean())
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    brier = float(((df["pred_p"] - df["win"]) ** 2).mean())

    summary = {
        "samples": int(len(df)),
        "win_rate": win_rate * 100.0,
        "avg_return": float(df[return_col].mean()),
        "expectancy": float(expectancy),
        "brier": brier,
    }

    return {"by_bin": by_bin, "summary": summary}


def estimate_edge_from_confidence(confidence: float, calib_by_bin: pd.DataFrame) -> dict:
    c = _safe_float(confidence, 50.0)
    if calib_by_bin is None or calib_by_bin.empty:
        return {"pred_win_rate_pct": c, "expected_return_pct": 0.0, "samples": 0}

    # Use closest bin center to avoid brittle exact-bin matching.
    tmp = calib_by_bin.copy()
    tmp["center"] = tmp["conf_bin"].astype(str).str.split("-").apply(lambda x: (float(x[0]) + float(x[1])) / 2 if len(x) == 2 else 50.0)
    tmp["dist"] = (tmp["center"] - c).abs()
    row = tmp.sort_values(["dist", "samples"], ascending=[True, False]).iloc[0]

    return {
        "pred_win_rate_pct": float(row.get("win_rate_pct", c)),
        "expected_return_pct": float(row.get("avg_return", 0.0)),
        "samples": int(row.get("samples", 0)),
    }


def recommend_size_multiplier(
    expected_return_pct: float,
    pred_win_rate_pct: float,
    profile: str = "balanced",
    min_mult: float = 0.3,
    max_mult: float = 1.4,
) -> float:
    exp_ret = _safe_float(expected_return_pct, 0.0)
    win = _safe_float(pred_win_rate_pct, 50.0)

    profile_key = str(profile)
    if profile_key == "defensive":
        profile_key = "conservative"

    profile_bias = {
        "aggressive": 0.08,
        "balanced": 0.0,
        "conservative": -0.10,
    }.get(profile_key, 0.0)

    bounds = {
        "aggressive": (0.45, 1.55),
        "balanced": (0.3, 1.4),
        "conservative": (0.2, 1.2),
    }.get(profile_key, (min_mult, max_mult))

    mult = 1.0 + profile_bias
    if exp_ret >= 1.0:
        mult += 0.25
    elif exp_ret >= 0.3:
        mult += 0.12
    elif exp_ret <= -1.0:
        mult -= 0.35
    elif exp_ret <= -0.3:
        mult -= 0.18

    if win >= 62.0:
        mult += 0.12
    elif win >= 56.0:
        mult += 0.06
    elif win <= 42.0:
        mult -= 0.18
    elif win <= 48.0:
        mult -= 0.08

    lo = max(0.0, bounds[0])
    hi = max(lo, bounds[1])
    return float(max(lo, min(hi, mult)))
