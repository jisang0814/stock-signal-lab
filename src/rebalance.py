from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rebalance_plan(
    symbols: list[str],
    current_w: np.ndarray,
    target_w: np.ndarray,
    total_capital: float,
    threshold: float = 0.02,
    fee_bps: float = 5.0,
    tax_bps: float = 0.0,
) -> dict:
    current_w = np.asarray(current_w, dtype=float)
    target_w = np.asarray(target_w, dtype=float)

    current_w = current_w / current_w.sum() if current_w.sum() > 0 else current_w
    target_w = target_w / target_w.sum() if target_w.sum() > 0 else target_w

    delta = target_w - current_w
    trade_w = np.where(np.abs(delta) >= threshold, delta, 0.0)

    trade_notional = trade_w * float(total_capital)
    turnover = 0.5 * np.abs(trade_w).sum()

    cost_rate = (fee_bps + tax_bps) / 10000.0
    total_cost = float(np.abs(trade_notional).sum() * cost_rate)

    plan_df = pd.DataFrame(
        {
            "symbol": symbols,
            "current_w": current_w,
            "target_w": target_w,
            "delta_w": delta,
            "trade_w": trade_w,
            "trade_notional": trade_notional,
            "action": ["BUY" if x > 0 else ("SELL" if x < 0 else "HOLD") for x in trade_w],
        }
    )

    return {
        "plan": plan_df,
        "turnover": float(turnover),
        "total_cost": total_cost,
    }
