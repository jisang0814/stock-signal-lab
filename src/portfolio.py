from __future__ import annotations


def evaluate_position(avg_buy_price: float, quantity: float, current_price: float) -> dict:
    invested = avg_buy_price * quantity
    current_value = current_price * quantity
    pnl = current_value - invested

    pnl_pct = 0.0
    if invested > 0:
        pnl_pct = (pnl / invested) * 100

    return {
        "invested": invested,
        "current_value": current_value,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
    }
