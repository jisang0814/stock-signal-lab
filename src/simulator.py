from __future__ import annotations

import pandas as pd

from .quality import estimate_edge_from_confidence, recommend_size_multiplier
from .signals import evaluate_signal


def run_execution_simulation(
    df: pd.DataFrame,
    profile: str = "balanced",
    event_risk_score: float = 0.0,
    initial_cash: float = 100000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 8.0,
    adv_limit_pct: float = 5.0,
    order_size_pct: float = 20.0,
    quality_adjust: bool = False,
    calib_by_bin: pd.DataFrame | None = None,
) -> dict:
    if df.empty:
        return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame(), "final_equity": initial_cash}

    cash = float(initial_cash)
    qty = 0.0
    trades = []
    curve = []

    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0
    adv_limit = max(0.0, adv_limit_pct / 100.0)
    order_size = max(0.0, order_size_pct / 100.0)

    for ts, row in df.iterrows():
        close = float(row.get("close", 0))
        vol = float(row.get("volume", 0))
        if close <= 0:
            continue

        signal = evaluate_signal(row, profile=profile, event_risk_score=event_risk_score)
        quality_mult = 1.0
        if quality_adjust and calib_by_bin is not None and not calib_by_bin.empty:
            edge = estimate_edge_from_confidence(signal.confidence, calib_by_bin)
            quality_mult = recommend_size_multiplier(
                expected_return_pct=float(edge.get("expected_return_pct", 0.0)),
                pred_win_rate_pct=float(edge.get("pred_win_rate_pct", 50.0)),
                profile=profile,
            )

        # 주문 명목금액(포트폴리오의 일정 비율)
        equity = cash + qty * close
        desired_notional = equity * order_size * quality_mult
        daily_value = vol * close
        max_fill_notional = daily_value * adv_limit
        fill_ratio = 1.0 if desired_notional <= max_fill_notional or max_fill_notional <= 0 else max_fill_notional / desired_notional

        if signal.action == "추가매수" and cash > 0:
            notional = min(cash, desired_notional * fill_ratio)
            if notional > 0:
                exec_price = close * (1 + slip_rate)
                buy_qty = notional / exec_price
                fee = notional * fee_rate
                total_cost = notional + fee
                if total_cost <= cash:
                    cash -= total_cost
                    qty += buy_qty
                    trades.append(
                        {
                            "ts": ts,
                            "side": "BUY",
                            "price": exec_price,
                            "qty": buy_qty,
                            "notional": notional,
                            "fee": fee,
                            "fill_ratio": fill_ratio,
                            "signal": signal.action,
                        }
                    )

        elif signal.action == "매도" and qty > 0:
            sell_qty = qty * min(1.0, order_size * fill_ratio)
            if sell_qty > 0:
                exec_price = close * (1 - slip_rate)
                proceeds = sell_qty * exec_price
                fee = proceeds * fee_rate
                cash += proceeds - fee
                qty -= sell_qty
                trades.append(
                    {
                        "ts": ts,
                        "side": "SELL",
                        "price": exec_price,
                        "qty": sell_qty,
                        "notional": proceeds,
                        "fee": fee,
                        "fill_ratio": fill_ratio,
                        "signal": signal.action,
                    }
                )

        curve.append(
            {
                "ts": ts,
                "close": close,
                "cash": cash,
                "qty": qty,
                "equity": cash + qty * close,
                "quality_mult": quality_mult,
            }
        )

    curve_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    final_equity = float(curve_df.iloc[-1]["equity"]) if not curve_df.empty else initial_cash

    return {
        "equity_curve": curve_df,
        "trades": trades_df,
        "final_equity": final_equity,
    }
