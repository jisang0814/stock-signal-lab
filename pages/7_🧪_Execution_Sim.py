from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from src.data_providers import fetch_price_history
from src.indicators import add_indicators
from src.quality import summarize_calibration
from src.simulator import run_execution_simulation
from src.trade_log import load_signal_log

st.title("🧪 Execution Simulator")
st.caption("수수료/슬리피지/부분체결(유동성 제한) 기반 주문 시뮬레이션")

c1, c2, c3 = st.columns(3)
with c1:
    symbol = st.text_input("종목", value="AAPL")
with c2:
    market = st.selectbox("시장", ["US", "KR"], index=0)
with c3:
    period = st.selectbox("기간", ["6mo", "1y", "2y"], index=1)

profile = st.selectbox("시그널 프리셋", ["aggressive", "balanced", "conservative"], index=1)

p1, p2, p3 = st.columns(3)
with p1:
    fee_bps = st.slider("수수료(bps)", 0.0, 30.0, 5.0, 0.5)
with p2:
    slip_bps = st.slider("슬리피지(bps)", 0.0, 50.0, 8.0, 0.5)
with p3:
    adv_limit = st.slider("일거래대금 대비 체결한도(%)", 1.0, 30.0, 5.0, 1.0)

q1, q2 = st.columns(2)
with q1:
    order_size = st.slider("주문 크기(자산 대비 %)", 5.0, 50.0, 20.0, 1.0)
with q2:
    initial_cash = st.number_input("초기자본", min_value=1000.0, value=100000.0, step=1000.0)

r1, r2 = st.columns(2)
with r1:
    quality_adjust = st.checkbox("품질 기반 주문 크기 자동조정", value=True)
with r2:
    calib_samples = st.slider("교정 샘플 수", 60, 300, 140, 10)


@st.cache_data(ttl=600)
def _current_price(symbol: str, market: str) -> float:
    _, px_df = fetch_price_history(symbol, market, period="3mo", interval="1d")
    return float(px_df.iloc[-1]["close"])


@st.cache_data(ttl=600)
def _profile_calib(profile_name: str, market_name: str, max_rows: int) -> dict:
    logs = load_signal_log(limit=2200)
    if logs.empty:
        return {"by_bin": pd.DataFrame(), "summary": {"samples": 0}}
    logs["ts_dt"] = pd.to_datetime(logs["ts"], errors="coerce")
    logs = logs.dropna(subset=["ts_dt"]).sort_values("ts_dt", ascending=False)
    logs = logs[logs["profile"].astype(str) == profile_name]
    logs = logs[logs["market"].astype(str).str.upper() == market_name.upper()]
    logs = logs.head(max_rows)
    rows = []
    for _, row in logs.iterrows():
        entry = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
        if entry <= 0:
            continue
        s = str(row.get("symbol", "")).strip()
        m = str(row.get("market", "US")).strip()
        action = str(row.get("action", "보유"))
        conf = float(pd.to_numeric(row.get("confidence", 50), errors="coerce") or 50)
        if not s:
            continue
        try:
            now = _current_price(s, m)
        except Exception:
            continue
        ret = (entry / now - 1.0) * 100 if action == "매도" and now > 0 else (now / entry - 1.0) * 100
        rows.append({"confidence": conf, "aligned_return_pct": ret})
    if not rows:
        return {"by_bin": pd.DataFrame(), "summary": {"samples": 0}}
    return summarize_calibration(pd.DataFrame(rows), bins=5, min_samples_per_bin=5)

if st.button("시뮬레이션 실행", type="primary"):
    try:
        _, df = fetch_price_history(symbol, market, period=period, interval="1d")
        ind = add_indicators(df)
        calib = _profile_calib(profile, market, calib_samples) if quality_adjust else {"by_bin": pd.DataFrame(), "summary": {"samples": 0}}
        out = run_execution_simulation(
            ind,
            profile=profile,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            slippage_bps=slip_bps,
            adv_limit_pct=adv_limit,
            order_size_pct=order_size,
            quality_adjust=quality_adjust,
            calib_by_bin=calib.get("by_bin", pd.DataFrame()),
        )

        curve = out["equity_curve"]
        trades = out["trades"]

        if curve.empty:
            st.warning("시뮬레이션 결과가 비어있습니다.")
            st.stop()

        buy_hold = initial_cash * (curve["close"].iloc[-1] / curve["close"].iloc[0])
        sim_ret = (out["final_equity"] / initial_cash - 1) * 100
        bh_ret = (buy_hold / initial_cash - 1) * 100

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("최종자산", f"{out['final_equity']:,.2f}")
        with m2:
            st.metric("전략 수익률", f"{sim_ret:+.2f}%")
        with m3:
            st.metric("Buy&Hold 수익률", f"{bh_ret:+.2f}%")
        with m4:
            avg_mult = float(pd.to_numeric(curve.get("quality_mult"), errors="coerce").fillna(1.0).mean()) if "quality_mult" in curve.columns else 1.0
            st.metric("평균 주문 배수", f"{avg_mult:.2f}x")

        fig = px.line(curve, x="ts", y=["equity", "cash"], title="Equity Curve")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        if not trades.empty:
            st.subheader("체결 로그")
            st.dataframe(trades, use_container_width=True, hide_index=True)
        else:
            st.info("체결이 발생하지 않았습니다.")
    except Exception as exc:
        st.error(f"시뮬레이션 실패: {exc}")
