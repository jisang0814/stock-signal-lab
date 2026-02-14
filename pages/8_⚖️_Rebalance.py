from __future__ import annotations

import numpy as np
import streamlit as st

from src.rebalance import compute_rebalance_plan

st.title("⚖️ Rebalance")
st.caption("리밸런싱 임계치/비용을 반영한 자동 리밸런싱 계획")

symbols_text = st.text_input("종목 목록", value="AAPL,MSFT,NVDA,005930")
current_w_text = st.text_input("현재 비중", value="0.25,0.25,0.25,0.25")
target_w_text = st.text_input("목표 비중", value="0.3,0.25,0.2,0.25")

c1, c2, c3, c4 = st.columns(4)
with c1:
    total_capital = st.number_input("총 자본", min_value=1000.0, value=100000.0, step=1000.0)
with c2:
    threshold = st.slider("리밸런싱 임계치(%)", 0.0, 20.0, 2.0, 0.5) / 100.0
with c3:
    fee_bps = st.slider("수수료(bps)", 0.0, 30.0, 5.0, 0.5)
with c4:
    tax_bps = st.slider("세금/기타 비용(bps)", 0.0, 50.0, 0.0, 0.5)

if st.button("리밸런싱 계획 생성", type="primary"):
    try:
        symbols = [x.strip().upper() for x in symbols_text.split(",") if x.strip()]
        cw = np.array([float(x.strip()) for x in current_w_text.split(",")], dtype=float)
        tw = np.array([float(x.strip()) for x in target_w_text.split(",")], dtype=float)
        if len(symbols) != len(cw) or len(symbols) != len(tw):
            raise ValueError("종목/현재비중/목표비중 길이가 동일해야 합니다.")

        out = compute_rebalance_plan(
            symbols=symbols,
            current_w=cw,
            target_w=tw,
            total_capital=total_capital,
            threshold=threshold,
            fee_bps=fee_bps,
            tax_bps=tax_bps,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("예상 회전율", f"{out['turnover']*100:.2f}%")
        with m2:
            st.metric("예상 총 비용", f"{out['total_cost']:,.2f}")

        st.dataframe(out["plan"], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"리밸런싱 계산 실패: {exc}")
