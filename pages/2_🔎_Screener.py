from __future__ import annotations

import streamlit as st

from src.data_providers import symbol_with_name
from src.ui import ensure_df_or_stop, load_snapshot_df, render_common_sidebar

st.title("🔎 Screener")
st.caption("점수/시그널/RSI 필터로 후보 종목을 선별합니다.")

universe, period, interval = render_common_sidebar()
df = load_snapshot_df(universe, period, interval)
ensure_df_or_stop(df)
df = df.copy()
df["symbol_name"] = df.apply(lambda r: symbol_with_name(str(r.get("symbol", "")), str(r.get("market", "US"))), axis=1)

f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    score_min = st.slider("최소 점수", 0, 100, 60)
with f2:
    score_max = st.slider("최대 점수", 0, 100, 100)
with f3:
    signal_filter = st.multiselect("시그널", ["추가매수", "보유", "매도"], default=["추가매수", "보유", "매도"])
with f4:
    rsi_range = st.slider("RSI 범위", 0, 100, (30, 75))
with f5:
    min_volume_value_m = st.slider("최소 거래대금(백만)", 0, 500, 10)

screened = df[
    (df["score"] >= score_min)
    & (df["score"] <= score_max)
    & (df["signal"].isin(signal_filter))
    & (df["rsi14"].between(rsi_range[0], rsi_range[1]))
    & (df["volume_value"] >= (min_volume_value_m * 1_000_000))
].sort_values(["score", "confidence"], ascending=False)

st.metric("필터 통과 종목 수", len(screened))

st.dataframe(
    screened[
        [
            "symbol_name",
            "market",
            "price",
            "change_pct",
            "signal",
            "score",
            "confidence",
            "rsi14",
            "market_cap",
            "volume_value",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)
