from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.profile_reco_log import load_recommendation_log

st.title("🗂 Reco Track")
st.caption("추천 프리셋의 시간 흐름과 시장별 변화를 추적합니다.")

log_df = load_recommendation_log(limit=5000)
if log_df.empty:
    st.info("추천 프리셋 로그가 없습니다. Analysis에서 추천 로그를 먼저 기록하세요.")
    st.stop()

log_df["ts"] = pd.to_datetime(log_df["ts"], errors="coerce")
log_df = log_df.dropna(subset=["ts"]).copy()
log_df["market"] = log_df["market"].astype(str).str.upper()

f1, f2 = st.columns(2)
with f1:
    market_filter = st.multiselect("시장", sorted(log_df["market"].unique().tolist()), default=sorted(log_df["market"].unique().tolist()))
with f2:
    days = st.selectbox("기간", [30, 90, 180, 365, 9999], index=1, format_func=lambda x: "전체" if x == 9999 else f"최근 {x}일")

flt = log_df.copy()
if market_filter:
    flt = flt[flt["market"].isin(market_filter)]
if days != 9999:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    flt = flt[flt["ts"] >= cutoff]

if flt.empty:
    st.warning("필터 조건에 맞는 로그가 없습니다.")
    st.stop()

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("로그 수", len(flt))
with k2:
    st.metric("평균 score", f"{pd.to_numeric(flt['score'], errors='coerce').fillna(0).mean():.2f}")
with k3:
    top = flt["recommended_profile"].value_counts()
    st.metric("최다 추천", top.index[0] if not top.empty else "-")

st.subheader("추천 이력")
st.dataframe(
    flt.sort_values("ts", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.subheader("시장별 추천 분포")
dist = (
    flt.groupby(["market", "recommended_profile"], dropna=False)
    .size()
    .reset_index(name="count")
)
fig1 = px.bar(dist, x="market", y="count", color="recommended_profile", barmode="stack")
fig1.update_layout(height=360)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("시간별 추천 점수")
line_df = flt.sort_values("ts")
fig2 = px.line(
    line_df,
    x="ts",
    y="score",
    color="recommended_profile",
    markers=True,
)
fig2.update_layout(height=360)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("일별 대표 추천")
daily = line_df.copy()
daily["d"] = daily["ts"].dt.date
daily_rank = (
    daily.sort_values(["d", "score"], ascending=[True, False])
    .groupby(["d", "market"], as_index=False)
    .first()[["d", "market", "recommended_profile", "score", "samples", "hit_rate", "avg_ret"]]
)
st.dataframe(daily_rank.sort_values(["d", "market"], ascending=[False, True]), use_container_width=True, hide_index=True)
