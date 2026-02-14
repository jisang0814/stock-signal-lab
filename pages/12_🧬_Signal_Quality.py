from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_providers import fetch_price_history
from src.data_providers import symbol_with_name
from src.quality import summarize_calibration
from src.trade_log import load_signal_log

st.title("🧬 Signal Quality")
st.caption("신호 신뢰도 대비 실제 성과(교정/캘리브레이션) 점검")


@st.cache_data(ttl=600)
def _current_price(symbol: str, market: str) -> float:
    _, px = fetch_price_history(symbol, market, period="3mo", interval="1d")
    return float(px.iloc[-1]["close"])


def _aligned_return(action: str, entry: float, now: float) -> float:
    if entry <= 0 or now <= 0:
        return 0.0
    if action == "매도":
        return (entry / now - 1.0) * 100.0
    return (now / entry - 1.0) * 100.0


log_df = load_signal_log(limit=2000)
if log_df.empty:
    st.info("시그널 로그가 없습니다.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    market_opt = st.selectbox("시장", ["ALL", "US", "KR"], index=0)
with c2:
    profile_opt = st.selectbox("프리셋", ["ALL", "aggressive", "balanced", "conservative"], index=0)
with c3:
    max_rows = st.slider("평가 샘플 수", 50, 500, 180, 10)

flt = log_df.copy()
flt["ts"] = pd.to_datetime(flt["ts"], errors="coerce")
flt = flt.dropna(subset=["ts"]).sort_values("ts", ascending=False)
if market_opt != "ALL":
    flt = flt[flt["market"].astype(str).str.upper() == market_opt]
if profile_opt != "ALL":
    flt = flt[flt["profile"].astype(str) == profile_opt]
flt = flt.head(max_rows)

rows = []
for _, r in flt.iterrows():
    symbol = str(r.get("symbol", "")).strip()
    market = str(r.get("market", "US")).strip()
    action = str(r.get("action", "보유"))
    conf = float(pd.to_numeric(r.get("confidence", 50), errors="coerce") or 50)
    entry = float(pd.to_numeric(r.get("price", 0), errors="coerce") or 0)
    if not symbol or entry <= 0:
        continue
    try:
        now = _current_price(symbol, market)
    except Exception:
        continue
    ret = _aligned_return(action, entry, now)
    rows.append(
        {
            "ts": r.get("ts"),
            "symbol": symbol,
            "symbol_name": symbol_with_name(symbol, market),
            "market": market,
            "profile": str(r.get("profile", "")),
            "action": action,
            "confidence": conf,
            "aligned_return_pct": ret,
            "win": 1 if ret > 0 else 0,
        }
    )

outcome_df = pd.DataFrame(rows)
if outcome_df.empty:
    st.warning("평가 가능한 샘플이 없습니다.")
    st.stop()

cal = summarize_calibration(outcome_df, bins=5, min_samples_per_bin=5)
sm = cal["summary"]
bin_df = cal["by_bin"]

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("샘플 수", f"{sm['samples']}")
with m2:
    st.metric("정방향 승률", f"{sm['win_rate']:.1f}%")
with m3:
    st.metric("평균 수익률", f"{sm['avg_return']:+.2f}%")
with m4:
    st.metric("기대값", f"{sm['expectancy']:+.2f}%")

st.metric("Brier Score", f"{sm['brier']:.4f}")

st.subheader("신뢰도 구간별 실제 성과")
if not bin_df.empty:
    st.dataframe(
        bin_df[["conf_bin", "samples", "pred_p_pct", "win_rate_pct", "avg_return"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("구간별 표본이 부족합니다.")

st.subheader("샘플 상세")
show_cols = ["ts", "symbol_name", "market", "profile", "action", "confidence", "aligned_return_pct", "win"]
st.dataframe(outcome_df.sort_values("ts", ascending=False)[show_cols], use_container_width=True, hide_index=True)
