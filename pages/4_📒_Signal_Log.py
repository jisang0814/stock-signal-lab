from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_providers import fetch_price_history, symbol_with_name
from src.trade_log import load_signal_log, summarize_logs

st.title("📒 Signal Log")
st.caption("저장한 시그널 기록과 추적 성과를 확인합니다.")

log_df = load_signal_log(limit=2000)
summary = summarize_logs(log_df)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("기록 수", summary["rows"])
with k2:
    st.metric("추가매수 기록", summary["buy_signals"])
with k3:
    st.metric("매도 기록", summary["sell_signals"])
with k4:
    st.metric("평균 점수", f"{summary['avg_score']:.1f}")

if log_df.empty:
    st.info("저장된 로그가 없습니다. Analysis 페이지에서 시그널을 기록하세요.")
    st.stop()

st.markdown("### 로그 필터")
f1, f2, f3, f4 = st.columns(4)
with f1:
    days_filter = st.selectbox("기간", [7, 30, 90, 180, 365, 9999], index=2, format_func=lambda x: "전체" if x == 9999 else f"최근 {x}일")
with f2:
    market_filter = st.multiselect("시장", sorted(log_df["market"].dropna().astype(str).unique().tolist()), default=sorted(log_df["market"].dropna().astype(str).unique().tolist()))
with f3:
    profile_filter = st.multiselect("프리셋", sorted(log_df["profile"].dropna().astype(str).unique().tolist()), default=sorted(log_df["profile"].dropna().astype(str).unique().tolist()))
with f4:
    min_score = st.slider("최소 점수", 0, 100, 0)

filtered_log = log_df.copy()
filtered_log["symbol_name"] = filtered_log.apply(
    lambda r: symbol_with_name(str(r.get("symbol", "")), str(r.get("market", "US"))),
    axis=1,
)
filtered_log["ts_dt"] = pd.to_datetime(filtered_log["ts"], errors="coerce")
if days_filter != 9999:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_filter)
    filtered_log = filtered_log[filtered_log["ts_dt"] >= cutoff]
if market_filter:
    filtered_log = filtered_log[filtered_log["market"].astype(str).isin(market_filter)]
if profile_filter:
    filtered_log = filtered_log[filtered_log["profile"].astype(str).isin(profile_filter)]
filtered_log = filtered_log[pd.to_numeric(filtered_log["score"], errors="coerce").fillna(0) >= min_score]

st.subheader("기록 목록")
show_cols = [
    "ts",
    "symbol_name",
    "market",
    "profile",
    "action",
    "score",
    "confidence",
    "price",
    "stop",
    "tp1",
    "rr_tp1",
]
avail_cols = [c for c in show_cols if c in filtered_log.columns]
st.dataframe(filtered_log.sort_values("ts", ascending=False)[avail_cols], use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("최근 50건 추적 성과")

recent = filtered_log.sort_values("ts", ascending=False).head(50).copy()
apply_slippage = st.checkbox("슬리피지 반영 성과 보기", value=True)
horizon_days = st.slider("사후 검증 기간(일)", 3, 30, 10)
rows = []


@st.cache_data(ttl=600)
def _load_live_snapshot(symbol: str, market: str) -> tuple[float, float]:
    _, px_df = fetch_price_history(symbol, market, period="3mo", interval="1d")
    now_price = float(px_df.iloc[-1]["close"])
    volume_value = float(px_df.iloc[-1]["volume"]) * now_price
    return now_price, volume_value


@st.cache_data(ttl=600)
def _load_history(symbol: str, market: str) -> pd.DataFrame:
    _, px_df = fetch_price_history(symbol, market, period="2y", interval="1d")
    hist = px_df.copy()
    hist = hist.sort_index()
    hist["date"] = pd.to_datetime(hist.index).tz_localize(None).date
    return hist


def _slippage_bps(volume_value: float) -> float:
    # 유동성에 따른 단순 슬리피지 모델 (왕복)
    if volume_value >= 300_000_000:
        return 6.0
    if volume_value >= 100_000_000:
        return 10.0
    if volume_value >= 30_000_000:
        return 18.0
    if volume_value >= 10_000_000:
        return 28.0
    return 45.0


for _, row in recent.iterrows():
    symbol = str(row.get("symbol", "")).strip()
    market = str(row.get("market", "US")).strip()
    entry = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
    action = str(row.get("action", "보유"))

    if not symbol or entry <= 0:
        continue

    try:
        now_price, volume_value = _load_live_snapshot(symbol, market)

        if action == "추가매수":
            ret = (now_price / entry - 1.0) * 100
        elif action == "매도":
            ret = (entry / now_price - 1.0) * 100 if now_price > 0 else 0.0
        else:
            ret = (now_price / entry - 1.0) * 100

        slip_bps = _slippage_bps(volume_value)
        slip_pct = slip_bps / 100.0
        ret_net = ret - slip_pct if apply_slippage else ret

        rows.append(
            {
                "ts": row.get("ts"),
                "symbol": symbol,
                "symbol_name": symbol_with_name(symbol, market),
                "profile": row.get("profile", ""),
                "market": market,
                "action": action,
                "entry": entry,
                "now": now_price,
                "volume_value": volume_value,
                "slippage_bps": slip_bps,
                "return_raw_pct": ret,
                "return_net_pct": ret_net,
            }
        )
    except Exception:
        continue

if rows:
    perf_df = pd.DataFrame(rows)
    st.dataframe(
        perf_df[
            [
                "ts",
                "symbol_name",
                "profile",
                "market",
                "action",
                "entry",
                "now",
                "volume_value",
                "slippage_bps",
                "return_raw_pct",
                "return_net_pct",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("추적 건수", len(perf_df))
    with p2:
        st.metric("평균 수익률(원시)", f"{perf_df['return_raw_pct'].mean():+.2f}%")
    with p3:
        st.metric("평균 수익률(슬립반영)", f"{perf_df['return_net_pct'].mean():+.2f}%")

    p4, p5 = st.columns(2)
    with p4:
        win_rate_raw = (perf_df["return_raw_pct"] > 0).mean() * 100
        st.metric("승률(원시)", f"{win_rate_raw:.1f}%")
    with p5:
        win_rate_net = (perf_df["return_net_pct"] > 0).mean() * 100
        st.metric("승률(슬립반영)", f"{win_rate_net:.1f}%")

    st.markdown("---")
    st.subheader("프리셋별 성과 비교")
    preset_df = (
        perf_df.groupby("profile", dropna=False)
        .agg(
            count=("symbol", "count"),
            avg_raw=("return_raw_pct", "mean"),
            avg_net=("return_net_pct", "mean"),
            win_raw=("return_raw_pct", lambda s: (s > 0).mean() * 100),
            win_net=("return_net_pct", lambda s: (s > 0).mean() * 100),
            avg_slip_bps=("slippage_bps", "mean"),
        )
        .reset_index()
        .sort_values("avg_net", ascending=False)
    )
    st.dataframe(preset_df, use_container_width=True, hide_index=True)
else:
    st.warning("추적 가능한 최근 로그가 없습니다.")

st.markdown("---")
st.subheader("N일 후 실제 성과 라벨링")
label_rows = []
label_source = filtered_log.sort_values("ts", ascending=False).head(200).copy()
for _, row in label_source.iterrows():
    symbol = str(row.get("symbol", "")).strip()
    market = str(row.get("market", "US")).strip()
    action = str(row.get("action", "보유"))
    ts = pd.to_datetime(row.get("ts"), errors="coerce")
    if not symbol or pd.isna(ts):
        continue

    try:
        hist = _load_history(symbol, market)
        if hist.empty:
            continue

        entry_date = ts.date()
        future_date = (ts + pd.Timedelta(days=horizon_days)).date()

        entry_rows = hist[hist["date"] >= entry_date]
        future_rows = hist[hist["date"] >= future_date]
        if entry_rows.empty or future_rows.empty:
            continue

        entry = float(entry_rows.iloc[0]["close"])
        future = float(future_rows.iloc[0]["close"])
        if entry <= 0 or future <= 0:
            continue

        if action == "추가매수":
            fwd_ret = (future / entry - 1.0) * 100
            is_hit = fwd_ret > 0
        elif action == "매도":
            fwd_ret = (entry / future - 1.0) * 100
            is_hit = fwd_ret > 0
        else:
            fwd_ret = (future / entry - 1.0) * 100
            is_hit = abs(fwd_ret) <= 3

        label_rows.append(
            {
                "ts": row.get("ts"),
                "symbol": symbol,
                "symbol_name": symbol_with_name(symbol, market),
                "profile": row.get("profile", ""),
                "market": market,
                "action": action,
                f"fwd_{horizon_days}d_ret_pct": fwd_ret,
                "hit": "HIT" if is_hit else "MISS",
            }
        )
    except Exception:
        continue

if label_rows:
    label_df = pd.DataFrame(label_rows)
    label_cols = [
        "ts",
        "symbol_name",
        "market",
        "profile",
        "action",
        f"fwd_{horizon_days}d_ret_pct",
        "hit",
    ]
    st.dataframe(label_df[[c for c in label_cols if c in label_df.columns]], use_container_width=True, hide_index=True)

    l1, l2, l3 = st.columns(3)
    with l1:
        hit_rate = (label_df["hit"] == "HIT").mean() * 100
        st.metric("라벨링 건수", len(label_df))
    with l2:
        st.metric("HIT 비율", f"{hit_rate:.1f}%")
    with l3:
        st.metric("평균 선행 수익", f"{label_df[f'fwd_{horizon_days}d_ret_pct'].mean():+.2f}%")

    st.markdown("#### 프리셋별 라벨링 성과")
    by_profile = (
        label_df.groupby("profile", dropna=False)
        .agg(
            count=("symbol", "count"),
            hit_rate=("hit", lambda s: (s == "HIT").mean() * 100),
            avg_fwd_ret=(f"fwd_{horizon_days}d_ret_pct", "mean"),
        )
        .reset_index()
        .sort_values("hit_rate", ascending=False)
    )
    st.dataframe(by_profile, use_container_width=True, hide_index=True)

    # 최근 90일 기준 자동 추천 프로필
    recent_90_cut = pd.Timestamp.now() - pd.Timedelta(days=90)
    labeled_90 = label_df[pd.to_datetime(label_df["ts"], errors="coerce") >= recent_90_cut].copy()
    min_auto_samples = st.slider("자동추천 최소 표본수(로그)", 5, 40, 12, 1)
    if not labeled_90.empty:
        auto_rank = (
            labeled_90.groupby("profile", dropna=False)
            .agg(
                count=("symbol", "count"),
                hit_rate=("hit", lambda s: (s == "HIT").mean() * 100),
                avg_fwd_ret=(f"fwd_{horizon_days}d_ret_pct", "mean"),
            )
            .reset_index()
        )
        auto_rank = auto_rank[auto_rank["count"] >= min_auto_samples]
        if auto_rank.empty:
            st.info("자동추천에 필요한 표본수가 부족합니다.")
            auto_rank = None
        if auto_rank is not None:
            auto_rank["score"] = auto_rank["hit_rate"] * 0.7 + auto_rank["avg_fwd_ret"] * 0.3
            auto_rank = auto_rank.sort_values(["score", "count"], ascending=False)
            best = auto_rank.iloc[0]
            st.success(
                f"최근 90일 자동 추천 프리셋: {best['profile']} "
                f"(HIT {best['hit_rate']:.1f}%, 평균 {best['avg_fwd_ret']:+.2f}%)"
            )
else:
    st.info("라벨링 가능한 기록이 아직 충분하지 않습니다.")
