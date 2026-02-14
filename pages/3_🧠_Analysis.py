from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from src.benchmarks import benchmark_for_market
from src.data_providers import KR_UNIVERSE, US_UNIVERSE, fetch_price_history, symbol_with_name
from src.event_risk import get_earnings_event_risk
from src.indicators import add_indicators
from src.macro_events import (
    add_macro_event,
    compute_macro_risk,
    export_macro_events_csv_bytes,
    load_macro_events,
    overwrite_macro_events,
)
from src.performance import compare_with_benchmark
from src.portfolio import evaluate_position
from src.presets import PROFILE_PRESETS
from src.profile_reco_log import append_recommendation_log
from src.quality import estimate_edge_from_confidence, recommend_size_multiplier, summarize_calibration
from src.signals import (
    backtest_signal_quality,
    evaluate_signal,
    recommend_position_size,
    walk_forward_signal_quality,
)
from src.trade_log import append_signal_log, load_signal_log
from src.ui import render_common_sidebar

st.title("🧠 Analysis")
st.caption("개별 종목 시그널과 내 포지션 성과를 수치로 분석합니다.")

universe, period, interval = render_common_sidebar()


@st.cache_data(ttl=600)
def _current_price_for_symbol(symbol: str, market: str) -> float:
    _, px_df = fetch_price_history(symbol, market, period="3mo", interval="1d")
    return float(px_df.iloc[-1]["close"])


def _recommend_profile_from_recent_logs(
    horizon_days: int = 10,
    market_filter: str | None = None,
    min_samples: int = 12,
) -> str | None:
    logs = load_signal_log(limit=800)
    if logs.empty:
        return None

    logs["ts_dt"] = pd.to_datetime(logs["ts"], errors="coerce")
    recent = logs[logs["ts_dt"] >= (pd.Timestamp.now() - pd.Timedelta(days=90))].copy()
    if market_filter:
        recent = recent[recent["market"].astype(str).str.upper() == market_filter.upper()]
    if recent.empty:
        return None

    rows = []
    for _, row in recent.sort_values("ts", ascending=False).head(120).iterrows():
        symbol = str(row.get("symbol", "")).strip()
        market = str(row.get("market", "US")).strip()
        action = str(row.get("action", "보유"))
        entry = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
        profile = str(row.get("profile", "")).strip()
        if not symbol or not profile or entry <= 0:
            continue
        try:
            now = _current_price_for_symbol(symbol, market)
            if action == "추가매수":
                ret = (now / entry - 1.0) * 100
                hit = ret > 0
            elif action == "매도":
                ret = (entry / now - 1.0) * 100 if now > 0 else 0.0
                hit = ret > 0
            else:
                ret = (now / entry - 1.0) * 100
                hit = abs(ret) <= 3
            rows.append({"profile": profile, "ret": ret, "hit": 1 if hit else 0})
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    rank = (
        df.groupby("profile")
        .agg(count=("profile", "count"), hit_rate=("hit", "mean"), avg_ret=("ret", "mean"))
        .reset_index()
    )
    rank = rank[rank["count"] >= min_samples]
    if rank.empty:
        return None
    rank["score"] = rank["hit_rate"] * 70 + rank["avg_ret"] * 0.3
    rank = rank.sort_values(["score", "count"], ascending=False)
    if rank.empty:
        return None
    best = str(rank.iloc[0]["profile"])
    return best if best in PROFILE_PRESETS else None


def _recommend_profile_rank(
    market_filter: str | None = None,
    min_samples: int = 12,
) -> pd.DataFrame:
    logs = load_signal_log(limit=800)
    if logs.empty:
        return pd.DataFrame()
    logs["ts_dt"] = pd.to_datetime(logs["ts"], errors="coerce")
    recent = logs[logs["ts_dt"] >= (pd.Timestamp.now() - pd.Timedelta(days=90))].copy()
    if market_filter:
        recent = recent[recent["market"].astype(str).str.upper() == market_filter.upper()]
    if recent.empty:
        return pd.DataFrame()

    rows = []
    for _, row in recent.sort_values("ts", ascending=False).head(120).iterrows():
        symbol = str(row.get("symbol", "")).strip()
        market = str(row.get("market", "US")).strip()
        action = str(row.get("action", "보유"))
        entry = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
        profile = str(row.get("profile", "")).strip()
        if not symbol or not profile or entry <= 0:
            continue
        try:
            now = _current_price_for_symbol(symbol, market)
            if action == "추가매수":
                ret = (now / entry - 1.0) * 100
                hit = ret > 0
            elif action == "매도":
                ret = (entry / now - 1.0) * 100 if now > 0 else 0.0
                hit = ret > 0
            else:
                ret = (now / entry - 1.0) * 100
                hit = abs(ret) <= 3
            rows.append({"profile": profile, "ret": ret, "hit": 1 if hit else 0})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    rank = (
        df.groupby("profile")
        .agg(count=("profile", "count"), hit_rate=("hit", "mean"), avg_ret=("ret", "mean"))
        .reset_index()
    )
    rank = rank[rank["count"] >= min_samples]
    if rank.empty:
        return pd.DataFrame()
    rank["score"] = rank["hit_rate"] * 70 + rank["avg_ret"] * 0.3
    return rank.sort_values(["score", "count"], ascending=False)


@st.cache_data(ttl=600)
def _profile_calibration(profile: str, market_filter: str, max_rows: int = 180) -> dict:
    logs = load_signal_log(limit=2000)
    if logs.empty:
        return {"by_bin": pd.DataFrame(), "summary": {"samples": 0, "win_rate": 0.0, "avg_return": 0.0, "expectancy": 0.0, "brier": 0.0}}

    logs["ts_dt"] = pd.to_datetime(logs["ts"], errors="coerce")
    logs = logs.dropna(subset=["ts_dt"]).sort_values("ts_dt", ascending=False)
    logs = logs[logs["profile"].astype(str) == profile]
    logs = logs[logs["market"].astype(str).str.upper() == market_filter.upper()]
    logs = logs.head(max_rows)
    if logs.empty:
        return {"by_bin": pd.DataFrame(), "summary": {"samples": 0, "win_rate": 0.0, "avg_return": 0.0, "expectancy": 0.0, "brier": 0.0}}

    rows = []
    for _, row in logs.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        market = str(row.get("market", "US")).strip()
        action = str(row.get("action", "보유"))
        entry = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
        conf = float(pd.to_numeric(row.get("confidence", 50), errors="coerce") or 50)
        if not symbol or entry <= 0:
            continue
        try:
            now = _current_price_for_symbol(symbol, market)
            ret = (entry / now - 1.0) * 100 if action == "매도" and now > 0 else (now / entry - 1.0) * 100
            rows.append({"confidence": conf, "aligned_return_pct": ret})
        except Exception:
            continue

    if not rows:
        return {"by_bin": pd.DataFrame(), "summary": {"samples": 0, "win_rate": 0.0, "avg_return": 0.0, "expectancy": 0.0, "brier": 0.0}}
    return summarize_calibration(pd.DataFrame(rows), bins=5, min_samples_per_bin=5)


@st.cache_data(ttl=900)
def _event_window_performance(symbol: str, market: str, bench_symbol: str, bench_market: str) -> pd.DataFrame:
    macro_df = load_macro_events()
    if macro_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().date()
    ev = macro_df[(macro_df["date"] < today) & (macro_df["market"].isin(["ALL", market]))].copy()
    ev = ev.sort_values("date", ascending=False).head(20)
    if ev.empty:
        return pd.DataFrame()

    _, asset_df = fetch_price_history(symbol, market, period="2y", interval="1d")
    _, bench_df = fetch_price_history(bench_symbol, bench_market, period="2y", interval="1d")
    asset = asset_df.copy().sort_index()
    bench = bench_df.copy().sort_index()
    asset["d"] = pd.to_datetime(asset.index).tz_localize(None).date
    bench["d"] = pd.to_datetime(bench.index).tz_localize(None).date

    rows = []
    for _, r in ev.iterrows():
        d0 = r["date"]
        d_pre = (pd.Timestamp(d0) - pd.Timedelta(days=5)).date()
        d_post = (pd.Timestamp(d0) + pd.Timedelta(days=5)).date()

        a0 = asset[asset["d"] >= d0]
        ap = asset[asset["d"] >= d_pre]
        an = asset[asset["d"] >= d_post]
        b0 = bench[bench["d"] >= d0]
        bp = bench[bench["d"] >= d_pre]
        bn = bench[bench["d"] >= d_post]
        if a0.empty or ap.empty or an.empty or b0.empty or bp.empty or bn.empty:
            continue

        a_pre = (float(a0.iloc[0]["close"]) / float(ap.iloc[0]["close"]) - 1.0) * 100
        a_post = (float(an.iloc[0]["close"]) / float(a0.iloc[0]["close"]) - 1.0) * 100
        b_pre = (float(b0.iloc[0]["close"]) / float(bp.iloc[0]["close"]) - 1.0) * 100
        b_post = (float(bn.iloc[0]["close"]) / float(b0.iloc[0]["close"]) - 1.0) * 100

        rows.append(
            {
                "date": d0,
                "event": r.get("name", ""),
                "asset_pre5_pct": a_pre,
                "asset_post5_pct": a_post,
                "bench_pre5_pct": b_pre,
                "bench_post5_pct": b_post,
                "post5_alpha_pct": a_post - b_post,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=900)
def _sector_event_review(
    market: str,
    bench_symbol: str,
    bench_market: str,
    max_events: int = 6,
) -> pd.DataFrame:
    macro_df = load_macro_events()
    if macro_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().date()
    ev = macro_df[(macro_df["date"] < today) & (macro_df["market"].isin(["ALL", market]))].copy()
    ev = ev.sort_values("date", ascending=False).head(max_events)
    if ev.empty:
        return pd.DataFrame()

    universe = US_UNIVERSE if market == "US" else KR_UNIVERSE
    _, bench_df = fetch_price_history(bench_symbol, bench_market, period="2y", interval="1d")
    bench = bench_df.copy().sort_index()
    bench["d"] = pd.to_datetime(bench.index).tz_localize(None).date

    rows = []

    def classify_event(name: str) -> str:
        t = (name or "").lower()
        if "fomc" in t:
            return "FOMC"
        if "cpi" in t:
            return "CPI"
        if "고용" in t or "jobs" in t or "payroll" in t:
            return "고용지표"
        return "기타"

    for symbol in universe:
        try:
            resolved, asset_df = fetch_price_history(symbol, market, period="2y", interval="1d")
            asset = asset_df.copy().sort_index()
            asset["d"] = pd.to_datetime(asset.index).tz_localize(None).date

            if market == "US":
                try:
                    sector = str(yf.Ticker(resolved).info.get("sector", "Unknown") or "Unknown")
                except Exception:
                    sector = "Unknown"
            else:
                sector = "Korea"

            for _, e in ev.iterrows():
                d0 = e["date"]
                d_post = (pd.Timestamp(d0) + pd.Timedelta(days=5)).date()
                a0 = asset[asset["d"] >= d0]
                an = asset[asset["d"] >= d_post]
                b0 = bench[bench["d"] >= d0]
                bn = bench[bench["d"] >= d_post]
                if a0.empty or an.empty or b0.empty or bn.empty:
                    continue
                a_post = (float(an.iloc[0]["close"]) / float(a0.iloc[0]["close"]) - 1.0) * 100
                b_post = (float(bn.iloc[0]["close"]) / float(b0.iloc[0]["close"]) - 1.0) * 100
                rows.append(
                    {
                        "sector": sector,
                        "symbol": symbol,
                        "event": e.get("name", ""),
                        "event_type": classify_event(str(e.get("name", ""))),
                        "asset_post5_pct": a_post,
                        "bench_post5_pct": b_post,
                        "alpha_post5_pct": a_post - b_post,
                    }
                )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


left, right = st.columns([1, 2])
with left:
    default_symbol = "005930" if universe == "KR" else "AAPL"
    symbol = st.text_input("종목 코드/티커", value=default_symbol)
    market = st.selectbox("시장", ["US", "KR"], index=1 if universe == "KR" else 0)

    st.markdown("### 내 포지션")
    avg_buy = st.number_input("평균 매수가", min_value=0.0, value=0.0, step=10.0)
    qty = st.number_input("수량", min_value=0.0, value=0.0, step=1.0)
    account_size = st.number_input("총 운용자금", min_value=0.0, value=10000.0, step=1000.0)
    risk_pct = st.slider("1회 거래 리스크(%)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    auto_profile = st.checkbox("최근 성과 기반 프리셋 자동 적용", value=True)
    min_profile_samples = st.slider("자동추천 최소 표본수", 5, 40, 12, 1)
    suggested_profile = (
        _recommend_profile_from_recent_logs(
            horizon_days=10,
            market_filter=market,
            min_samples=min_profile_samples,
        )
        if auto_profile
        else None
    )
    profile_index = (
        list(PROFILE_PRESETS.keys()).index(suggested_profile)
        if suggested_profile in PROFILE_PRESETS
        else 1
    )
    if suggested_profile:
        st.caption(f"자동 추천 프리셋: {PROFILE_PRESETS[suggested_profile]['label']} ({suggested_profile})")
        rank_df = _recommend_profile_rank(market_filter=market, min_samples=min_profile_samples)
        if not rank_df.empty:
            with st.expander("자동 추천 상세 랭킹"):
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
            if st.button("추천 프리셋 로그 기록", use_container_width=True):
                best = rank_df.iloc[0]
                append_recommendation_log(
                    {
                        "market": market,
                        "recommended_profile": best["profile"],
                        "samples": int(best["count"]),
                        "hit_rate": float(best["hit_rate"] * 100.0),
                        "avg_ret": float(best["avg_ret"]),
                        "score": float(best["score"]),
                    }
                )
                st.success("추천 프리셋 로그를 저장했습니다.")

    profile = st.selectbox(
        "전략 프리셋",
        options=list(PROFILE_PRESETS.keys()),
        format_func=lambda x: PROFILE_PRESETS[x]["label"],
        index=profile_index,
    )
    apply_event_risk = st.checkbox("실적발표 리스크 반영", value=True)
    apply_macro_risk = st.checkbox("거시 이벤트 리스크 반영", value=True)

    with st.expander("거시 이벤트 캘린더 관리"):
        ev_date = st.date_input("이벤트 날짜")
        ev_name = st.text_input("이벤트명", value="FOMC / CPI / 고용지표")
        ev_market = st.selectbox("대상 시장", ["ALL", "US", "KR"], index=0)
        ev_severity = st.slider("중요도", 1, 3, 2)
        if st.button("이벤트 추가"):
            add_macro_event(ev_date, ev_name, ev_market, ev_severity)
            st.success("거시 이벤트를 추가했습니다.")

        macro_df = load_macro_events()
        st.dataframe(macro_df, use_container_width=True, hide_index=True)
        st.download_button(
            "이벤트 CSV 다운로드",
            data=export_macro_events_csv_bytes(),
            file_name="macro_events.csv",
            mime="text/csv",
            use_container_width=True,
        )
        up = st.file_uploader("이벤트 CSV 업로드", type=["csv"])
        if up is not None:
            try:
                up_df = pd.read_csv(up)
                overwrite_macro_events(up_df)
                st.success("이벤트 CSV를 반영했습니다. 새로고침 후 확인하세요.")
            except Exception as exc:
                st.error(f"CSV 처리 실패: {exc}")

if symbol:
    try:
        ticker, df = fetch_price_history(symbol, market, period=period, interval=interval)
        ind_df = add_indicators(df)
        latest = ind_df.iloc[-1]
        event = get_earnings_event_risk(ticker)
        macro_event = compute_macro_risk(market)
        earning_risk_score = event["risk_score"] if apply_event_risk else 0.0
        macro_risk_score = macro_event["risk_score"] if apply_macro_risk else 0.0
        event_risk_score = min(1.0, max(earning_risk_score, macro_risk_score))
        signal = evaluate_signal(latest, profile=profile, event_risk_score=event_risk_score)
        current_price = float(latest["close"])
        calib = _profile_calibration(profile=profile, market_filter=market, max_rows=180)
        edge = estimate_edge_from_confidence(signal.confidence, calib.get("by_bin", pd.DataFrame()))

        # Multi-timeframe confirmation (1w)
        _, df_w = fetch_price_history(symbol, market, period="2y", interval="1wk")
        ind_df_w = add_indicators(df_w)
        signal_w = evaluate_signal(ind_df_w.iloc[-1], profile=profile, event_risk_score=event_risk_score)

        # Benchmark comparison
        bench_symbol, bench_market, bench_name = benchmark_for_market(market)
        _, bench_df = fetch_price_history(bench_symbol, bench_market, period=period, interval=interval)
        perf_cmp = compare_with_benchmark(ind_df, bench_df)

        with left:
            st.metric("현재가", f"{current_price:,.2f}")
            st.metric("시그널", signal.action)
            st.metric("점수", f"{signal.score:.1f}/100")
            st.metric("신뢰도", f"{signal.confidence:.1f}%")
            st.metric("교정 승률", f"{edge['pred_win_rate_pct']:.1f}%")
            st.metric("교정 기대값", f"{edge['expected_return_pct']:+.2f}%")
            regime_label = {"bull": "상승", "sideways": "횡보", "bear": "하락"}.get(signal.regime, "중립")
            st.metric("시장 국면", regime_label)
            st.metric("이벤트 리스크", event["risk_label"])
            st.caption(event["note"])
            st.metric("거시 리스크", macro_event["risk_label"])
            st.caption(macro_event["note"])
            st.metric("종합 이벤트 점수", f"{event_risk_score:.2f}")
            st.metric("주봉 시그널", signal_w.action)

            if signal.action == signal_w.action:
                st.success("일봉/주봉 시그널이 일치합니다.")
            else:
                st.warning("일봉/주봉 시그널이 불일치합니다. 포지션 축소 또는 분할 접근을 권장합니다.")

            if avg_buy > 0 and qty > 0:
                perf = evaluate_position(avg_buy, qty, current_price)
                st.metric("투입 금액", f"{perf['invested']:,.0f}")
                st.metric("현재 평가", f"{perf['current_value']:,.0f}")
                st.metric("손익", f"{perf['pnl']:,.0f} ({perf['pnl_pct']:+.2f}%)")
            else:
                st.info("평단/수량 입력 시 손익이 계산됩니다.")

            pos = recommend_position_size(
                account_size=account_size,
                risk_pct=risk_pct,
                entry_price=signal.trade_plan["entry"],
                stop_price=signal.trade_plan["stop"],
            )
            size_mult = recommend_size_multiplier(
                expected_return_pct=float(edge.get("expected_return_pct", 0.0)),
                pred_win_rate_pct=float(edge.get("pred_win_rate_pct", signal.confidence)),
                profile=profile,
            )
            st.markdown("### 리스크 기반 권장 포지션")
            st.metric("리스크 예산", f"{pos['risk_budget']:,.2f}")
            st.metric("권장 수량", f"{pos['qty']:,.2f}")
            st.metric("권장 포지션 금액", f"{pos['position_value']:,.2f}")
            st.metric("품질 조정 배수", f"{size_mult:.2f}x")
            st.metric("품질 조정 수량", f"{pos['qty'] * size_mult:,.2f}")
            st.metric("품질 조정 금액", f"{pos['position_value'] * size_mult:,.2f}")

        with right:
            cdf = ind_df.tail(240)
            fig = px.line(cdf, y=["close", "sma20", "sma60"], height=420)
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

            rr = signal.reasons
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("RSI(14)", f"{rr['rsi14']:.2f}")
            with r2:
                st.metric("MACD Hist", f"{rr['macd_hist']:.4f}")
            with r3:
                st.metric("20일 모멘텀", f"{rr['momentum20_pct']:+.2f}%")
            with r4:
                st.metric("20일 변동성", f"{rr['volatility20_pct']:.2f}%")

            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.metric("Entry", f"{signal.trade_plan['entry']:,.2f}")
            with t2:
                st.metric("Stop", f"{signal.trade_plan['stop']:,.2f}")
            with t3:
                st.metric("TP1", f"{signal.trade_plan['tp1']:,.2f}")
            with t4:
                st.metric("R:R(TP1)", f"{signal.trade_plan['rr_tp1']:.2f}")
            st.caption(f"ATR 스탑 배수: {signal.trade_plan['stop_mult_atr']:.2f}x")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Trend", f"{signal.components['trend']:.1f}")
            with c2:
                st.metric("Momentum", f"{signal.components['momentum']:.1f}")
            with c3:
                st.metric("Quality", f"{signal.components['quality']:.1f}")

            quality = backtest_signal_quality(
                ind_df,
                lookahead=10,
                profile=profile,
                event_risk_score=event_risk_score,
            )
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.metric("검증 거래수", f"{int(quality['trades'])}")
            with q2:
                st.metric("승률", f"{quality['win_rate']:.1f}%")
            with q3:
                st.metric("평균수익", f"{quality['avg_return_pct']:+.2f}%")
            with q4:
                st.metric("기대값", f"{quality['expectancy_pct']:+.2f}%")

            wf = walk_forward_signal_quality(
                ind_df,
                lookahead=10,
                window=120,
                step=20,
                profile=profile,
                event_risk_score=event_risk_score,
            )
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric("WF Fold 수", f"{int(wf['folds'])}")
            with w2:
                st.metric("WF 평균 거래수", f"{wf['trades_avg']:.1f}")
            with w3:
                st.metric("WF 승률 평균", f"{wf['win_rate_avg']:.1f}%")
            with w4:
                st.metric("WF 승률 표준편차", f"{wf['win_rate_std']:.1f}")
            st.metric("WF 기대값 평균", f"{wf['expectancy_avg']:+.2f}%")

            st.markdown("### 벤치마크 대비")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("자산 수익률", f"{perf_cmp['asset_return_pct']:+.2f}%")
            with b2:
                st.metric(f"{bench_name} 수익률", f"{perf_cmp['bench_return_pct']:+.2f}%")
            with b3:
                st.metric("알파", f"{perf_cmp['alpha_pct']:+.2f}%")
            with b4:
                st.metric("Sharpe Edge", f"{perf_cmp['sharpe_edge']:+.2f}")

            b5, b6 = st.columns(2)
            with b5:
                st.metric("자산 MDD", f"{perf_cmp['asset_mdd_pct']:.2f}%")
            with b6:
                st.metric(f"{bench_name} MDD", f"{perf_cmp['bench_mdd_pct']:.2f}%")

            st.markdown("### 이벤트 전후 성과 회고")
            event_perf = _event_window_performance(symbol, market, bench_symbol, bench_market)
            if event_perf.empty:
                st.info("회고 가능한 과거 이벤트 데이터가 부족합니다.")
            else:
                st.dataframe(event_perf, use_container_width=True, hide_index=True)
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.metric("평균 자산 Post+5", f"{event_perf['asset_post5_pct'].mean():+.2f}%")
                with e2:
                    st.metric("평균 벤치 Post+5", f"{event_perf['bench_post5_pct'].mean():+.2f}%")
                with e3:
                    st.metric("평균 Post+5 알파", f"{event_perf['post5_alpha_pct'].mean():+.2f}%")

            st.markdown("### 섹터별 이벤트 회고")
            sec_df = _sector_event_review(market, bench_symbol, bench_market, max_events=6)
            if sec_df.empty:
                st.info("섹터 회고 데이터가 부족합니다.")
            else:
                sec_agg = (
                    sec_df.groupby("sector", dropna=False)
                    .agg(
                        samples=("symbol", "count"),
                        avg_asset_post5=("asset_post5_pct", "mean"),
                        avg_bench_post5=("bench_post5_pct", "mean"),
                        avg_alpha_post5=("alpha_post5_pct", "mean"),
                    )
                    .reset_index()
                    .sort_values("avg_alpha_post5", ascending=False)
                )
                st.dataframe(sec_agg, use_container_width=True, hide_index=True)
                type_agg = (
                    sec_df.groupby("event_type", dropna=False)
                    .agg(
                        samples=("symbol", "count"),
                        avg_alpha_post5=("alpha_post5_pct", "mean"),
                    )
                    .reset_index()
                    .sort_values("avg_alpha_post5", ascending=False)
                )
                st.markdown("#### 이벤트 종류별 알파")
                st.dataframe(type_agg, use_container_width=True, hide_index=True)

            if st.button("현재 시그널 기록", use_container_width=True):
                append_signal_log(
                    {
                        "symbol": symbol,
                        "ticker": ticker,
                        "market": market,
                        "profile": profile,
                        "action": signal.action,
                        "score": signal.score,
                        "confidence": signal.confidence,
                        "regime": signal.regime,
                        "price": current_price,
                        "stop": signal.trade_plan["stop"],
                        "tp1": signal.trade_plan["tp1"],
                        "rr_tp1": signal.trade_plan["rr_tp1"],
                        "event_risk": event["risk_score"],
                        "event_days": event["days_to_event"],
                    }
                )
                st.success("시그널 로그에 저장했습니다.")

            st.info(f"분석 대상: {symbol_with_name(symbol, market)} | 조회 티커: {ticker}")
    except Exception as exc:
        st.error(f"분석 실패: {exc}")
