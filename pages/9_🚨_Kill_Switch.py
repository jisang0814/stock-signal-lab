from __future__ import annotations

import streamlit as st

from src.alerts import send_telegram_alert_with_cooldown
from src.data_providers import fetch_price_history
from src.event_risk import get_earnings_event_risk
from src.indicators import add_indicators
from src.macro_events import compute_macro_risk
from src.performance import summarize_price_performance
from src.risk_controls import evaluate_kill_switch

st.title("🚨 Risk Kill-Switch")
st.caption("변동성/MDD/이벤트 리스크 기반 자동 리스크 오프 판단")

c1, c2 = st.columns(2)
with c1:
    symbol = st.text_input("종목", value="AAPL")
with c2:
    market = st.selectbox("시장", ["US", "KR"], index=0)

k1, k2, k3 = st.columns(3)
with k1:
    vol_thr = st.slider("변동성 임계(%)", 1.0, 10.0, 4.0, 0.5)
with k2:
    mdd_thr = st.slider("MDD 임계(%)", -40.0, -5.0, -15.0, 1.0)
with k3:
    event_thr = st.slider("이벤트 리스크 임계", 0.0, 1.0, 0.75, 0.05)

a1, a2 = st.columns(2)
with a1:
    auto_alert = st.checkbox("킬스위치 활성 시 Telegram 자동 알림", value=True)
with a2:
    cooldown_min = st.slider("알림 쿨다운(분)", 5, 240, 60, 5)

if st.button("킬스위치 평가", type="primary"):
    try:
        ticker, df = fetch_price_history(symbol, market, period="1y", interval="1d")
        ind = add_indicators(df)
        latest = ind.iloc[-1]

        vol_pct = float(latest.get("volatility20", 0) * 100)
        perf = summarize_price_performance(ind)
        mdd = float(perf.get("max_drawdown_pct", 0))

        er = get_earnings_event_risk(ticker)
        mr = compute_macro_risk(market)
        event_score = max(float(er.get("risk_score", 0)), float(mr.get("risk_score", 0)))

        out = evaluate_kill_switch(
            vol_pct=vol_pct,
            mdd_pct=mdd,
            event_risk_score=event_score,
            vol_threshold=vol_thr,
            mdd_threshold=mdd_thr,
            event_threshold=event_thr,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("현재 변동성(20d)", f"{vol_pct:.2f}%")
        with m2:
            st.metric("현재 MDD", f"{mdd:.2f}%")
        with m3:
            st.metric("이벤트 리스크", f"{event_score:.2f}")

        st.metric("Risk Mode", out["risk_mode"])
        if out["active"]:
            st.error(out["action"])
            for r in out["reasons"]:
                st.write(f"- {r}")
            if auto_alert:
                alert_key = f"kill_switch:{market}:{symbol.strip().upper()}"
                body = (
                    f"symbol={symbol} market={market}\n"
                    f"vol20={vol_pct:.2f}% mdd={mdd:.2f}% event={event_score:.2f}\n"
                    f"reasons={'; '.join(out['reasons'])}"
                )
                alert_out = send_telegram_alert_with_cooldown(
                    title="Kill-Switch Triggered",
                    body=body,
                    alert_key=alert_key,
                    cooldown_minutes=cooldown_min,
                )
                if alert_out.get("ok"):
                    st.success("Telegram 알림 발송 완료")
                elif alert_out.get("skipped"):
                    st.info(alert_out.get("message", "쿨다운으로 발송 생략"))
                else:
                    st.warning(f"Telegram 발송 실패: {alert_out.get('message')}")
        else:
            st.success(out["action"])
    except Exception as exc:
        st.error(f"평가 실패: {exc}")
