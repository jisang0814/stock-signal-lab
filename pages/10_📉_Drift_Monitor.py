from __future__ import annotations

import pandas as pd
import streamlit as st

from src.alerts import send_telegram_alert_with_cooldown
from src.data_providers import fetch_price_history
from src.drift import detect_signal_drift
from src.trade_log import load_signal_log

st.title("📉 Drift Monitor")
st.caption("최근 성과 붕괴(드리프트) 감지")

log_df = load_signal_log(limit=2000)
if log_df.empty:
    st.info("시그널 로그가 없습니다.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    recent_days = st.slider("최근 윈도우(일)", 7, 90, 30)
with c2:
    baseline_days = st.slider("베이스라인(일)", 60, 365, 180)
with c3:
    min_samples = st.slider("최소 표본수", 10, 100, 20)

a1, a2 = st.columns(2)
with a1:
    auto_alert = st.checkbox("드리프트 감지 시 Telegram 자동 알림", value=True)
with a2:
    cooldown_min = st.slider("알림 쿨다운(분)", 5, 240, 60, 5, key="drift_cooldown")

mkt = st.multiselect("시장", sorted(log_df["market"].dropna().astype(str).unique().tolist()), default=sorted(log_df["market"].dropna().astype(str).unique().tolist()))
prof = st.multiselect("프리셋", sorted(log_df["profile"].dropna().astype(str).unique().tolist()), default=sorted(log_df["profile"].dropna().astype(str).unique().tolist()))

flt = log_df.copy()
if mkt:
    flt = flt[flt["market"].astype(str).isin(mkt)]
if prof:
    flt = flt[flt["profile"].astype(str).isin(prof)]

rows = []
for _, row in flt.sort_values("ts", ascending=False).head(150).iterrows():
    symbol = str(row.get("symbol", "")).strip()
    market = str(row.get("market", "US")).strip()
    action = str(row.get("action", "보유"))
    entry = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
    if not symbol or entry <= 0:
        continue
    try:
        _, px_df = fetch_price_history(symbol, market, period="3mo", interval="1d")
        now = float(px_df.iloc[-1]["close"])
        if action == "추가매수":
            ret = (now / entry - 1.0) * 100
        elif action == "매도":
            ret = (entry / now - 1.0) * 100 if now > 0 else 0.0
        else:
            ret = (now / entry - 1.0) * 100
        rows.append({"ts": row.get("ts"), "return_net_pct": ret})
    except Exception:
        continue

perf_df = pd.DataFrame(rows)
out = detect_signal_drift(perf_df, recent_days=recent_days, baseline_days=baseline_days, min_samples=min_samples)

if out.get("status") == "drift":
    st.error("성능 드리프트 감지: 방어 프리셋 전환 또는 진입 강도 축소 필요")
    if auto_alert:
        m_key = ",".join(sorted([str(x) for x in mkt])) if mkt else "ALL"
        p_key = ",".join(sorted([str(x) for x in prof])) if prof else "ALL"
        alert_key = f"drift:{m_key}:{p_key}:{recent_days}:{baseline_days}"
        body = (
            f"market={m_key} profile={p_key}\n"
            f"mean_delta={out.get('mean_delta', 0):+.2f}pp "
            f"win_delta={out.get('win_delta', 0):+.2f}pp\n"
            f"samples={out.get('recent_samples', 0)}/{out.get('baseline_samples', 0)}"
        )
        alert_out = send_telegram_alert_with_cooldown(
            title="Signal Drift Detected",
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
elif out.get("status") == "stable":
    st.success("성능 안정 구간")
else:
    st.info(out.get("message", ""))

st.write(out)
if not perf_df.empty:
    st.dataframe(perf_df.sort_values("ts", ascending=False), use_container_width=True, hide_index=True)
