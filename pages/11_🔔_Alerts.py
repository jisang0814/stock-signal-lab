from __future__ import annotations

import streamlit as st

from src.alerts import (
    append_alert_history,
    build_alert_message,
    clear_alert_history,
    load_alert_config,
    load_alert_history,
    save_alert_config,
    send_telegram_alert,
)

st.title("🔔 Alerts")
st.caption("알림 설정 및 테스트 발송")

cfg = load_alert_config()
tg = cfg.get("telegram", {})

enabled = st.checkbox("Telegram 사용", value=bool(tg.get("enabled", False)))
bot_token = st.text_input("Bot Token", value=str(tg.get("bot_token", "")), type="password")
chat_id = st.text_input("Chat ID", value=str(tg.get("chat_id", "")))

if st.button("설정 저장", use_container_width=True):
    cfg["telegram"] = {
        "enabled": enabled,
        "bot_token": bot_token,
        "chat_id": chat_id,
    }
    save_alert_config(cfg)
    st.success("알림 설정 저장 완료")

st.markdown("---")
title = st.text_input("알림 제목", value="Risk Alert")
body = st.text_area("알림 내용", value="킬스위치 활성: 신규 진입 차단")

if st.button("테스트 알림 발송", type="primary", use_container_width=True):
    msg = build_alert_message(title, body)
    result = send_telegram_alert(msg, cfg=load_alert_config())
    append_alert_history("manual:test", title, "sent" if result.get("ok") else "failed", result.get("message", ""))
    if result.get("ok"):
        st.success(f"발송 성공: {result.get('message')}")
    else:
        st.error(f"발송 실패: {result.get('message')}")
    st.json(result)

st.markdown("---")
st.subheader("알림 이력")
h1, h2 = st.columns(2)
with h1:
    limit = st.slider("조회 개수", 20, 500, 200, 20)
with h2:
    if st.button("이력 초기화", use_container_width=True):
        clear_alert_history()
        st.success("알림 이력을 초기화했습니다.")

hist = load_alert_history(limit=limit)
if hist is None or getattr(hist, "empty", True):
    st.info("알림 이력이 없습니다.")
else:
    st.dataframe(hist.sort_values("ts", ascending=False), use_container_width=True, hide_index=True)
