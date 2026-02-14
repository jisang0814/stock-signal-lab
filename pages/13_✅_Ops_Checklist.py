from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.alerts import load_alert_config, load_alert_history
from src.trade_log import load_signal_log

st.title("✅ Ops Checklist")
st.caption("실전 운용 전/중 점검 체크리스트")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECK_PATH = DATA_DIR / "ops_checklist.json"


DEFAULT_ITEMS = [
    "알림 설정(Telegram) 활성화 확인",
    "Kill-Switch 임계값 점검",
    "Drift Monitor 임계값 점검",
    "오늘 시장 개장 전 핵심 이벤트 확인",
    "신규 진입 전 포지션 사이징 검토",
    "최근 알림 실패 원인 점검",
]


def load_checks() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECK_PATH.exists():
        payload = {
            "updated_at": "",
            "items": {k: False for k in DEFAULT_ITEMS},
        }
        CHECK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    try:
        return json.loads(CHECK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"updated_at": "", "items": {k: False for k in DEFAULT_ITEMS}}


def save_checks(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


payload = load_checks()
items = payload.get("items", {})
for k in DEFAULT_ITEMS:
    items.setdefault(k, False)

st.subheader("운영 상태 요약")
try:
    cfg = load_alert_config()
    tg = cfg.get("telegram", {})
    tg_ok = bool(tg.get("enabled", False) and str(tg.get("bot_token", "")).strip() and str(tg.get("chat_id", "")).strip())
except Exception:
    tg_ok = False

hist = load_alert_history(limit=200)
last_alert_ts = "-"
last_alert_status = "N/A"
failed_24h = 0
if hist is not None and not hist.empty:
    hist2 = hist.copy()
    hist2["ts_dt"] = pd.to_datetime(hist2["ts"], errors="coerce")
    hist2 = hist2.dropna(subset=["ts_dt"])
    if not hist2.empty:
        row = hist2.sort_values("ts_dt", ascending=False).iloc[0]
        last_alert_ts = str(row.get("ts", "-"))
        last_alert_status = str(row.get("status", "N/A")).upper()
        cut = pd.Timestamp.now() - pd.Timedelta(hours=24)
        failed_24h = int(((hist2["status"].astype(str).str.lower() == "failed") & (hist2["ts_dt"] >= cut)).sum())

log_df = load_signal_log(limit=3000)
log_count = int(len(log_df))

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Telegram 설정", "OK" if tg_ok else "CHECK")
with m2:
    st.metric("최근 알림 상태", last_alert_status)
with m3:
    st.metric("24h 알림 실패", f"{failed_24h}")
with m4:
    st.metric("시그널 로그 수", f"{log_count}")

st.caption(f"최근 알림 시각: {last_alert_ts}")

st.markdown("---")
st.subheader("체크리스트")
new_items = {}
for key in DEFAULT_ITEMS:
    new_items[key] = st.checkbox(key, value=bool(items.get(key, False)))

c1, c2 = st.columns(2)
with c1:
    if st.button("체크리스트 저장", type="primary", use_container_width=True):
        out = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "items": new_items,
        }
        save_checks(out)
        st.success("체크리스트 저장 완료")
with c2:
    if st.button("전체 해제", use_container_width=True):
        out = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "items": {k: False for k in DEFAULT_ITEMS},
        }
        save_checks(out)
        st.info("전체 해제 완료")

saved_at = payload.get("updated_at", "")
if saved_at:
    st.caption(f"마지막 저장: {saved_at}")
