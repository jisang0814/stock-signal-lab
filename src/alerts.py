from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CFG_PATH = DATA_DIR / "alerts_config.json"
STATE_PATH = DATA_DIR / "alerts_state.json"
HISTORY_PATH = DATA_DIR / "alerts_history.csv"


def _ensure_cfg() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CFG_PATH.exists():
        CFG_PATH.write_text(json.dumps({"telegram": {"enabled": False, "bot_token": "", "chat_id": ""}}, ensure_ascii=False, indent=2))
    if not STATE_PATH.exists():
        STATE_PATH.write_text(json.dumps({"last_sent": {}}, ensure_ascii=False, indent=2))
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("ts,alert_key,title,status,message\n", encoding="utf-8")


def load_alert_config() -> dict:
    _ensure_cfg()
    return json.loads(CFG_PATH.read_text())


def save_alert_config(cfg: dict) -> None:
    _ensure_cfg()
    CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def load_alert_state() -> dict:
    _ensure_cfg()
    return json.loads(STATE_PATH.read_text())


def save_alert_state(state: dict) -> None:
    _ensure_cfg()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def send_telegram_alert(message: str, cfg: dict) -> dict:
    tg = cfg.get("telegram", {})
    if not tg.get("enabled"):
        return {"ok": False, "message": "telegram 비활성화"}

    token = (tg.get("bot_token") or "").strip()
    chat_id = (tg.get("chat_id") or "").strip()
    if not token or not chat_id:
        return {"ok": False, "message": "bot_token/chat_id 미설정"}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        r = requests.post(url, json=payload, timeout=8)
        ok = r.status_code == 200
        return {"ok": ok, "message": f"status={r.status_code}", "response": r.text[:200]}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def build_alert_message(title: str, body: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{ts}] {title}\n{body}"


def _is_in_cooldown(alert_key: str, cooldown_minutes: int) -> tuple[bool, str]:
    state = load_alert_state()
    last_sent_map = state.get("last_sent", {})
    last_ts = str(last_sent_map.get(alert_key, "")).strip()
    if not last_ts:
        return False, ""
    try:
        dt = datetime.fromisoformat(last_ts)
    except Exception:
        return False, ""
    until = dt + timedelta(minutes=max(0, int(cooldown_minutes)))
    if datetime.now() < until:
        return True, until.strftime("%Y-%m-%d %H:%M:%S")
    return False, ""


def send_telegram_alert_with_cooldown(
    title: str,
    body: str,
    alert_key: str,
    cooldown_minutes: int = 60,
    force: bool = False,
) -> dict:
    if not force:
        blocked, until = _is_in_cooldown(alert_key, cooldown_minutes)
        if blocked:
            out = {"ok": False, "skipped": True, "message": f"쿨다운 중 (until {until})"}
            append_alert_history(alert_key, title, "skipped", out.get("message", ""))
            return out

    cfg = load_alert_config()
    result = send_telegram_alert(build_alert_message(title, body), cfg=cfg)
    if result.get("ok"):
        state = load_alert_state()
        last_sent_map = state.get("last_sent", {})
        last_sent_map[str(alert_key)] = datetime.now().isoformat(timespec="seconds")
        state["last_sent"] = last_sent_map
        save_alert_state(state)
        append_alert_history(alert_key, title, "sent", result.get("message", ""))
    else:
        append_alert_history(alert_key, title, "failed", result.get("message", ""))
    return result


def append_alert_history(alert_key: str, title: str, status: str, message: str) -> None:
    _ensure_cfg()
    ts = datetime.now().isoformat(timespec="seconds")
    safe = lambda x: str(x).replace("\n", " ").replace(",", ";")
    line = f"{safe(ts)},{safe(alert_key)},{safe(title)},{safe(status)},{safe(message)}\n"
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def load_alert_history(limit: int = 300):
    _ensure_cfg()
    try:
        import pandas as pd

        df = pd.read_csv(HISTORY_PATH)
        if len(df) > limit:
            return df.tail(limit).reset_index(drop=True)
        return df
    except Exception:
        return None


def clear_alert_history() -> None:
    _ensure_cfg()
    HISTORY_PATH.write_text("ts,alert_key,title,status,message\n", encoding="utf-8")
