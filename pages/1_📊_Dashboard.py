from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import plotly.express as px
import streamlit as st

from src.alerts import load_alert_history
from src.data_providers import symbol_with_name
from src.market import generate_ai_comment
from src.ui import ensure_df_or_stop, load_snapshot_df, render_common_sidebar

st.title("📊 Dashboard")
st.caption("시장 히트맵, 상위 랭킹, AI 스타일 분석 요약")

universe, period, interval = render_common_sidebar()
df = load_snapshot_df(universe, period, interval)
ensure_df_or_stop(df)
df = df.copy()
df["symbol_name"] = df.apply(lambda r: symbol_with_name(str(r.get("symbol", "")), str(r.get("market", "US"))), axis=1)

if universe in ["ALL", "US"]:
    us_count = int((df["market"].astype(str).str.upper() == "US").sum()) if "market" in df.columns else 0
    if us_count == 0:
        st.warning("US 데이터가 비어 있습니다. 사이드바에서 `캐시 새로고침` 후 다시 확인하세요.")
if universe in ["ALL", "KR"]:
    kr_count = int((df["market"].astype(str).str.upper() == "KR").sum()) if "market" in df.columns else 0
    if kr_count == 0:
        st.warning("KR 데이터가 비어 있습니다. 사이드바에서 `캐시 새로고침` 후 다시 확인하세요.")

ops_path = Path(__file__).resolve().parents[1] / "data" / "ops_checklist.json"
ops_items = {}
if ops_path.exists():
    try:
        ops_payload = json.loads(ops_path.read_text(encoding="utf-8"))
        ops_items = ops_payload.get("items", {}) if isinstance(ops_payload, dict) else {}
    except Exception:
        ops_items = {}

if ops_items:
    total_items = len(ops_items)
    checked_items = int(sum(1 for v in ops_items.values() if bool(v)))
    unchecked = [k for k, v in ops_items.items() if not bool(v)]
    if unchecked:
        st.warning(f"Ops Checklist 미완료 {len(unchecked)}건 ({checked_items}/{total_items} 완료): {', '.join(unchecked[:3])}")
        st.page_link("pages/13_✅_Ops_Checklist.py", label="Ops Checklist 페이지로 이동", icon="✅")
    else:
        st.success(f"Ops Checklist 완료 ({checked_items}/{total_items})")
        st.page_link("pages/13_✅_Ops_Checklist.py", label="Ops Checklist 확인", icon="✅")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("분석 종목 수", f"{len(df)}")
with k2:
    st.metric("추가매수 신호", f"{(df['signal'] == '추가매수').sum()}")
with k3:
    st.metric("매도 신호", f"{(df['signal'] == '매도').sum()}")
with k4:
    st.metric("평균 점수", f"{df['score'].mean():.1f}")
with k5:
    hist = load_alert_history(limit=50)
    if hist is not None and not hist.empty:
        last = hist.sort_values("ts", ascending=False).iloc[0]
        status = str(last.get("status", "-")).upper()
        st.metric("최근 알림", status)
    else:
        st.metric("최근 알림", "N/A")

st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
hist = load_alert_history(limit=50)
if hist is not None and not hist.empty:
    last = hist.sort_values("ts", ascending=False).iloc[0]
    status_raw = str(last.get("status", "")).strip().lower()
    status_label = {"sent": "SENT", "failed": "FAILED", "skipped": "SKIPPED"}.get(status_raw, status_raw.upper() or "N/A")
    status_bg = {"sent": "#e8f7ee", "failed": "#fdecec", "skipped": "#fff8e1"}.get(status_raw, "#eef2f7")
    status_fg = {"sent": "#137333", "failed": "#b42318", "skipped": "#8a6d1d"}.get(status_raw, "#344054")
    status_bd = {"sent": "#a6e3b8", "failed": "#f5b8b3", "skipped": "#f2df9d"}.get(status_raw, "#cfd8e3")
    st.markdown(
        (
            f"<div style='padding:10px 12px;border:1px solid {status_bd};"
            f"background:{status_bg};border-radius:10px;'>"
            f"<div style='font-size:12px;color:#475467;'>최근 알림 상태</div>"
            f"<div style='font-size:18px;font-weight:700;color:{status_fg};'>{status_label}</div>"
            f"<div style='font-size:12px;color:#475467;margin-top:4px;'>"
            f"{str(last.get('ts', ''))} | {str(last.get('title', ''))}"
            f"</div></div>"
        ),
        unsafe_allow_html=True,
    )
    recent5 = hist.sort_values("ts", ascending=False).head(5).copy()
    recent5["status_badge"] = recent5["status"].astype(str).str.lower().map(
        {
            "sent": "🟢 SENT",
            "failed": "🔴 FAILED",
            "skipped": "🟡 SKIPPED",
        }
    ).fillna("⚪ N/A")
    st.caption("최근 5건 알림 타임라인")
    st.dataframe(
        recent5[["ts", "title", "status_badge", "alert_key", "message"]],
        use_container_width=True,
        hide_index=True,
    )

h1, h2 = st.columns([2, 1])
with h1:
    st.subheader("시장 히트맵 (일일 변동률)")
    heat_df = df.copy()
    heat_df["group"] = heat_df["market"]
    heat_df["size"] = heat_df["market_cap"].where(heat_df["market_cap"] > 0, heat_df["volume_value"])

    fig = px.treemap(
        heat_df,
        path=["group", "symbol_name"],
        values="size",
        color="change_pct",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, use_container_width=True)

with h2:
    st.subheader("Top 5")
    top_mode = st.radio("기준", ["시가총액", "거래대금", "점수"], horizontal=True)
    sort_col = {"시가총액": "market_cap", "거래대금": "volume_value", "점수": "score"}[top_mode]
    top5 = df.sort_values(sort_col, ascending=False).head(5)
    st.dataframe(
        top5[["symbol_name", "market", "price", "change_pct", "signal", "score"]],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.subheader("AI 투자 분석 카드")
card_base = df.sort_values(["score", "confidence"], ascending=False).head(3)
cols = st.columns(3)

for idx, (_, row) in enumerate(card_base.iterrows()):
    with cols[idx]:
        st.markdown(f"### {row['symbol_name']}")
        st.caption(f"가격 {row['price']:,.2f} | 변동률 {row['change_pct']:+.2f}%")
        st.write(generate_ai_comment(row))
        st.metric("Action", row["signal"])
        st.metric("Score", f"{row['score']:.1f}")
