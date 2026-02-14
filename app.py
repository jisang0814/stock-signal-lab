from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Value-style Stock Lab", page_icon="📊", layout="wide")

st.title("📊 Value-style Stock Lab")
st.caption("국내/미국 주식 시그널 분석 플랫폼")

st.markdown(
    """
### 페이지 구성
- `📊 Dashboard`: 시장 히트맵, Top 5, AI 투자 분석 카드
- `🔎 Screener`: 점수/시그널/RSI 기반 후보 스크리닝
- `🧠 Analysis`: 개별 종목 시그널 + 내 포지션 손익 분석

좌측 페이지 메뉴에서 이동해 사용하세요.
"""
)

st.info("실거래 주문 기능은 없으며, 분석/연구 목적 도구입니다.")
