from __future__ import annotations

import pandas as pd
import streamlit as st

from .data_providers import KR_UNIVERSE, US_UNIVERSE
from .market import build_snapshots, snapshots_to_df


def render_common_sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.header("공통 설정")
        universe = st.selectbox(
            "유니버스",
            ["US", "KR", "ALL"],
            index=2,
            key="ui_universe",
        )
        period = st.selectbox(
            "조회 기간",
            ["3mo", "6mo", "1y"],
            index=1,
            key="ui_period",
        )
        interval = st.selectbox(
            "봉",
            ["1d", "1wk"],
            index=0,
            key="ui_interval",
        )

        if st.button("캐시 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.success("캐시를 초기화했습니다.")

    return universe, period, interval


@st.cache_data(ttl=600)
def load_snapshot_df(universe_key: str, period: str, interval: str) -> pd.DataFrame:
    if universe_key == "US":
        symbols = US_UNIVERSE
        rows = build_snapshots(symbols, period=period, interval=interval)
        return snapshots_to_df(rows)
    elif universe_key == "KR":
        symbols = KR_UNIVERSE
        rows = build_snapshots(symbols, period=period, interval=interval)
        return snapshots_to_df(rows)
    else:
        # ALL은 시장별로 분리 로딩 후 병합: 혼합 요청 시 특정 시장 누락되는 현상 방지
        us_df = snapshots_to_df(build_snapshots(US_UNIVERSE, period=period, interval=interval))
        kr_df = snapshots_to_df(build_snapshots(KR_UNIVERSE, period=period, interval=interval))
        df = pd.concat([us_df, kr_df], ignore_index=True)

        missing_us = df.empty or ("market" not in df.columns) or (df["market"].astype(str).str.upper() == "US").sum() == 0
        missing_kr = df.empty or ("market" not in df.columns) or (df["market"].astype(str).str.upper() == "KR").sum() == 0

        if missing_us:
            retry_us = snapshots_to_df(build_snapshots(US_UNIVERSE, period="6mo", interval="1d"))
            if not retry_us.empty:
                df = pd.concat([df, retry_us], ignore_index=True)
        if missing_kr:
            retry_kr = snapshots_to_df(build_snapshots(KR_UNIVERSE, period="6mo", interval="1d"))
            if not retry_kr.empty:
                df = pd.concat([df, retry_kr], ignore_index=True)

        if not df.empty and "symbol" in df.columns and "market" in df.columns:
            df = df.drop_duplicates(subset=["symbol", "market"], keep="first").reset_index(drop=True)

        return df


def ensure_df_or_stop(df: pd.DataFrame):
    if df.empty:
        st.error("스냅샷 데이터를 불러오지 못했습니다. 잠시 후 다시 시도하세요.")
        st.stop()
