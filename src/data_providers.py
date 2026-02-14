from __future__ import annotations

from typing import Iterable
from pathlib import Path

import pandas as pd
import yfinance as yf


KR_UNIVERSE = [
    "005930", "000660", "035420", "035720", "051910",
    "005380", "068270", "105560", "207940", "006400",
]

US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "AVGO", "NFLX",
]

KR_COMPANY_NAMES = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "051910": "LG화학",
    "005380": "현대차",
    "068270": "셀트리온",
    "105560": "KB금융",
    "207940": "삼성바이오로직스",
    "006400": "삼성SDI",
}

US_COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "AVGO": "Broadcom",
    "NFLX": "Netflix",
}


def _configure_yfinance_cache() -> None:
    # Sandbox/배포 환경에서 기본 캐시 경로가 읽기 전용일 수 있어 프로젝트 내부로 고정
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "yf_tz_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass


_configure_yfinance_cache()


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out.columns = [str(c).lower() for c in out.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"필수 OHLCV 컬럼 누락: {missing}")

    out = out[required].dropna()
    return out


def _download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return _normalize_ohlcv(df)


def resolve_ticker(symbol: str, market: str) -> list[str]:
    symbol = symbol.strip().upper()

    if market == "US":
        return [symbol]

    # KR: 6-digit code base, try KOSPI/KOSDAQ suffixes
    if symbol.isdigit() and len(symbol) == 6:
        return [f"{symbol}.KS", f"{symbol}.KQ"]

    return [symbol]


def fetch_price_history(
    symbol: str,
    market: str,
    period: str = "6mo",
    interval: str = "1d",
) -> tuple[str, pd.DataFrame]:
    errors = []
    for ticker in resolve_ticker(symbol, market):
        try:
            df = _download_history(ticker, period, interval)
            if not df.empty:
                return ticker, df
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    raise ValueError("가격 데이터를 가져오지 못했습니다. " + " | ".join(errors))


def get_universe(market: str) -> Iterable[str]:
    if market == "KR":
        return KR_UNIVERSE
    if market == "US":
        return US_UNIVERSE
    if market == "ALL":
        return [*KR_UNIVERSE, *US_UNIVERSE]
    raise ValueError(f"지원하지 않는 market: {market}")


def detect_market(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.isdigit() and len(s) == 6:
        return "KR"
    return "US"


def _normalize_symbol_key(symbol: str) -> str:
    s = str(symbol).strip().upper()
    # e.g. 005930.KS -> 005930
    if "." in s:
        s = s.split(".")[0]
    return s


def company_name(symbol: str, market: str | None = None) -> str:
    key = _normalize_symbol_key(symbol)
    m = (market or detect_market(key)).upper()
    if m == "KR":
        return KR_COMPANY_NAMES.get(key, key)
    return US_COMPANY_NAMES.get(key, key)


def symbol_with_name(symbol: str, market: str | None = None) -> str:
    key = _normalize_symbol_key(symbol)
    name = company_name(key, market)
    return f"{name} ({key})"
