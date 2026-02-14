from __future__ import annotations


def benchmark_for_market(market: str) -> tuple[str, str, str]:
    m = (market or "US").upper()
    if m == "KR":
        return "^KS11", "KR", "KOSPI"
    return "SPY", "US", "S&P 500"
