from __future__ import annotations

from dataclasses import dataclass


# 2026-02-14 기준 수동 입력 베이스라인(필요 시 주기 업데이트)
# 점수 범위: 0~100
US_FUNDAMENTAL_SCORES: dict[str, float] = {
    "MSFT": 84.0,
    "GOOGL": 80.0,
    "AMZN": 70.0,
    "META": 73.0,
    "NVDA": 77.0,
    "AVGO": 75.0,
    "BRK-B": 68.0,
    "LLY": 72.0,
    "COST": 61.0,
    "V": 69.0,
}


@dataclass
class HybridScoreResult:
    fundamental_score: float
    technical_score: float
    confidence: float
    technical_adjusted: float
    hybrid_score: float
    label: str


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def get_fundamental_score(symbol: str, market: str, default_value: float = 50.0) -> float:
    m = str(market).upper()
    s = str(symbol).upper()

    if m == "US":
        return float(US_FUNDAMENTAL_SCORES.get(s, default_value))

    # KR/기타 시장은 아직 연결 전: 중립값
    return float(default_value)


def compute_hybrid_score(
    fundamental_score: float,
    technical_score: float,
    confidence: float,
    w_fund: float = 0.6,
    w_tech: float = 0.4,
) -> HybridScoreResult:
    fund = _clamp(float(fundamental_score), 0.0, 100.0)
    tech = _clamp(float(technical_score), 0.0, 100.0)
    conf = _clamp(float(confidence), 0.0, 100.0)

    conf01 = conf / 100.0
    technical_adjusted = tech * (0.6 + 0.4 * conf01)
    hybrid_score = (w_fund * fund) + (w_tech * technical_adjusted)

    if hybrid_score >= 75:
        label = "ACCUMULATE"
    elif hybrid_score >= 60:
        label = "HOLD/SCALE"
    else:
        label = "WATCH"

    return HybridScoreResult(
        fundamental_score=round(fund, 1),
        technical_score=round(tech, 1),
        confidence=round(conf, 1),
        technical_adjusted=round(technical_adjusted, 1),
        hybrid_score=round(hybrid_score, 1),
        label=label,
    )
