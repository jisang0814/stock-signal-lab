from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "profile_reco_log.csv"


def _ensure_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        pd.DataFrame(
            columns=[
                "ts",
                "market",
                "recommended_profile",
                "samples",
                "hit_rate",
                "avg_ret",
                "score",
            ]
        ).to_csv(LOG_PATH, index=False)


def append_recommendation_log(row: dict) -> None:
    _ensure_file()
    payload = {"ts": datetime.now().isoformat(timespec="seconds"), **row}
    pd.DataFrame([payload]).to_csv(LOG_PATH, mode="a", index=False, header=False)


def load_recommendation_log(limit: int = 2000) -> pd.DataFrame:
    _ensure_file()
    df = pd.read_csv(LOG_PATH)
    if len(df) > limit:
        return df.tail(limit).reset_index(drop=True)
    return df
