from __future__ import annotations


PROFILE_PRESETS = {
    "conservative": {
        "label": "보수",
        "entry_threshold": 74.0,
        "sell_threshold": 34.0,
        "trend_boost": 0.0,
        "momentum_boost": -0.05,
        "quality_boost": 0.2,
        "stop_mult_adj": 0.3,
    },
    "balanced": {
        "label": "균형",
        "entry_threshold": 70.0,
        "sell_threshold": 35.0,
        "trend_boost": 0.0,
        "momentum_boost": 0.0,
        "quality_boost": 0.0,
        "stop_mult_adj": 0.0,
    },
    "aggressive": {
        "label": "공격",
        "entry_threshold": 66.0,
        "sell_threshold": 30.0,
        "trend_boost": 0.0,
        "momentum_boost": 0.18,
        "quality_boost": -0.1,
        "stop_mult_adj": -0.15,
    },
}


def get_profile_config(profile: str) -> dict:
    aliases = {
        "defensive": "conservative",
    }
    key = aliases.get(str(profile), str(profile))
    return PROFILE_PRESETS.get(key, PROFILE_PRESETS["balanced"])
