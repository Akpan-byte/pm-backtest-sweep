# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Created core/candle_utils.py, the candlestick-metrics module for the
#     StarTrading strategies. Separates body/wick math, ATR, and candle
#     classification (doji vs strong body) from detection and signal logic.
# WHY: Granular compartmentalization; see docs/BLUEPRINTS.md.

"""Candlestick metrics and classification for the StarTrading strategies.

Bars are dicts: {timestamp, open, high, low, close, volume}.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import logging

log = logging.getLogger("strategies.core.candle_utils")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Candle body must be at least this fraction of (body+wick) to count as a
# strong displacement candle (Blueprint 1: ~70% body, <30% wicks).
STRONG_BODY_RATIO = 0.70

# A candle whose body fraction is below this is treated as a doji / 50_50 candle.
DOJI_BODY_RATIO = 0.30

# Movement candle must be at least this multiple of ATR(20) to be "significantly
# larger than recent candles" (Blueprint 1).
MOVEMENT_CANDLE_ATR_MULT = 2.5


def candle_body_size(candle: dict) -> float:
    return abs(candle["close"] - candle["open"])


def candle_range(candle: dict) -> float:
    return candle["high"] - candle["low"]


def candle_wick_size(candle: dict) -> float:
    body = candle_body_size(candle)
    return candle_range(candle) - body


def candle_body_ratio(candle: dict) -> float:
    """body / (body + wick). 1.0 = no wicks, 0.0 = doji."""
    body = candle_body_size(candle)
    rng = candle_range(candle)
    if rng <= 0:
        return 0.0
    return body / rng


def is_doji_candle(candle: dict, threshold: float = DOJI_BODY_RATIO) -> bool:
    return candle_body_ratio(candle) <= threshold


def is_strong_body_candle(candle: dict, threshold: float = STRONG_BODY_RATIO) -> bool:
    return candle_body_ratio(candle) >= threshold


def calculate_atr(bars: list[dict], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        high_low = bars[i]["high"] - bars[i]["low"]
        high_close = abs(bars[i]["high"] - bars[i - 1]["close"])
        low_close = abs(bars[i]["low"] - bars[i - 1]["close"])
        trs.append(max(high_low, high_close, low_close))
    return sum(trs[-period:]) / period
