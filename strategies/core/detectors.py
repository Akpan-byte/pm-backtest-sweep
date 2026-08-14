# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Created core/detectors.py, the market-structure detection module for the
#     StarTrading strategies. Holds ESTABLISHED_MOVEMENT, FVG/BPR/DIRTY_BPR,
#     EQH/EQL, and PROTECTIVE_SWING logic in one navigable place.
# WHY: Granular compartmentalization; see docs/BLUEPRINTS.md.

"""Market-structure detectors for the StarTrading strategies.

Bars are dicts: {timestamp, open, high, low, close, volume}.  Timeframes are
strings: "1m","5m","15m","1h","4h","1d".
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import logging

from . import candle_utils as cu

log = logging.getLogger("strategies.core.detectors")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# Higher-timeframe bias
# ---------------------------------------------------------------------------

def is_established_movement(daily_candles: list[dict], min_consecutive: int = 2) -> tuple[bool, str | None]:
    """Return (confirmed, direction) for ESTABLISHED_MOVEMENT.

    Requires >= min_consecutive daily candles closing in the same direction
    with strong bodies (body ratio above the doji threshold) and minimal
    opposing wicks.
    """
    if len(daily_candles) < min_consecutive:
        return False, None
    recent = daily_candles[-min_consecutive:]
    if all(c["close"] > c["open"] and not cu.is_doji_candle(c) for c in recent):
        return True, "UP"
    if all(c["close"] < c["open"] and not cu.is_doji_candle(c) for c in recent):
        return True, "DOWN"
    return False, None


def is_consolidation_phase(daily_candles: list[dict], min_consecutive: int = 2) -> bool:
    """Opposite of ESTABLISHED_MOVEMENT (Blueprint 2 phase filter)."""
    confirmed, _ = is_established_movement(daily_candles, min_consecutive)
    return not confirmed


# ---------------------------------------------------------------------------
# FVG / BPR
# ---------------------------------------------------------------------------

def detect_fvg(bars: list[dict], timeframe: str = "1m") -> list[dict]:
    """Detect Fair Value Gaps.

    Bullish FVG: low of bar i+1 > high of bar i-1 (gap up).
    Bearish FVG: high of bar i+1 < low of bar i-1 (gap down).
    Returns list of {timeframe, type, direction, high, low, timestamp}.
    """
    fvgs = []
    if len(bars) < 3:
        return fvgs
    for i in range(1, len(bars) - 1):
        prev, nxt = bars[i - 1], bars[i + 1]
        if nxt["low"] > prev["high"]:
            fvgs.append({
                "timeframe": timeframe,
                "type": "FVG",
                "direction": "UP",
                "high": nxt["low"],
                "low": prev["high"],
                "timestamp": bars[i]["timestamp"],
            })
        elif nxt["high"] < prev["low"]:
            fvgs.append({
                "timeframe": timeframe,
                "type": "FVG",
                "direction": "DOWN",
                "high": prev["low"],
                "low": nxt["high"],
                "timestamp": bars[i]["timestamp"],
            })
    return fvgs


def detect_bpr(bars: list[dict], timeframe: str = "5m") -> list[dict]:
    """Detect Balanced Price Ranges (overlapping FVGs).

    A BPR is a balance zone formed when a bullish FVG (gap up) and a bearish
    FVG (gap down) overlap.  Opposite-direction FVGs are required because two
    same-direction FVG ranges can never overlap (the inner bound would demand
    a candle's high < its own low).  Direction is inferred from which imbalance
    extends further: if the upside gap tops the downside gap it is an UP BPR.
    Returns list of {timeframe, type, direction, high, low, timestamp, fvg_overlap}.
    """
    fvgs = detect_fvg(bars, timeframe)
    bprs = []
    for a in range(len(fvgs)):
        for b in range(a + 1, len(fvgs)):
            f1, f2 = fvgs[a], fvgs[b]
            if f1["direction"] == f2["direction"]:
                continue
            overlap_high = min(f1["high"], f2["high"])
            overlap_low = max(f1["low"], f2["low"])
            if overlap_high <= overlap_low:
                continue
            bull = f1 if f1["direction"] == "UP" else f2
            bear = f2 if f1["direction"] == "UP" else f1
            direction = "UP" if bull["high"] >= bear["high"] else "DOWN"
            bprs.append({
                "timeframe": timeframe,
                "type": "BPR",
                "direction": direction,
                "high": overlap_high,
                "low": overlap_low,
                "timestamp": f2["timestamp"],
                "fvg_overlap": True,
            })
    return bprs


def detect_dirty_bpr(bars: list[dict], timeframe: str = "5m") -> list[dict]:
    """DIRTY_BPR: an FVG pierced by a wick but no full body close through it."""
    fvgs = detect_fvg(bars, timeframe)
    dirty = []
    for f in fvgs:
        for c in bars:
            if c["timestamp"] <= f["timestamp"]:
                continue
            if c["high"] >= f["low"] and c["close"] < f["low"]:
                dirty.append({**f, "type": "DIRTY_BPR", "timestamp": c["timestamp"]})
                break
            if c["low"] <= f["high"] and c["close"] > f["high"]:
                dirty.append({**f, "type": "DIRTY_BPR", "timestamp": c["timestamp"]})
                break
    return dirty


# ---------------------------------------------------------------------------
# EQH / EQL
# ---------------------------------------------------------------------------

def detect_eqh_eql(highs: list[float], lows: list[float], threshold_pct: float = 0.001) -> dict:
    """Find Relative Equal Highs/Lows.

    Two swing points are "equal" if within threshold_pct of each other and
    visually obvious.  Returns {highs:[levels], lows:[levels]}.
    """
    eq_highs = []
    eq_lows = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / max(highs[i], 1e-9) <= threshold_pct:
                eq_highs.append((highs[i] + highs[j]) / 2)
                break
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) / max(lows[i], 1e-9) <= threshold_pct:
                eq_lows.append((lows[i] + lows[j]) / 2)
                break
    return {"highs": eq_highs, "lows": eq_lows}


# ---------------------------------------------------------------------------
# Protective swing
# ---------------------------------------------------------------------------

def find_protective_swing(tf_bars: list[dict], trend_direction: str, fvgs: list[dict]) -> dict | None:
    """Return a swing extreme that traded into an FVG and rejected.

    For UP trend we want a swing LOW that dipped into a DOWN FVG and closed
    back above it (the "wall" behind which we hide the stop).
    """
    if not tf_bars:
        return None
    tf_label = tf_bars[0].get("timeframe", "?")
    for c in tf_bars:
        for f in fvgs:
            if f["direction"] == "DOWN" and trend_direction == "UP":
                if c["low"] <= f["high"] and c["close"] > f["low"]:
                    return {
                        "timeframe": tf_label,
                        "direction": "DOWN",
                        "price_level": f["low"],
                        "timestamp": c["timestamp"],
                        "fvg_respected": True,
                    }
            if f["direction"] == "UP" and trend_direction == "DOWN":
                if c["high"] >= f["low"] and c["close"] < f["high"]:
                    return {
                        "timeframe": tf_label,
                        "direction": "UP",
                        "price_level": f["high"],
                        "timestamp": c["timestamp"],
                        "fvg_respected": True,
                    }
    return None
