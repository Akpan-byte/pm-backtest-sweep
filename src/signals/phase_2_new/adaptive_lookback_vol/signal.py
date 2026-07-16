# CHANGE_SUMMARY
# 2026-07-16  assistant
#   - Created adaptive_lookback_vol signal module per INTERFACE.md and STRATEGY_SPECS.md.
#   - Computes 20-tick realized volatility and compares it to the recent vol history.
#   - High-vol regime (top 25%): follow 5-tick high/low momentum breakouts.
#   - Low-vol regime (bottom 25%): fade spot extremes using a 20-tick z-score > 1.5.
#   - Respects time guard (rem_sec/elapsed_sec > 5) and entry-price cap [0.05, 0.85].
# WHY: Implements the adaptive_lookback_vol strategy for Polymarket BTC 5m up/down
#      markets, keeping all state local and avoiding network calls.

import statistics
from typing import Any, Dict, List

# Per-market persistent state keyed by market_id, as required by INTERFACE.md.
_STATE: Dict[str, Dict[str, Any]] = {}

# Strategy constants.
_VOL_LOOKBACK = 20          # ticks for realized-vol estimate
_BREAKOUT_LOOKBACK = 5      # short lookback for high-vol momentum breakout
_FADE_LOOKBACK = 20         # long lookback for low-vol mean-reversion fade
_Z_SCORE_THRESHOLD = 1.5    # fade trigger threshold
_VOL_HISTORY_MAX = 100      # rolling window of vol estimates for percentile ranking
_VOL_PERCENTILE_MIN = 20    # min samples needed before declaring a vol regime


def _returns(prices: List[float]) -> List[float]:
    """Return percentage returns for the given price series."""
    return [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
        if prices[i - 1] != 0.0
    ]


def _realized_vol(prices: List[float], lookback: int = _VOL_LOOKBACK) -> float:
    """Standard deviation of returns over the last `lookback` ticks."""
    if len(prices) < lookback + 1:
        return 0.0
    rets = _returns(prices[-(lookback + 1):])
    if len(rets) < 2:
        return 0.0
    return statistics.stdev(rets)


def _update_vol_history(market_id: str, vol: float) -> None:
    """Append the current vol estimate to the market's rolling vol history."""
    state = _STATE.setdefault(market_id, {})
    vols: List[float] = state.setdefault("vol_history", [])
    vols.append(vol)
    if len(vols) > _VOL_HISTORY_MAX:
        vols.pop(0)


def _vol_regime(vols: List[float]) -> str:
    """Return 'high', 'low', or 'mid' based on current vol's percentile rank."""
    if len(vols) < _VOL_PERCENTILE_MIN:
        return "unknown"
    sorted_vols = sorted(vols)
    n = len(sorted_vols)
    q25 = sorted_vols[n // 4]
    q75 = sorted_vols[(3 * n) // 4]
    current = vols[-1]
    if current >= q75:
        return "high"
    if current <= q25:
        return "low"
    return "mid"


def _neutral(spot_price: float, reason: str) -> Dict[str, Any]:
    """Return a neutral (no-trigger) result dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": spot_price,
        "entry_price": 0.0,
        "source": "ADAPTIVE_LOOKBACK_VOL",
        "reason": reason,
    }


def adaptive_lookback_vol_signal(**kwargs: Any) -> Dict[str, Any]:
    """Adaptive lookback volatility signal for Polymarket BTC 5m up/down markets.

    Regime-switching signal: in high-vol regimes it follows short-term momentum
    breakouts; in low-vol regimes it fades spot extremes via z-score.
    """
    spot_price = float(kwargs.get("spot_price", 0.0))
    strike = float(kwargs.get("strike", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    spot_history: List[float] = kwargs.get("spot_history", [])
    market_id = str(kwargs.get("market_id", ""))

    neutral = _neutral(spot_price, "no signal")

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5.0 or elapsed_sec <= 5.0:
        neutral["reason"] = "time guard"
        return neutral

    # Need enough spot history for vol and z-score estimates.
    if len(spot_history) < _VOL_LOOKBACK + 1:
        neutral["reason"] = "insufficient spot history"
        return neutral

    if strike <= 0.0:
        neutral["reason"] = "invalid strike"
        return neutral

    # Compute current realized vol and store it for percentile ranking.
    vol = _realized_vol(spot_history, _VOL_LOOKBACK)
    _update_vol_history(market_id, vol)

    vols = _STATE.get(market_id, {}).get("vol_history", [])
    regime = _vol_regime(vols)

    if regime == "unknown":
        neutral["reason"] = "building vol history"
        return neutral

    if regime == "mid":
        neutral["reason"] = "mid vol regime"
        return neutral

    direction = None
    entry_price = 0.0
    confidence = 0.0
    reason = ""

    if regime == "high":
        # High vol: follow 5-tick high/low breakout.
        window = spot_history[-_BREAKOUT_LOOKBACK:]
        if spot_price >= max(window):
            direction = "YES"
            entry_price = yp
            ret_5 = (spot_price - window[0]) / window[0] if window[0] != 0.0 else 0.0
            confidence = min(1.0, max(0.0, abs(ret_5) / 0.001))
            reason = f"high vol 5-tick high breakout (vol={vol:.6f})"
        elif spot_price <= min(window):
            direction = "NO"
            entry_price = np_val
            ret_5 = (spot_price - window[0]) / window[0] if window[0] != 0.0 else 0.0
            confidence = min(1.0, max(0.0, abs(ret_5) / 0.001))
            reason = f"high vol 5-tick low breakout (vol={vol:.6f})"
        else:
            neutral["reason"] = "high vol but no 5-tick breakout"
            return neutral

    elif regime == "low":
        # Low vol: fade extremes using a 20-tick z-score.
        window = spot_history[-_FADE_LOOKBACK:]
        mean = statistics.mean(window)
        std = statistics.stdev(window)
        if std <= 0.0:
            neutral["reason"] = "zero std in fade window"
            return neutral
        z = (spot_price - mean) / std
        if z > _Z_SCORE_THRESHOLD:
            direction = "NO"
            entry_price = np_val
            confidence = min(1.0, max(0.0, (z - _Z_SCORE_THRESHOLD) / _Z_SCORE_THRESHOLD))
            reason = f"low vol z-score fade high (z={z:.3f})"
        elif z < -_Z_SCORE_THRESHOLD:
            direction = "YES"
            entry_price = yp
            confidence = min(1.0, max(0.0, (abs(z) - _Z_SCORE_THRESHOLD) / _Z_SCORE_THRESHOLD))
            reason = f"low vol z-score fade low (z={z:.3f})"
        else:
            neutral["reason"] = f"low vol but z-score {z:.3f} within threshold"
            return neutral

    # Entry-price cap: only enter inside [0.05, 0.85].
    if not (0.05 <= entry_price <= 0.85):
        neutral["reason"] = f"{direction} entry price {entry_price:.4f} outside cap"
        neutral["entry_price"] = entry_price
        return neutral

    return {
        "triggered": True,
        "direction": direction,
        "confidence": confidence,
        "signal_price": spot_price,
        "entry_price": entry_price,
        "source": "ADAPTIVE_LOOKBACK_VOL",
        "reason": reason,
    }
