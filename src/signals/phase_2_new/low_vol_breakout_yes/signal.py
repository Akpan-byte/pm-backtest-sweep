# CHANGE_SUMMARY
# 2026-07-16  assistant
#   - Created low_vol_breakout_yes signal module per INTERFACE.md and the active wave-2 strategy plan.
#   - 10t vol < 0.5x 30t vol and 10-tick high -> YES
#   - Respects the standard time guard (no trade in first/last 5s) and entry-price
#     cap [0.05, 0.85].
# WHY: Wave-1 strategies were mostly too restrictive and produced zero trades. This
#      module is intentionally simple so it actually fires on realistic 5m BTC data.
from typing import Any, Dict, List

# Per-market persistent state keyed by market_id, as required by INTERFACE.md.
_STATE: Dict[str, Any] = {}


def _neutral(spot_price: float, reason: str, source: str) -> Dict[str, Any]:
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": spot_price,
        "entry_price": 0.0,
        "source": source,
        "reason": reason,
    }

def _returns(prices: List[float]) -> List[float]:
    return [(prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0.0
            for i in range(1, len(prices))]

def _vol(returns: List[float], window: int) -> float:
    if len(returns) < window or window < 2:
        return 0.0
    s = returns[-window:]
    n = len(s)
    if n < 2:
        return 0.0
    mean = sum(s) / n
    var = sum((r - mean) ** 2 for r in s) / (n - 1)
    return var ** 0.5



def low_vol_breakout_yes_signal(**kwargs: Any) -> Dict[str, Any]:
    """Simple 5m BTC up/down signal: 10t vol < 0.5x 30t vol and 10-tick high -> YES"""
    spot_price = float(kwargs.get("spot_price", 0.0))
    strike = float(kwargs.get("strike", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    spot_history: List[float] = kwargs.get("spot_history", [])
    yp_history: List[float] = kwargs.get("yp_history", [])
    np_history: List[float] = kwargs.get("np_history", [])
    source = "LOW_VOL_BREAKOUT_YES"

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5.0 or elapsed_sec <= 5.0:
        return _neutral(spot_price, "time guard", source)

    if len(spot_history) < 31:
        return _neutral(spot_price, "insufficient spot history", source)
    rets = _returns(spot_history)
    vol10 = _vol(rets, 10)
    vol30 = _vol(rets, 30)
    if vol30 <= 0.0:
        return _neutral(spot_price, "zero prior vol", source)
    if vol10 < vol30 * 0.5 and spot_price >= max(spot_history[-10:]):
        entry = yp
        if 0.05 <= entry <= 0.85:
            return {
                "triggered": True, "direction": "YES",
                "confidence": 0.6,
                "signal_price": spot_price, "entry_price": entry,
                "source": source,
                "reason": f"vol collapse {vol10:.6f}<{vol30*0.5:.6f}, 10t high breakout",
            }
    return _neutral(spot_price, "no signal", source)

