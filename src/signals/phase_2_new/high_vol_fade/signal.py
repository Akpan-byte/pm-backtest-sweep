# CHANGE_SUMMARY
# 2026-07-16  assistant
#   - Created high_vol_fade signal module per INTERFACE.md and the active wave-2 strategy plan.
#   - 10t vol > 2x 30t vol with yp/np extreme -> fade
#   - Respects the standard time guard (no trade in first/last 5s) and entry-price
#     cap [0.05, 0.85].
# 2026-07-16  kilo
#   - Switched entry_price to the ask side for taker fills:
#     YES direction uses yes_ask (fallback to yp), NO direction uses no_ask
#     (fallback to np_val) when ask is missing or invalid.
#   - The [0.05, 0.85] entry-price guard is unchanged.
#
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



def high_vol_fade_signal(**kwargs: Any) -> Dict[str, Any]:
    """Simple 5m BTC up/down signal: 10t vol > 2x 30t vol with yp/np extreme -> fade"""
    spot_price = float(kwargs.get("spot_price", 0.0))
    strike = float(kwargs.get("strike", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    spot_history: List[float] = kwargs.get("spot_history", [])
    yp_history: List[float] = kwargs.get("yp_history", [])
    np_history: List[float] = kwargs.get("np_history", [])
    source = "HIGH_VOL_FADE"

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
    if vol10 > vol30 * 2.0:
        if yp > 0.85 and np_val <= 0.85:
            entry = float(kwargs.get("no_ask", np_val) or np_val)  # taker fill at ask
            if 0.05 <= entry <= 0.85:
                return {
                    "triggered": True, "direction": "NO",
                    "confidence": min(1.0, max(0.0, (yp - 0.85) / 0.15)),
                    "signal_price": spot_price, "entry_price": entry,
                    "source": source,
                    "reason": f"vol spike {vol10:.6f}>{vol30*2:.6f}, fade extreme yp {yp:.3f}",
                }
        if np_val > 0.85 and yp <= 0.85:
            entry = float(kwargs.get("yes_ask", yp) or yp)  # taker fill at ask
            if 0.05 <= entry <= 0.85:
                return {
                    "triggered": True, "direction": "YES",
                    "confidence": min(1.0, max(0.0, (np_val - 0.85) / 0.15)),
                    "signal_price": spot_price, "entry_price": entry,
                    "source": source,
                    "reason": f"vol spike {vol10:.6f}>{vol30*2:.6f}, fade extreme np {np_val:.3f}",
                }
    return _neutral(spot_price, "no signal", source)

