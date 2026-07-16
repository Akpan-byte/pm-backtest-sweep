# CHANGE_SUMMARY
# 2026-07-16  assistant
#   - Created closing_fade signal module per INTERFACE.md and the active wave-2 strategy plan.
#   - rem <= 60s, fade last 3 ticks, entry <= 0.80
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



def closing_fade_signal(**kwargs: Any) -> Dict[str, Any]:
    """Simple 5m BTC up/down signal: rem <= 60s, fade last 3 ticks, entry <= 0.80"""
    spot_price = float(kwargs.get("spot_price", 0.0))
    strike = float(kwargs.get("strike", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    spot_history: List[float] = kwargs.get("spot_history", [])
    yp_history: List[float] = kwargs.get("yp_history", [])
    np_history: List[float] = kwargs.get("np_history", [])
    source = "CLOSING_FADE"

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5.0 or elapsed_sec <= 5.0:
        return _neutral(spot_price, "time guard", source)

    if rem_sec > 60.0:
        return _neutral(spot_price, "outside closing window", source)
    if len(spot_history) < 4:
        return _neutral(spot_price, "insufficient spot history", source)
    last = spot_history[-4:]
    ups = all(last[i] > last[i-1] for i in range(1, 4))
    downs = all(last[i] < last[i-1] for i in range(1, 4))
    if ups and np_val <= 0.80:
        entry = kwargs.get("no_ask", np_val)
        if 0.05 <= entry <= 0.85:
            return {
                "triggered": True, "direction": "NO",
                "confidence": 0.6,
                "signal_price": spot_price, "entry_price": entry,
                "source": source,
                "reason": f"last 3 ticks up near expiry, fading NO",
            }
    if downs and yp <= 0.80:
        entry = kwargs.get("yes_ask", yp)
        if 0.05 <= entry <= 0.85:
            return {
                "triggered": True, "direction": "YES",
                "confidence": 0.6,
                "signal_price": spot_price, "entry_price": entry,
                "source": source,
                "reason": f"last 3 ticks down near expiry, fading YES",
            }
    return _neutral(spot_price, "no signal", source)

