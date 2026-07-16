# CHANGE_SUMMARY
# 2026-07-16  assistant
#   - Created book_pressure_yes signal module per INTERFACE.md and the active wave-2 strategy plan.
#   - yp rising faster than np falling over 3 ticks -> YES
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



def book_pressure_yes_signal(**kwargs: Any) -> Dict[str, Any]:
    """Simple 5m BTC up/down signal: yp rising faster than np falling over 3 ticks -> YES"""
    spot_price = float(kwargs.get("spot_price", 0.0))
    strike = float(kwargs.get("strike", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    spot_history: List[float] = kwargs.get("spot_history", [])
    yp_history: List[float] = kwargs.get("yp_history", [])
    np_history: List[float] = kwargs.get("np_history", [])
    source = "BOOK_PRESSURE_YES"

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5.0 or elapsed_sec <= 5.0:
        return _neutral(spot_price, "time guard", source)

    if len(yp_history) < 4 or len(np_history) < 4:
        return _neutral(spot_price, "insufficient book history", source)
    yp_inc = yp_history[-1] - yp_history[-4]
    np_dec = np_history[-4] - np_history[-1]
    if yp_inc > np_dec * 1.1 and yp <= 0.80:
        entry = yp
        if 0.05 <= entry <= 0.85:
            return {
                "triggered": True, "direction": "YES",
                "confidence": min(1.0, max(0.0, yp_inc / 0.05)),
                "signal_price": spot_price, "entry_price": entry,
                "source": source,
                "reason": f"yp rise {yp_inc:.4f} exceeds np decay {np_dec:.4f}",
            }
    return _neutral(spot_price, "no signal", source)

