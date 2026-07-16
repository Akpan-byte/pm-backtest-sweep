# CHANGE_SUMMARY
# 2026-07-16  (subagent)
#   - Created /config/new_signals/time_of_day_bias/signal.py.
#   - Implements the Wave 1 time_of_day_bias strategy per INTERFACE.md and
#     STRATEGY_SPECS.md.
#   - Detects ET-hour windows (breakout vs mean-reversion) and triggers on
#     20-tick spot highs/lows with fixed 0.5 confidence.
#   - Enforces time guard (rem_sec <= 5 or elapsed_sec <= 5) and entry price
#     cap (0.05 <= entry_price <= 0.85).
# WHY: Deliver a self-contained, offline-safe 5m BTC up/down signal module.

"""Time-of-day bias signal for Polymarket BTC 5m UP/DOWN markets."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo

    _NY_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - older Python or missing zoneinfo data
    ZoneInfo = None  # type: ignore[misc, assignment]
    _NY_TZ = timezone(timedelta(hours=-5))

# Module-level state per INTERFACE rule 3. This strategy is stateless, but the
# dict is kept available for future extensions (e.g. per-market cooldowns).
_STATE: Dict[str, Any] = {}


def _parse_et_hour(start_date_iso: Optional[str]) -> Optional[float]:
    """Parse an ISO timestamp and return the hour as a float in ET."""
    if not start_date_iso:
        return None
    try:
        s = start_date_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_et = dt.astimezone(_NY_TZ)
        return dt_et.hour + dt_et.minute / 60.0
    except Exception:
        return None


def _window_type(hour: float) -> str:
    """Return 'breakout' or 'mean_reversion' for the given ET hour."""
    # Breakout-friendly windows: 09:30-11:30 and 14:00-16:00 ET.
    if (9.5 <= hour < 11.5) or (14.0 <= hour < 16.0):
        return "breakout"
    # Mean-reversion-friendly windows: 11:30-14:00 and overnight (16:00-09:30).
    return "mean_reversion"


def _is_new_20tick_high(spot_price: float, spot_history: List[float]) -> bool:
    """True if spot_price exceeds the maximum of the prior 20 ticks."""
    if not spot_history or len(spot_history) < 20:
        return False
    window = spot_history[-20:]
    return spot_price > max(window)


def _is_new_20tick_low(spot_price: float, spot_history: List[float]) -> bool:
    """True if spot_price is below the minimum of the prior 20 ticks."""
    if not spot_history or len(spot_history) < 20:
        return False
    window = spot_history[-20:]
    return spot_price < min(window)


def _neutral(
    signal_price: float,
    source: str,
    reason: str,
) -> Dict[str, Any]:
    """Return a neutral/non-triggered signal dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": signal_price,
        "entry_price": 0.0,
        "source": source,
        "reason": reason,
    }


def time_of_day_bias_signal(**kwargs) -> Dict[str, Any]:
    """Generate a time-of-day bias signal.

    Follows the new signal interface documented in /config/new_signals/INTERFACE.md.
    """
    spot_price: Optional[float] = kwargs.get("spot_price")
    yp: Optional[float] = kwargs.get("yp")
    np_val: Optional[float] = kwargs.get("np_val")
    rem_sec: Optional[float] = kwargs.get("rem_sec")
    elapsed_sec: Optional[float] = kwargs.get("elapsed_sec")
    spot_history: List[float] = kwargs.get("spot_history") or []
    start_date_iso: Optional[str] = kwargs.get("start_date_iso")
    source = "TIME_OF_DAY_BIAS"

    if spot_price is None or yp is None or np_val is None:
        return _neutral(
            spot_price if spot_price is not None else 0.0,
            source,
            "missing required price inputs",
        )

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec is None or elapsed_sec is None or rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral(spot_price, source, "time guard blocked (first/last 5s)")

    hour = _parse_et_hour(start_date_iso)
    if hour is None:
        return _neutral(spot_price, source, "could not parse start_date_iso to ET hour")

    wtype = _window_type(hour)
    new_high = _is_new_20tick_high(spot_price, spot_history)
    new_low = _is_new_20tick_low(spot_price, spot_history)

    direction: Optional[str] = None
    entry_price: float = 0.0
    reason = ""

    if wtype == "breakout":
        if new_high:
            direction = "YES"
            entry_price = kwargs.get("yes_ask", yp)
            reason = f"ET {hour:.2f} breakout window, new 20-tick high"
        elif new_low:
            direction = "NO"
            entry_price = kwargs.get("no_ask", np_val)
            reason = f"ET {hour:.2f} breakout window, new 20-tick low"
    else:  # mean_reversion
        if new_high:
            direction = "NO"
            entry_price = kwargs.get("no_ask", np_val)
            reason = f"ET {hour:.2f} mean-reversion window, fade new 20-tick high"
        elif new_low:
            direction = "YES"
            entry_price = kwargs.get("yes_ask", yp)
            reason = f"ET {hour:.2f} mean-reversion window, fade new 20-tick low"

    if direction is None:
        return _neutral(
            spot_price,
            source,
            f"ET {hour:.2f} {wtype} window, no 20-tick extreme",
        )

    # Entry price cap: only enter within [0.05, 0.85].
    if not (0.05 <= entry_price <= 0.85):
        return _neutral(
            spot_price,
            source,
            f"entry price {entry_price:.4f} outside [0.05, 0.85]",
        )

    return {
        "triggered": True,
        "direction": direction,
        "confidence": 0.5,
        "signal_price": spot_price,
        "entry_price": entry_price,
        "source": source,
        "reason": reason,
    }
