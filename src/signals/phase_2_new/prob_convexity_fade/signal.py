# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Created prob_convexity_fade signal module.
#   - Implemented prob_convexity_fade_signal per INTERFACE.md and the
#     "prob_convexity_fade" section of STRATEGY_SPECS.md.
#   - Interprets "fade" as fading the recent probability move when it is
#     disproportionately large relative to the realized prob/spot slope near
#     probability extremes (prob > 0.65 -> fade NO; prob < 0.35 -> fade YES).
# 2026-07-16  kilo
#   - Switched entry_price to the ask side for taker fills:
#     YES direction uses yes_ask (fallback to yp), NO direction uses no_ask
#     (fallback to np_val) when ask is missing or invalid.
#   - The [0.05, 0.85] entry-price guard is unchanged.
#
# WHY: Add Wave 1 standalone BTC 5m UP/DOWN signal to the new_signals package.

"""
prob_convexity_fade signal.

Idea: The binary price is a probability, so the same spot move should have a
smaller probability impact when the market is already near 0 or 1. Fade moves
that look too large relative to the spot change.

Logic:
- Compute prob from YES/NO bid prices.
- Compute recent spot return over the last 10 ticks.
- Compute implied delta from strike distance.
- Compute realized delta over the last 30 ticks via simple OLS regression of
  prob vs spot.
- If the recent prob/spot slope is > 1.5x the realized delta and prob is in an
  extreme region, fade the move:
    - prob > 0.65 with an upward prob move -> NO
    - prob < 0.35 with a downward prob move -> YES
- Confidence scales with the overreaction ratio.
"""

from typing import Any, Dict

# Module-level state, keyed by market_id, for any cross-snapshot data.
_STATE: Dict[str, Any] = {}

# Strategy hyperparameters.
N_RECENT = 10
M_REALIZED = 30
OVERREACTION_THRESHOLD = 1.5
PROB_HIGH = 0.65
PROB_LOW = 0.35


def _neutral(signal_price: float = 0.0, reason: str = "neutral") -> Dict[str, Any]:
    """Return a neutral (no-trade) result dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": float(signal_price),
        "entry_price": 0.0,
        "source": "PROB_CONVEXITY_FADE",
        "reason": reason,
    }


def _compute_prob(yp: float, np_val: float) -> float:
    """Implied probability from YES and NO bid prices."""
    denom = yp + np_val
    return yp / denom if denom > 0 else 0.5


def _ols_slope(x: list, y: list) -> float:
    """Simple OLS slope of y vs x. Returns 0.0 if undefined."""
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    var = sum((xi - x_mean) ** 2 for xi in x)
    return cov / var if var != 0 else 0.0


def prob_convexity_fade_signal(
    spot_price: float = None,
    strike: float = None,
    rem_sec: float = None,
    elapsed_sec: float = None,
    duration_sec: float = None,
    yp: float = None,
    np_val: float = None,
    yes_ask: float = None,
    no_ask: float = None,
    spot_history: list = None,
    yp_history: list = None,
    np_history: list = None,
    tf_hint: str = None,
    market_id: str = None,
    start_date_iso: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a probability-convexity fade signal for a BTC 5m binary market.

    See /config/new_signals/INTERFACE.md for the full keyword interface.
    """
    signal_price = float(spot_price) if spot_price is not None else 0.0
    source = "PROB_CONVEXITY_FADE"

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec is None or elapsed_sec is None:
        return _neutral(signal_price, "missing time fields")
    if rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral(
            signal_price,
            f"time guard: rem={rem_sec}s elapsed={elapsed_sec}s",
        )

    # Validate required price fields.
    if spot_price is None or strike is None or yp is None or np_val is None:
        return _neutral(signal_price, "missing required price fields")
    if strike <= 0:
        return _neutral(signal_price, "invalid strike")

    spot = float(spot_price)
    strike_f = float(strike)
    yp_f = float(yp)
    np_f = float(np_val)

    # Need enough history for the 10-tick recent window and 30-tick regression.
    spot_history = spot_history or []
    yp_history = yp_history or []
    np_history = np_history or []
    if (
        len(spot_history) < M_REALIZED
        or len(yp_history) < M_REALIZED
        or len(np_history) < M_REALIZED
    ):
        return _neutral(signal_price, "insufficient history")

    # Current implied probability and historical probability series.
    prob = _compute_prob(yp_f, np_f)
    prob_history = [
        _compute_prob(y, n) for y, n in zip(yp_history, np_history)
    ]

    # Recent spot return over the last N ticks.
    ret = (spot - spot_history[-N_RECENT]) / spot_history[-N_RECENT]

    # Implied delta (signed distance from strike, normalized by strike).
    # Computed per the spec; not directly used in the trigger but retained
    # for reasoning completeness.
    delta = (prob - 0.5) / (spot - strike_f + 1e-9) * strike_f

    # Realized delta: OLS slope of prob vs spot over the last M ticks.
    realized_delta = _ols_slope(
        spot_history[-M_REALIZED:], prob_history[-M_REALIZED:]
    )

    # Recent prob/spot slope over the last N ticks.
    recent_prob_slope = (prob - prob_history[-N_RECENT]) / (
        spot - spot_history[-N_RECENT] + 1e-9
    )

    # Overreaction ratio: how much larger is the recent prob/spot slope than
    # the realized slope?
    ratio = abs(recent_prob_slope) / (abs(realized_delta) + 1e-9)

    if ratio <= OVERREACTION_THRESHOLD:
        return _neutral(
            signal_price,
            f"no overreaction: ratio={ratio:.3f} "
            f"ret={ret:.4%} delta={delta:.4f} "
            f"realized_delta={realized_delta:.6f}",
        )

    # Fade the move only when prob is in an extreme region and the move is
    # toward that extreme. This matches the strategy name and the stated idea
    # of fading overreaction near 0 or 1.
    direction = None
    entry_price = 0.0
    confidence = 0.0
    reason = ""

    if prob > PROB_HIGH and recent_prob_slope > 0:
        direction = "NO"
        entry_price = float(no_ask if no_ask is not None else np_f)  # taker fill at ask
        prob_factor = (prob - PROB_HIGH) / (1.0 - PROB_HIGH)
        reason = (
            f"prob={prob:.3f} > {PROB_HIGH} with upward overreaction "
            f"ratio={ratio:.3f}; fade -> NO"
        )
    elif prob < PROB_LOW and recent_prob_slope < 0:
        direction = "YES"
        entry_price = float(yes_ask if yes_ask is not None else yp_f)  # taker fill at ask
        prob_factor = (PROB_LOW - prob) / PROB_LOW
        reason = (
            f"prob={prob:.3f} < {PROB_LOW} with downward overreaction "
            f"ratio={ratio:.3f}; fade -> YES"
        )
    else:
        return _neutral(
            signal_price,
            f"prob/slope mismatch: prob={prob:.3f} "
            f"slope={recent_prob_slope:.6f} ratio={ratio:.3f}",
        )

    # Entry price cap: only enter within [0.05, 0.85].
    if entry_price < 0.05 or entry_price > 0.85:
        return _neutral(
            signal_price,
            f"entry price {entry_price:.3f} outside [0.05, 0.85]",
        )

    # Confidence scales with overreaction ratio and how extreme prob is.
    confidence = min(1.0, ((ratio - OVERREACTION_THRESHOLD) / 0.5) * prob_factor)

    return {
        "triggered": True,
        "direction": direction,
        "confidence": float(confidence),
        "signal_price": signal_price,
        "entry_price": float(entry_price),
        "source": source,
        "reason": reason,
    }
