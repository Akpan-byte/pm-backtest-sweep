# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Created prob_convexity_trend signal module.
#   - Implemented prob_convexity_trend_signal per INTERFACE.md and the
#     "prob_convexity_trend" section of STRATEGY_SPECS.md.
# WHY: Add Wave 1 standalone BTC 5m UP/DOWN signal to the new_signals package.

"""
prob_convexity_trend signal.

Idea: When spot keeps trending and the binary probability is lagging
(underpriced convexity), buy the direction.

Logic:
- Require spot has moved > 0.1% from strike (using current spot vs strike).
- If spot is above strike and prob < 0.6, trigger YES.
- If spot is below strike and prob > 0.4, trigger NO.
- Confidence scales with distance from strike and how lagging prob is.
"""

_STATE = {}


def prob_convexity_trend_signal(
    *,
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
):
    """
    Return a signal dict per INTERFACE.md.

    Parameters
    ----------
    **kwargs : ignored
        The function accepts the full interface and ignores unused fields.

    Returns
    -------
    dict
        {
            "triggered": bool,
            "direction": "YES" | "NO" | None,
            "confidence": float,
            "signal_price": float,
            "entry_price": float,
            "source": str,
            "reason": str,
        }
    """
    source = "PROB_CONVEXITY_TREND"
    signal_price = float(spot_price) if spot_price is not None else 0.0

    # Time guard: avoid first/last 5 seconds of the market.
    if rem_sec is None or elapsed_sec is None:
        return _neutral(signal_price, source, "missing time fields")
    if rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral(
            signal_price, source, f"time guard: rem={rem_sec}s elapsed={elapsed_sec}s"
        )

    # Need enough spot history to satisfy the "last 10 ticks" requirement.
    spot_history = spot_history or []
    if len(spot_history) < 10:
        return _neutral(signal_price, source, "insufficient spot history")

    # Validate required price fields.
    if spot_price is None or strike is None:
        return _neutral(signal_price, source, "missing spot/strike")

    spot = float(spot_price)
    strike_f = float(strike)

    # Implied probability from top-of-book YES/NO bids.
    yp_f = float(yp) if yp is not None else 0.0
    np_f = float(np_val) if np_val is not None else 0.0
    denom = yp_f + np_f
    prob = yp_f / denom if denom > 0 else 0.5

    # Distance from strike, measured as a signed percentage.
    dist = (spot - strike_f) / strike_f
    abs_dist = abs(dist)

    # Require spot has moved > 0.1% from strike.
    if abs_dist <= 0.001:
        return _neutral(
            signal_price, source, f"spot move too small: {abs_dist:.4%}"
        )

    direction = None
    entry_price = 0.0
    reason = ""

    if spot > strike_f:
        # Spot is above strike but probability has not caught up.
        if prob >= 0.6:
            return _neutral(
                signal_price, source, f"prob not lagging above strike: {prob:.3f}"
            )
        direction = "YES"
        entry_price = yp_f
        prob_factor = (0.6 - prob) / 0.6
        reason = (
            f"spot above strike by {dist:+.4%} with lagging prob {prob:.3f}; "
            "expect convexity catch-up"
        )
    else:
        # Spot is below strike but probability has not caught up.
        if prob <= 0.4:
            return _neutral(
                signal_price, source, f"prob not lagging below strike: {prob:.3f}"
            )
        direction = "NO"
        entry_price = np_f
        prob_factor = (prob - 0.4) / 0.6
        reason = (
            f"spot below strike by {dist:+.4%} with lagging prob {prob:.3f}; "
            "expect convexity catch-up"
        )

    # Entry price cap: only enter within [0.05, 0.85].
    if entry_price < 0.05 or entry_price > 0.85:
        return _neutral(
            signal_price,
            source,
            f"entry price {entry_price:.3f} outside [0.05, 0.85]",
        )

    # Confidence scales with distance from strike and how lagging prob is.
    confidence = min(1.0, (abs_dist / 0.001) * prob_factor)

    return {
        "triggered": True,
        "direction": direction,
        "confidence": float(confidence),
        "signal_price": signal_price,
        "entry_price": float(entry_price),
        "source": source,
        "reason": reason,
    }


def _neutral(signal_price, source, reason):
    """Return a neutral (non-triggered) signal dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": signal_price,
        "entry_price": 0.0,
        "source": source,
        "reason": reason,
    }
