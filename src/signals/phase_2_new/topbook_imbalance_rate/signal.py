# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Created topbook_imbalance_rate signal for Polymarket BTC 5m up/down markets.
#   - Tracks YES bid size (ub) and NO ask size (da) per market in _STATE.
#   - Triggers YES when YES bid size rises strongly over the last 3 ticks and
#     yp <= 0.80.
#   - Triggers NO when NO ask size rises strongly over the last 3 ticks and
#     np_val <= 0.80.
#   - Confidence scales with the magnitude of the size change relative to the
#     recent average size.
# WHY: Implements the topbook_imbalance_rate strategy spec from
#      /config/new_signals/STRATEGY_SPECS.md.

from collections import deque

# Module-level state keyed by market_id. Each entry keeps a short rolling
# history of YES bid size (ub) and NO ask size (da) so we can compute per-tick
# changes and look back over the last 3 ticks.
_STATE = {}

# Tunables
_IMBALANCE_LOOKBACK_TICKS = 3
_SIZE_HISTORY_MAXLEN = 10
# "Strongly positive" threshold: the sum of the last N size increases must exceed
# this fraction of the recent average size. Confidence normalizes at 2x this.
_STRONG_FRAC = 0.05
_CONFIDENCE_FRAC = 0.10


def _neutral(spot_price, reason):
    """Return a neutral (no-trade) signal dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": float(spot_price) if spot_price is not None else 0.0,
        "entry_price": 0.0,
        "source": "TOPBOOK_IMBALANCE_RATE",
        "reason": reason,
    }


def topbook_imbalance_rate_signal(
    spot_price=None,
    strike=None,
    rem_sec=None,
    elapsed_sec=None,
    duration_sec=None,
    yp=None,
    np_val=None,
    yes_ask=None,
    no_ask=None,
    spot_history=None,
    yp_history=None,
    np_history=None,
    tf_hint=None,
    market_id=None,
    start_date_iso=None,
    **kwargs,
):
    """
    Top-of-book imbalance rate signal.

    Uses the rate of change of YES bid size (ub) vs NO ask size (da) as a
    microstructure signal. Book sizes are expected in kwargs as ``ub`` and
    ``da``; if they are missing the signal returns neutral safely.
    """
    # Time guards: do not trade in the first/last 5 seconds of the market.
    if rem_sec is None or elapsed_sec is None:
        return _neutral(spot_price, "missing time fields")
    if rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral(spot_price, "time guard")

    # Required top-of-book sizes. They are not in the documented INTERFACE but
    # are required by this strategy; safely return neutral if absent.
    ub = kwargs.get("ub")
    da = kwargs.get("da")
    if ub is None or da is None:
        return _neutral(spot_price, "missing ub/da")

    # Validate prices.
    if yp is None or np_val is None or yp <= 0 or np_val <= 0:
        return _neutral(spot_price, "invalid prices")

    key = market_id if market_id is not None else "default"
    state = _STATE.setdefault(
        key,
        {
            "ub_history": deque(maxlen=_SIZE_HISTORY_MAXLEN),
            "da_history": deque(maxlen=_SIZE_HISTORY_MAXLEN),
        },
    )

    ub_hist = state["ub_history"]
    da_hist = state["da_history"]

    # Always store the latest sizes so the history is warm for future calls.
    ub_hist.append(float(ub))
    da_hist.append(float(da))

    # Need at least (lookback + 1) observations to compute ``lookback`` deltas.
    if len(ub_hist) <= _IMBALANCE_LOOKBACK_TICKS:
        return _neutral(spot_price, "warming up")

    recent_ub_deltas = [
        ub_hist[-(i + 1)] - ub_hist[-(i + 2)]
        for i in range(_IMBALANCE_LOOKBACK_TICKS)
    ]
    recent_da_deltas = [
        da_hist[-(i + 1)] - da_hist[-(i + 2)]
        for i in range(_IMBALANCE_LOOKBACK_TICKS)
    ]

    avg_ub = sum(ub_hist) / len(ub_hist)
    avg_da = sum(da_hist) / len(da_hist)

    # YES side: YES bid size rising strongly over the last 3 ticks.
    if all(d > 0 for d in recent_ub_deltas) and yp <= 0.80:
        sum_d_ub = sum(recent_ub_deltas)
        if avg_ub > 0 and sum_d_ub > _STRONG_FRAC * avg_ub:
            entry_price = float(yp)
            if 0.05 <= entry_price <= 0.85:
                confidence = min(1.0, sum_d_ub / (_CONFIDENCE_FRAC * avg_ub))
                return {
                    "triggered": True,
                    "direction": "YES",
                    "confidence": float(confidence),
                    "signal_price": float(spot_price),
                    "entry_price": entry_price,
                    "source": "TOPBOOK_IMBALANCE_RATE",
                    "reason": (
                        f"YES bid size rising {sum_d_ub:.2f} over last "
                        f"{_IMBALANCE_LOOKBACK_TICKS} ticks (avg {avg_ub:.2f})"
                    ),
                }

    # NO side: NO ask size rising strongly over the last 3 ticks.
    if all(d > 0 for d in recent_da_deltas) and np_val <= 0.80:
        sum_d_da = sum(recent_da_deltas)
        if avg_da > 0 and sum_d_da > _STRONG_FRAC * avg_da:
            entry_price = float(np_val)
            if 0.05 <= entry_price <= 0.85:
                confidence = min(1.0, sum_d_da / (_CONFIDENCE_FRAC * avg_da))
                return {
                    "triggered": True,
                    "direction": "NO",
                    "confidence": float(confidence),
                    "signal_price": float(spot_price),
                    "entry_price": entry_price,
                    "source": "TOPBOOK_IMBALANCE_RATE",
                    "reason": (
                        f"NO ask size rising {sum_d_da:.2f} over last "
                        f"{_IMBALANCE_LOOKBACK_TICKS} ticks (avg {avg_da:.2f})"
                    ),
                }

    return _neutral(spot_price, "no topbook imbalance")
