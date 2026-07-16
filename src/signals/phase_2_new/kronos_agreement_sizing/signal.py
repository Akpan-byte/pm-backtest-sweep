# CHANGE_SUMMARY
# 2026-07-16  (subagent)
#   - Implemented kronos_agreement_sizing_signal per STRATEGY_SPECS.md.
#   - Reads pre-computed Kronos 5m close forecasts from /tmp/kronos_forecasts_5m.parquet
#     at module import and returns neutral if the file is missing or unreadable.
#   - Always returns triggered=False; provides direction/confidence for a meta sizer.
#   - Applies INTERFACE.md time guards (rem_sec/elapsed_sec > 5) and entry-price cap
#     (0.05 <= entry_price <= 0.85).
# WHY: Provide a lightweight sizing helper that exposes Kronos forecast agreement
#      without firing standalone trades.

"""Kronos agreement sizing helper for Polymarket BTC 5m up/down markets."""

from datetime import datetime, timezone

# Module-level state required by the interface.
_STATE = {}

# Forecast file path referenced in STRATEGY_SPECS.md.
_FORECAST_PATH = "/tmp/kronos_forecasts_5m.parquet"

# Column name candidates for the predicted close in the forecast file.
_PREDICTED_CLOSE_COLS = (
    "predicted_close",
    "forecast",
    "pred_close",
    "prediction",
    "pred_price",
    "close",
    "price",
)


def _load_kronos_forecasts(path):
    """Load the Kronos forecast parquet/CSV at import time; return None on any failure."""
    try:
        import pandas as pd

        return pd.read_parquet(path)
    except Exception:
        pass

    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()
    except Exception:
        pass

    return None


_KRONOS_DF = _load_kronos_forecasts(_FORECAST_PATH)


def _parse_iso(value):
    """Parse an ISO timestamp string to a timezone-aware datetime, or None."""
    if not value:
        return None
    try:
        # datetime.fromisoformat in older Pythons does not accept a trailing 'Z'.
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _find_forecast_row(market_id, start_date_iso):
    """Return the predicted close for a market, or None if unavailable."""
    if _KRONOS_DF is None or not market_id:
        return None

    df = _KRONOS_DF
    market_col = None
    if "market_id" in df.columns:
        market_col = "market_id"

    try:
        if market_col is not None:
            rows = df[df[market_col] == market_id]
        elif df.index.name == "market_id":
            rows = df.loc[[market_id]]
        else:
            return None
    except Exception:
        return None

    if rows.empty:
        return None

    # If multiple forecasts exist, pick the row whose timestamp is closest to
    # the market open time.
    if len(rows) > 1 and "forecast_timestamp" in rows.columns:
        start_dt = _parse_iso(start_date_iso)
        if start_dt is not None:
            try:
                rows = rows.copy()
                rows["_diff"] = (
                    rows["forecast_timestamp"].apply(_parse_iso) - start_dt
                ).abs()
                rows = rows.sort_values("_diff")
            except Exception:
                pass

    row = rows.iloc[0]

    for col in _PREDICTED_CLOSE_COLS:
        if col in row.index:
            val = row[col]
            try:
                return float(val)
            except Exception:
                continue
    return None


def _get_forecast(market_id, start_date_iso):
    """Cache and return the per-market forecast; keyed by market_id in _STATE."""
    cache_key = f"{market_id}_forecast"
    if cache_key not in _STATE:
        _STATE[cache_key] = _find_forecast_row(market_id, start_date_iso)
    return _STATE[cache_key]


def _neutral(spot_price, reason):
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": float(spot_price),
        "entry_price": 0.0,
        "source": "KRONOS_AGREEMENT_SIZING",
        "reason": reason,
    }


def kronos_agreement_sizing_signal(**kwargs):
    """Return Kronos forecast direction/confidence for a meta sizer.

    This signal never triggers on its own (``triggered`` is always ``False``).
    It loads ``/tmp/kronos_forecasts_5m.parquet`` once at module import and
    returns neutral if the file is missing or the current market has no forecast.
    """
    spot_price = float(kwargs.get("spot_price", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    market_id = kwargs.get("market_id", "")
    start_date_iso = kwargs.get("start_date_iso", "")

    # Global time guards from INTERFACE.md.
    if rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral(spot_price, "time guard blocks first/last 5 seconds")

    pred_close = _get_forecast(market_id, start_date_iso)
    if pred_close is None:
        return _neutral(spot_price, "no Kronos forecast available")

    up_threshold = spot_price * 1.0001
    down_threshold = spot_price * 0.9999

    if pred_close > up_threshold:
        direction = "YES"
        entry_price = kwargs.get("yes_ask", yp)
    elif pred_close < down_threshold:
        direction = "NO"
        entry_price = kwargs.get("no_ask", np_val)
    else:
        return _neutral(
            spot_price,
            f"forecast inside neutral band (pred={pred_close:.2f}, spot={spot_price:.2f})",
        )

    # Entry price cap from INTERFACE.md.
    if not (0.05 <= entry_price <= 0.85):
        return _neutral(
            spot_price,
            f"entry price {entry_price:.3f} outside [0.05, 0.85] cap",
        )

    # Confidence scales with predicted move magnitude, capped at 1.0.
    move = abs(pred_close - spot_price) / spot_price
    confidence = min(1.0, move / 0.001)

    return {
        "triggered": False,
        "direction": direction,
        "confidence": float(confidence),
        "signal_price": float(spot_price),
        "entry_price": float(entry_price),
        "source": "KRONOS_AGREEMENT_SIZING",
        "reason": (
            f"Kronos forecast agrees {direction} "
            f"(pred={pred_close:.2f}, spot={spot_price:.2f}, "
            f"move={move:.6f})"
        ),
    }
