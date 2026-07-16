# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Created kronos_5m_filter signal module per STRATEGY_SPECS.md.
#   - Loads /tmp/kronos_forecasts_5m.parquet once at import; returns neutral if missing.
#   - Triggers YES/NO when the precomputed 5m close forecast diverges from spot
#     beyond the thresholds defined in the spec, subject to time/price guards.
# WHY: Provides a standalone Kronos-mini forecast filter signal for Polymarket
#      BTC 5m UP/DOWN markets.

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd  # type: ignore

_STATE: Dict[str, Any] = {}

_FORECASTS: Optional[Any] = None
_FORECAST_LOAD_ERROR: Optional[str] = None


def _load_forecasts(path: str = "/tmp/kronos_forecasts_5m.parquet") -> Optional[Any]:
    """Load Kronos 5m close forecasts from parquet.

    Expected columns: market_id, timestamp, predicted_close.
    Returns None (and sets _FORECAST_LOAD_ERROR) on any failure.
    """
    global _FORECASTS, _FORECAST_LOAD_ERROR
    try:
        import os

        if not os.path.exists(path):
            _FORECAST_LOAD_ERROR = f"forecast file not found: {path}"
            return None

        df = pd.read_parquet(path)
        required = {"market_id", "timestamp", "predicted_close"}
        if not required.issubset(set(df.columns)):
            _FORECAST_LOAD_ERROR = (
                f"missing columns; need {required}, got {set(df.columns)}"
            )
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index(["market_id", "timestamp"]).sort_index()
        return df
    except Exception as exc:  # noqa: BLE001
        _FORECAST_LOAD_ERROR = str(exc)
        return None


try:
    _FORECASTS = _load_forecasts()
except Exception:  # noqa: BLE001
    _FORECASTS = None


def _market_expiry_ts(start_date_iso: str, duration_sec: float) -> Optional[datetime]:
    """Compute the market expiry timestamp from its open ISO time and duration."""
    try:
        iso = start_date_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt + timedelta(seconds=float(duration_sec))
    except Exception:  # noqa: BLE001
        return None


def _neutral(reason: str, spot_price: float) -> Dict[str, Any]:
    """Return a neutral (non-triggered) signal dict."""
    return {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "signal_price": float(spot_price),
        "entry_price": 0.0,
        "source": "KRONOS_5M_FILTER",
        "reason": reason,
    }


def kronos_5m_filter_signal(**kwargs) -> Dict[str, Any]:
    """Kronos-mini 5m close forecast filter signal.

    Uses precomputed forecasts to confirm or block a directional entry.
    Follows the new signal interface defined in INTERFACE.md.
    """
    spot_price = float(kwargs.get("spot_price", 0.0))
    yp = float(kwargs.get("yp", 0.0))
    np_val = float(kwargs.get("np_val", 0.0))
    rem_sec = float(kwargs.get("rem_sec", 0.0))
    elapsed_sec = float(kwargs.get("elapsed_sec", 0.0))
    duration_sec = float(kwargs.get("duration_sec", 300.0))
    market_id = kwargs.get("market_id", "")
    start_date_iso = kwargs.get("start_date_iso", "")

    # Time guard: do not trade in the first or last 5 seconds.
    if rem_sec <= 5 or elapsed_sec <= 5:
        return _neutral("time guard: too early or too late", spot_price)

    # Forecast availability guard.
    if _FORECASTS is None:
        return _neutral(
            f"forecast unavailable: {_FORECAST_LOAD_ERROR or 'unknown'}",
            spot_price,
        )

    expiry_ts = _market_expiry_ts(start_date_iso, duration_sec)
    if expiry_ts is None:
        return _neutral("invalid start_date_iso", spot_price)

    # Look up the forecast for this market's expiry (5m close).
    try:
        key = (str(market_id), expiry_ts)
        if key not in _FORECASTS.index:
            return _neutral("no forecast for market expiry", spot_price)

        forecast_row = _FORECASTS.loc[key]
        if isinstance(forecast_row, pd.Series):
            predicted_close = float(forecast_row["predicted_close"])
        else:
            # Duplicate index entries; take the first row.
            predicted_close = float(forecast_row.iloc[0]["predicted_close"])
    except Exception as exc:  # noqa: BLE001
        return _neutral(f"forecast lookup failed: {exc}", spot_price)

    # Determine direction from forecast vs spot.
    if predicted_close > spot_price * 1.0001:
        direction = "YES"
        entry_price = yp
        threshold_desc = "predicted close > spot * 1.0001"
    elif predicted_close < spot_price * 0.9999:
        direction = "NO"
        entry_price = np_val
        threshold_desc = "predicted close < spot * 0.9999"
    else:
        return _neutral("forecast within neutral band around spot", spot_price)

    # Entry price cap per INTERFACE.
    if not (0.05 <= entry_price <= 0.85):
        return _neutral(
            f"entry price {entry_price:.4f} outside [0.05, 0.85]",
            spot_price,
        )

    # Confidence scales with predicted move magnitude, capped at 1.0.
    move = abs(predicted_close - spot_price) / spot_price if spot_price > 0 else 0.0
    confidence = min(1.0, max(0.1, move * 1000.0))

    return {
        "triggered": True,
        "direction": direction,
        "confidence": float(confidence),
        "signal_price": float(spot_price),
        "entry_price": float(entry_price),
        "source": "KRONOS_5M_FILTER",
        "reason": (
            f"Kronos 5m close forecast {predicted_close:.2f} "
            f"({threshold_desc}), entry {direction} @ {entry_price:.4f}"
        ),
    }
