"""Data loading and preprocessing helpers for the Flexing Joe ORB backtest."""
from __future__ import annotations

from typing import Optional

import pandas as pd


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load a 1-minute OHLCV CSV.

    The CSV is expected to contain the columns ``timestamp, open, high, low,
    close, volume`` (case-insensitive). ``timestamp`` is parsed as UTC, the
    frame is sorted, column names are normalized to lowercase, and the
    timestamp is set as the DataFrame's timezone-aware DatetimeIndex.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    # Ensure numeric price/volume columns.
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resample_bars(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars to ``minutes``-minute bars.

    The input frame must have a UTC-aware DatetimeIndex. The output is a
    DataFrame with an America/New_York DatetimeIndex.
    """
    if minutes < 1:
        raise ValueError("minutes must be >= 1")

    df = df_1m.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    # Resample using proper OHLCV aggregation.
    resampled = df.resample(f"{minutes}min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop bars with no price data (e.g. weekends/holidays).
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled["volume"] = resampled["volume"].fillna(0)

    # Convert to ET.
    resampled.index = resampled.index.tz_convert("America/New_York")
    return resampled


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Standard exponential moving average."""
    if period < 1:
        raise ValueError("period must be >= 1")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def get_session_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Return a boolean mask for bars whose ET time is within ``start`` and ``end``.

    ``start`` and ``end`` are "HH:MM" strings interpreted in the frame's local
    time (expected to be ``America/New_York``).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    ts = df.index
    if ts.tz is None:
        ts = ts.tz_localize("America/New_York")
    elif str(ts.tz) != "America/New_York":
        ts = ts.tz_convert("America/New_York")
    time_str = ts.strftime("%H:%M")
    return (time_str >= start) & (time_str <= end)


def load_optional_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load an optional cross-instrument CSV (VIX, ES, NQ, etc.).

    Returns ``None`` if ``path`` is ``None`` or the file cannot be read.
    """
    if path is None:
        return None
    try:
        return load_ohlcv_csv(path)
    except Exception:
        return None
