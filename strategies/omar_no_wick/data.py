"""
Data loader for the fractal IV reversal strategy backtester.

Supports CSV / Parquet OHLCV files and timeframe resampling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import resample_ohlcv


# Map human-friendly timeframe labels to pandas offsets.
_TIMEFRAMES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
}


def load_ohlcv(path: str | Path, timeframe: str = "1m") -> pd.DataFrame:
    """
    Load OHLCV market data from a CSV or Parquet file.

    Parameters
    ----------
    path : str | Path
        Path to the data file. Supported extensions: .csv, .parquet, .pq.
    timeframe : str, optional
        Target timeframe: '1m', '5m', '15m', '1h'. The source data is assumed
        to be 1-minute bars and will be resampled if needed, by default '1m'.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with lowercase columns and a DatetimeIndex.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    df.columns = [c.lower() for c in df.columns]

    # Locate timestamp column and set index.
    ts_col = None
    for col in ("timestamp", "datetime", "date", "time"):
        if col in df.columns:
            ts_col = col
            break

    if ts_col is None:
        raise ValueError("No timestamp column found (expected timestamp/datetime/date/time)")

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()

    # Normalize column names if abbreviations are used.
    rename_map = {}
    for col in df.columns:
        if col in ("o", "open"):
            rename_map[col] = "open"
        elif col in ("h", "high"):
            rename_map[col] = "high"
        elif col in ("l", "low"):
            rename_map[col] = "low"
        elif col in ("c", "close"):
            rename_map[col] = "close"
        elif col in ("v", "volume"):
            rename_map[col] = "volume"
    df = df.rename(columns=rename_map)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    # Keep only required columns to avoid surprises.
    df = df[list(required)]

    # Resample if a higher timeframe is requested.
    freq = _TIMEFRAMES.get(timeframe)
    if freq is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use: {list(_TIMEFRAMES)}")

    if timeframe != "1m":
        df = resample_ohlcv(df, freq)

    return df
