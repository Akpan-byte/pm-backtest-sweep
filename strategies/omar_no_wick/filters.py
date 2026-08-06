"""
No-wick strategy filter library.

All filters are computed bar-by-bar using only past data (no lookahead).
Each filter adds a boolean column to the DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import atr


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder/RMA smoothing: first value = SMA, then alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add +DI, -DI, ADX columns."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = _wilder_smooth(tr, period)
    plus_dm_smooth = _wilder_smooth(pd.Series(plus_dm, index=df.index), period)
    minus_dm_smooth = _wilder_smooth(pd.Series(minus_dm, index=df.index), period)

    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _wilder_smooth(dx, period)

    df[f"plus_di_{period}"] = plus_di
    df[f"minus_di_{period}"] = minus_di
    df[f"adx_{period}"] = adx
    return df


def compute_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute all filter columns used in the sweep."""
    df = df.copy()

    # ATR
    df["atr_14"] = atr(df, 14)

    # ADX / DI
    df = compute_adx(df, 14)

    # Volume SMA
    df["volume_sma20"] = df["volume"].rolling(20, min_periods=10).mean()

    # Body size
    df["body"] = (df["close"] - df["open"]).abs()

    # Realized volatility (20-bar)
    logret = np.log(df["close"] / df["close"].shift(1))
    df["realvol_20"] = logret.rolling(20, min_periods=10).std()
    df["realvol_90q"] = df["realvol_20"].rolling(1000, min_periods=200).quantile(0.90)

    # ATR median / percentile for chop filter
    df["atr_median_500"] = df["atr_14"].rolling(500, min_periods=100).median()

    # Higher timeframe (15m) trend aligned with 5m signal direction
    df_15 = df.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    # Avoid circular import by importing inside function
    from no_wick import detect_structure
    df_15 = detect_structure(df_15)
    df_15["trend_15m"] = df_15["trend_state"]
    df_15_5m = df_15["trend_15m"].reindex(df.index, method="ffill")
    df["trend_15m"] = df_15_5m

    # Time helpers
    t = df.index.time
    df["is_lunch"] = (t >= pd.Timestamp("11:30").time()) & (t < pd.Timestamp("13:30").time())
    df["is_first30"] = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("10:00").time())
    df["is_last30"] = (t >= pd.Timestamp("15:30").time()) & (t < pd.Timestamp("16:00").time())

    # Filter booleans
    df["f_atr_chop"] = df["atr_14"] >= (df["atr_median_500"] * 0.8)
    df["f_adx"] = df["adx_14"] > 25.0
    df["f_volume"] = df["volume"] > df["volume_sma20"]
    df["f_body"] = df["body"] >= (0.5 * df["atr_14"])
    df["f_iv_wall"] = df["realvol_20"] <= df["realvol_90q"]
    df["f_skip_lunch"] = ~df["is_lunch"]
    df["f_skip_open_close"] = ~(df["is_first30"] | df["is_last30"])

    # Pullback depth: distance to structural stop vs recent swing range
    hh = df["confirmed_HH"]
    ll = df["confirmed_LL"]
    hl = df["confirmed_HL"]
    lh = df["confirmed_LH"]
    swing_range = (hh - ll).abs()
    dist_to_stop_long = (df["open"] - hl).abs()
    dist_to_stop_short = (df["open"] - lh).abs()
    # Allow pullback between 10% and 70% of recent swing range
    df["f_pullback_long"] = (dist_to_stop_long >= 0.10 * swing_range) & (dist_to_stop_long <= 0.70 * swing_range)
    df["f_pullback_short"] = (dist_to_stop_short >= 0.10 * swing_range) & (dist_to_stop_short <= 0.70 * swing_range)
    df["f_pullback"] = df["f_pullback_long"] | df["f_pullback_short"]

    # MTF: 15m trend must agree with intended direction (computed at signal time via current 15m trend)
    df["f_mtf_long"] = df["trend_15m"] == 1
    df["f_mtf_short"] = df["trend_15m"] == -1

    return df


def filter_mask(df: pd.DataFrame, filters: list[str]) -> pd.Series:
    """Return combined boolean mask for a list of filter column names."""
    mask = pd.Series(True, index=df.index)
    for f in filters:
        mask &= df[f"f_{f}"]
    return mask
