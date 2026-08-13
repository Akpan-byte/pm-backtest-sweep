"""Signal generation and daily bias computation for Flexing Joe ORB."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import compute_ema, resample_bars
from .models import DailyBias, Signal, StrategyConfig


def _et_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a tz-aware ET DatetimeIndex from a frame with a timestamp column or index."""
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    else:
        ts = pd.to_datetime(df["timestamp"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York")


def _time_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Return minute-of-day (0-1439) for an ET DatetimeIndex."""
    return index.hour.values * 60 + index.minute.values


def _date_values(index: pd.DatetimeIndex) -> np.ndarray:
    """Return integer YYYYMMDD session-date values for grouping/filtering.

    Futures sessions run 18:00 ET previous day through 17:00 ET current day and
    are labelled by the RTH date (the calendar date that contains 09:30 ET).
    Bars at or after 18:00 ET belong to the next session.
    """
    norm = index.normalize()
    # Vectorized session rollover at 18:00 ET.
    offset = pd.to_timedelta((index.hour >= 18).astype(int), unit="D")
    session = norm + offset
    return session.year * 10000 + session.month * 100 + session.day


def _get_prior_day_16_close(prior_day_df: pd.DataFrame) -> Optional[float]:
    """Return the 16:00 ET close of the prior trading day, if available."""
    if prior_day_df is None or prior_day_df.empty:
        return None
    et = _et_index(prior_day_df)
    mins = _time_minutes(et)
    mask = mins == 16 * 60
    if not mask.any():
        return float(prior_day_df["close"].iloc[-1])
    return float(prior_day_df.loc[mask, "close"].iloc[-1])


def _compute_prior_day_stats(prior_day_df: pd.DataFrame) -> Dict[str, float]:
    """High, low, open, close for the prior trading day."""
    return {
        "high": float(prior_day_df["high"].max()),
        "low": float(prior_day_df["low"].min()),
        "open": float(prior_day_df["open"].iloc[0]),
        "close": float(prior_day_df["close"].iloc[-1]),
    }


def _london_orb_range(df_1m: pd.DataFrame) -> tuple[float, float]:
    """High/low of the 03:00-03:30 AM ET London ORB window."""
    et = _et_index(df_1m)
    mins = _time_minutes(et)
    mask = (mins >= 3 * 60) & (mins <= 3 * 60 + 30)
    if not mask.any():
        return np.nan, np.nan
    window = df_1m.loc[mask]
    return float(window["high"].max()), float(window["low"].min())


def _prior_day_candle_type(prior_day_df: pd.DataFrame) -> tuple[bool, bool]:
    """Classify prior day as inside day / doji vs trending."""
    if prior_day_df is None or len(prior_day_df) < 2:
        return False, False

    stats = _compute_prior_day_stats(prior_day_df)
    day_range = stats["high"] - stats["low"]
    body = abs(stats["close"] - stats["open"])

    is_doji = (day_range > 0) and (body / day_range < 0.2)

    is_inside = False
    try:
        df_30 = resample_bars(prior_day_df, 30)
        if len(df_30) >= 12:
            hl = df_30["high"] - df_30["low"]
            hc = (df_30["close"] - df_30["high"]).abs()
            lc = (df_30["close"] - df_30["low"]).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr = tr.rolling(10, min_periods=5).mean().iloc[-1]
            if atr > 0 and day_range < 1.5 * atr:
                is_inside = True
    except Exception:
        is_inside = False

    return is_inside, is_doji


def _value_at_time(df: Optional[pd.DataFrame], date_str: str, time_str: str) -> Optional[float]:
    """Return the close of ``df`` at ``time_str`` ET on ``date_str``."""
    if df is None or df.empty:
        return None
    et = _et_index(df)
    dates = _date_values(et)
    target_ts = pd.Timestamp(date_str, tz="America/New_York")
    target_date = target_ts.year * 10000 + target_ts.month * 100 + target_ts.day
    h, m = (int(x) for x in time_str.split(":"))
    target_min = h * 60 + m
    mins = _time_minutes(et)
    mask = (dates == target_date) & (mins == target_min)
    if not mask.any():
        return None
    return float(df.loc[mask, "close"].iloc[-1])


def _compute_vix_bias(
    vix_df: Optional[pd.DataFrame],
    date_str: str,
    prior_date_str: str,
) -> tuple[Optional[float], Optional[bool]]:
    """Return (vix_value_now, vix_rising)."""
    if vix_df is None or vix_df.empty:
        return None, None
    now_val = _value_at_time(vix_df, date_str, "09:30")
    prior_val = _value_at_time(vix_df, prior_date_str, "16:00")
    if now_val is None or prior_val is None or prior_val == 0:
        return now_val, None
    return now_val, now_val > prior_val


def _compute_es_nq_aligned(
    es_df: Optional[pd.DataFrame],
    nq_df: Optional[pd.DataFrame],
    date_str: str,
    prior_date_str: str,
) -> Optional[bool]:
    """True if ES and NQ both gap in the same direction."""
    def gap_pct(df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty:
            return None
        prior_close = _value_at_time(df, prior_date_str, "16:00")
        today_open = _value_at_time(df, date_str, "09:30")
        if prior_close is None or today_open is None or prior_close == 0:
            return None
        return (today_open - prior_close) / prior_close

    es_gap = gap_pct(es_df) if es_df is not None else None
    nq_gap = gap_pct(nq_df) if nq_df is not None else None
    if es_gap is None or nq_gap is None:
        return None
    return (es_gap > 0 and nq_gap > 0) or (es_gap < 0 and nq_gap < 0)


def compute_daily_bias(
    df_1m: pd.DataFrame,
    prior_day_df: pd.DataFrame,
    config: StrategyConfig,
    optional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    date_str: Optional[str] = None,
    prior_date_str: Optional[str] = None,
) -> DailyBias:
    """Compute pre-market directional bias for one trading day."""
    optional_dfs = optional_dfs or {}

    et = _et_index(df_1m)
    if date_str is None:
        date_str = et[0].strftime("%Y-%m-%d")
    today_open = float(df_1m["open"].iloc[0])

    prior_stats = _compute_prior_day_stats(prior_day_df)
    prior_close_16 = _get_prior_day_16_close(prior_day_df)
    if prior_close_16 is None:
        prior_close_16 = prior_stats["close"]

    gap_pct = (
        (today_open - prior_close_16) / prior_close_16 * 100.0
        if prior_close_16
        else 0.0
    )

    above_pdh = today_open > prior_stats["high"]
    below_pdl = today_open < prior_stats["low"]
    inside_pdh_pdl = (not above_pdh) and (not below_pdl)

    london_high, london_low = _london_orb_range(df_1m)
    above_london = (
        today_open > london_high if pd.notna(london_high) and pd.notna(london_low) else False
    )
    below_london = (
        today_open < london_low if pd.notna(london_high) and pd.notna(london_low) else False
    )

    prior_inside, prior_doji = _prior_day_candle_type(prior_day_df)

    if prior_date_str is None:
        prior_date_str = (_et_index(prior_day_df)[0]).strftime("%Y-%m-%d")
    vix_value, vix_rising = _compute_vix_bias(
        optional_dfs.get("vix"), date_str, prior_date_str
    )
    es_nq_aligned = _compute_es_nq_aligned(
        optional_dfs.get("es"), optional_dfs.get("nq"), date_str, prior_date_str
    )

    bias_score = 0
    if gap_pct > 0:
        bias_score += 1
    elif gap_pct < 0:
        bias_score -= 1

    if above_pdh:
        bias_score += 1
    elif below_pdl:
        bias_score -= 1

    if above_london:
        bias_score += 1
    elif below_london:
        bias_score -= 1

    if vix_rising is True:
        bias_score -= 1
    elif vix_rising is False:
        bias_score += 1

    if es_nq_aligned is True:
        if bias_score > 0:
            bias_score += 1
        elif bias_score < 0:
            bias_score -= 1

    allow_long = True
    allow_short = True

    if bias_score >= 2:
        allow_short = False
    elif bias_score <= -2:
        allow_long = False

    if config.min_gap_pct is not None and gap_pct < config.min_gap_pct:
        allow_long = allow_short = False
    if config.max_gap_pct is not None and gap_pct > config.max_gap_pct:
        allow_long = allow_short = False

    if config.require_vix and vix_value is None:
        allow_long = allow_short = False

    if config.require_es_nq_alignment and es_nq_aligned is not True:
        allow_long = allow_short = False

    return DailyBias(
        date=date_str,
        gap_pct=round(gap_pct, 4),
        above_pdh=above_pdh,
        below_pdl=below_pdl,
        inside_pdh_pdl=inside_pdh_pdl,
        london_orb_high=round(london_high, 4) if pd.notna(london_high) else None,
        london_orb_low=round(london_low, 4) if pd.notna(london_low) else None,
        above_london_orb=above_london,
        below_london_orb=below_london,
        prior_day_inside=prior_inside,
        prior_day_doji=prior_doji,
        vix_value=round(vix_value, 4) if vix_value is not None else None,
        vix_rising=vix_rising,
        es_nq_aligned=es_nq_aligned,
        bias_score=bias_score,
        allow_long=allow_long,
        allow_short=allow_short,
    )


def _extract_orb_range(df_30m_day: pd.DataFrame) -> tuple[float, float, float]:
    """Return (orb_high, orb_low, orb_range) from the 09:30 30-min bar."""
    et = _et_index(df_30m_day)
    mins = _time_minutes(et)
    mask = mins == 9 * 60 + 30
    if not mask.any():
        raise ValueError("No 09:30 30-min bar found for ORB range")
    orb_bar = df_30m_day.loc[mask].iloc[0]
    orb_high = float(orb_bar["high"])
    orb_low = float(orb_bar["low"])
    return orb_high, orb_low, orb_high - orb_low


def generate_signals_for_day(
    df_1m_day: pd.DataFrame,
    df_2m_day: pd.DataFrame,
    df_10m_day: pd.DataFrame,
    df_30m_day: pd.DataFrame,
    bias: DailyBias,
    config: StrategyConfig,
) -> List[Signal]:
    """Generate ORB signals for a single session following the PDF rules."""
    signals: List[Signal] = []

    try:
        orb_high, orb_low, orb_range = _extract_orb_range(df_30m_day)
    except ValueError:
        return signals

    if orb_range <= 0:
        return signals

    et_10m = _et_index(df_10m_day)
    mins_10m = _time_minutes(et_10m)
    close_10m = df_10m_day["close"].to_numpy(dtype=float)

    df2 = df_2m_day.copy()
    df2["ema20"] = compute_ema(df2["close"], config.ema_period)
    et_2m = _et_index(df2)
    mins_2m = _time_minutes(et_2m)

    # Convert 2m data to numpy arrays for fast loop access.
    m2_opens = df2["open"].to_numpy(dtype=float)
    m2_highs = df2["high"].to_numpy(dtype=float)
    m2_lows = df2["low"].to_numpy(dtype=float)
    m2_closes = df2["close"].to_numpy(dtype=float)
    m2_emas = df2["ema20"].to_numpy(dtype=float)
    m2_mins = mins_2m
    m2_timestamps = et_2m

    # Build 1m running HOD/LOD lookup by minute-of-day.
    et_1m = _et_index(df_1m_day)
    mins_1m = _time_minutes(et_1m)
    m1_highs_arr = df_1m_day["high"].to_numpy(dtype=float)
    m1_lows_arr = df_1m_day["low"].to_numpy(dtype=float)
    minute_highs = np.full(1440, -np.inf, dtype=float)
    minute_lows = np.full(1440, np.inf, dtype=float)
    for m, h, l in zip(mins_1m, m1_highs_arr, m1_lows_arr):
        if h > minute_highs[m]:
            minute_highs[m] = h
        if l < minute_lows[m]:
            minute_lows[m] = l
    # Forward-fill missing minutes so cumulative max/min works.
    valid_high = minute_highs != -np.inf
    valid_low = minute_lows != np.inf
    if valid_high.any():
        minute_highs = np.maximum.accumulate(np.where(valid_high, minute_highs, -np.inf))
    if valid_low.any():
        minute_lows = np.minimum.accumulate(np.where(valid_low, minute_lows, np.inf))

    end_minutes = (
        int(config.session_end_time.split(":")[0]) * 60
        + int(config.session_end_time.split(":")[1])
    )

    long_taken = False
    short_taken = False
    entry_count = 0

    for idx_10 in range(len(close_10m)):
        if mins_10m[idx_10] < 10 * 60:
            continue

        direction: Optional[int] = None
        c10 = close_10m[idx_10]
        if c10 > orb_high and bias.allow_long:
            direction = 1
        elif c10 < orb_low and bias.allow_short:
            direction = -1

        if direction is None:
            continue

        # Enforce one-trade-per-direction variant.
        if config.one_trade_per_direction:
            if direction == 1 and long_taken:
                continue
            if direction == -1 and short_taken:
                continue

        # The 10-min bar timestamp is the interval start; its close is only known
        # at interval end (ts_10 + 10 minutes).  Entries must occur after that.
        start_minutes = mins_10m[idx_10] + 10
        if start_minutes > end_minutes:
            continue

        # Find first 2m bar index >= start_minutes.
        win_start = int(np.searchsorted(m2_mins, start_minutes, side="left"))
        if win_start >= len(m2_mins) - 1:
            continue

        # Vectorized pullback search within the entry window.
        ema_win = m2_emas[win_start:]
        valid_ema = ~np.isnan(ema_win)
        if not valid_ema.any():
            continue

        if direction == 1:
            cond = valid_ema & (m2_lows[win_start:] <= ema_win) & (m2_closes[win_start:] > ema_win)
        else:
            cond = valid_ema & (m2_highs[win_start:] >= ema_win) & (m2_closes[win_start:] < ema_win)

        hits = np.flatnonzero(cond)
        if len(hits) == 0:
            continue

        i = int(hits[0])
        entry_idx = win_start + i + 1
        if entry_idx >= len(m2_mins):
            continue

        entry_minutes = int(m2_mins[entry_idx])
        if entry_minutes > end_minutes:
            continue

        entry_price = float(m2_opens[entry_idx])
        entry_ts = m2_timestamps[entry_idx]

        # Stop below/above the opposite side of the ORB range (the recent
        # structural extreme).  Target is a fixed multiple of the ORB range only;
        # do not use HOD/LOD reached before entry.
        if direction == 1:
            stop_price = orb_low
            target_price = entry_price + config.target_multiple * orb_range
        else:
            stop_price = orb_high
            target_price = entry_price - config.target_multiple * orb_range

        sig = Signal(
            timestamp=entry_ts,
            direction=direction,
            entry_price=round(entry_price, 4),
            stop_price=round(stop_price, 4),
            target_price=round(target_price, 4),
            contracts=config.contracts_per_trade,
            reason=f"ORB_{'LONG' if direction == 1 else 'SHORT'}",
        )
        signals.append(sig)
        entry_count += 1

        if direction == 1:
            long_taken = True
        else:
            short_taken = True

        if config.one_trade_per_day:
            return signals
        if config.one_trade_per_direction:
            if direction == 1:
                bias.allow_long = False
            else:
                bias.allow_short = False
        if entry_count >= config.max_entries_per_day:
            return signals

    return signals


def _compute_orb_range_history(df_30m: pd.DataFrame) -> pd.Series:
    """Return a Series mapping date_val -> ORB range for every session."""
    et = _et_index(df_30m)
    df = df_30m.copy()
    df["date_val"] = _date_values(et)
    ranges = {}
    for date_val in sorted(df["date_val"].unique()):
        day_30m = df[df["date_val"] == date_val]
        try:
            _, _, orb_range = _extract_orb_range(day_30m)
            ranges[date_val] = orb_range
        except ValueError:
            ranges[date_val] = np.nan
    return pd.Series(ranges).sort_index()


def generate_all_signals(
    df_1m: pd.DataFrame,
    config: StrategyConfig,
    optional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Signal]:
    """Loop over trading days, compute bias, generate signals, return flat list."""
    optional_dfs = optional_dfs or {}

    df = df_1m.copy()
    et = _et_index(df)
    df["date_val"] = _date_values(et)

    df_2m = resample_bars(df_1m, 2)
    df_10m = resample_bars(df_1m, 10)
    df_30m = resample_bars(df_1m, 30)

    df_2m["date_val"] = _date_values(_et_index(df_2m))
    df_10m["date_val"] = _date_values(_et_index(df_10m))
    df_30m["date_val"] = _date_values(_et_index(df_30m))

    dates = sorted(df["date_val"].unique())
    all_signals: List[Signal] = []

    # Pre-compute ORB ranges and rolling median for the volatility filter.
    # Use only prior sessions to avoid lookahead bias.
    orb_range_hist = _compute_orb_range_history(df_30m)
    orb_range_median = orb_range_hist.rolling(
        config.orb_range_lookback, min_periods=max(1, config.orb_range_lookback // 2)
    ).median()

    for i, date_val in enumerate(dates):
        day_1m = df[df["date_val"] == date_val].drop(columns="date_val")
        if len(day_1m) < config.orb_minutes:
            continue

        if i == 0:
            continue
        prior_date_val = dates[i - 1]
        prior_day_1m = df[df["date_val"] == prior_date_val].drop(columns="date_val")
        if prior_day_1m.empty:
            continue

        # Volatility filter: skip low-ORB-range sessions.
        if config.min_orb_range_multiple is not None:
            today_orb_range = orb_range_hist.get(date_val)
            median_orb_range = orb_range_median.get(prior_date_val)
            if (
                pd.notna(today_orb_range)
                and pd.notna(median_orb_range)
                and median_orb_range > 0
                and today_orb_range < config.min_orb_range_multiple * median_orb_range
            ):
                continue

        date_str = f"{date_val // 10000:04d}-{(date_val // 100) % 100:02d}-{date_val % 100:02d}"
        prior_date_str = f"{prior_date_val // 10000:04d}-{(prior_date_val // 100) % 100:02d}-{prior_date_val % 100:02d}"
        bias = compute_daily_bias(
            day_1m, prior_day_1m, config, optional_dfs,
            date_str=date_str, prior_date_str=prior_date_str,
        )

        day_2m = df_2m[df_2m["date_val"] == date_val].drop(columns="date_val")
        day_10m = df_10m[df_10m["date_val"] == date_val].drop(columns="date_val")
        day_30m = df_30m[df_30m["date_val"] == date_val].drop(columns="date_val")

        if day_2m.empty or day_10m.empty or day_30m.empty:
            continue

        day_signals = generate_signals_for_day(
            day_1m, day_2m, day_10m, day_30m, bias, config
        )
        all_signals.extend(day_signals)

    all_signals.sort(key=lambda s: s.timestamp)

    # Deduplicate signals that share the same entry timestamp, direction, and
    # price.  The per-day loop can emit repeats when multiple 10-minute bars
    # project onto the same 2-minute pullback entry.
    seen: set = set()
    unique_signals: List[Signal] = []
    for s in all_signals:
        key = (s.timestamp, s.direction, s.entry_price)
        if key in seen:
            continue
        seen.add(key)
        unique_signals.append(s)

    return unique_signals
