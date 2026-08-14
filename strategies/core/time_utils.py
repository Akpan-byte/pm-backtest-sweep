# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Created core/time_utils.py, the granular time-module for the StarTrading
#     intraday strategies. Holds all ET/UTC conversions and session windows so
#     every strategy imports one place for time logic.
# WHY: Compartmentalize the strategy suite so each concern lives in its own
#      navigable module (see docs/BLUEPRINTS.md).

"""Time helpers for the StarTrading intraday strategies.

All session windows use America/New_York (ET).  The daily rollover used by the
MOS session strategy is 00:00 UTC.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import logging

log = logging.getLogger("strategies.core.time_utils")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# NY equity session.
MARKET_OPEN_ET = (9, 30)
MARKET_CLOSE_ET = (16, 0)
# 8:30 ET liquidity-sweep gate (Blueprint 1).
NY_SWEEP_GATE_ET = (8, 30)
# 08:00 ET filter lock (Blueprint 4).
NY_FILTER_ET = (8, 0)
# 14:00 ET hard exit (Blueprint 1).
HARD_EXIT_ET = (14, 0)
# MOS session entry window length in seconds (Blueprint 3, 00:00 UTC).
MOS_ENTRY_SECONDS = 5


def get_et_now() -> datetime:
    return datetime.now(EST)


def get_utc_now() -> datetime:
    return datetime.now(UTC)


def _at(dt: datetime | None, tz: ZoneInfo) -> datetime:
    return (dt or datetime.now(tz)).astimezone(tz)


def seconds_since_market_open_et(dt: datetime | None = None) -> int:
    """Whole seconds since 09:30:00 ET today (negative before open)."""
    et = _at(dt, EST)
    open_dt = et.replace(hour=MARKET_OPEN_ET[0], minute=MARKET_OPEN_ET[1], second=0, microsecond=0)
    return int((et - open_dt).total_seconds())


def is_market_hours_et(dt: datetime | None = None) -> bool:
    et = _at(dt, EST)
    open_dt = et.replace(hour=MARKET_OPEN_ET[0], minute=MARKET_OPEN_ET[1], second=0, microsecond=0)
    close_dt = et.replace(hour=MARKET_CLOSE_ET[0], minute=MARKET_CLOSE_ET[1], second=0, microsecond=0)
    return open_dt <= et < close_dt


def ny_open_et() -> datetime:
    """8:30 AM ET cutoff used by Blueprint 1 liquidity-sweep filter."""
    now = get_et_now()
    return now.replace(hour=NY_SWEEP_GATE_ET[0], minute=NY_SWEEP_GATE_ET[1], second=0, microsecond=0)


def hard_session_exit_et() -> datetime:
    """2:00 PM ET hard exit for Blueprint 1."""
    now = get_et_now()
    return now.replace(hour=HARD_EXIT_ET[0], minute=HARD_EXIT_ET[1], second=0, microsecond=0)


def is_after_ny_open_filter_time(dt: datetime | None = None) -> bool:
    """Blueprint 4 lock: only act after 08:00 EST."""
    et = _at(dt, EST)
    filter_dt = et.replace(hour=NY_FILTER_ET[0], minute=NY_FILTER_ET[1], second=0, microsecond=0)
    return et >= filter_dt


def is_mos_session_time(dt: datetime | None = None) -> bool:
    """Blueprint 3 entry: within the first seconds of 00:00 UTC."""
    utc = _at(dt, UTC)
    return utc.hour == 0 and utc.minute == 0 and utc.second < MOS_ENTRY_SECONDS
