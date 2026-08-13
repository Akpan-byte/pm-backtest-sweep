"""Shared dataclasses for the Flexing Joe ORB backtest."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: int  # +1 Long, -1 Short
    entry_price: float
    stop_price: float
    target_price: float
    contracts: int = 1
    reason: str = ""


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    contracts: int
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    exit_reason: str


@dataclass
class StrategyConfig:
    """Runtime parameters for one backtest run."""
    symbol: str = "NQ"
    data_path: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    point_value: float = 20.0
    tick_size: float = 0.25
    commission_per_contract: float = 2.50
    slippage_points: float = 0.25
    initial_account_size: float = 50_000.0
    contracts_per_trade: int = 1

    # Risk limits
    daily_loss_limit: float = 900.0
    trailing_drawdown_limit: float = 2_000.0
    session_start_time: str = "09:30"
    session_end_time: str = "16:00"

    # ORB / entry parameters
    orb_minutes: int = 30
    confirm_timeframe_minutes: int = 10
    entry_timeframe_minutes: int = 2
    ema_period: int = 20
    target_multiple: float = 2.0  # target = entry +/- orb_range * multiple
    max_entries_per_day: int = 999  # for reentries variant
    one_trade_per_day: bool = False
    one_trade_per_direction: bool = False

    # Filters (all optional)
    require_vix: bool = False
    vix_path: Optional[str] = None
    require_es_nq_alignment: bool = False
    es_path: Optional[str] = None
    nq_path: Optional[str] = None
    macro_event_dates_path: Optional[str] = None
    min_gap_pct: Optional[float] = None
    max_gap_pct: Optional[float] = None

    # Volatility filter: skip sessions whose ORB range is too small relative to
    # the median ORB range of the previous N sessions.  Helps avoid low-vol
    # chop where the counter-strategy also loses money.
    orb_range_lookback: int = 20
    min_orb_range_multiple: Optional[float] = None

    # Validation / MC
    mc_runs: int = 50_000
    bootstrap_samples: int = 50_000
    prop_mc_runs: int = 20_000
    prop_bootstrap_samples: int = 20_000
    random_seed: int = 42


@dataclass
class DailyBias:
    date: str
    gap_pct: float
    above_pdh: bool
    below_pdl: bool
    inside_pdh_pdl: bool
    london_orb_high: float
    london_orb_low: float
    above_london_orb: bool
    below_london_orb: bool
    prior_day_inside: bool
    prior_day_doji: bool
    vix_value: Optional[float] = None
    vix_rising: Optional[bool] = None
    es_nq_aligned: Optional[bool] = None
    bias_score: int = 0  # positive = bullish, negative = bearish, 0 = neutral
    allow_long: bool = True
    allow_short: bool = True
