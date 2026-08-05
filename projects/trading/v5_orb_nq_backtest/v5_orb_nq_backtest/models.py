"""Data models for the pure YM ORB v5 strategy engine."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class TFParams:
    """Timeframe-specific ORB parameters."""

    or_min: int
    trig: float
    sint: float
    lock: float


@dataclass
class DayParams:
    """Daily scaling parameters (mutated once per session at 09:30 ET)."""

    S: float = 1.0
    sl_dist: float = 20.0
    buf: float = 20.0


@dataclass
class StrategyConfig:
    """Strategy configuration."""

    mode: str = "paper"
    initial_capital: float = 50000.0
    risk_per_trade: float = 166.67
    max_drawdown: float = 2000.0
    daily_loss_limit: float = 900.0
    buffer_pts: float = 20.0
    sl_pts: float = 20.0
    baseline_index: float = 29174.0
    max_entries: int = 4
    max_contracts: int = 5


@dataclass
class TFPosition:
    """A single open position for one timeframe."""

    direction: str
    entry_price: float
    entry_time: Optional[datetime]
    qty: int
    virtual_sl: float
    max_r: float = 0.0


@dataclass
class TradeRecord:
    """Closed-trade record produced by TFEngine.close()."""

    tf: str
    direction: str
    entry_price: float
    exit_price: float
    qty: int
    gross: float
    net: float
    exit_reason: str
    entry_time: str
    exit_time: str
    duration_mins: int
