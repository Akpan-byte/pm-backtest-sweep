"""Flexing Joe ORB futures backtest package."""
from __future__ import annotations

from .backtest import run_backtest
from .data import load_ohlcv_csv, resample_bars
from .execution import FuturesExecutionEngine
from .mc_bootstrap import attach_mc_and_bootstrap
from .metrics import summarize_metrics
from .models import Bar, DailyBias, Signal, StrategyConfig, Trade
from .prop_firm import attach_prop_firm_analysis
from .signals import compute_daily_bias, generate_all_signals, generate_signals_for_day

__all__ = [
    "Bar",
    "DailyBias",
    "Signal",
    "StrategyConfig",
    "Trade",
    "load_ohlcv_csv",
    "resample_bars",
    "compute_daily_bias",
    "generate_signals_for_day",
    "generate_all_signals",
    "FuturesExecutionEngine",
    "run_backtest",
    "summarize_metrics",
    "attach_mc_and_bootstrap",
    "attach_prop_firm_analysis",
]
