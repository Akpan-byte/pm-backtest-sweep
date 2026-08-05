#!/usr/bin/env python3
"""
Configuration for the v5 ORB NQ backtest.

Derived from the YM ORB v5 live bot config. NQ-specific values are overridden:
- TICK_VALUE: NQ is $20 per point (YM is $5 per point, one tick = one point).
- BASELINE_INDEX: NQ opening price at the start of the dataset so S ≈ 1.0.
- max_entries: configurable per variant; default is 2 per timeframe as requested.
"""

from __future__ import annotations

# NQ futures: $20 per index point, 0.25 tick size.
TICK_VALUE = 20.0
TICK_SIZE = 0.25
SYMBOL = "NQ"

# Set to the first open price of the 10-year NQ dataset so daily scaling starts
# near 1.0. This is a v5-ism: S = open_price / baseline_index, and sl_dist/buffer
# are scaled by S.
BASELINE_INDEX = 4524.25

# ORB parameters copied verbatim from YM v5 live bot.
TIMEFRAMES: dict[str, dict[str, float]] = {
    "1m":  {"or_min": 571, "trig": 0.05, "sint": 1.50, "lock": 0.90},
    "3m":  {"or_min": 573, "trig": 0.05, "sint": 0.50, "lock": 0.50},
    "5m":  {"or_min": 575, "trig": 0.05, "sint": 2.00, "lock": 0.90},
    "15m": {"or_min": 585, "trig": 0.05, "sint": 1.79, "lock": 0.90},
    "30m": {"or_min": 600, "trig": 0.25, "sint": 0.50, "lock": 0.50},
    "60m": {"or_min": 630, "trig": 0.05, "sint": 1.79, "lock": 0.75},
}

# Default strategy config for the NQ backtest.
DEFAULT_STRATEGY_CONFIG = {
    "mode": "paper",
    "initial_capital": 50_000.0,
    "risk_per_trade": 166.67,
    "max_drawdown": 2_000.0,
    "daily_loss_limit": 900.0,
    "buffer_pts": 20.0,
    "sl_pts": 20.0,
    "baseline_index": BASELINE_INDEX,
    "max_entries": 2,
    "max_contracts": 5,
}
