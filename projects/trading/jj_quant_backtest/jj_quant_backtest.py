#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-24  coder
#   - Created parallelized JJ Simon NQ Fair-Price quant suite backtest runner.
#   - Supports chunked execution: --chunk_id 0 --total_chunks 20
#   - Full quant suite: MC 20k, bootstrap 20k, Markov, Bayesian, 4 regressions,
#     PSR, DSR, walk-forward, drawdown analysis.
#   - Comprehensive error logging at every stage.
# WHY: Need to run 972 configs x 3 instruments across 20 GitHub Actions workers
#      and 16 laptop threads simultaneously.

"""
JJ Simon NQ Fair-Price Strategy — Full Quant Suite Backtest

Parallelizable runner that:
  1. Loads 1-min data for one instrument
  2. Runs all parameter configs for that instrument
  3. Computes full quant suite per config
  4. Saves results as JSON chunk

Usage:
  # Single instrument (for testing)
  python jj_quant_backtest.py --instrument NQ --chunk_id 0 --total_chunks 1

  # Full sweep (one instrument, all configs)
  python jj_quant_backtest.py --instrument NQ --chunk_id 0 --total_chunks 20

  # Aggregation (after all chunks complete)
  python aggregate_results.py --results_dir results/
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
LOGS_DIR = SCRIPT_DIR / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Data: local absolute path or relative to project root (CI)
LOCAL_DATA = Path("/config/fvg_execution_engine/backtests/data/gdrive_raw")
CI_DATA = SCRIPT_DIR / "fvg_execution_engine/backtests/data/gdrive_raw"
DATA_DIR = LOCAL_DATA if LOCAL_DATA.exists() else CI_DATA

# Quant suite import
QUANT_SUITE_PATH = Path("/config/.hermes/hermes-agent/skills/finance/quant_suite")
sys.path.insert(0, str(QUANT_SUITE_PATH))
import quant_suite as qs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("jj_quant_backtest")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# File handler (verbose)
fh = logging.FileHandler(LOG_FILE, mode="w")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(fh)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MC_RUNS = 20_000
BOOTSTRAP_RUNS = 20_000
MC_BATCH_SIZE = 2_000
STARTING_BALANCE = 100.0
RISK_PCT = 0.005
BOS_LOOKBACKS = [3, 5, 7]
MEAN_REVERSION_DISTANCES = [25.0, 38.0, 50.0]
NEWS_SPIKE_THRESHOLDS = [40.0, 60.0, 80.0]
DYNAMIC_CANDLE_TRIGGERS = [20.0, 25.0, 30.0]
PM_SESSION_OPTIONS = [False, True]
FEE_MODELS = [
    (0.0, "no_fees"),
    (0.00010, "hl_maker_1bps"),
    (0.00035, "hl_taker_3.5bps"),
]
PROFILES = {
    "50k": {"sl_pts": 25.0, "tp_pts": 38.0, "label": "50k"},
    "150k": {"sl_pts": 50.0, "tp_pts": 75.0, "label": "150k"},
}

INSTRUMENTS = {
    "NQ": {"csv": "NQ_1min.csv.gz", "point_value": 2.0, "tick_size": 0.25},
    "ES": {"csv": "ES_1min.csv.gz", "point_value": 1.0, "tick_size": 0.25},
    "YM": {"csv": "YM_1min.csv.gz", "point_value": 5.0, "tick_size": 1.0},
}

# Session timing (minute-of-day, US Eastern)
SCHEDULED_NEWS_START = 510   # 08:30
SCHEDULED_NEWS_END = 540     # 09:00
OPENING_CONT_START = 570     # 09:30
OPENING_CONT_END = 580       # 09:40
MEAN_REVERSION_START = 580   # 09:40
HARD_CUTOFF = 660            # 11:00
PM_START = 840               # 14:00
PM_REVERSION_START = 850     # 14:10
PM_END = 900                 # 15:00

# Anchor candle times (hour, minute)
ANCHOR_829 = (8, 29)
ANCHOR_929 = (9, 29)
ANCHOR_1359 = (13, 59)

MAX_MORNING_TRADES = 3
MAX_CONSECUTIVE_LOSSES = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_instrument_data(symbol: str) -> pd.DataFrame:
    """Load and cache 1-min CSV for an instrument."""
    info = INSTRUMENTS[symbol]
    csv_path = DATA_DIR / info["csv"]

    if not csv_path.exists():
        logger.error("Data file not found: %s", csv_path)
        raise FileNotFoundError(f"Missing data: {csv_path}")

    logger.info("Loading %s from %s (%.1f MB)", symbol, csv_path, csv_path.stat().st_size / 1e6)
    t0 = time.time()

    try:
        if csv_path.suffix == ".gz":
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        else:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    except Exception as e:
        logger.error("Failed to load CSV %s: %s", csv_path, e)
        raise

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Pre-compute minute-of-day and date (US Eastern)
    # Assume data is already in ET or UTC — we treat it as-is
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    # Pre-compute anchor flags
    df["is_829"] = (df["hour"] == ANCHOR_829[0]) & (df["minute"] == ANCHOR_829[1])
    df["is_929"] = (df["hour"] == ANCHOR_929[0]) & (df["minute"] == ANCHOR_929[1])
    df["is_1359"] = (df["hour"] == ANCHOR_1359[0]) & (df["minute"] == ANCHOR_1359[1])

    elapsed = time.time() - t0
    logger.info(
        "Loaded %s: %d bars, %s to %s (%.1fs)",
        symbol, len(df),
        df["timestamp"].min().date(),
        df["timestamp"].max().date(),
        elapsed,
    )
    return df


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
@dataclass
class JJConfig:
    """One parameter configuration for the JJ Simon strategy."""
    profile: str
    sl_pts: float
    tp_pts: float
    bos_lookback: int
    mean_reversion_distance: float
    news_spike_threshold: float
    dynamic_candle_trigger: float
    enable_pm_session: bool
    fee_rate: float
    fee_label: str
    point_value: float
    tick_size: float

    @property
    def config_id(self) -> str:
        return (
            f"{self.profile}_bos{self.bos_lookback}_"
            f"mrd{int(self.mean_reversion_distance)}_"
            f"nst{int(self.news_spike_threshold)}_"
            f"dct{int(self.dynamic_candle_trigger)}_"
            f"pm{int(self.enable_pm_session)}_"
            f"{self.fee_label}"
        )


def generate_all_configs(instrument: str) -> list[JJConfig]:
    """Generate all parameter combinations for an instrument."""
    info = INSTRUMENTS[instrument]
    configs = []

    for profile_name, profile_vals in PROFILES.items():
        for bos_lb in BOS_LOOKBACKS:
            for mrd in MEAN_REVERSION_DISTANCES:
                for nst in NEWS_SPIKE_THRESHOLDS:
                    for dct in DYNAMIC_CANDLE_TRIGGERS:
                        for pm in PM_SESSION_OPTIONS:
                            for fee_rate, fee_label in FEE_MODELS:
                                configs.append(JJConfig(
                                    profile=profile_name,
                                    sl_pts=profile_vals["sl_pts"],
                                    tp_pts=profile_vals["tp_pts"],
                                    bos_lookback=bos_lb,
                                    mean_reversion_distance=mrd,
                                    news_spike_threshold=nst,
                                    dynamic_candle_trigger=dct,
                                    enable_pm_session=pm,
                                    fee_rate=fee_rate,
                                    fee_label=fee_label,
                                    point_value=info["point_value"],
                                    tick_size=info["tick_size"],
                                ))

    logger.info("Generated %d configs for %s", len(configs), instrument)
    return configs


# ---------------------------------------------------------------------------
# JJ Simon Strategy Engine (bar-by-bar, vectorized data access)
# ---------------------------------------------------------------------------
def run_jj_simon_backtest(
    df: pd.DataFrame,
    config: JJConfig,
) -> tuple[list[dict], float, float]:
    """Run the JJ Simon strategy on 1-min bars.

    Returns:
        (trades_list, final_balance, total_fees)
    """
    n = len(df)
    if n < 100:
        logger.warning("Insufficient bars (%d) for backtest", n)
        return [], STARTING_BALANCE, 0.0

    # Extract arrays for speed
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    minutes = df["minute_of_day"].values.astype(np.int32)
    dates = df["date"].values
    is_829 = df["is_829"].values
    is_929 = df["is_929"].values
    is_1359 = df["is_1359"].values
    timestamps = df["timestamp"].values

    # State
    balance = STARTING_BALANCE
    total_fees = 0.0
    trades = []

    # Session state
    current_date = None
    news_anchor = None  # (body_high, body_low, midpoint)
    ny_anchor = None
    pm_anchor = None
    reset_anchor = None
    news_event_pending = False
    consolidation_bars = []
    spike_bar = None

    # Position state
    position_open = False
    position_direction = 0  # 1=LONG, -1=SHORT
    entry_price = 0.0
    entry_time = None
    current_sl = 0.0
    current_tp = 0.0
    open_qty = 0

    # Trade limits
    morning_trades = 0
    consecutive_losses = 0

    # Trigger detection buffer
    bos_buffer_highs = np.zeros(config.bos_lookback, dtype=np.float64)
    bos_buffer_lows = np.zeros(config.bos_lookback, dtype=np.float64)
    bos_buf_idx = 0
    bos_buf_count = 0
    prev_body = 0.0

    for i in range(n):
        ts = timestamps[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        mod = int(minutes[i])
        d = dates[i]

        # ---- Date change -> reset ----
        if d != current_date:
            current_date = d
            news_anchor = None
            ny_anchor = None
            pm_anchor = None
            reset_anchor = None
            news_event_pending = False
            consolidation_bars = []
            spike_bar = None
            morning_trades = 0
            consecutive_losses = 0
            position_open = False
            position_direction = 0
            entry_price = 0.0
            current_sl = 0.0
            current_tp = 0.0
            open_qty = 0
            bos_buf_count = 0
            bos_buf_idx = 0
            prev_body = 0.0

        # ---- Anchor detection ----
        body = abs(c - o)
        if is_829[i]:
            bh = max(o, c)
            bl = min(o, c)
            news_anchor = (bh, bl, (bh + bl) / 2.0)
        if is_929[i]:
            bh = max(o, c)
            bl = min(o, c)
            ny_anchor = (bh, bl, (bh + bl) / 2.0)
        if is_1359[i]:
            bh = max(o, c)
            bl = min(o, c)
            pm_anchor = (bh, bl, (bh + bl) / 2.0)

        # ---- Determine session phase ----
        if mod < SCHEDULED_NEWS_START:
            phase = "pre_market"
        elif SCHEDULED_NEWS_START <= mod < SCHEDULED_NEWS_END:
            phase = "news_phase"
        elif OPENING_CONT_START <= mod < OPENING_CONT_END:
            phase = "opening_continuation"
        elif MEAN_REVERSION_START <= mod < HARD_CUTOFF:
            phase = "mean_reversion"
        elif config.enable_pm_session and PM_START <= mod < PM_END:
            phase = "afternoon"
        elif mod >= HARD_CUTOFF:
            phase = "hard_cutoff"
        else:
            phase = "post_session"

        # ---- Select active anchor ----
        active_anchor = None
        if reset_anchor is not None:
            active_anchor = reset_anchor
        elif phase == "news_phase":
            active_anchor = news_anchor
        elif phase in ("opening_continuation", "mean_reversion"):
            active_anchor = ny_anchor
        elif phase == "afternoon":
            active_anchor = pm_anchor

        # ---- News detection (>threshold spike outside scheduled windows) ----
        candle_range = h - l
        if (
            not position_open
            and phase not in ("news_phase", "opening_continuation", "pre_market")
            and candle_range > config.news_spike_threshold
            and not news_event_pending
        ):
            news_event_pending = True
            spike_bar = (o, h, l, c)
            consolidation_bars = []
            # Invalidate previous anchor
            if active_anchor is not None:
                reset_anchor = None  # Will be rebuilt from consolidation

        # ---- Consolidation anchor after spike ----
        if news_event_pending and spike_bar is not None:
            consolidation_bars.append((o, h, l, c))
            if len(consolidation_bars) >= 3:
                avg_high = np.mean([b[1] for b in consolidation_bars[:3]])
                avg_low = np.mean([b[2] for b in consolidation_bars[:3]])
                reset_anchor = (avg_high, avg_low, (avg_high + avg_low) / 2.0)
                active_anchor = reset_anchor
                news_event_pending = False
                spike_bar = None
                consolidation_bars = []

        # ---- Exit logic (if position open) ----
        if position_open:
            exit_price = None
            exit_reason = None

            if position_direction == 1:  # LONG
                if l <= current_sl:
                    exit_price = current_sl
                    exit_reason = "stop_loss"
                elif current_tp > 0 and h >= current_tp:
                    exit_price = current_tp
                    exit_reason = "take_profit"
            else:  # SHORT
                if h >= current_sl:
                    exit_price = current_sl
                    exit_reason = "stop_loss"
                elif current_tp > 0 and l <= current_tp:
                    exit_price = current_tp
                    exit_reason = "take_profit"

            # Hard cutoff
            if exit_price is None and mod >= HARD_CUTOFF:
                exit_price = c
                exit_reason = "hard_cutoff"

            # Break-even at 09:30
            if (
                exit_price is None
                and mod == OPENING_CONT_START
                and entry_price > 0
            ):
                tick = config.tick_size
                if position_direction == 1:
                    new_sl = entry_price + tick
                    if new_sl > current_sl:
                        current_sl = new_sl
                else:
                    new_sl = entry_price - tick
                    if new_sl < current_sl:
                        current_sl = new_sl

            if exit_price is not None:
                # Close trade
                if position_direction == 1:
                    gross_pts = exit_price - entry_price
                else:
                    gross_pts = entry_price - exit_price

                gross = gross_pts * config.point_value * open_qty
                fee = abs(gross) * config.fee_rate
                net = gross - fee
                balance += net
                total_fees += fee

                trade = {
                    "entry_time": str(entry_time),
                    "exit_time": str(ts),
                    "direction": "long" if position_direction == 1 else "short",
                    "entry_price": round(float(entry_price), 4),
                    "exit_price": round(float(exit_price), 4),
                    "qty": round(float(open_qty), 4),
                    "gross": round(float(gross), 4),
                    "fee": round(float(fee), 4),
                    "net": round(float(net), 4),
                    "exit_reason": exit_reason,
                    "balance_after": round(float(balance), 4),
                }
                trades.append(trade)

                if net < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                position_open = False
                position_direction = 0
                entry_price = 0.0
                current_sl = 0.0
                current_tp = 0.0
                open_qty = 0

            # Don't enter on exit bar
            continue

        # ---- Skip if not in tradeable phase ----
        if phase not in ("news_phase", "opening_continuation", "mean_reversion", "afternoon"):
            prev_body = body
            # Update BOS buffer
            if bos_buf_count < config.bos_lookback:
                bos_buffer_highs[bos_buf_count] = h
                bos_buffer_lows[bos_buf_count] = l
                bos_buf_count += 1
            else:
                bos_buffer_highs[bos_buf_idx] = h
                bos_buffer_lows[bos_buf_idx] = l
                bos_buf_idx = (bos_buf_idx + 1) % config.bos_lookback
            continue

        # ---- Trade limits ----
        if morning_trades >= MAX_MORNING_TRADES:
            prev_body = body
            if bos_buf_count < config.bos_lookback:
                bos_buffer_highs[bos_buf_count] = h
                bos_buffer_lows[bos_buf_count] = l
                bos_buf_count += 1
            else:
                bos_buffer_highs[bos_buf_idx] = h
                bos_buffer_lows[bos_buf_idx] = l
                bos_buf_idx = (bos_buf_idx + 1) % config.bos_lookback
            continue
        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            prev_body = body
            if bos_buf_count < config.bos_lookback:
                bos_buffer_highs[bos_buf_count] = h
                bos_buffer_lows[bos_buf_count] = l
                bos_buf_count += 1
            else:
                bos_buffer_highs[bos_buf_idx] = h
                bos_buffer_lows[bos_buf_idx] = l
                bos_buf_idx = (bos_buf_idx + 1) % config.bos_lookback
            continue

        # ---- Trigger detection ----
        if active_anchor is None:
            prev_body = body
            if bos_buf_count < config.bos_lookback:
                bos_buffer_highs[bos_buf_count] = h
                bos_buffer_lows[bos_buf_count] = l
                bos_buf_count += 1
            else:
                bos_buffer_highs[bos_buf_idx] = h
                bos_buffer_lows[bos_buf_idx] = l
                bos_buf_idx = (bos_buf_idx + 1) % config.bos_lookback
            continue

        # Displacement detection
        is_displacement = False
        disp_bullish = False
        if prev_body > 0 and body > prev_body:
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            if body > 0 and upper_wick < body * 0.3 and lower_wick < body * 0.3:
                is_displacement = True
                disp_bullish = c > o

        # BOS detection
        is_bos = False
        bos_bullish = False
        if bos_buf_count >= config.bos_lookback:
            highest_high = np.max(bos_buffer_highs[:bos_buf_count])
            lowest_low = np.min(bos_buffer_lows[:bos_buf_count])
            if c > highest_high:
                is_bos = True
                bos_bullish = True
            elif c < lowest_low:
                is_bos = True
                bos_bullish = False

        # ---- Entry logic ----
        anchor_mid = active_anchor[2]
        distance = c - anchor_mid
        entry_signal = None
        entry_direction = 0

        if phase == "news_phase":
            if abs(distance) >= config.mean_reversion_distance:
                if distance > 0 and is_bos and not bos_bullish:
                    entry_signal = "mean_reversion"
                    entry_direction = -1
                elif distance < 0 and is_bos and bos_bullish:
                    entry_signal = "mean_reversion"
                    entry_direction = 1
                elif distance > 0 and is_displacement and not disp_bullish:
                    entry_signal = "mean_reversion"
                    entry_direction = -1
                elif distance < 0 and is_displacement and disp_bullish:
                    entry_signal = "mean_reversion"
                    entry_direction = 1

        elif phase == "opening_continuation":
            if is_displacement:
                entry_signal = "opening_continuation"
                entry_direction = 1 if disp_bullish else -1
            elif is_bos:
                entry_signal = "opening_continuation"
                entry_direction = 1 if bos_bullish else -1

        elif phase == "mean_reversion":
            if abs(distance) >= config.mean_reversion_distance:
                if distance > 0 and is_bos and not bos_bullish:
                    entry_signal = "mean_reversion"
                    entry_direction = -1
                elif distance < 0 and is_bos and bos_bullish:
                    entry_signal = "mean_reversion"
                    entry_direction = 1
                elif distance > 0 and is_displacement and not disp_bullish:
                    entry_signal = "mean_reversion"
                    entry_direction = -1
                elif distance < 0 and is_displacement and disp_bullish:
                    entry_signal = "mean_reversion"
                    entry_direction = 1

        elif phase == "afternoon":
            if mod < PM_REVERSION_START:
                # Continuation (first 10 min)
                if is_displacement:
                    entry_signal = "pm_continuation"
                    entry_direction = 1 if disp_bullish else -1
                elif is_bos:
                    entry_signal = "pm_continuation"
                    entry_direction = 1 if bos_bullish else -1
            else:
                # Reversion (after 10 min)
                if abs(distance) >= config.mean_reversion_distance:
                    if distance > 0 and (is_bos and not bos_bullish) or (is_displacement and not disp_bullish):
                        entry_signal = "pm_reversion"
                        entry_direction = -1
                    elif distance < 0 and (is_bos and bos_bullish) or (is_displacement and disp_bullish):
                        entry_signal = "pm_reversion"
                        entry_direction = 1

        # ---- Execute entry ----
        if entry_signal is not None and entry_direction != 0:
            # Dynamic candle adjustment
            sl_pts = config.sl_pts
            tp_pts = config.tp_pts
            if body > config.dynamic_candle_trigger:
                sl_pts = max(config.sl_pts, 50.0)
                tp_pts = max(config.tp_pts, 75.0)

            # Position sizing
            risk_dollars = balance * RISK_PCT
            risk_per_contract = sl_pts * config.point_value
            if risk_per_contract > 0:
                qty = max(1, round(risk_dollars / risk_per_contract))
            else:
                qty = 1

            if entry_direction == 1:
                entry_sl = c - sl_pts
                entry_tp = c + tp_pts
            else:
                entry_sl = c + sl_pts
                entry_tp = c - tp_pts

            position_open = True
            position_direction = entry_direction
            entry_price = c
            entry_time = ts
            current_sl = entry_sl
            current_tp = entry_tp
            open_qty = qty
            morning_trades += 1

        # ---- Update BOS buffer ----
        prev_body = body
        if bos_buf_count < config.bos_lookback:
            bos_buffer_highs[bos_buf_count] = h
            bos_buffer_lows[bos_buf_count] = l
            bos_buf_count += 1
        else:
            bos_buffer_highs[bos_buf_idx] = h
            bos_buffer_lows[bos_buf_idx] = l
            bos_buf_idx = (bos_buf_idx + 1) % config.bos_lookback

    # Force close any open position at end of data
    if position_open and entry_price > 0:
        exit_price = closes[-1]
        if position_direction == 1:
            gross_pts = exit_price - entry_price
        else:
            gross_pts = entry_price - exit_price
        gross = gross_pts * config.point_value * open_qty
        fee = abs(gross) * config.fee_rate
        net = gross - fee
        balance += net
        total_fees += fee
        trades.append({
            "entry_time": str(entry_time),
            "exit_time": str(timestamps[-1]),
            "direction": "long" if position_direction == 1 else "short",
            "entry_price": round(float(entry_price), 4),
            "exit_price": round(float(exit_price), 4),
            "qty": round(float(open_qty), 4),
            "gross": round(float(gross), 4),
            "fee": round(float(fee), 4),
            "net": round(float(net), 4),
            "exit_reason": "end_of_data",
            "balance_after": round(float(balance), 4),
        })

    return trades, balance, total_fees


# ---------------------------------------------------------------------------
# Quant suite computation (20k MC, 20k bootstrap)
# ---------------------------------------------------------------------------
def compute_quant_suite(
    trades: list[dict],
    balance: float,
    starting_balance: float,
    total_fees: float,
    years: float,
    seed: int,
) -> dict:
    """Full quant suite with 20k MC and 20k bootstrap."""
    if not trades:
        return {"error": "no trades", "trade_count": 0}

    np.random.seed(seed)

    pnl = np.array([t["net"] for t in trades], dtype=np.float64)
    risk_dollar = np.array(
        [max(t["balance_after"] - t["net"], 1e-9) for t in trades], dtype=np.float64
    ) * RISK_PCT
    pnl_r = pnl / np.maximum(risk_dollar, 1e-12)

    trades_df = pd.DataFrame({"pnl": pnl, "pnl_r": pnl_r})

    results = {}

    # PnL summary
    try:
        results["pnl_summary"] = qs.calculate_pnl_summary(
            trades_df, starting_balance=starting_balance, years=years
        )
    except Exception as e:
        logger.error("pnl_summary failed: %s", e)
        results["pnl_summary"] = {"error": str(e)}

    # Core metrics
    try:
        results["metrics"] = qs.calculate_metrics(
            trades_df, starting_balance=starting_balance
        )
    except Exception as e:
        logger.error("metrics failed: %s", e)
        results["metrics"] = {"error": str(e)}

    # PSR
    try:
        results["psr"] = qs.calculate_psr(pnl_r, benchmark_sr=0.0)
    except Exception as e:
        logger.error("psr failed: %s", e)
        results["psr"] = {"error": str(e)}

    # DSR
    try:
        trial_sharpes = []
        rng = np.random.default_rng(seed + 1)
        for _ in range(100):
            sample = rng.choice(pnl_r, size=len(pnl_r), replace=True)
            if len(sample) > 1 and np.std(sample) > 0:
                trial_sharpes.append(np.mean(sample) / np.std(sample))
        results["dsr"] = qs.calculate_dsr(pnl_r, trial_sharpes)
    except Exception as e:
        logger.error("dsr failed: %s", e)
        results["dsr"] = {"error": str(e)}

    # Bayesian win rate
    try:
        wins = int(np.sum(pnl > 0))
        losses = int(np.sum(pnl <= 0))
        results["bayesian_winrate"] = qs.calculate_bayesian_winrate(wins, losses)
    except Exception as e:
        logger.error("bayesian failed: %s", e)
        results["bayesian_winrate"] = {"error": str(e)}

    # Markov transitions
    try:
        results["markov"] = qs.calculate_markov_transitions(pnl_r)
    except Exception as e:
        logger.error("markov failed: %s", e)
        results["markov"] = {"error": str(e)}

    # Monte Carlo (20k runs)
    try:
        results["monte_carlo"] = _fast_mc(
            pnl_r, starting_balance, RISK_PCT, MC_RUNS, seed
        )
    except Exception as e:
        logger.error("monte_carlo failed: %s", e)
        results["monte_carlo"] = {"error": str(e)}

    # Monte Carlo ratchet (20k runs)
    try:
        results["monte_carlo_ratchet"] = _fast_mc_ratchet(
            pnl_r, starting_balance, RISK_PCT, MC_RUNS, seed + 3
        )
    except Exception as e:
        logger.error("mc_ratchet failed: %s", e)
        results["monte_carlo_ratchet"] = {"error": str(e)}

    # Regressions on equity curve
    try:
        equity_curve = starting_balance + np.cumsum(pnl)
        results["regressions"] = qs.run_regressions(equity_curve)
    except Exception as e:
        logger.error("regressions failed: %s", e)
        results["regressions"] = {"error": str(e)}

    # Walk-forward
    try:
        results["walk_forward"] = qs.run_walk_forward(trades_df, folds=5)
    except Exception as e:
        logger.error("walk_forward failed: %s", e)
        results["walk_forward"] = {"error": str(e)}

    # Ratchet PnL summary
    try:
        results["ratchet_summary"] = qs.calculate_ratchet_pnl_summary(
            trades_df, starting_balance=starting_balance,
            risk_pct=RISK_PCT, years=years,
        )
    except Exception as e:
        logger.error("ratchet_summary failed: %s", e)
        results["ratchet_summary"] = {"error": str(e)}

    # Bootstrap Sharpe CI (20k)
    try:
        results["bootstrap_sharpe_ci"] = _bootstrap_sharpe_ci(
            pnl_r, ci=0.95, n_boot=BOOTSTRAP_RUNS, seed=seed + 2
        )
    except Exception as e:
        logger.error("bootstrap_sharpe_ci failed: %s", e)
        results["bootstrap_sharpe_ci"] = {"error": str(e)}

    results["total_fees"] = round(total_fees, 6)
    results["trade_count"] = len(trades)

    return results


MC_MIN_BALANCE = 1.0  # Floor to prevent NaN cascade


def _fast_mc(pnl_r, starting_balance, risk_pct, runs, seed, batch_size=MC_BATCH_SIZE):
    n = len(pnl_r)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    terminal = np.zeros(runs)
    max_dd = np.zeros(runs)
    for bs in range(0, runs, batch_size):
        be = min(bs + batch_size, runs)
        b = be - bs
        sampled = rng.choice(pnl_r, size=(b, n), replace=True)
        bal = np.full(b, starting_balance, dtype=np.float64)
        peak = np.full(b, starting_balance, dtype=np.float64)
        bdd = np.zeros(b, dtype=np.float64)
        for j in range(n):
            bal = bal * (1.0 + risk_pct * sampled[:, j])
            bal = np.where(np.isfinite(bal), bal, MC_MIN_BALANCE)
            bal = np.maximum(bal, MC_MIN_BALANCE)
            peak = np.maximum(peak, bal)
            dd = (peak - bal) / np.maximum(peak, 1e-12)
            dd = np.where(np.isfinite(dd), dd, 0.0)
            bdd = np.maximum(bdd, dd)
        terminal[bs:be] = bal
        max_dd[bs:be] = bdd
    return {
        "runs": runs,
        "P10_balance": float(np.nanpercentile(terminal, 10)),
        "P50_balance": float(np.nanpercentile(terminal, 50)),
        "P90_balance": float(np.nanpercentile(terminal, 90)),
        "P50_max_dd": float(np.nanpercentile(max_dd, 50)),
        "P95_max_dd": float(np.nanpercentile(max_dd, 95)),
    }


def _fast_mc_ratchet(pnl_r, starting_balance, risk_pct, runs, seed, batch_size=MC_BATCH_SIZE):
    n = len(pnl_r)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    terminal = np.zeros(runs)
    max_dd = np.zeros(runs)
    for bs in range(0, runs, batch_size):
        be = min(bs + batch_size, runs)
        b = be - bs
        sampled = rng.choice(pnl_r, size=(b, n), replace=True)
        bal = np.full(b, starting_balance, dtype=np.float64)
        locked = np.full(b, starting_balance * risk_pct, dtype=np.float64)
        peak = np.full(b, starting_balance, dtype=np.float64)
        bdd = np.zeros(b, dtype=np.float64)
        for j in range(n):
            pnl_step = locked * sampled[:, j]
            bal += pnl_step
            bal = np.where(np.isfinite(bal), bal, MC_MIN_BALANCE)
            bal = np.maximum(bal, MC_MIN_BALANCE)
            peak = np.maximum(peak, bal)
            dd = (peak - bal) / np.maximum(peak, 1e-12)
            dd = np.where(np.isfinite(dd), dd, 0.0)
            bdd = np.maximum(bdd, dd)
            new_risk = bal * risk_pct
            win_mask = sampled[:, j] > 0
            locked = np.where(win_mask & (new_risk > locked), new_risk, locked)
        terminal[bs:be] = bal
        max_dd[bs:be] = bdd
    return {
        "runs": runs,
        "P10_balance": float(np.nanpercentile(terminal, 10)),
        "P50_balance": float(np.nanpercentile(terminal, 50)),
        "P90_balance": float(np.nanpercentile(terminal, 90)),
        "P50_max_dd": float(np.nanpercentile(max_dd, 50)),
        "P95_max_dd": float(np.nanpercentile(max_dd, 95)),
    }


def _bootstrap_sharpe_ci(pnl_r, ci=0.95, n_boot=20000, seed=42):
    if len(pnl_r) < 4:
        return {"lower": 0.0, "upper": 0.0, "median": 0.0, "mean": 0.0}
    rng = np.random.default_rng(seed)
    n = len(pnl_r)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = pnl_r[idx]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    sharpes = np.where(stds > 0, means / stds, 0.0)
    alpha = 1 - ci
    return {
        "lower": float(np.percentile(sharpes, alpha / 2 * 100)),
        "upper": float(np.percentile(sharpes, (1 - alpha / 2) * 100)),
        "median": float(np.median(sharpes)),
        "mean": float(np.mean(sharpes)),
    }


# ---------------------------------------------------------------------------
# Chunk worker
# ---------------------------------------------------------------------------
def run_chunk(
    instrument: str,
    chunk_id: int,
    total_chunks: int,
) -> dict:
    """Run a chunk of configs for one instrument."""
    logger.info(
        "=== CHUNK %d/%d for %s ===", chunk_id + 1, total_chunks, instrument
    )
    t_start = time.time()

    # Load data
    try:
        df = load_instrument_data(instrument)
    except Exception as e:
        logger.error("FATAL: Cannot load data for %s: %s", instrument, e)
        return {"error": str(e), "instrument": instrument, "chunk_id": chunk_id}

    # Generate configs
    all_configs = generate_all_configs(instrument)
    total = len(all_configs)

    # Select this chunk's configs
    chunk_configs = [c for i, c in enumerate(all_configs) if i % total_chunks == chunk_id]
    logger.info(
        "Chunk %d/%d: running %d of %d configs",
        chunk_id + 1, total_chunks, len(chunk_configs), total,
    )

    # Compute time span
    first_ts = df["timestamp"].iloc[0]
    last_ts = df["timestamp"].iloc[-1]
    years = (last_ts - first_ts).total_seconds() / (365.25 * 24 * 3600)

    results = []
    errors = []
    t_chunk = time.time()

    for idx, config in enumerate(chunk_configs):
        config_t0 = time.time()
        try:
            trades, balance, total_fees = run_jj_simon_backtest(df, config)

            quant = compute_quant_suite(
                trades, balance, STARTING_BALANCE, total_fees, years, seed=42 + idx
            )

            result = {
                "instrument": instrument,
                "config_id": config.config_id,
                "profile": config.profile,
                "sl_pts": config.sl_pts,
                "tp_pts": config.tp_pts,
                "bos_lookback": config.bos_lookback,
                "mean_reversion_distance": config.mean_reversion_distance,
                "news_spike_threshold": config.news_spike_threshold,
                "dynamic_candle_trigger": config.dynamic_candle_trigger,
                "enable_pm_session": config.enable_pm_session,
                "fee_rate": config.fee_rate,
                "fee_label": config.fee_label,
                "point_value": config.point_value,
                "trade_count": len(trades),
                "final_balance": round(balance, 4),
                "total_fees": round(total_fees, 4),
                "years": round(years, 2),
                "quant_suite": quant,
                "elapsed_sec": round(time.time() - config_t0, 2),
            }
            results.append(result)

            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(
                    "  [%s] Config %d/%d done: %d trades, bal=$%.2f (%.1fs)",
                    instrument, idx + 1, len(chunk_configs),
                    len(trades), balance, time.time() - config_t0,
                )

        except Exception as e:
            error_msg = f"Config {config.config_id} failed: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            errors.append({
                "config_id": config.config_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    elapsed = time.time() - t_chunk
    total_elapsed = time.time() - t_start

    summary = {
        "instrument": instrument,
        "chunk_id": chunk_id,
        "total_chunks": total_chunks,
        "configs_attempted": len(chunk_configs),
        "configs_completed": len(results),
        "configs_failed": len(errors),
        "errors": errors,
        "results": results,
        "elapsed_sec": round(total_elapsed, 2),
        "avg_sec_per_config": round(elapsed / max(len(chunk_configs), 1), 2),
    }

    # Save chunk
    out_path = RESULTS_DIR / f"{instrument}_chunk{chunk_id:03d}_of_{total_chunks}.json"
    try:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Saved results to %s", out_path)
    except Exception as e:
        logger.error("Failed to save results: %s", e)

    logger.info(
        "=== CHUNK %d/%d COMPLETE: %d/%d configs, %.1fs total ===",
        chunk_id + 1, total_chunks,
        len(results), len(chunk_configs),
        total_elapsed,
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="JJ Simon Quant Suite Backtest")
    parser.add_argument("--instrument", required=True, choices=["NQ", "ES", "YM"])
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--quick", action="store_true", help="Reduce MC/bootstrap to 2000 for faster testing")
    args = parser.parse_args()

    if args.quick:
        global MC_RUNS, BOOTSTRAP_RUNS
        MC_RUNS = 2_000
        BOOTSTRAP_RUNS = 2_000
        logger.info("QUICK MODE: MC=%s, Bootstrap=%s", f"{MC_RUNS:,}", f"{BOOTSTRAP_RUNS:,}")

    logger.info("=" * 60)
    logger.info("JJ SIMON QUANT SUITE BACKTEST")
    logger.info("Instrument: %s | Chunk: %d/%d", args.instrument, args.chunk_id + 1, args.total_chunks)
    logger.info("MC runs: %s | Bootstrap runs: %s", f"{MC_RUNS:,}", f"{BOOTSTRAP_RUNS:,}")
    logger.info("Log file: %s", LOG_FILE)
    logger.info("=" * 60)

    summary = run_chunk(args.instrument, args.chunk_id, args.total_chunks)

    if "error" in summary:
        logger.error("Chunk failed: %s", summary["error"])
        sys.exit(1)

    logger.info(
        "Done: %d/%d configs completed, %d errors",
        summary["configs_completed"],
        summary["configs_attempted"],
        summary["configs_failed"],
    )


if __name__ == "__main__":
    main()
