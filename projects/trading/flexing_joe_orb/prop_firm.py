"""Prop-firm payout modeling for Flexing Joe ORB backtests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PropFirmConfig:
    """Parameters for one prop-firm payout scenario."""

    account_start: float = 50_000.0
    profit_target: float = 8_000.0
    daily_loss_limit: float = 900.0
    max_loss_limit: float = 2_000.0  # EOD or trailing drawdown limit
    payout_cap: float = 4_000.0
    payout_rate: float = 0.4
    consistency_max_day_pct: float = 0.40
    min_winning_day_profit: float = 150.0  # 0 for consistency accounts
    winning_days_required: int = 5  # standard: 5 separate winning days
    consecutive_days_required: int = 3  # consistency: 3 consecutive days
    drawdown_mode: str = "eod"  # "eod" or "trailing"
    account_type: str = "standard"  # "standard" or "consistency"


# Scenario definitions.
_PROP_FIRM_SCENARIOS: Dict[str, PropFirmConfig] = {
    "50k_standard_eod": PropFirmConfig(
        account_start=50_000.0,
        profit_target=8_000.0,
        daily_loss_limit=900.0,
        max_loss_limit=2_000.0,
        payout_cap=4_000.0,
        payout_rate=0.4,
        consistency_max_day_pct=0.40,
        min_winning_day_profit=150.0,
        winning_days_required=5,
        consecutive_days_required=5,
        drawdown_mode="eod",
        account_type="standard",
    ),
    "50k_standard_trailing": PropFirmConfig(
        account_start=50_000.0,
        profit_target=8_000.0,
        daily_loss_limit=900.0,
        max_loss_limit=2_000.0,
        payout_cap=4_000.0,
        payout_rate=0.4,
        consistency_max_day_pct=0.40,
        min_winning_day_profit=150.0,
        winning_days_required=5,
        consecutive_days_required=5,
        drawdown_mode="trailing",
        account_type="standard",
    ),
    "50k_consistency_eod": PropFirmConfig(
        account_start=50_000.0,
        profit_target=8_000.0,
        daily_loss_limit=900.0,
        max_loss_limit=2_000.0,
        payout_cap=4_000.0,
        payout_rate=0.4,
        consistency_max_day_pct=0.40,
        min_winning_day_profit=0.0,
        winning_days_required=0,
        consecutive_days_required=3,
        drawdown_mode="eod",
        account_type="consistency",
    ),
    "50k_consistency_trailing": PropFirmConfig(
        account_start=50_000.0,
        profit_target=8_000.0,
        daily_loss_limit=900.0,
        max_loss_limit=2_000.0,
        payout_cap=4_000.0,
        payout_rate=0.4,
        consistency_max_day_pct=0.40,
        min_winning_day_profit=0.0,
        winning_days_required=0,
        consecutive_days_required=3,
        drawdown_mode="trailing",
        account_type="consistency",
    ),
    "150k_standard_eod": PropFirmConfig(
        account_start=150_000.0,
        profit_target=24_000.0,
        daily_loss_limit=3_500.0,
        max_loss_limit=4_500.0,
        payout_cap=12_000.0,
        payout_rate=0.4,
        consistency_max_day_pct=0.40,
        min_winning_day_profit=150.0,
        winning_days_required=5,
        consecutive_days_required=5,
        drawdown_mode="eod",
        account_type="standard",
    ),
    "150k_standard_trailing": PropFirmConfig(
        account_start=150_000.0,
        profit_target=24_000.0,
        daily_loss_limit=3_500.0,
        max_loss_limit=4_500.0,
        payout_cap=12_000.0,
        payout_rate=0.4,
        consistency_max_day_pct=0.40,
        min_winning_day_profit=150.0,
        winning_days_required=5,
        consecutive_days_required=5,
        drawdown_mode="trailing",
        account_type="standard",
    ),
    "150k_consistency_eod": PropFirmConfig(
        account_start=150_000.0,
        profit_target=24_000.0,
        daily_loss_limit=3_500.0,
        max_loss_limit=4_500.0,
        payout_cap=12_000.0,
        payout_rate=0.4,
        consistency_max_day_pct=0.40,
        min_winning_day_profit=0.0,
        winning_days_required=0,
        consecutive_days_required=3,
        drawdown_mode="eod",
        account_type="consistency",
    ),
    "150k_consistency_trailing": PropFirmConfig(
        account_start=150_000.0,
        profit_target=24_000.0,
        daily_loss_limit=3_500.0,
        max_loss_limit=4_500.0,
        payout_cap=12_000.0,
        payout_rate=0.4,
        consistency_max_day_pct=0.40,
        min_winning_day_profit=0.0,
        winning_days_required=0,
        consecutive_days_required=3,
        drawdown_mode="trailing",
        account_type="consistency",
    ),
}


def _frequency_stats(intervals: List[int], total_days: int) -> Dict[str, Any]:
    """Convert a list of day-intervals into readable frequency stats."""
    if not intervals:
        return {
            "avg_days_between": None,
            "median_days_between": None,
            "per_year": 0.0,
            "per_month": 0.0,
            "per_week": 0.0,
        }
    arr = np.array(intervals, dtype=float)
    years = max(total_days, 1) / 365.25
    return {
        "avg_days_between": float(np.mean(arr)),
        "median_days_between": float(np.median(arr)),
        "per_year": float(len(intervals) / years),
        "per_month": float(len(intervals) / years / 12.0),
        "per_week": float(len(intervals) / years / 52.0),
    }


def model_prop_firm_payouts(
    daily_pnl: Dict[str, float],
    config: PropFirmConfig,
) -> Dict[str, Any]:
    """Simulate prop-firm payout cycles with corrected accounting.

    Standard rule: at least 5 separate winning days (each profit >= $150) within
    the current cycle, total cycle profit >= profit_target, max day <= 40%.

    Consistency rule: 3 consecutive trading days, total >= profit_target,
    max day <= 40% of window profit.

    On payout: full window profit removed, account resets to start.
    On blow: account resets to start.
    """
    series = pd.Series(daily_pnl).sort_index()
    if series.empty:
        return _empty_result(config)

    values = series.values.astype(float)
    total_days = len(values)

    balance = config.account_start
    cum_profit = 0.0
    window_max = -np.inf
    winning_days = 0
    active_days = 0
    payouts: List[Dict[str, Any]] = []
    payout_intervals: List[int] = []
    reset_intervals: List[int] = []
    first_payout_days: Optional[int] = None
    resets = 0
    days_since_payout = 0
    days_since_reset = 0

    for i, pnl_raw in enumerate(values):
        days_since_payout += 1
        days_since_reset += 1
        pnl = max(float(pnl_raw), -config.daily_loss_limit)
        balance += pnl
        cum_profit += pnl
        active_days += 1
        window_max = max(window_max, pnl)

        if pnl >= config.min_winning_day_profit:
            winning_days += 1

        # Drawdown check.
        blew = False
        if config.drawdown_mode == "eod":
            if balance <= config.account_start - config.max_loss_limit:
                blew = True
        else:  # trailing
            if cum_profit <= -config.max_loss_limit:
                blew = True

        if blew:
            resets += 1
            reset_intervals.append(days_since_reset)
            days_since_reset = 0
            days_since_payout = 0
            balance = config.account_start
            cum_profit = 0.0
            active_days = 0
            window_max = -np.inf
            winning_days = 0
            continue

        # Payout check.
        eligible = False
        if config.account_type == "standard":
            if active_days >= config.consecutive_days_required and winning_days >= config.winning_days_required:
                eligible = True
        else:  # consistency
            if active_days >= config.consecutive_days_required:
                eligible = True

        if eligible and cum_profit >= config.profit_target and window_max / cum_profit <= config.consistency_max_day_pct:
            payout = min(cum_profit * config.payout_rate, config.payout_cap)
            payouts.append(
                {
                    "day_index": i + 1,
                    "active_days": active_days,
                    "winning_days": winning_days,
                    "window_profit": float(cum_profit),
                    "payout": float(payout),
                }
            )
            if first_payout_days is None:
                first_payout_days = i + 1
            payout_intervals.append(days_since_payout)
            days_since_payout = 0
            days_since_reset = 0
            balance = config.account_start
            cum_profit = 0.0
            active_days = 0
            window_max = -np.inf
            winning_days = 0

    total_payout_dollars = sum(p["payout"] for p in payouts)
    payout_freq = _frequency_stats(payout_intervals, total_days)
    reset_freq = _frequency_stats(reset_intervals, total_days)

    return {
        "account_start": config.account_start,
        "profit_target": config.profit_target,
        "daily_loss_limit": config.daily_loss_limit,
        "max_loss_limit": config.max_loss_limit,
        "payout_cap": config.payout_cap,
        "payout_rate": config.payout_rate,
        "consistency_max_day_pct": config.consistency_max_day_pct,
        "min_winning_day_profit": config.min_winning_day_profit,
        "winning_days_required": config.winning_days_required,
        "consecutive_days_required": config.consecutive_days_required,
        "drawdown_mode": config.drawdown_mode,
        "account_type": config.account_type,
        "total_payouts": len(payouts),
        "total_payout_dollars": float(total_payout_dollars),
        "avg_payout": float(total_payout_dollars / len(payouts)) if payouts else 0.0,
        "first_payout_days": first_payout_days,
        "payout_list": payouts,
        "final_balance": float(balance),
        "resets": resets,
        **{f"payout_{k}": v for k, v in payout_freq.items()},
        **{f"reset_{k}": v for k, v in reset_freq.items()},
    }


def _empty_result(config: PropFirmConfig) -> Dict[str, Any]:
    return {
        "account_start": config.account_start,
        "profit_target": config.profit_target,
        "daily_loss_limit": config.daily_loss_limit,
        "max_loss_limit": config.max_loss_limit,
        "payout_cap": config.payout_cap,
        "payout_rate": config.payout_rate,
        "consistency_max_day_pct": config.consistency_max_day_pct,
        "min_winning_day_profit": config.min_winning_day_profit,
        "winning_days_required": config.winning_days_required,
        "consecutive_days_required": config.consecutive_days_required,
        "drawdown_mode": config.drawdown_mode,
        "account_type": config.account_type,
        "total_payouts": 0,
        "total_payout_dollars": 0.0,
        "avg_payout": 0.0,
        "first_payout_days": None,
        "payout_list": [],
        "final_balance": config.account_start,
        "resets": 0,
        "payout_avg_days_between": None,
        "payout_median_days_between": None,
        "payout_per_year": 0.0,
        "payout_per_month": 0.0,
        "payout_per_week": 0.0,
        "reset_avg_days_between": None,
        "reset_median_days_between": None,
        "reset_per_year": 0.0,
        "reset_per_month": 0.0,
        "reset_per_week": 0.0,
    }


def run_prop_firm_analysis(daily_pnl: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Run all prop-firm scenarios."""
    return {
        name: model_prop_firm_payouts(daily_pnl, cfg)
        for name, cfg in _PROP_FIRM_SCENARIOS.items()
    }


def _vectorized_prop_paths(
    samples: np.ndarray,
    cfg: PropFirmConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized prop-firm simulation.

    Returns total_payouts, total_payout_dollars, first_payout_day,
    final_balance, resets, avg_days_between_payouts.
    """
    num_runs, n = samples.shape
    balance = np.full(num_runs, cfg.account_start, dtype=float)
    cum_profit = np.zeros(num_runs, dtype=float)
    window_max = np.full(num_runs, -np.inf, dtype=float)
    winning_days = np.zeros(num_runs, dtype=np.int64)
    active_days = np.zeros(num_runs, dtype=np.int64)
    total_payouts = np.zeros(num_runs, dtype=np.int64)
    total_payout_dollars = np.zeros(num_runs, dtype=float)
    first_payout_day = np.full(num_runs, np.nan, dtype=float)
    resets = np.zeros(num_runs, dtype=np.int64)
    days_since_payout = np.zeros(num_runs, dtype=np.int64)
    sum_days_between = np.zeros(num_runs, dtype=float)

    for d in range(n):
        days_since_payout += 1
        pnl = np.maximum(samples[:, d], -cfg.daily_loss_limit)
        balance += pnl
        cum_profit += pnl
        active_days += 1
        window_max = np.maximum(window_max, pnl)
        winning_days += (pnl >= cfg.min_winning_day_profit).astype(np.int64)

        # Drawdown check.
        if cfg.drawdown_mode == "eod":
            reset_mask = balance <= cfg.account_start - cfg.max_loss_limit
        else:
            reset_mask = cum_profit <= -cfg.max_loss_limit

        resets[reset_mask] += 1
        balance[reset_mask] = cfg.account_start
        cum_profit[reset_mask] = 0.0
        active_days[reset_mask] = 0
        window_max[reset_mask] = -np.inf
        winning_days[reset_mask] = 0
        days_since_payout[reset_mask] = 0

        # Payout check.
        if cfg.account_type == "standard":
            enough = (active_days >= cfg.consecutive_days_required) & (winning_days >= cfg.winning_days_required)
        else:
            enough = active_days >= cfg.consecutive_days_required

        hit_target = cum_profit >= cfg.profit_target
        valid_window = window_max > -np.inf
        consistency_ok = np.zeros(num_runs, dtype=bool)
        consistency_ok[valid_window] = window_max[valid_window] / np.maximum(cum_profit[valid_window], 1e-9) <= cfg.consistency_max_day_pct

        payout_mask = enough & hit_target & consistency_ok
        if not payout_mask.any():
            continue

        payout_amount = np.minimum(cum_profit * cfg.payout_rate, cfg.payout_cap)
        total_payouts[payout_mask] += 1
        total_payout_dollars[payout_mask] += payout_amount[payout_mask]
        sum_days_between[payout_mask] += days_since_payout[payout_mask]
        newly_paid = payout_mask & np.isnan(first_payout_day)
        first_payout_day[newly_paid] = d + 1
        balance[payout_mask] = cfg.account_start
        cum_profit[payout_mask] = 0.0
        active_days[payout_mask] = 0
        window_max[payout_mask] = -np.inf
        winning_days[payout_mask] = 0
        days_since_payout[payout_mask] = 0

    final_balance = balance + cum_profit
    avg_days_between = np.where(
        total_payouts > 1,
        sum_days_between / (total_payouts - 1),
        np.nan,
    )
    return total_payouts, total_payout_dollars, first_payout_day, final_balance, resets, avg_days_between


def prop_firm_monte_carlo(
    daily_pnl: Dict[str, float],
    num_runs: int = 20_000,
    random_seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Resample daily PnL series and collect prop-firm outcome distributions."""
    if num_runs <= 0:
        return {}
    rng = np.random.default_rng(random_seed)
    series = pd.Series(daily_pnl).sort_index()
    if series.empty:
        return {}

    values = series.values.astype(float)
    n = len(values)
    years = n / 365.25

    results: Dict[str, Dict[str, Any]] = {}
    for name, base_cfg in _PROP_FIRM_SCENARIOS.items():
        samples = rng.choice(values, size=(num_runs, n), replace=True)
        counts, dollars, first_days, balances, resets, avg_interval = _vectorized_prop_paths(samples, base_cfg)

        results[name] = {
            "total_payouts_mean": float(np.mean(counts)),
            "total_payouts_p5": float(np.percentile(counts, 5)),
            "total_payouts_p50": float(np.percentile(counts, 50)),
            "total_payouts_p95": float(np.percentile(counts, 95)),
            "total_payout_dollars_mean": float(np.mean(dollars)),
            "total_payout_dollars_p5": float(np.percentile(dollars, 5)),
            "total_payout_dollars_p50": float(np.percentile(dollars, 50)),
            "total_payout_dollars_p95": float(np.percentile(dollars, 95)),
            "avg_days_between_payouts_mean": (
                float(np.nanmean(avg_interval)) if not np.all(np.isnan(avg_interval)) else None
            ),
            "avg_days_between_payouts_p50": (
                float(np.nanpercentile(avg_interval, 50)) if not np.all(np.isnan(avg_interval)) else None
            ),
            "payouts_per_year_mean": float(np.mean(counts) / years),
            "first_payout_days_mean": (
                float(np.nanmean(first_days)) if not np.all(np.isnan(first_days)) else None
            ),
            "first_payout_days_p50": (
                float(np.nanpercentile(first_days, 50)) if not np.all(np.isnan(first_days)) else None
            ),
            "resets_mean": float(np.mean(resets)),
            "resets_per_year_mean": float(np.mean(resets) / years),
            "final_balance_mean": float(np.mean(balances)),
            "final_balance_p5": float(np.percentile(balances, 5)),
            "final_balance_p95": float(np.percentile(balances, 95)),
        }

    return results


def prop_firm_bootstrap_ci(
    daily_pnl: Dict[str, float],
    num_samples: int = 20_000,
    random_seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Bootstrapped 95% CI on prop-firm outputs per scenario."""
    if num_samples <= 0:
        return {}
    rng = np.random.default_rng(random_seed)
    series = pd.Series(daily_pnl).sort_index()
    if series.empty:
        return {}

    values = series.values.astype(float)
    n = len(values)
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100
    years = n / 365.25

    results: Dict[str, Dict[str, Any]] = {}
    for name, base_cfg in _PROP_FIRM_SCENARIOS.items():
        samples = rng.choice(values, size=(num_samples, n), replace=True)
        counts, dollars, first_days, balances, resets, avg_interval = _vectorized_prop_paths(samples, base_cfg)

        results[name] = {
            "total_payouts_ci_lower": float(np.percentile(counts, lower_pct)),
            "total_payouts_ci_upper": float(np.percentile(counts, upper_pct)),
            "total_payouts_mean": float(np.mean(counts)),
            "total_payout_dollars_ci_lower": float(np.percentile(dollars, lower_pct)),
            "total_payout_dollars_ci_upper": float(np.percentile(dollars, upper_pct)),
            "total_payout_dollars_mean": float(np.mean(dollars)),
            "avg_days_between_payouts_ci_lower": (
                float(np.nanpercentile(avg_interval, lower_pct))
                if not np.all(np.isnan(avg_interval))
                else None
            ),
            "avg_days_between_payouts_ci_upper": (
                float(np.nanpercentile(avg_interval, upper_pct))
                if not np.all(np.isnan(avg_interval))
                else None
            ),
            "payouts_per_year_ci_lower": float(np.percentile(counts, lower_pct) / years),
            "payouts_per_year_ci_upper": float(np.percentile(counts, upper_pct) / years),
            "first_payout_days_ci_lower": (
                float(np.nanpercentile(first_days, lower_pct))
                if not np.all(np.isnan(first_days))
                else None
            ),
            "first_payout_days_ci_upper": (
                float(np.nanpercentile(first_days, upper_pct))
                if not np.all(np.isnan(first_days))
                else None
            ),
            "first_payout_days_mean": (
                float(np.nanmean(first_days)) if not np.all(np.isnan(first_days)) else None
            ),
            "resets_ci_lower": float(np.percentile(resets, lower_pct)),
            "resets_ci_upper": float(np.percentile(resets, upper_pct)),
            "resets_mean": float(np.mean(resets)),
        }

    return results


def attach_prop_firm_analysis(
    result: Dict[str, Any],
    prop_mc_runs: int = 20_000,
    prop_bootstrap_samples: int = 20_000,
) -> Dict[str, Any]:
    """Add prop-firm payout tables, Monte Carlo, and bootstrap CI to ``result``."""
    daily_pnl = result.get("daily_pnl", {})
    result["prop_firm_payouts"] = run_prop_firm_analysis(daily_pnl)
    result["prop_monte_carlo_20k"] = prop_firm_monte_carlo(
        daily_pnl, num_runs=prop_mc_runs
    )
    result["prop_bootstrap_20k"] = prop_firm_bootstrap_ci(
        daily_pnl, num_samples=prop_bootstrap_samples
    )
    return result
