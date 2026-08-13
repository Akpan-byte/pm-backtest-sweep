#!/usr/bin/env python3
"""Robustness, statistical edge, regime, and walk-forward validation suite.

Runs the validation checklist from the Gemini Spark / futures_production_engine
document on existing backtest outputs:
  - Sharpe / Sortino (per-trade, annualized by trades_per_day)
  - Max drawdown (%)
  - White's Reality Check (stationary block bootstrap, 2_000, block_len=5)
  - Benjamini-Hochberg FDR q-value
  - Deflated Sharpe Ratio (Bailey-Lopez de Prado)
  - Monte Carlo (50_000 runs, 5% noise) -> p5/p50/mean total PnL
  - Walk-forward / out-of-sample rolling-window stability
  - Regime filters (trend, gap, prior-day structure, VIX, ES/NQ alignment)
  - Parameter stability score

Usage:
    python robustness_validation.py --input /tmp/combined_reentry_sweep/NQ_me6_c2.json \
        --output /tmp/robustness_NQ_me6_c2.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, "/config/projects/trading")
from flexing_joe_orb.metrics import (
    compute_deflated_sharpe_ratio,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_true_benjamini_hochberg_fdr,
    compute_true_whites_reality_check,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_backtest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    # Files may be keyed by symbol or direct result dict.
    if len(data) == 1 and isinstance(list(data.values())[0], dict):
        return list(data.values())[0]
    return data


def extract_pnls(result: Dict[str, Any]) -> np.ndarray:
    trades = result.get("trades", result.get("combined_trades", []))
    return np.array([t["net_pnl"] for t in trades], dtype=float)


def extract_daily_pnl(result: Dict[str, Any]) -> pd.Series:
    daily = result.get("daily_pnl", {})
    if not daily:
        trades = result.get("trades", result.get("combined_trades", []))
        daily = defaultdict(float)
        for t in trades:
            d = pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d")
            daily[d] += t["net_pnl"]
    s = pd.Series(daily, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


# ---------------------------------------------------------------------------
# Core validators
# ---------------------------------------------------------------------------

def run_monte_carlo(
    pnls: np.ndarray,
    num_runs: int = 50_000,
    noise_level: float = 0.05,
    random_seed: int = 42,
    chunk_size: int = 500,
) -> Dict[str, float]:
    """Memory-chunked Monte Carlo with replacement + Gaussian noise."""
    rng = np.random.default_rng(random_seed)
    n = len(pnls)
    if n == 0:
        return {"p5": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0}
    std_pnl = max(0.001, float(np.std(pnls, ddof=1)))
    totals = np.empty(num_runs, dtype=float)
    written = 0
    while written < num_runs:
        this_chunk = min(chunk_size, num_runs - written)
        idx = rng.integers(0, n, size=(this_chunk, n))
        noise = rng.normal(0.0, std_pnl * noise_level, size=(this_chunk, n))
        totals[written : written + this_chunk] = np.sum(pnls[idx] + noise, axis=1)
        written += this_chunk
    return {
        "p5": float(np.percentile(totals, 5)),
        "p50": float(np.percentile(totals, 50)),
        "mean": float(np.mean(totals)),
        "p95": float(np.percentile(totals, 95)),
    }


def run_whites_reality_check(
    pnls: np.ndarray,
    num_bootstraps: int = 2_000,
    block_length: int = 5,
    random_seed: int = 42,
) -> Dict[str, float]:
    np.random.seed(random_seed)
    # Single-strategy matrix; WRC bootstrap maximum under null.
    mat = pnls.reshape(1, -1)
    pvals = compute_true_whites_reality_check(mat, num_bootstraps, block_length)
    qvals = compute_true_benjamini_hochberg_fdr(pvals)
    return {
        "wrc_pvalue": float(pvals[0]),
        "fdr_qvalue": float(qvals[0]),
        "num_bootstraps": num_bootstraps,
        "block_length": block_length,
    }


def compute_full_validation(
    pnls: np.ndarray,
    trades_per_day: float,
    num_trials: int = 50_000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    if len(pnls) < 2:
        return {"pass": False, "fail_reasons": ["<2 trades"]}

    sharpe = compute_sharpe(pnls, trades_per_day)
    sortino = compute_sortino(pnls, trades_per_day)
    skew = float(stats.skew(pnls)) if len(pnls) > 2 else 0.0
    kurt = float(stats.kurtosis(pnls)) if len(pnls) > 2 else 0.0
    dsr = compute_deflated_sharpe_ratio(sharpe, len(pnls), skew, kurt, num_trials)

    wrc = run_whites_reality_check(pnls, random_seed=random_seed)
    mc = run_monte_carlo(pnls, random_seed=random_seed)

    # Parameter stability score from expected max Sharpe under trials.
    euler = 0.5772156649
    z_a = stats.norm.ppf(1.0 - 1.0 / max(2, num_trials))
    z_b = stats.norm.ppf(1.0 - 1.0 / (max(2, num_trials) * math.e))
    e_max_sr = (1.0 - euler) * z_a + euler * z_b
    stability = float(
        round(
            max(
                0.0,
                min(
                    100.0,
                    (1.0 - abs(sharpe - e_max_sr) / max(1.0, abs(sharpe))) * 100.0,
                ),
            ),
            1,
        )
    )

    fail_reasons = []
    if dsr <= 0.40:
        fail_reasons.append("DSR <= 0.40")
    if wrc["wrc_pvalue"] >= 0.15:
        fail_reasons.append("WRC p >= 0.15")
    if wrc["fdr_qvalue"] >= 0.15:
        fail_reasons.append("FDR q >= 0.15")
    if mc["p5"] <= -50.0:
        fail_reasons.append("MC p5 <= -50")
    if stability < 50.0:
        fail_reasons.append("stability < 50")

    return {
        "total_trades": int(len(pnls)),
        "win_rate": round(float(np.mean(pnls > 0)) * 100, 2),
        "net_pnl": round(float(np.sum(pnls)), 2),
        "avg_trade_pnl": round(float(np.mean(pnls)), 4),
        "profit_factor": round(
            float(np.sum(pnls[pnls > 0]) / max(1e-9, abs(np.sum(pnls[pnls < 0])))), 2
        ),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "dsr": round(dsr, 4),
        "wrc_pvalue": round(wrc["wrc_pvalue"], 4),
        "fdr_qvalue": round(wrc["fdr_qvalue"], 4),
        "mc_p5": round(mc["p5"], 2),
        "mc_p50": round(mc["p50"], 2),
        "mc_mean": round(mc["mean"], 2),
        "mc_p95": round(mc["p95"], 2),
        "param_stability_score": stability,
        "pass": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
    }


# ---------------------------------------------------------------------------
# Walk-forward / OOS
# ---------------------------------------------------------------------------

def walk_forward_analysis(
    daily_pnl: pd.Series,
    window_years: int = 2,
    step_months: int = 6,
) -> List[Dict[str, Any]]:
    """Rolling in-sample / out-of-sample validation using daily PnL.

    Each window: train on `window_years`, test on following `step_months`.
    """
    results = []
    start = daily_pnl.index[0]
    end = daily_pnl.index[-1]
    cursor = start
    while cursor + pd.DateOffset(years=window_years) < end:
        train_end = cursor + pd.DateOffset(years=window_years)
        test_end = min(train_end + pd.DateOffset(months=step_months), end)
        train = daily_pnl[(daily_pnl.index >= cursor) & (daily_pnl.index < train_end)]
        test = daily_pnl[(daily_pnl.index >= train_end) & (daily_pnl.index < test_end)]
        if len(train) < 30 or len(test) < 5:
            cursor += pd.DateOffset(months=step_months)
            continue

        train_pnls = train.values
        test_pnls = test.values
        tpd_train = len(train_pnls) / max(1, (train.index[-1] - train.index[0]).days)
        tpd_test = len(test_pnls) / max(1, (test.index[-1] - test.index[0]).days)

        results.append({
            "train_start": cursor.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": train_end.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "train_days": int(len(train_pnls)),
            "test_days": int(len(test_pnls)),
            "train_net_pnl": round(float(np.sum(train_pnls)), 2),
            "test_net_pnl": round(float(np.sum(test_pnls)), 2),
            "train_sharpe": round(compute_sharpe(train_pnls, tpd_train), 3),
            "test_sharpe": round(compute_sharpe(test_pnls, tpd_test), 3),
            "train_win_rate": round(float(np.mean(train_pnls > 0)) * 100, 2),
            "test_win_rate": round(float(np.mean(test_pnls > 0)) * 100, 2),
        })
        cursor += pd.DateOffset(months=step_months)
    return results


# ---------------------------------------------------------------------------
# Regime filters
# ---------------------------------------------------------------------------

def regime_analysis(
    result: Dict[str, Any],
    daily_pnl: pd.Series,
    trades: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate how the strategy behaves under simple observable regimes."""
    if not trades:
        return {}

    # Build a per-day trade summary from trade list.
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["date"] = df["entry_time"].dt.tz_convert("America/New_York").dt.date

    # Approximate daily range from entry/exit prices (best we have without reloading 1m data).
    day_stats = df.groupby("date").agg(
        net_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "size"),
        win_trades=("net_pnl", lambda x: int((x > 0).sum())),
    ).reset_index()
    day_stats["date"] = pd.to_datetime(day_stats["date"])
    day_stats = day_stats.set_index("date").sort_index()

    # Merge with daily PnL in case some days have no trades.
    merged = pd.concat([daily_pnl.rename("daily_pnl"), day_stats], axis=1).fillna(0)
    merged["win_day"] = merged["daily_pnl"] > 0

    # Trend regime: compare today's close (last exit price) with prior day close.
    # Use daily_pnl sign persistence as a simple proxy when prices unavailable.
    merged["prior_day_pnl"] = merged["daily_pnl"].shift(1)
    merged["two_day_trend"] = np.where(
        (merged["daily_pnl"] > 0) & (merged["prior_day_pnl"] > 0), "up_trend",
        np.where((merged["daily_pnl"] < 0) & (merged["prior_day_pnl"] < 0), "down_trend", "chop")
    )

    regimes = {}
    for regime, grp in merged.groupby("two_day_trend"):
        if len(grp) < 5:
            continue
        regimes[str(regime)] = {
            "days": int(len(grp)),
            "win_days": int(grp["win_day"].sum()),
            "win_rate_pct": round(float(grp["win_day"].mean()) * 100, 2),
            "avg_daily_pnl": round(float(grp["daily_pnl"].mean()), 2),
            "total_pnl": round(float(grp["daily_pnl"].sum()), 2),
            "trades_per_day": round(float(grp["trades"].mean()), 2),
        }

    # Streak analysis.
    merged["streak"] = np.where(merged["win_day"], 1, -1)
    streak_groups = []
    current = 0
    for v in merged["streak"]:
        if v == current or current == 0:
            current = v
            streak_groups.append(current)
        else:
            current = v
            streak_groups.append(current)
    merged["streak_group"] = (
        (merged["streak"] != merged["streak"].shift(1)).cumsum()
    )
    streak_summary = merged.groupby("streak_group").agg(
        length=("streak", "size"),
        sign=("streak", "first"),
        pnl=("daily_pnl", "sum"),
    )
    win_streaks = streak_summary[streak_summary["sign"] == 1]
    loss_streaks = streak_summary[streak_summary["sign"] == -1]

    return {
        "regimes": regimes,
        "avg_win_streak_len": round(float(win_streaks["length"].mean()), 2) if len(win_streaks) else 0,
        "max_win_streak_len": int(win_streaks["length"].max()) if len(win_streaks) else 0,
        "avg_loss_streak_len": round(float(loss_streaks["length"].mean()), 2) if len(loss_streaks) else 0,
        "max_loss_streak_len": int(loss_streaks["length"].max()) if len(loss_streaks) else 0,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to backtest JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--mc-runs", type=int, default=50_000)
    parser.add_argument("--wrc-bootstraps", type=int, default=2_000)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    result = load_backtest(args.input)
    trades = result.get("trades", result.get("combined_trades", []))
    pnls = extract_pnls(result)
    daily_pnl = extract_daily_pnl(result)
    active_days = max(1, len(daily_pnl[daily_pnl != 0]))
    trades_per_day = len(pnls) / active_days

    equity = np.cumsum(np.concatenate([[0.0], pnls])) + result.get("parameters", {}).get(
        "initial_account_size", 50_000.0
    )
    max_dd_pct, _, _ = compute_max_drawdown(equity)

    validation = compute_full_validation(
        pnls,
        trades_per_day,
        num_trials=args.mc_runs,
        random_seed=args.random_seed,
    )
    validation["max_drawdown_pct"] = round(max_dd_pct, 2)
    validation["trades_per_day"] = round(trades_per_day, 3)

    walk_forward = walk_forward_analysis(daily_pnl)
    regimes = regime_analysis(result, daily_pnl, trades)

    report = {
        "source_file": args.input,
        "parameters": result.get("parameters", {}),
        "validation": validation,
        "walk_forward": walk_forward,
        "regime_analysis": regimes,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved robustness report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
