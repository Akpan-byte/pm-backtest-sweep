"""Monte Carlo and bootstrap confidence intervals for backtest results."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .models import StrategyConfig


def run_monte_carlo(
    pnls: np.ndarray,
    num_runs: int,
    noise_level: float = 0.05,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Resample trades with replacement and add Gaussian noise.

    Returns the 5th percentile, median, and mean of the total PnL distribution.
    """
    rng = np.random.default_rng(random_seed)
    pnls = np.asarray(pnls, dtype=float)
    total_trades = len(pnls)
    if total_trades == 0:
        return {"p5": 0.0, "p50": 0.0, "mean": 0.0}

    std_pnl = max(0.001, float(np.std(pnls, ddof=1)))
    idx_mat = rng.integers(0, total_trades, size=(num_runs, total_trades))
    noise = rng.normal(0.0, std_pnl * noise_level, size=(num_runs, total_trades))
    sim_totals = np.sum(pnls[idx_mat] + noise, axis=1)

    return {
        "p5": float(np.percentile(sim_totals, 5)),
        "p50": float(np.percentile(sim_totals, 50)),
        "mean": float(np.mean(sim_totals)),
    }


def run_bootstrap_ci(
    pnls: np.ndarray,
    num_samples: int,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Bootstrapped confidence intervals for total and per-trade mean PnL."""
    rng = np.random.default_rng(random_seed)
    pnls = np.asarray(pnls, dtype=float)
    n = len(pnls)
    if n == 0:
        return {
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "mean_pnl_ci_lower": 0.0,
            "mean_pnl_ci_upper": 0.0,
        }

    idx_mat = rng.integers(0, n, size=(num_samples, n))
    samples = pnls[idx_mat]
    totals = np.sum(samples, axis=1)
    means = np.mean(samples, axis=1)

    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    return {
        "ci_lower": float(np.percentile(totals, lower_pct)),
        "ci_upper": float(np.percentile(totals, upper_pct)),
        "mean": float(np.mean(totals)),
        "median": float(np.percentile(totals, 50)),
        "std": float(np.std(totals, ddof=1)),
        "mean_pnl_ci_lower": float(np.percentile(means, lower_pct)),
        "mean_pnl_ci_upper": float(np.percentile(means, upper_pct)),
    }


def attach_mc_and_bootstrap(
    result: Dict[str, Any],
    config: StrategyConfig,
) -> Dict[str, Any]:
    """Add ``monte_carlo_50k`` and ``bootstrap_ci_50k`` to ``result``.

    If ``config.mc_runs`` or ``config.bootstrap_samples`` is 0, that section is
    skipped (useful for fast chunk runs where the aggregator will recompute on
    the combined series).
    """
    trades = result.get("trades", [])
    if not trades:
        result["monte_carlo_50k"] = {"p5": 0.0, "p50": 0.0, "mean": 0.0}
        result["bootstrap_ci_50k"] = {
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "mean_pnl_ci_lower": 0.0,
            "mean_pnl_ci_upper": 0.0,
        }
        return result

    # Use daily PnL for MC/bootstrap to reduce memory and preserve serial
    # structure.  Daily series is ~100x shorter than the trade list.
    daily_pnl = result.get("daily_pnl", {})
    if daily_pnl:
        pnls = np.array(list(daily_pnl.values()), dtype=float)
    else:
        pnls = np.array([t["net_pnl"] for t in trades], dtype=float)
    if config.mc_runs > 0:
        result["monte_carlo_50k"] = run_monte_carlo(
            pnls, num_runs=config.mc_runs, random_seed=config.random_seed
        )
    if config.bootstrap_samples > 0:
        result["bootstrap_ci_50k"] = run_bootstrap_ci(
            pnls, num_samples=config.bootstrap_samples, random_seed=config.random_seed
        )
    return result
