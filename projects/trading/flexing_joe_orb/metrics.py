"""Statistical metrics for the Flexing Joe ORB backtest."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from .models import Trade


def compute_win_rate(pnls: np.ndarray) -> float:
    """Percentage of positive PnLs."""
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return 0.0
    return float(np.sum(pnls > 0) / len(pnls) * 100.0)


def compute_profit_factor(pnls: np.ndarray) -> float:
    """Gross profits divided by gross losses (absolute value)."""
    pnls = np.asarray(pnls, dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    if losses.sum() == 0:
        return 999.0 if wins.sum() > 0 else 0.0
    return float(wins.sum() / abs(losses.sum()))


def compute_sharpe(pnls: np.ndarray, trades_per_day: Optional[float] = None) -> float:
    """Annualized Sharpe ratio using per-trade PnLs."""
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) < 2:
        return 0.0
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls, ddof=1))
    if std_pnl <= 0:
        return 0.0
    tpd = trades_per_day if trades_per_day and trades_per_day > 0 else 1.0
    return float(mean_pnl / std_pnl * np.sqrt(252 * tpd))


def compute_sortino(pnls: np.ndarray, trades_per_day: Optional[float] = None) -> float:
    """Annualized Sortino ratio using downside deviation of per-trade PnLs."""
    pnls = np.asarray(pnls, dtype=float)
    downside = pnls[pnls < 0]
    if len(downside) < 2:
        return 0.0
    std_down = float(np.std(downside, ddof=1))
    if std_down <= 0:
        return 0.0
    tpd = trades_per_day if trades_per_day and trades_per_day > 0 else 1.0
    return float(np.mean(pnls) / std_down * np.sqrt(252 * tpd))


def compute_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """Return max drawdown percentage and the peak/trough indices."""
    equity = np.asarray(equity_curve, dtype=float)
    if len(equity) == 0:
        return 0.0, 0, 0
    running_max = np.maximum.accumulate(equity)
    # Avoid divide-by-zero on zero/negative peaks; use inf-safe division.
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_max > 0, (running_max - equity) / running_max, 0.0)
    trough_idx = int(np.argmax(drawdowns))
    peak_idx = int(np.argmax(equity[: trough_idx + 1])) if trough_idx >= 0 else 0
    return float(drawdowns[trough_idx] * 100.0), peak_idx, trough_idx


def compute_deflated_sharpe_ratio(
    sharpe: float,
    total_trades: int,
    skew: float,
    kurt: float,
    num_trials: int,
) -> float:
    """Bailey-López de Prado Deflated Sharpe Ratio."""
    if total_trades < 2 or num_trials < 2:
        return 0.0

    var_term = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * (sharpe ** 2)
    denom = np.sqrt(max(0.0001, var_term))

    euler_mascheroni = 0.5772156649
    z_a = stats.norm.ppf(1.0 - 1.0 / num_trials)
    z_b = stats.norm.ppf(1.0 - 1.0 / (num_trials * math.e))
    e_max_sr = (1.0 - euler_mascheroni) * z_a + euler_mascheroni * z_b

    dsr = stats.norm.cdf(((sharpe - e_max_sr) * np.sqrt(total_trades - 1)) / denom)
    return float(dsr)


def _stationary_bootstrap_sample(data: np.ndarray, block_length: int) -> np.ndarray:
    """Politis & Romano stationary block bootstrap sample."""
    n = len(data)
    out = np.empty(n, dtype=data.dtype)
    idx = np.random.randint(0, n)
    for i in range(n):
        if np.random.random() < 1.0 / block_length:
            idx = np.random.randint(0, n)
        out[i] = data[idx]
        idx = (idx + 1) % n
    return out


def compute_true_whites_reality_check(
    returns_matrix: np.ndarray,
    num_bootstraps: int = 2_000,
    block_length: int = 5,
) -> np.ndarray:
    """White's Reality Check via stationary block bootstrap.

    ``returns_matrix`` shape: (N_strategies, T_trades).  Returns a p-value per
    strategy adjusted for multiple testing via the bootstrap distribution of the
    maximum mean return.
    """
    returns_matrix = np.asarray(returns_matrix)
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2D")
    N, T = returns_matrix.shape
    if N == 0 or T == 0:
        return np.zeros(N)

    mean_returns = np.mean(returns_matrix, axis=1)
    centered = returns_matrix - mean_returns[:, None]

    max_stats = np.empty(num_bootstraps)
    for b in range(num_bootstraps):
        boot_sample = np.empty((N, T), dtype=returns_matrix.dtype)
        for i in range(N):
            boot_sample[i, :] = _stationary_bootstrap_sample(centered[i, :], block_length)
        max_stats[b] = np.max(np.mean(boot_sample, axis=1))

    pvalues = np.array([np.mean(max_stats >= mean_returns[i]) for i in range(N)])
    return np.clip(pvalues, 1.0 / (num_bootstraps + 1), 1.0)


def compute_true_benjamini_hochberg_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    q_sorted = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    original_q = np.empty(n)
    original_q[sorted_idx] = q_sorted
    return original_q


def summarize_metrics(
    trades: List[Trade],
    daily_pnl: Dict[str, float],
    returns_matrix: Optional[np.ndarray] = None,
    initial_equity: float = 0.0,
) -> Dict[str, Any]:
    """Compute the full metrics suite from executed trades and daily PnL."""
    if not trades:
        return {
            "win_rate": 0.0,
            "total_trades": 0,
            "net_pnl": 0.0,
            "avg_trade_pnl": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown_dollars": 0.0,
            "max_drawdown_pct": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "dsr": 0.0,
            "wrc_pvalue": 0.5,
            "fdr_qvalue": 0.5,
            "trades_per_day": 0.0,
        }

    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    total_trades = len(pnls)
    win_rate = compute_win_rate(pnls)
    net_pnl = float(np.sum(pnls))
    avg_trade_pnl = float(np.mean(pnls))
    profit_factor = compute_profit_factor(pnls)

    # Trades per day from daily PnL map.
    active_days = len(daily_pnl)
    trades_per_day = total_trades / max(1, active_days)

    sharpe = compute_sharpe(pnls, trades_per_day)
    sortino = compute_sortino(pnls, trades_per_day)

    skew = float(stats.skew(pnls)) if total_trades > 2 else 0.0
    kurt = float(stats.kurtosis(pnls)) if total_trades > 2 else 0.0

    # Build equity curve from trades.
    equity = [float(initial_equity)]
    for p in pnls:
        equity.append(equity[-1] + p)
    max_dd_pct, _, _ = compute_max_drawdown(np.array(equity))
    max_dd_dollars = float(np.max(np.maximum.accumulate(equity) - equity))

    # Deflated Sharpe Ratio.
    num_trials = (
        returns_matrix.shape[0]
        if returns_matrix is not None and returns_matrix.ndim == 2
        else max(2, total_trades)
    )
    dsr = compute_deflated_sharpe_ratio(sharpe, total_trades, skew, kurt, num_trials)

    # White's Reality Check and FDR.
    if returns_matrix is not None and returns_matrix.ndim == 2:
        wrc_pvalues = compute_true_whites_reality_check(returns_matrix)
        fdr_qvalues = compute_true_benjamini_hochberg_fdr(wrc_pvalues)
        wrc_pvalue = float(wrc_pvalues[0]) if len(wrc_pvalues) else 0.5
        fdr_qvalue = float(fdr_qvalues[0]) if len(fdr_qvalues) else 0.5
    else:
        wrc_pvalue = 0.5
        fdr_qvalue = 0.5

    return {
        "win_rate": round(win_rate, 2),
        "total_trades": total_trades,
        "net_pnl": round(net_pnl, 2),
        "avg_trade_pnl": round(avg_trade_pnl, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown_dollars": round(max_dd_dollars, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "dsr": round(dsr, 4),
        "wrc_pvalue": round(wrc_pvalue, 4),
        "fdr_qvalue": round(fdr_qvalue, 4),
        "trades_per_day": round(trades_per_day, 4),
    }
