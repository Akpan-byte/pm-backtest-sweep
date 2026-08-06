"""
Shared utilities for the fractal IV reversal strategy backtester.

Provides:
    - resample_ohlcv: aggregate 1-min bars to higher timeframes.
    - atr: Average True Range.
    - returns: simple / log returns.
    - metrics: Sharpe, Sortino, max drawdown, win rate, profit factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation of the standard normal CDF."""
    if np.isnan(x):
        return 0.0
    sign = np.sign(x)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989423 * np.exp(-x * x / 2.0)
    prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    if sign > 0:
        prob = 1.0 - prob
    return float(prob)


# -----------------------------------------------------------------------------
# Resampling
# -----------------------------------------------------------------------------

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample a 1-minute OHLCV DataFrame to a higher timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume (case-insensitive).
    freq : str
        Pandas offset alias, e.g. '5min', '15min', '1h'.

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV with a DatetimeIndex.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        # Try common timestamp column names.
        for col in ("timestamp", "datetime", "date", "time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex or a timestamp column")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(freq).agg(agg).dropna()


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame.
    period : int, optional
        Lookback window, by default 14.

    Returns
    -------
    pd.Series
        ATR values aligned with the input index.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def returns(prices: pd.Series, log: bool = False) -> pd.Series:
    """
    Compute returns from a price series.

    Parameters
    ----------
    prices : pd.Series
    log : bool, optional
        If True, return log returns, by default False.

    Returns
    -------
    pd.Series
    """
    if log:
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


# -----------------------------------------------------------------------------
# Performance metrics
# -----------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.
    """
    if returns.empty or returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    excess = returns - risk_free / periods_per_year
    return excess.mean() / returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """
    Annualized Sortino ratio (downside deviation only).
    """
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0 or np.isnan(downside.std()):
        return 0.0
    excess = returns.mean() - risk_free / periods_per_year
    return excess / downside.std() * np.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a negative fraction.
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def win_rate(trades: list[float]) -> float:
    """
    Fraction of profitable trades.

    Parameters
    ----------
    trades : list[float]
        Per-trade PnL values.
    """
    if not trades:
        return 0.0
    wins = sum(1 for pnl in trades if pnl > 0)
    return wins / len(trades)


def profit_factor(trades: list[float]) -> float:
    """
    Gross profit / gross loss. Returns inf if there are no losses.
    """
    if not trades:
        return 0.0
    gross_profit = sum(pnl for pnl in trades if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trades if pnl < 0))
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0,
                               periods_per_year: int = 252) -> float:
    """
    PSR of the observed Sharpe ratio versus a benchmark SR (Bailey-López de Prado).
    Returns a probability (0-1).
    """
    if returns.empty or returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    n = len(returns)
    sr = sharpe_ratio(returns, periods_per_year)
    skew = returns.skew()
    kurt = returns.kurtosis()
    if pd.isna(skew):
        skew = 0.0
    if pd.isna(kurt):
        kurt = 3.0
    var_sr = (1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) / (n - 1)
    if var_sr <= 0 or np.isnan(var_sr):
        return 0.0
    z = (sr - benchmark_sr) / np.sqrt(var_sr)
    return float(_norm_cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, trials: int = 1,
                          periods_per_year: int = 252) -> float:
    """
    DSR: PSR corrected for multiple trials / backtest overfitting.
    trials = number of independent strategy configurations tested.
    """
    if returns.empty or returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    n = len(returns)
    sr = sharpe_ratio(returns, periods_per_year)
    skew = returns.skew()
    kurt = returns.kurtosis()
    if pd.isna(skew):
        skew = 0.0
    if pd.isna(kurt):
        kurt = 3.0
    var_sr = (1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) / (n - 1)
    if var_sr <= 0 or np.isnan(var_sr):
        return 0.0
    # Approximate expected max SR under the null (independent trials)
    from math import gamma
    try:
        e_max = np.sqrt(var_sr) * ((1 - 0.57721566) * np.log(trials) + 0.57721566)
    except Exception:
        e_max = np.sqrt(var_sr) * np.sqrt(2 * np.log(trials)) if trials > 1 else 0.0
    benchmark = max(0.0, e_max)
    z = (sr - benchmark) / np.sqrt(var_sr)
    return float(_norm_cdf(z))
