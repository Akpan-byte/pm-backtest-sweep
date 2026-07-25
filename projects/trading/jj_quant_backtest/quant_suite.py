#!/usr/bin/env python3
"""
🌙 Moon Dev's Quant Suite Engine (SciPy-Free Version)
====================================================
Built with love by the AkpanBrain Mesh.
A high-performance reusable library for institutional-grade trading backtest validation.
This version implements all custom statistical algorithms in pure python/numpy to bypass
environment library corruptions.
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Any, Tuple, Optional

# --- CUSTOM STATISTICAL FUNCTIONS ---

def norm_cdf(z: float) -> float:
    """Standard Normal Cumulative Distribution Function (Phi) using error function."""
    try:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    except (ValueError, OverflowError):
        return 1.0 if z > 0 else 0.0

def skewness(x: np.ndarray) -> float:
    """Calculate sample skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return 0.0
    return (n / ((n - 1.0) * (n - 2.0))) * np.sum(((x - mean) / std) ** 3)

def pearson_kurtosis(x: np.ndarray) -> float:
    """Calculate Pearson Kurtosis (Fisher Excess Kurtosis + 3.0)."""
    n = len(x)
    if n < 4:
        return 3.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return 3.0
    m4 = np.mean((x - mean) ** 4)
    m2 = np.mean((x - mean) ** 2)
    return m4 / (m2 ** 2) if m2 != 0 else 3.0

def custom_linregress(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Calculate slope, intercept, and R2 of linear regression."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.sum((x - mean_x) * (y - mean_y))
    var_x = np.sum((x - mean_x) ** 2)
    if var_x == 0:
        return 0.0, mean_y, 0.0
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    
    # R2
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - mean_y) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return float(slope), float(intercept), float(r2)

# --- 1. CORE PERFORMANCE METRICS ---

def calculate_pnl_summary(trades_df: pd.DataFrame, starting_balance: float = 100.0, years: float = 1.0) -> Dict[str, Any]:
    """
    Calculate a comprehensive P&L summary from $starting_balance capital.
    Returns all dollar-value metrics including CAGR, expectancy, gross profit/loss,
    win/loss breakdowns, and terminal balance with total return percentage.
    
    Args:
        trades_df: DataFrame with 'pnl' (dollar P&L) and 'pnl_r' (R-multiple) columns
        starting_balance: Starting capital in dollars (default $100)
        years: Number of years the data covers (for CAGR calculation)
    Returns:
        Dict with full P&L breakdown
    """
    if trades_df.empty:
        return {"error": "No trades to summarize"}
    
    pnl = trades_df['pnl'].values
    pnl_r = trades_df['pnl_r'].values if 'pnl_r' in trades_df.columns else pnl
    
    # Win / Loss split
    win_mask  = pnl_r > 0
    loss_mask = pnl_r < 0
    flat_mask = pnl_r == 0
    
    win_count  = int(np.sum(win_mask))
    loss_count = int(np.sum(loss_mask))
    flat_count = int(np.sum(flat_mask))
    total      = len(pnl_r)
    
    win_rate = win_count / total if total > 0 else 0.0
    
    # Dollar amounts
    gross_profit = float(np.sum(pnl[win_mask]))   if win_count  > 0 else 0.0
    gross_loss   = float(np.sum(pnl[loss_mask]))  if loss_count > 0 else 0.0
    net_pnl      = float(np.sum(pnl))
    
    avg_win_dollar  = gross_profit / win_count   if win_count  > 0 else 0.0
    avg_loss_dollar = gross_loss   / loss_count  if loss_count > 0 else 0.0
    avg_trade_pnl   = net_pnl / total            if total      > 0 else 0.0
    
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
    
    # Equity curve
    equity_curve = starting_balance + np.cumsum(pnl)
    terminal_balance = float(equity_curve[-1])
    total_return_pct = (terminal_balance - starting_balance) / starting_balance * 100.0
    
    # CAGR
    if years > 0 and terminal_balance > 0:
        cagr_pct = ((terminal_balance / starting_balance) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = 0.0
    
    # Drawdown
    peaks    = np.maximum.accumulate(equity_curve)
    dd_curve = peaks - equity_curve
    dd_pct   = dd_curve / peaks * 100.0
    max_dd_dollar = float(np.max(dd_curve)) if len(dd_curve) > 0 else 0.0
    max_dd_pct    = float(np.max(dd_pct))   if len(dd_pct)   > 0 else 0.0
    
    # Expectancy per trade (R-multiple weighted)
    expectancy_r     = float(np.mean(pnl_r))
    expectancy_dollar = float(np.mean(pnl))
    
    # Avg R per win and loss
    avg_win_r  = float(np.mean(pnl_r[win_mask]))  if win_count  > 0 else 0.0
    avg_loss_r = float(np.mean(pnl_r[loss_mask])) if loss_count > 0 else 0.0
    
    return {
        # Capital Summary
        "starting_balance":      starting_balance,
        "terminal_balance":      round(terminal_balance, 4),
        "net_pnl":               round(net_pnl, 4),
        "total_return_pct":      round(total_return_pct, 2),
        "cagr_pct":              round(cagr_pct, 2),
        # Trade Counts
        "total_trades":          total,
        "wins":                  win_count,
        "losses":                loss_count,
        "flats":                 flat_count,
        "win_rate":              round(win_rate, 4),
        # Dollar Breakdown
        "gross_profit":          round(gross_profit, 4),
        "gross_loss":            round(gross_loss, 4),
        "profit_factor":         round(profit_factor, 3),
        "avg_win_dollar":        round(avg_win_dollar, 4),
        "avg_loss_dollar":       round(avg_loss_dollar, 4),
        "avg_trade_pnl":         round(avg_trade_pnl, 4),
        "expectancy_dollar":     round(expectancy_dollar, 4),
        # R-Multiple Summary
        "expectancy_r":          round(expectancy_r, 4),
        "avg_win_r":             round(avg_win_r, 3),
        "avg_loss_r":            round(avg_loss_r, 3),
        # Risk Summary
        "max_drawdown_dollar":   round(max_dd_dollar, 4),
        "max_drawdown_pct":      round(max_dd_pct, 2),
        # Duration
        "backtest_years":        years
    }


def calculate_metrics(trades_df: pd.DataFrame, starting_balance: float = 100.0) -> Dict[str, Any]:
    """Calculate baseline performance statistics from a trade history DataFrame."""
    if trades_df.empty:
        return {}
        
    pnl = trades_df['pnl'].values
    pnl_r = trades_df['pnl_r'].values if 'pnl_r' in trades_df.columns else pnl
    
    wins = pnl_r > 0
    losses = pnl_r <= 0
    
    win_count = np.sum(wins)
    loss_count = np.sum(losses)
    total_trades = len(pnl_r)
    
    win_rate = win_count / total_trades if total_trades > 0 else 0.0
    
    total_profit = np.sum(pnl[pnl > 0])
    total_loss = np.sum(pnl[pnl < 0])
    profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
    
    mean_r = np.mean(pnl_r)
    std_r = np.std(pnl_r)
    robust_sharpe = mean_r / std_r if std_r != 0 else 0.0
    
    # Calculate drawdown curve
    equity_curve = starting_balance + np.cumsum(pnl)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks * 100.0
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    return {
        "total_trades": total_trades,
        "wins": int(win_count),
        "losses": int(loss_count),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy_r": float(mean_r),
        "robust_sharpe": float(robust_sharpe),
        "max_drawdown_pct": float(max_dd),
        "terminal_balance": float(equity_curve[-1]) if len(equity_curve) > 0 else starting_balance
    }

# --- 2. ADVANCED SHARPE METRICS (PSR & DSR) ---

def calculate_psr(returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    Measures probability that the true Sharpe Ratio exceeds benchmark_sr.
    """
    n = len(returns)
    if n < 4:
        return 0.5
        
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
        
    sr = mean / std
    
    # Skewness and Kurtosis
    skew = skewness(returns)
    kurt = pearson_kurtosis(returns)
    
    # Variance of Sharpe Ratio
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1.0)
    
    if var_sr <= 0:
        return 0.5
        
    t_stat = (sr - benchmark_sr) / np.sqrt(var_sr)
    return float(norm_cdf(t_stat))

def calculate_dsr(returns: np.ndarray, all_sharpes: List[float]) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR).
    Adjusts PSR for selection bias (multiple testing) based on a trial set of Sharpe Ratios.
    """
    if not all_sharpes:
        return calculate_psr(returns, 0.0)
        
    n_trials = len(all_sharpes)
    if n_trials <= 1:
        return calculate_psr(returns, 0.0)
        
    std_sharpe = np.std(all_sharpes, ddof=1)
    if std_sharpe == 0:
        return calculate_psr(returns, 0.0)
        
    # Expected maximum Sharpe under the null hypothesis (approximate quantile calculation)
    # Using the standard normal percentile approximation for standard maxima
    alpha_n = stats_norm_ppf(1.0 - 1.0 / n_trials) if 'stats_norm_ppf' in globals() else 1.96
    expected_max_sr = std_sharpe * alpha_n
    
    return calculate_psr(returns, expected_max_sr)

def stats_norm_ppf(p: float) -> float:
    """Approximation of the standard normal percent point function (inverse CDF)."""
    # Winitzki approximation of inverse error function
    if p <= 0.0 or p >= 1.0:
        return 0.0
    x = 2.0 * p - 1.0
    a = 0.147
    ln_term = math.log(1.0 - x**2)
    val = (2.0 / (math.pi * a) + ln_term / 2.0)
    inner = val**2 - ln_term / a
    erf_inv = np.sign(x) * math.sqrt(math.sqrt(inner) - val)
    return float(erf_inv * math.sqrt(2.0))

# --- 3. BAYESIAN WIN-RATE CONJUGATE MODEL ---

def calculate_bayesian_winrate(wins: int, losses: int, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> Dict[str, Any]:
    """
    Calculate posterior win-rate distribution using a Beta-Binomial conjugate prior model.
    Uses Normal approximation for credible intervals which is extremely accurate for samples > 10.
    """
    post_alpha = prior_alpha + wins
    post_beta = prior_beta + losses
    
    total = post_alpha + post_beta
    mean = post_alpha / total
    std = math.sqrt((post_alpha * post_beta) / (total**2 * (total + 1.0)))
    
    # 95% Credible Interval (Normal approximation: mean +/- 1.96 * std)
    ci_lower = max(0.0, mean - 1.96 * std)
    ci_upper = min(1.0, mean + 1.96 * std)
    
    return {
        "prior_alpha": prior_alpha,
        "prior_beta": prior_beta,
        "posterior_alpha": post_alpha,
        "posterior_beta": post_beta,
        "posterior_mean": float(mean),
        "posterior_std": float(std),
        "credible_interval_95": (float(ci_lower), float(ci_upper))
    }

# --- 4. MARKOV SEQUENCE STREAKS ---

def calculate_markov_transitions(pnl_r: np.ndarray) -> Dict[str, float]:
    """
    Compute first-order Markov chain transition state probabilities.
    Answers: what is the probability of a win given the previous trade outcome?
    """
    n = len(pnl_r)
    if n < 2:
        return {"P_W_W": 0.5, "P_L_W": 0.5, "P_W_L": 0.5, "P_L_L": 0.5}
        
    wins = (pnl_r > 0).astype(int)
    
    # Streaks counts
    ww, wl, lw, ll = 0, 0, 0, 0
    
    for i in range(n - 1):
        prev = wins[i]
        curr = wins[i+1]
        
        if prev == 1 and curr == 1:
            ww += 1
        elif prev == 1 and curr == 0:
            wl += 1
        elif prev == 0 and curr == 1:
            lw += 1
        elif prev == 0 and curr == 0:
            ll += 1
            
    total_w_prev = ww + wl
    total_l_prev = lw + ll
    
    p_w_w = ww / total_w_prev if total_w_prev > 0 else 0.5
    p_l_w = wl / total_w_prev if total_w_prev > 0 else 0.5
    p_w_l = lw / total_l_prev if total_l_prev > 0 else 0.5
    p_l_l = ll / total_l_prev if total_l_prev > 0 else 0.5
    
    return {
        "P_W_W": float(p_w_w),
        "P_L_W": float(p_l_w),
        "P_W_L": float(p_w_l),
        "P_L_L": float(p_l_l)
    }

# --- 5. HIGH-PERFORMANCE MONTE CARLO BOOTSTRAP ---

def run_monte_carlo(pnl_r: np.ndarray, starting_balance: float = 100.0, risk_pct: float = 0.005, runs: int = 10000) -> Dict[str, Any]:
    """
    Perform 10,000 randomized Monte Carlo simulations (random resampling with replacement)
    to compute expected balance percentiles and maximum drawdown envelopes.
    """
    n = len(pnl_r)
    if n == 0:
        return {}
        
    # We pre-allocate arrays to optimize execution speed
    terminal_balances = np.zeros(runs)
    max_drawdowns = np.zeros(runs)
    
    for r in range(runs):
        # Resample returns with replacement
        sampled_r = np.random.choice(pnl_r, size=n, replace=True)
        
        # Simulate compounding equity curve
        balance = starting_balance
        equity_curve = np.zeros(n)
        
        # Compensating positioning sizing formula:
        for i in range(n):
            pnl_val = balance * risk_pct * sampled_r[i]
            balance += pnl_val
            equity_curve[i] = balance
            
        terminal_balances[r] = balance
        
        # Calculate drawdown curve
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (peaks - equity_curve) / peaks * 100.0
        max_drawdowns[r] = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
    return {
        "runs": runs,
        "P10_balance": float(np.percentile(terminal_balances, 10)),
        "P50_balance": float(np.percentile(terminal_balances, 50)),  # Median
        "P90_balance": float(np.percentile(terminal_balances, 90)),
        "P50_max_dd": float(np.percentile(max_drawdowns, 50)),
        "P95_max_dd": float(np.percentile(max_drawdowns, 95))       # Worst case Max DD
    }

# --- 6. EQUITY-CURVE REGRESSION ANALYSIS ---

def run_regressions(equity_curve: np.ndarray) -> Dict[str, Any]:
    """
    Fit four regression models (Linear, Quadratic, Exponential, Logarithmic)
    using custom algebraic formulations to evaluate returns stability.
    """
    n = len(equity_curve)
    if n < 5:
        return {}
        
    x = np.arange(1, n + 1)
    y = equity_curve
    
    # 1. Linear Regression
    slope, intercept, r2_linear = custom_linregress(x, y)
    
    # 2. Quadratic Regression
    poly_coefs = np.polyfit(x, y, 2)
    y_quad_pred = poly_coefs[0] * x**2 + poly_coefs[1] * x + poly_coefs[2]
    ss_res_quad = np.sum((y - y_quad_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_quad = 1.0 - (ss_res_quad / ss_tot) if ss_tot != 0 else 0.0
    acceleration = poly_coefs[0]
    
    # 3. Exponential Regression
    y_clamped = np.maximum(y, 0.01)
    ln_y = np.log(y_clamped)
    slope_exp, intercept_exp, r2_exp = custom_linregress(x, ln_y)
    growth_rate = slope_exp
    
    # 4. Logarithmic Regression
    ln_x = np.log(x)
    slope_log, intercept_log, r2_log = custom_linregress(ln_x, y)
    decay_rate = slope_log
    
    return {
        "linear": {
            "r2": float(r2_linear),
            "slope": float(slope),
            "intercept": float(intercept)
        },
        "quadratic": {
            "r2": float(r2_quad),
            "acceleration_beta2": float(acceleration)
        },
        "exponential": {
            "r2": float(r2_exp),
            "growth_rate_b": float(growth_rate)
        },
        "logarithmic": {
            "r2": float(r2_log),
            "coefficient_b": float(decay_rate)
        }
    }

# --- 7. WALK-FORWARD CHRONOLOGICAL FOLDS ---

def run_walk_forward(trades_df: pd.DataFrame, folds: int = 5) -> Dict[str, Any]:
    """
    Split the dataset chronologically into 5 equal folds.
    Evaluates out-of-sample stability over time.
    """
    if trades_df.empty or len(trades_df) < folds * 2:
        return {}
        
    trades_df = trades_df.sort_index().reset_index(drop=True)
    n = len(trades_df)
    fold_size = n // folds
    
    fold_metrics = []
    
    for i in range(folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < folds - 1 else n
        
        fold_df = trades_df.iloc[start_idx:end_idx]
        fold_res = calculate_metrics(fold_df)
        
        fold_metrics.append({
            "fold": i + 1,
            "trades": len(fold_df),
            "win_rate": fold_res.get("win_rate", 0.0),
            "expectancy_r": fold_res.get("expectancy_r", 0.0),
            "robust_sharpe": fold_res.get("robust_sharpe", 0.0),
            "pnl": float(np.sum(fold_df['pnl'].values))
        })
        
    # Calculate Walk-Forward Efficiency (ratio of OOS average Sharpe vs first fold as baseline)
    sharpes = [f["robust_sharpe"] for f in fold_metrics]
    mean_oos_sharpe = np.mean(sharpes[1:]) if len(sharpes) > 1 else 0.0
    wfe = mean_oos_sharpe / sharpes[0] if sharpes[0] != 0 else 0.0
    
    return {
        "folds": fold_metrics,
        "walk_forward_efficiency": float(wfe),
        "sharpe_stability_std": float(np.std(sharpes))
    }


# --- 8. RATCHET COMPOUNDING SIMULATOR ---
# "Win-lock" compounding: sizing increases on wins, locks in place on losses.
# This gives compound growth on winning streaks while protecting against
# the recovery-lag problem of standard compounding (where losses reduce
# future trade sizing and extend the drawdown recovery period).

def simulate_ratchet_equity(
    pnl_r: np.ndarray,
    starting_balance: float = 100.0,
    risk_pct: float = 0.005
) -> Dict[str, Any]:
    """
    Simulate a ratchet (win-lock) compounding equity curve.
    
    Rules:
    - Start: dollar_risk = starting_balance * risk_pct
    - On a WIN:  new_dollar_risk = current_balance * risk_pct  (size up)
    - On a LOSS or FLAT: dollar_risk stays unchanged  (lock — don't size down)
    
    This separates compounding upside from the recovery-lag problem inherent
    in standard percentage-risk compounding, where losses shrink the next
    trade's risk dollar amount and slow recovery.
    
    Args:
        pnl_r: Array of R-multiples per trade (1.0 = 1R win, -1.0 = 1R loss)
        starting_balance: Starting capital in dollars
        risk_pct: Base risk percentage (e.g. 0.005 = 0.5% per trade)
    Returns:
        Dict with equity_curve, per_trade_risk_dollars, terminal_balance,
        max_drawdown_dollar, max_drawdown_pct, net_pnl, total_return_pct
    """
    n = len(pnl_r)
    if n == 0:
        return {}

    balance = starting_balance
    # Locked risk dollar — starts at base, only ratchets up on wins
    locked_risk_dollar = starting_balance * risk_pct

    equity_curve      = np.zeros(n)
    risk_dollar_trace = np.zeros(n)

    for i in range(n):
        r = pnl_r[i]
        pnl_dollar = locked_risk_dollar * r
        balance += pnl_dollar
        if not np.isfinite(balance) or balance < 1.0:
            balance = 1.0
        equity_curve[i]      = balance
        risk_dollar_trace[i] = locked_risk_dollar

        # Ratchet rule: only update sizing on a win
        if r > 0:
            new_risk = balance * risk_pct
            if new_risk > locked_risk_dollar:          # only ever increase
                locked_risk_dollar = new_risk

    terminal_balance  = float(equity_curve[-1])
    net_pnl           = terminal_balance - starting_balance
    total_return_pct  = net_pnl / starting_balance * 100.0

    peaks     = np.maximum.accumulate(equity_curve)
    dd_dollar = peaks - equity_curve
    dd_pct    = dd_dollar / peaks * 100.0
    max_dd_dollar = float(np.max(dd_dollar)) if n > 0 else 0.0
    max_dd_pct    = float(np.max(dd_pct))   if n > 0 else 0.0

    return {
        "equity_curve":        equity_curve,
        "risk_dollar_trace":   risk_dollar_trace,
        "terminal_balance":    round(terminal_balance, 4),
        "net_pnl":             round(net_pnl, 4),
        "total_return_pct":    round(total_return_pct, 2),
        "max_drawdown_dollar": round(max_dd_dollar, 4),
        "max_drawdown_pct":    round(max_dd_pct, 2),
        "final_risk_dollar":   round(float(locked_risk_dollar), 4),
        "starting_balance":    starting_balance,
        "risk_pct":            risk_pct
    }


def run_monte_carlo_ratchet(
    pnl_r: np.ndarray,
    starting_balance: float = 100.0,
    risk_pct: float = 0.005,
    runs: int = 10000
) -> Dict[str, Any]:
    """
    Run 10,000 Monte Carlo simulations using the ratchet compounding model.
    Each run randomly resamples the R-multiple series (bootstrap with replacement)
    and simulates a ratchet equity curve for that permutation.
    
    Outputs percentile distributions for terminal balance and max drawdown,
    allowing direct comparison with standard compounding Monte Carlo results.
    """
    n = len(pnl_r)
    if n == 0:
        return {}

    terminal_balances = np.zeros(runs)
    max_drawdowns     = np.zeros(runs)

    for run_i in range(runs):
        sampled_r = np.random.choice(pnl_r, size=n, replace=True)

        balance          = starting_balance
        locked_risk_dol  = starting_balance * risk_pct
        equity_curve     = np.zeros(n)

        for i in range(n):
            r = sampled_r[i]
            balance += locked_risk_dol * r
            if not np.isfinite(balance) or balance < 1.0:
                balance = 1.0
            equity_curve[i] = balance
            if r > 0:
                new_risk = balance * risk_pct
                if new_risk > locked_risk_dol:
                    locked_risk_dol = new_risk

        terminal_balances[run_i] = balance

        peaks     = np.maximum.accumulate(equity_curve)
        dd_pct    = (peaks - equity_curve) / peaks * 100.0
        max_drawdowns[run_i] = float(np.max(dd_pct)) if n > 0 else 0.0

    return {
        "runs":         runs,
        "P10_balance":  float(np.percentile(terminal_balances, 10)),
        "P50_balance":  float(np.percentile(terminal_balances, 50)),
        "P90_balance":  float(np.percentile(terminal_balances, 90)),
        "P50_max_dd":   float(np.percentile(max_drawdowns, 50)),
        "P95_max_dd":   float(np.percentile(max_drawdowns, 95))
    }


def calculate_ratchet_pnl_summary(
    trades_df: pd.DataFrame,
    starting_balance: float = 100.0,
    risk_pct: float = 0.005,
    years: float = 1.0
) -> Dict[str, Any]:
    """
    Full P&L summary for the ratchet compounding sizing model.
    Calls simulate_ratchet_equity() internally and appends CAGR and
    trade-count statistics for reporting.
    
    Args:
        trades_df: DataFrame with 'pnl_r' column (R-multiples per trade)
        starting_balance: Starting capital (default $100)
        risk_pct: Base risk percentage per trade (default 0.5%)
        years: Backtest duration in years (for CAGR)
    """
    if trades_df.empty:
        return {"error": "No trades"}

    pnl_r = trades_df['pnl_r'].values

    # Run deterministic ratchet curve
    ratchet = simulate_ratchet_equity(pnl_r, starting_balance=starting_balance, risk_pct=risk_pct)
    equity_curve = ratchet["equity_curve"]

    # Win / loss split
    win_mask  = pnl_r > 0
    loss_mask = pnl_r < 0
    win_count  = int(np.sum(win_mask))
    loss_count = int(np.sum(loss_mask))
    total      = len(pnl_r)
    win_rate   = win_count / total if total > 0 else 0.0

    terminal_balance = ratchet["terminal_balance"]
    net_pnl          = ratchet["net_pnl"]
    total_return_pct = ratchet["total_return_pct"]

    # CAGR
    if years > 0 and terminal_balance > 0:
        cagr_pct = ((terminal_balance / starting_balance) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = 0.0

    # Per-trade ratchet P&L reconstruction for expectancy
    # Re-run to get per-trade dollar P&L
    locked_risk  = starting_balance * risk_pct
    bal          = starting_balance
    per_trade_pnl = []
    for r in pnl_r:
        pnl_d = locked_risk * r
        per_trade_pnl.append(pnl_d)
        bal += pnl_d
        if not np.isfinite(bal) or bal < 1.0:
            bal = 1.0
        if r > 0:
            new_r = bal * risk_pct
            if new_r > locked_risk:
                locked_risk = new_r

    pnl_arr       = np.array(per_trade_pnl)
    gross_profit  = float(np.sum(pnl_arr[win_mask]))  if win_count  > 0 else 0.0
    gross_loss    = float(np.sum(pnl_arr[loss_mask])) if loss_count > 0 else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
    expectancy_d  = float(np.mean(pnl_arr))
    avg_win_d     = gross_profit / win_count  if win_count  > 0 else 0.0
    avg_loss_d    = gross_loss   / loss_count if loss_count > 0 else 0.0

    return {
        # Capital
        "sizing_model":          "ratchet_compounding",
        "starting_balance":      starting_balance,
        "terminal_balance":      round(terminal_balance, 4),
        "net_pnl":               round(net_pnl, 4),
        "total_return_pct":      round(total_return_pct, 2),
        "cagr_pct":              round(cagr_pct, 2),
        # Trades
        "total_trades":          total,
        "wins":                  win_count,
        "losses":                loss_count,
        "win_rate":              round(win_rate, 4),
        # Dollar breakdown
        "gross_profit":          round(gross_profit, 4),
        "gross_loss":            round(gross_loss, 4),
        "profit_factor":         round(profit_factor, 3),
        "avg_win_dollar":        round(avg_win_d, 4),
        "avg_loss_dollar":       round(avg_loss_d, 4),
        "expectancy_dollar":     round(expectancy_d, 4),
        # Risk
        "max_drawdown_dollar":   ratchet["max_drawdown_dollar"],
        "max_drawdown_pct":      ratchet["max_drawdown_pct"],
        "final_risk_dollar":     ratchet["final_risk_dollar"],
        # Duration
        "backtest_years":        years
    }
