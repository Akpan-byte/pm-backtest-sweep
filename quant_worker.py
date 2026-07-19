#!/usr/bin/env python3
"""Worker script for parallelized quant suite. Runs a chunk of MC + bootstrap iterations."""
import argparse, gzip, json, os, pickle, sys, time, random
import numpy as np

# Add paths
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
QUANT_DIR = os.path.join(os.path.expanduser("~"), ".hermes", "hermes-agent", "skills", "finance", "quant_suite")
for p in (HERE, SRC, QUANT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

CAPITAL = 200.0
BASE_RISK = 0.005
STRATEGY = "tf_dema_lb20_dev002_emax85_alp0001"

def load_pnls_from_tarball():
    """Load DEMA per-trade PnLs from the trade tarballs."""
    trades_dir = "trades"
    if not os.path.exists(trades_dir):
        print("ERROR: trades/ directory not found. Run download step first.", file=sys.stderr)
        sys.exit(1)
    
    import glob
    files = sorted(glob.glob(f"{trades_dir}/*.trades.jsonl.gz"))
    if not files:
        print(f"ERROR: No .trades.jsonl.gz files in {trades_dir}/", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading PnLs from {len(files)} trade files...")
    pnls = []
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        if rec.get("strategy") == STRATEGY:
                            pnls.append(float(rec["pnl"]))
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            continue
    
    print(f"Loaded {len(pnls)} trades for {STRATEGY}")
    return np.array(pnls)

def run_monte_carlo(pnls, n_runs, start_idx=0):
    """Run Monte Carlo simulations."""
    rng = np.random.RandomState(42 + start_idx)
    n_trades = len(pnls)
    ruin_level = CAPITAL * 0.025  # 5% of capital
    
    final_equities = []
    max_dds = []
    ruin_count = 0
    
    for i in range(n_runs):
        perm = rng.permutation(n_trades)
        equity = CAPITAL
        peak = CAPITAL
        max_dd = 0.0
        hit_ruin = False
        
        for idx in perm:
            equity += pnls[idx] * BASE_RISK / 0.005  # Scale by risk
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            if equity <= ruin_level:
                hit_ruin = True
        
        final_equities.append(equity)
        max_dds.append(max_dd * peak)  # DD in dollars
        if hit_ruin:
            ruin_count += 1
    
    return {
        "final_equities": final_equities,
        "max_dds": max_dds,
        "ruin_count": ruin_count,
        "n_runs": n_runs,
    }

def run_bootstrap_ci(pnls, n_runs, start_idx=0):
    """Run bootstrap confidence intervals."""
    rng = np.random.RandomState(42 + start_idx)
    n_trades = len(pnls)
    
    bootstrap_pnls = []
    bootstrap_sharpes = []
    bootstrap_winrates = []
    
    for i in range(n_runs):
        sample = rng.choice(n_trades, size=n_trades, replace=True)
        sample_pnls = pnls[sample]
        
        total = np.sum(sample_pnls)
        bootstrap_pnls.append(total)
        
        mean = np.mean(sample_pnls)
        std = np.std(sample_pnls)
        sharpe = mean / std if std > 0 else 0
        bootstrap_sharpes.append(sharpe)
        
        wins = np.sum(sample_pnls > 0) / n_trades
        bootstrap_winrates.append(wins)
    
    return {
        "pnl_samples": bootstrap_pnls,
        "sharpe_samples": bootstrap_sharpes,
        "winrate_samples": bootstrap_winrates,
        "n_runs": n_runs,
    }

def compute_regressions(equity_curve):
    """Compute all regression types on equity curve."""
    x = np.arange(len(equity_curve))
    y = np.array(equity_curve)
    
    results = {}
    
    # Linear
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    results["linear"] = {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
    
    # Quadratic
    coeffs = np.polyfit(x, y, 2)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    results["quadratic"] = {"coeffs": [float(c) for c in coeffs], "r2": float(r2)}
    
    # Cubic
    coeffs = np.polyfit(x, y, 3)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    results["cubic"] = {"coeffs": [float(c) for c in coeffs], "r2": float(r2)}
    
    # Exponential (fit log(y) vs x)
    mask = y > 0
    if np.sum(mask) > 10:
        log_y = np.log(y[mask])
        x_m = x[mask]
        slope, intercept = np.polyfit(x_m, log_y, 1)
        y_pred = np.exp(slope * x_m + intercept)
        ss_res = np.sum((y[mask] - y_pred) ** 2)
        ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results["exponential"] = {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
    
    # Polynomial (degree 5)
    coeffs = np.polyfit(x, y, 5)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    results["polynomial_5"] = {"r2": float(r2)}
    
    return results

def compute_markov(pnls):
    """Compute Markov transition probabilities."""
    wins = pnls > 0
    n = len(wins)
    
    ww = wl = lw = ll = 0
    for i in range(n - 1):
        if wins[i] and wins[i+1]: ww += 1
        elif wins[i] and not wins[i+1]: wl += 1
        elif not wins[i] and wins[i+1]: lw += 1
        else: ll += 1
    
    w_total = ww + wl
    l_total = lw + ll
    
    return {
        "p_win_after_win": ww / w_total if w_total > 0 else 0,
        "p_loss_after_win": wl / w_total if w_total > 0 else 0,
        "p_win_after_loss": lw / l_total if l_total > 0 else 0,
        "p_loss_after_loss": ll / l_total if l_total > 0 else 0,
        "stationary_win_rate": float(np.mean(wins)),
    }

def compute_bayesian(pnls):
    """Bayesian win rate posterior with Beta(1,1) prior."""
    wins = int(np.sum(pnls > 0))
    losses = int(np.sum(pnls < 0))
    
    # Beta(wins+1, losses+1) posterior
    from statistics import mean
    # Approximate posterior mean and 95% CI
    a = wins + 1
    b = losses + 1
    posterior_mean = a / (a + b)
    
    # 95% CI using normal approximation
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    std = var ** 0.5
    ci_low = max(0, posterior_mean - 1.96 * std)
    ci_high = min(1, posterior_mean + 1.96 * std)
    
    # P(win_rate > 0.5)
    # Using normal approximation
    z = (0.5 - posterior_mean) / std if std > 0 else 0
    from math import erf, sqrt
    p_gt_05 = 0.5 * (1 + erf((posterior_mean - 0.5) / (std * sqrt(2)))) if std > 0 else 1.0
    
    return {
        "winrate_post_mean": posterior_mean,
        "winrate_ci_2.5": ci_low,
        "winrate_ci_97.5": ci_high,
        "p_winrate_gt_0.5": p_gt_05,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, required=True)
    parser.add_argument("--total-mc", type=int, default=20000)
    parser.add_argument("--total-bootstrap", type=int, default=20000)
    args = parser.parse_args()
    
    wid = args.worker_id
    nwk = args.total_workers
    
    # Load data
    pnls = load_pnls_from_tarball()
    
    # Split MC iterations
    mc_per_worker = args.total_mc // nwk
    mc_start = wid * mc_per_worker
    mc_count = mc_per_worker if wid < nwk - 1 else args.total_mc - mc_start
    
    # Split bootstrap iterations
    bs_per_worker = args.total_bootstrap // nwk
    bs_start = wid * bs_per_worker
    bs_count = bs_per_worker if wid < nwk - 1 else args.total_bootstrap - bs_start
    
    print(f"Worker {wid}/{nwk}: MC {mc_start}-{mc_start+mc_count}, BS {bs_start}-{bs_start+bs_count}")
    
    # Run MC
    t0 = time.time()
    mc = run_monte_carlo(pnls, mc_count, start_idx=mc_start)
    print(f"  MC done in {time.time()-t0:.1f}s")
    
    # Run Bootstrap
    t0 = time.time()
    bs = run_bootstrap_ci(pnls, bs_count, start_idx=bs_start)
    print(f"  Bootstrap done in {time.time()-t0:.1f}s")
    
    # Compute static metrics (same for all workers, but we compute once)
    markov = compute_markov(pnls)
    bayesian = compute_bayesian(pnls)
    
    # Equity curve for regressions
    equity = np.cumsum(np.concatenate([[CAPITAL], pnls * BASE_RISK / 0.005]))
    regressions = compute_regressions(equity)
    
    # Save results
    os.makedirs("quant_worker_results", exist_ok=True)
    result = {
        "worker_id": wid,
        "mc": mc,
        "bootstrap": bs,
        "markov": markov,
        "bayesian": bayesian,
        "regressions": regressions,
        "n_trades": len(pnls),
    }
    
    outpath = f"quant_worker_results/worker_{wid:02d}.json"
    with open(outpath, "w") as f:
        json.dump(result, f)
    print(f"Saved to {outpath}")

if __name__ == "__main__":
    main()
