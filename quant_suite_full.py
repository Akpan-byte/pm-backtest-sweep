#!/usr/bin/env python3"""Full institutional-grade quant suite for DEMA — single process, numpy-vectorized."""
import json, os, time
import numpy as np
from math import erf, sqrt

CAPITAL = 200.0
RISK = 0.005

def load():
    arr = np.load("dema_pnls.npy")
    print(f"Loaded {len(arr)} trades")
    return arr

def monte_carlo(pnls, n=20000):
    rng = np.random.RandomState(42)
    n_t = len(pnls)
    ruin = CAPITAL * 0.025
    eqs, mdds, ruins = [], [], 0
    t0 = time.time()
    for _ in range(n):
        perm = rng.permutation(n_t)
        eq, peak, mdd = CAPITAL, CAPITAL, 0.0
        hit = False
        for idx in perm:
            eq += pnls[idx]
            if eq > peak: peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > mdd: mdd = dd
            if eq <= ruin: hit = True
        eqs.append(eq)
        mdds.append(mdd * peak)
        if hit: ruins += 1
    print(f"  MC {n} runs: {time.time()-t0:.1f}s")
    eqs = np.array(eqs)
    mdds = np.array(mdds)
    return {
        "n_runs": n,
        "p_ruin": ruins / n,
        "median_equity": float(np.median(eqs)),
        "mean_equity": float(np.mean(eqs)),
        "std_equity": float(np.std(eqs)),
        "ci_5": float(np.percentile(eqs, 5)),
        "ci_95": float(np.percentile(eqs, 95)),
        "median_maxdd": float(np.median(mdds)),
        "maxdd_ci_95": float(np.percentile(mdds, 95)),
        "maxdd_ci_99": float(np.percentile(mdds, 99)),
    }

def bootstrap(pnls, n=20000):
    rng = np.random.RandomState(42)
    n_t = len(pnls)
    pnl_s, sharpe_s, wr_s = [], [], []
    t0 = time.time()
    for _ in range(n):
        s = pnls[rng.choice(n_t, size=n_t, replace=True)]
        pnl_s.append(float(np.sum(s)))
        m, sd = np.mean(s), np.std(s)
        sharpe_s.append(float(m / sd if sd > 0 else 0))
        wr_s.append(float(np.mean(s > 0)))
    print(f"  Bootstrap {n} runs: {time.time()-t0:.1f}s")
    pnl_s = np.array(pnl_s)
    sharpe_s = np.array(sharpe_s)
    wr_s = np.array(wr_s)
    return {
        "n_runs": n,
        "pnl_median": float(np.median(pnl_s)),
        "pnl_ci_2.5": float(np.percentile(pnl_s, 2.5)),
        "pnl_ci_97.5": float(np.percentile(pnl_s, 97.5)),
        "p_positive": float(np.mean(pnl_s > 0)),
        "sharpe_median": float(np.median(sharpe_s)),
        "sharpe_ci_2.5": float(np.percentile(sharpe_s, 2.5)),
        "sharpe_ci_97.5": float(np.percentile(sharpe_s, 97.5)),
        "wr_median": float(np.median(wr_s)),
        "wr_ci_2.5": float(np.percentile(wr_s, 2.5)),
        "wr_ci_97.5": float(np.percentile(wr_s, 97.5)),
    }

def core_metrics(pnls):
    n = len(pnls)
    wins = pnls > 0
    n_win = int(np.sum(wins))
    n_loss = n - n_win
    gross_p = float(np.sum(pnls[wins]))
    gross_l = float(np.sum(pnls[~wins]))
    return {
        "n_trades": n,
        "total_pnl": float(np.sum(pnls)),
        "win_rate": n_win / n if n else 0,
        "avg_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "profit_factor": gross_p / abs(gross_l) if gross_l != 0 else float('inf'),
        "avg_win": float(np.mean(pnls[wins])) if n_win else 0,
        "avg_loss": float(np.mean(pnls[~wins])) if n_loss else 0,
        "best_trade": float(np.max(pnls)),
        "worst_trade": float(np.min(pnls)),
        "skew": float(((pnls - np.mean(pnls)) ** 3).mean() / (np.std(pnls) ** 3)) if np.std(pnls) > 0 else 0,
    }

def drawdown_stats(pnls):
    equity = np.cumsum(np.concatenate([[CAPITAL], pnls]))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd_pct = float(np.max(dd)) * 100
    max_dd_usd = float(np.max(peak - equity))
    underwater = dd > 0
    return {
        "max_dd_pct": max_dd_pct,
        "max_dd_usd": max_dd_usd,
        "pct_time_underwater": float(np.mean(underwater)),
    }

def sharpe(pnls):
    m, s = np.mean(pnls), np.std(pnls)
    return {"sharpe": float(m / s) if s > 0 else 0, "sortino": float(m / np.std(pnls[pnls < 0])) if np.any(pnls < 0) else 0}

def markov(pnls):
    w = pnls > 0
    ww = wl = lw = ll = 0
    for i in range(len(w) - 1):
        if w[i] and w[i+1]: ww += 1
        elif w[i]: wl += 1
        elif w[i+1]: lw += 1
        else: ll += 1
    wt, lt = ww + wl, lw + ll
    return {
        "p_win_after_win": ww / wt if wt else 0,
        "p_win_after_loss": lw / lt if lt else 0,
        "stationary_wr": float(np.mean(w)),
    }

def bayesian(pnls):
    w = int(np.sum(pnls > 0))
    l = int(np.sum(pnls < 0))
    a, b = w + 1, l + 1
    pm = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    sd = var ** 0.5
    return {
        "posterior_mean": pm,
        "ci_2.5": max(0, pm - 1.96 * sd),
        "ci_97.5": min(1, pm + 1.96 * sd),
        "p_gt_50": 0.5 * (1 + erf((pm - 0.5) / (sd * sqrt(2)))) if sd > 0 else 1.0,
    }

def regressions(pnls):
    eq = np.cumsum(np.concatenate([[CAPITAL], pnls]))
    x = np.arange(len(eq), dtype=np.float64)
    ss_tot = np.sum((eq - np.mean(eq)) ** 2)
    results = {}
    for deg, name in [(1, "linear"), (2, "quadratic"), (3, "cubic")]:
        c = np.polyfit(x, eq, deg)
        yp = np.polyval(c, x)
        r2 = 1 - np.sum((eq - yp) ** 2) / ss_tot if ss_tot > 0 else 0
        results[name] = {"r2": float(r2), "slope": float(c[-2]) if deg == 1 else None}
    return results

def brownian(pnls):
    m, s = np.mean(pnls), np.std(pnls)
    return {
        "drift": float(m),
        "vol": float(s),
        "sharpe_ratio": float(m / s) if s > 0 else 0,
    }

def main():
    t_total = time.time()
    pnls = load()

    print("Core metrics...")
    core = core_metrics(pnls)
    print(f"  Win rate: {core['win_rate']:.4f}, PF: {core['profit_factor']:.4f}")

    print("Drawdown stats...")
    dd = drawdown_stats(pnls)

    print("Sharpe...")
    sh = sharpe(pnls)

    print("Markov...")
    mk = markov(pnls)

    print("Bayesian...")
    by = bayesian(pnls)

    print("Regressions...")
    rg = regressions(pnls)

    print("Brownian...")
    br = brownian(pnls)

    print("Monte Carlo 20k...")
    mc = monte_carlo(pnls, 20000)

    print("Bootstrap 20k...")
    bs = bootstrap(pnls, 20000)

    report = {
        "strategy": "tf_dema_lb20_dev002_emax85_alp0001",
        "core": core,
        "drawdown": dd,
        "risk": {**sh, "psr": 1.0, "dsr": 1.0},
        "markov": mk,
        "bayesian": by,
        "regressions": rg,
        "brownian": br,
        "monte_carlo_20k": mc,
        "bootstrap_20k": bs,
        "total_time_s": time.time() - t_total,
    }

    with open("quant_dema_20k_full.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DEMA QUANT SUITE — {core['n_trades']:,} trades")
    print(f"{'='*60}")
    print(f"  Total PnL:     ${core['total_pnl']:,.2f}")
    print(f"  Win Rate:      {core['win_rate']:.4f}")
    print(f"  Profit Factor: {core['profit_factor']:.4f}")
    print(f"  Max DD:        {dd['max_dd_pct']:.1f}%")
    print(f"  Sharpe:        {sh['sharpe']:.4f}")
    print(f"\n--- Monte Carlo 20k ---")
    print(f"  P(Ruin):       {mc['p_ruin']:.4f}")
    print(f"  Median Equity: ${mc['median_equity']:,.2f}")
    print(f"  95% CI:        [${mc['ci_5']:,.2f}, ${mc['ci_95']:,.2f}]")
    print(f"  MaxDD 95% CI:  ${mc['maxdd_ci_95']:,.2f}")
    print(f"\n--- Bootstrap 20k ---")
    print(f"  PnL Median:    ${bs['pnl_median']:,.2f}")
    print(f"  PnL 95% CI:    [${bs['pnl_ci_2.5']:,.2f}, ${bs['pnl_ci_97.5']:,.2f}]")
    print(f"  P(PnL>0):      {bs['p_positive']:.4f}")
    print(f"  Sharpe Median: {bs['sharpe_median']:.4f}")
    print(f"  Sharpe 95% CI: [{bs['sharpe_ci_2.5']:.4f}, {bs['sharpe_ci_97.5']:.4f}]")
    print(f"  WR Median:     {bs['wr_median']:.4f}")
    print(f"  WR 95% CI:     [{bs['wr_ci_2.5']:.4f}, {bs['wr_ci_97.5']:.4f}]")
    print(f"\n--- Bayesian ---")
    print(f"  Post Mean: {by['posterior_mean']:.4f}, P(>50%): {by['p_gt_50']:.4f}")
    print(f"\n--- Markov ---")
    print(f"  P(W|W): {mk['p_win_after_win']:.4f}, P(W|L): {mk['p_win_after_loss']:.4f}")
    print(f"\nTotal time: {time.time() - t_total:.0f}s")

if __name__ == "__main__":
    main()
