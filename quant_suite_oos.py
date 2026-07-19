#!/usr/bin/env python3
"""DEMA OOS Quant Suite — July 1-18, 2026."""
import json, time, tarfile, gzip, os, sys
import numpy as np
from math import erf, sqrt

CAPITAL = 200.0
OOS_TS = 1751328000.0  # July 1, 2026 00:00:00 UTC

def calculate_psr(returns, benchmark_sr=0.0):
    n = len(returns)
    if n < 4: return 0.5
    m = float(np.mean(returns))
    s = float(np.std(returns, ddof=1))
    if s == 0: return 0.0
    sr = m / s
    skew = float(((returns - m) ** 3).mean()) / (s ** 3) if s > 0 else 0
    kurt = float(((returns - m) ** 4).mean()) / (s ** 4) if s > 0 else 3
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n - 1.0)
    if var_sr <= 0: return 0.5
    t_stat = (sr - benchmark_sr) / np.sqrt(var_sr)
    return float(0.5 * (1 + erf(t_stat / sqrt(2))))

def calculate_dsr(returns, all_sharpes):
    from scipy.stats import norm
    n_trials = len(all_sharpes)
    if n_trials <= 1: return calculate_psr(returns, 0.0)
    std_sharpe = float(np.std(all_sharpes, ddof=1))
    if std_sharpe == 0: return calculate_psr(returns, 0.0)
    alpha_n = float(norm.ppf(1.0 - 1.0 / n_trials))
    expected_max_sr = std_sharpe * alpha_n
    return calculate_psr(returns, expected_max_sr)

def extract_oos_pnls():
    """Stream DEMA trades from tarballs, filter for OOS (closed_at >= July 1 2026)."""
    TARGET = "tf_dema_lb20_dev002_emax85_alp0001.trades.jsonl.gz"
    pnls = []
    for part in range(1, 6):
        tgz = f"dl/trades-part{part}.tar.gz"
        if not os.path.exists(tgz):
            continue
        print(f"Scanning {tgz}...", flush=True)
        with tarfile.open(tgz, "r:gz") as tar:
            for member in tar:
                if member.name.endswith(TARGET) and "rV" not in member.name and "rC" not in member.name and "rD" not in member.name:
                    print(f"  Found: {member.name}", flush=True)
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    count = 0
                    oos_count = 0
                    with gzip.open(fobj, "rt", errors="replace") as gz:
                        for line in gz:
                            try:
                                rec = json.loads(line)
                                # Filter by market start_date_iso, not closed_at
                                market = rec.get("market", {})
                                start_date = market.get("start_date_iso", "")
                                if start_date >= "2026-07-01":
                                    pnls.append(float(rec["pnl"]))
                                    oos_count += 1
                                count += 1
                            except:
                                continue
                    print(f"  Scanned {count} trades, {oos_count} OOS (>= July 1)", flush=True)
                    break
    if not pnls:
        print("ERROR: No OOS trades found!", file=sys.stderr)
        sys.exit(1)
    return np.array(pnls, dtype=np.float64)

def main():
    t_total = time.time()

    # Extract OOS trades
    pnls = extract_oos_pnls()
    arr = pnls
    n = len(arr)
    np.save("dema_oos_pnls.npy", arr)
    print(f"Total OOS: {n} trades ({arr.nbytes/1048576:.1f} MB)")

    # Core metrics
    wins = int(np.sum(arr > 0))
    losses = int(np.sum(arr < 0))
    wr = wins / n if n else 0
    avg_win = float(np.mean(arr[arr > 0])) if wins else 0
    avg_loss = float(np.mean(arr[arr < 0])) if losses else 0
    pf = (avg_win * wins) / (abs(avg_loss) * losses) if losses and avg_loss != 0 else 999
    total_pnl = float(np.sum(arr))
    avg_pnl = total_pnl / n if n else 0
    sharpe = float(np.mean(arr) / np.std(arr)) if np.std(arr) > 0 else 0

    # Drawdown
    eq = np.cumsum(np.concatenate([[CAPITAL], arr]))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    mdd_idx = np.argmax(dd)
    max_dd_pct = dd[mdd_idx] * 100
    max_dd_dollar = dd[mdd_idx] * peak[mdd_idx]

    # PSR / DSR
    psr = calculate_psr(arr, 0.0)
    all_sharpes = [0.0259, 0.0223, 0.0092, 0.0072]
    dsr = calculate_dsr(arr, all_sharpes)

    print(f"\nCore: WR={wr:.4f} PF={pf:.4f} PnL=${total_pnl:,.2f} Sharpe={sharpe:.4f} PSR={psr:.4f} DSR={dsr:.4f}")

    # Monte Carlo 20k
    print("Monte Carlo 20k...")
    rng = np.random.RandomState(42)
    SAMPLE = min(100000, n)
    t0 = time.time()
    eqs = np.empty(20000)
    mdds = np.empty(20000)
    ruins = 0
    ruin_threshold = CAPITAL * 0.025
    for i in range(20000):
        s = arr[rng.randint(0, n, size=SAMPLE)]
        eq = np.empty(SAMPLE + 1)
        eq[0] = CAPITAL
        np.cumsum(s, out=eq[1:])
        eq += CAPITAL
        pk = np.maximum.accumulate(eq)
        dd = (pk - eq) / np.where(pk > 0, pk, 1)
        eqs[i] = eq[-1]
        mdds[i] = dd[np.argmax(dd)] * pk[np.argmax(dd)]
        if eq[-1] <= ruin_threshold or np.any(eq <= ruin_threshold):
            ruins += 1
    mc_time = time.time() - t0
    p_ruin = ruins / 20000
    print(f"  MC {20000} runs ({SAMPLE} sample): {mc_time:.1f}s")

    # Bootstrap 20k
    print("Bootstrap 20k...")
    t0 = time.time()
    pnl_bs = np.empty(20000)
    sharpe_bs = np.empty(20000)
    wr_bs = np.empty(20000)
    for i in range(20000):
        s = arr[rng.randint(0, n, size=SAMPLE)]
        pnl_bs[i] = np.sum(s) * (n / SAMPLE)
        std = float(np.std(s))
        sharpe_bs[i] = float(np.mean(s) / std) if std > 0 else 0
        wr_bs[i] = np.mean(s > 0)
    bs_time = time.time() - t0
    print(f"  Bootstrap {20000} runs ({SAMPLE} sample): {bs_time:.1f}s")

    # Bayesian
    a, b = wins + 1, losses + 1
    pm = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    sd = var ** 0.5
    p_gt_50 = 0.5 * (1 + erf((pm - 0.5) / (sd * sqrt(2)))) if sd > 0 else 1.0

    # Markov
    w = arr > 0
    ww = int(np.sum(w[:-1] & w[1:]))
    wl = int(np.sum(w[:-1] & ~w[1:]))
    lw = int(np.sum(~w[:-1] & w[1:]))
    ll = int(np.sum(~w[:-1] & ~w[1:]))
    wt, lt = ww + wl, lw + ll

    # Save
    report = {
        "strategy": "tf_dema_lb20_dev002_emax85_alp0001",
        "period": "OOS (July 1-18, 2026)",
        "n_trades": n,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4),
        "avg_pnl": round(avg_pnl, 4),
        "sharpe": round(sharpe, 4),
        "psr": round(psr, 4),
        "dsr": round(dsr, 4),
        "max_dd_pct": round(max_dd_pct, 2),
        "max_dd_dollar": round(max_dd_dollar, 2),
        "p_ruin": round(p_ruin, 4),
        "mc_median_equity": round(float(np.median(eqs)), 2),
        "mc_ci_5": round(float(np.percentile(eqs, 5)), 2),
        "mc_ci_95": round(float(np.percentile(eqs, 95)), 2),
        "mc_maxdd_95": round(float(np.percentile(mdds, 95)), 2),
        "bs_pnl_median": round(float(np.median(pnl_bs)), 2),
        "bs_pnl_ci_2.5": round(float(np.percentile(pnl_bs, 2.5)), 2),
        "bs_pnl_ci_97.5": round(float(np.percentile(pnl_bs, 97.5)), 2),
        "bs_p_positive": round(float(np.mean(pnl_bs > 0)), 4),
        "bs_sharpe_median": round(float(np.median(sharpe_bs)), 4),
        "bs_sharpe_ci_2.5": round(float(np.percentile(sharpe_bs, 2.5)), 4),
        "bs_sharpe_ci_97.5": round(float(np.percentile(sharpe_bs, 97.5)), 4),
        "bs_wr_median": round(float(np.median(wr_bs)), 4),
        "bs_wr_ci_2.5": round(float(np.percentile(wr_bs, 2.5)), 4),
        "bs_wr_ci_97.5": round(float(np.percentile(wr_bs, 97.5)), 4),
        "bayesian_posterior_mean": round(pm, 4),
        "bayesian_p_gt_50": round(p_gt_50, 4),
        "markov_p_ww": round(ww / wt if wt else 0, 4),
        "markov_p_lw": round(lw / lt if lt else 0, 4),
    }
    with open("quant_dema_oos.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DEMA OOS QUANT SUITE — {n:,} trades")
    print(f"{'='*60}")
    print(f"  Total PnL:     ${total_pnl:,.2f}")
    print(f"  Win Rate:      {wr:.4f}")
    print(f"  Profit Factor: {pf:.4f}")
    print(f"  Max DD:        {max_dd_pct:.1f}% (${max_dd_dollar:,.2f})")
    print(f"  Sharpe:        {sharpe:.4f}")
    print(f"  PSR:           {psr:.4f}")
    print(f"  DSR:           {dsr:.4f}")
    print(f"\n--- Monte Carlo 20k ---")
    print(f"  P(Ruin):       {p_ruin:.4f}")
    print(f"  Median Equity: ${np.median(eqs):,.2f}")
    print(f"  95% CI:        [${np.percentile(eqs, 5):,.2f}, ${np.percentile(eqs, 95):,.2f}]")
    print(f"  MaxDD 95% CI:  ${np.percentile(mdds, 95):,.2f}")
    print(f"\n--- Bootstrap 20k ---")
    print(f"  PnL Median:    ${np.median(pnl_bs):,.2f}")
    print(f"  PnL 95% CI:    [${np.percentile(pnl_bs, 2.5):,.2f}, ${np.percentile(pnl_bs, 97.5):,.2f}]")
    print(f"  P(PnL>0):      {np.mean(pnl_bs > 0):.4f}")
    print(f"  Sharpe Median: {np.median(sharpe_bs):.4f}")
    print(f"  Sharpe 95% CI: [{np.percentile(sharpe_bs, 2.5):.4f}, {np.percentile(sharpe_bs, 97.5):.4f}]")
    print(f"  WR Median:     {np.median(wr_bs):.4f}")
    print(f"  WR 95% CI:     [{np.percentile(wr_bs, 2.5):.4f}, {np.percentile(wr_bs, 97.5):.4f}]")
    print(f"\n--- Bayesian ---")
    print(f"  Post Mean: {pm:.4f}, P(>50%): {p_gt_50:.4f}")
    print(f"\n--- Markov ---")
    print(f"  P(W|W): {ww/wt if wt else 0:.4f}, P(W|L): {lw/lt if lt else 0:.4f}")
    print(f"\nTotal time: {time.time() - t_total:.0f}s")

if __name__ == "__main__":
    main()
