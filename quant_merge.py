#!/usr/bin/env python3"""Merge results from 20 quant workers into final report."""
import json, glob, os, sys
import numpy as np

def merge():
    files = sorted(glob.glob("quant_worker_results/worker_*.json"))
    if not files:
        print("ERROR: No worker results found")
        sys.exit(1)
    
    print(f"Merging {len(files)} worker results...")
    
    # Load all workers
    workers = []
    for f in files:
        with open(f) as fh:
            workers.append(json.load(fh))
    
    n_trades = workers[0]["n_trades"]
    
    # Merge Monte Carlo
    all_final_eq = []
    all_max_dd = []
    total_ruin = 0
    total_mc = 0
    for w in workers:
        all_final_eq.extend(w["mc"]["final_equities"])
        all_max_dd.extend(w["mc"]["max_dds"])
        total_ruin += w["mc"]["ruin_count"]
        total_mc += w["mc"]["n_runs"]
    
    all_final_eq = np.array(all_final_eq)
    all_max_dd = np.array(all_max_dd)
    
    mc_result = {
        "n_runs": total_mc,
        "p_ruin": total_ruin / total_mc,
        "median_equity": float(np.median(all_final_eq)),
        "mean_equity": float(np.mean(all_final_eq)),
        "std_equity": float(np.std(all_final_eq)),
        "ci_5_equity": float(np.percentile(all_final_eq, 5)),
        "ci_95_equity": float(np.percentile(all_final_eq, 95)),
        "median_maxdd": float(np.median(all_max_dd)),
        "maxdd_ci_95": float(np.percentile(all_max_dd, 95)),
        "maxdd_ci_99": float(np.percentile(all_max_dd, 99)),
    }
    
    # Merge Bootstrap
    all_pnl = []
    all_sharpe = []
    all_wr = []
    total_bs = 0
    for w in workers:
        all_pnl.extend(w["bootstrap"]["pnl_samples"])
        all_sharpe.extend(w["bootstrap"]["sharpe_samples"])
        all_wr.extend(w["bootstrap"]["winrate_samples"])
        total_bs += w["bootstrap"]["n_runs"]
    
    all_pnl = np.array(all_pnl)
    all_sharpe = np.array(all_sharpe)
    all_wr = np.array(all_wr)
    
    bs_result = {
        "n_runs": total_bs,
        "pnl_median": float(np.median(all_pnl)),
        "pnl_ci_2.5": float(np.percentile(all_pnl, 2.5)),
        "pnl_ci_97.5": float(np.percentile(all_pnl, 97.5)),
        "pnl_mean": float(np.mean(all_pnl)),
        "p_positive": float(np.mean(all_pnl > 0)),
        "sharpe_median": float(np.median(all_sharpe)),
        "sharpe_ci_2.5": float(np.percentile(all_sharpe, 2.5)),
        "sharpe_ci_97.5": float(np.percentile(all_sharpe, 97.5)),
        "winrate_median": float(np.median(all_wr)),
        "winrate_ci_2.5": float(np.percentile(all_wr, 2.5)),
        "winrate_ci_97.5": float(np.percentile(all_wr, 97.5)),
    }
    
    # Use first worker's static metrics (same for all)
    markov = workers[0]["markov"]
    bayesian = workers[0]["bayesian"]
    regressions = workers[0]["regressions"]
    
    # Build final report
    report = {
        "strategy": "tf_dema_lb20_dev002_emax85_alp0001",
        "n_trades": n_trades,
        "capital": 200.0,
        "monte_carlo_20k": mc_result,
        "bootstrap_20k": bs_result,
        "markov": markov,
        "bayesian": bayesian,
        "regressions": regressions,
    }
    
    # Save
    with open("quant_dema_20k_full.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print(f"DEMA QUANT SUITE — {n_trades:,} trades, {total_mc:,} MC, {total_bs:,} Bootstrap")
    print("="*70)
    
    print(f"\n--- Monte Carlo ({total_mc:,} runs) ---")
    print(f"  P(Ruin):           {mc_result['p_ruin']:.4f}")
    print(f"  Median Equity:     ${mc_result['median_equity']:,.2f}")
    print(f"  Mean Equity:       ${mc_result['mean_equity']:,.2f}")
    print(f"  Equity 95% CI:     [${mc_result['ci_5_equity']:,.2f}, ${mc_result['ci_95_equity']:,.2f}]")
    print(f"  Median MaxDD:      ${mc_result['median_maxdd']:,.2f}")
    print(f"  MaxDD 95% CI:      ${mc_result['maxdd_ci_95']:,.2f}")
    print(f"  MaxDD 99% CI:      ${mc_result['maxdd_ci_99']:,.2f}")
    
    print(f"\n--- Bootstrap ({total_bs:,} runs) ---")
    print(f"  PnL Median:        ${bs_result['pnl_median']:,.2f}")
    print(f"  PnL 95% CI:        [${bs_result['pnl_ci_2.5']:,.2f}, ${bs_result['pnl_ci_97.5']:,.2f}]")
    print(f"  P(PnL > 0):        {bs_result['p_positive']:.4f}")
    print(f"  Sharpe Median:     {bs_result['sharpe_median']:.4f}")
    print(f"  Sharpe 95% CI:     [{bs_result['sharpe_ci_2.5']:.4f}, {bs_result['sharpe_ci_97.5']:.4f}]")
    print(f"  Win Rate Median:   {bs_result['winrate_median']:.4f}")
    print(f"  Win Rate 95% CI:   [{bs_result['winrate_ci_2.5']:.4f}, {bs_result['winrate_ci_97.5']:.4f}]")
    
    print(f"\n--- Markov ---")
    print(f"  P(Win|Win):   {markov['p_win_after_win']:.4f}")
    print(f"  P(Win|Loss):  {markov['p_win_after_loss']:.4f}")
    print(f"  Stationary WR: {markov['stationary_win_rate']:.4f}")
    
    print(f"\n--- Bayesian ---")
    print(f"  Posterior Mean:  {bayesian['winrate_post_mean']:.4f}")
    print(f"  95% Credible:    [{bayesian['winrate_ci_2.5']:.4f}, {bayesian['winrate_ci_97.5']:.4f}]")
    print(f"  P(WR > 50%):     {bayesian['p_winrate_gt_0.5']:.4f}")
    
    print(f"\n--- Regressions ---")
    for name, r in regressions.items():
        print(f"  {name:<15} R² = {r['r2']:.4f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    merge()
