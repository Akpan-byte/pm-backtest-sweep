#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-18  kilo
#   - Post-processes existing trade data to simulate equity curve trading
#     and drawdown-based position reduction, without re-running the backtest.
#   - Stdlib-only (no numpy) for GHA runner compatibility.
# WHY: User wants to reduce 50%+ drawdowns while keeping profitability.
from __future__ import annotations
import argparse, glob, gzip, json, math, os, statistics, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
TRADES_DIR = os.path.join(HERE, "results", "is_taker_regime_full")
OUTPUT_DIR = os.path.join(HERE, "results", "drawdown_study")

CAPITAL = 200.0
BASE_RISK_PCT = 0.005


def load_pnl_array(path: str) -> list[float]:
    """Load just the pnl values from a gzipped jsonl into a list."""
    pnls = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                pnls.append(float(json.loads(line)["pnl"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return pnls


def _metrics(pnls_taken: list, n_total: int, n_skipped: int) -> dict:
    """Compute standard metrics from the taken-trade PnL sequence."""
    n = len(pnls_taken)
    if n == 0:
        return {
            "final_equity": CAPITAL, "total_pnl": 0, "return_pct": 0,
            "peak_equity": CAPITAL, "max_dd_usd": 0, "max_dd_pct": 0,
            "pct_time_underwater": 0, "win_rate": 0, "wins": 0, "losses": 0,
            "profit_factor": None, "sharpe_per_trade": 0, "avg_pnl_per_trade": 0,
            "n_trades_taken": 0, "n_trades_skipped": n_skipped,
            "skip_pct": round(100 * n_skipped / n_total, 1) if n_total > 0 else 0,
            "n_trades_total": n_total,
        }
    
    # Rebuild equity curve from taken trades only
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    max_dd_pct = 0.0
    underwater = 0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    sum_pnl = 0.0
    
    for pnl in pnls_taken:
        equity += pnl
        sum_pnl += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = 100 * dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
        if dd > 0:
            underwater += 1
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss -= pnl
    
    mean_p = sum_pnl / n
    std_p = statistics.stdev(pnls_taken) if n > 1 else 0.0
    sharpe = mean_p / std_p if std_p > 0 else 0
    
    return {
        "final_equity": round(equity, 2),
        "total_pnl": round(sum_pnl, 2),
        "return_pct": round(100 * sum_pnl / CAPITAL, 2),
        "peak_equity": round(peak, 2),
        "max_dd_usd": round(max_dd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "pct_time_underwater": round(100 * underwater / n, 1) if n > 0 else 0,
        "win_rate": round(wins / n, 4) if n > 0 else 0,
        "wins": wins, "losses": losses,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "sharpe_per_trade": round(sharpe, 4),
        "avg_pnl_per_trade": round(mean_p, 6),
        "n_trades_taken": n,
        "n_trades_skipped": n_skipped,
        "skip_pct": round(100 * n_skipped / n_total, 1) if n_total > 0 else 0,
        "n_trades_total": n_total,
    }


def simulate_fixed(pnls: list) -> dict:
    """Baseline: take every trade, fixed 0.5% risk."""
    return _metrics(pnls, len(pnls), 0)


def simulate_equity_curve(pnls: list, sma_period: int = 100) -> dict:
    """Equity curve filter: skip trades when equity < SMA of equity."""
    n = len(pnls)
    taken = []
    skipped = 0
    equity = CAPITAL
    eq_history = deque(maxlen=sma_period + 1)
    eq_history.append(CAPITAL)
    
    for i, pnl in enumerate(pnls):
        if i >= sma_period:
            sma = sum(eq_history) / len(eq_history)
            if equity < sma:
                skipped += 1
                eq_history.append(equity)
                continue
        
        taken.append(pnl)
        equity += pnl
        eq_history.append(equity)
    
    return _metrics(taken, n, skipped)


def simulate_drawdown_gate(pnls: list, dd_threshold_pct: float = 10.0) -> dict:
    """Drawdown gate: skip trades when in drawdown > threshold from peak."""
    n = len(pnls)
    taken = []
    skipped = 0
    equity = CAPITAL
    peak = CAPITAL
    
    for pnl in pnls:
        dd_pct = 100 * (peak - equity) / peak if peak > 0 else 0
        
        if dd_pct > dd_threshold_pct:
            skipped += 1
            continue
        
        taken.append(pnl)
        equity += pnl
        if equity > peak:
            peak = equity
    
    return _metrics(taken, n, skipped)


def simulate_combined(pnls: list, sma_period: int = 100, dd_threshold_pct: float = 10.0) -> dict:
    """Combined: equity curve + drawdown gate."""
    n = len(pnls)
    taken = []
    skipped = 0
    equity = CAPITAL
    peak = CAPITAL
    eq_history = deque(maxlen=sma_period + 1)
    eq_history.append(CAPITAL)
    
    for i, pnl in enumerate(pnls):
        dd_pct = 100 * (peak - equity) / peak if peak > 0 else 0
        skip = False
        
        if dd_pct > dd_threshold_pct:
            skip = True
        
        if not skip and i >= sma_period:
            sma = sum(eq_history) / len(eq_history)
            if equity < sma:
                skip = True
        
        if skip:
            skipped += 1
            eq_history.append(equity)
            continue
        
        taken.append(pnl)
        equity += pnl
        if equity > peak:
            peak = equity
        eq_history.append(equity)
    
    return _metrics(taken, n, skipped)


def run_one_strategy(args):
    """Load data once, run all configs for one strategy."""
    name, path, configs = args
    t0 = time.time()
    pnls = load_pnl_array(path)
    t_load = time.time() - t0
    
    results = []
    for cfg in configs:
        t1 = time.time()
        mode = cfg["mode"]
        if mode == "fixed":
            r = simulate_fixed(pnls)
        elif mode == "equity_curve":
            r = simulate_equity_curve(pnls, sma_period=cfg.get("sma_period", 100))
        elif mode == "drawdown_gate":
            r = simulate_drawdown_gate(pnls, dd_threshold_pct=cfg.get("dd_threshold", 10.0))
        elif mode == "combined":
            r = simulate_combined(pnls, sma_period=cfg.get("sma_period", 100),
                                  dd_threshold_pct=cfg.get("dd_threshold", 10.0))
        else:
            r = simulate_fixed(pnls)
        r["strategy"] = name
        r["mode"] = mode
        r["compute_time_s"] = round(time.time() - t1, 1)
        results.append(r)
    
    total = time.time() - t0
    print(f"  {name}: {len(pnls):,} trades, load={t_load:.0f}s, total={total:.0f}s", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="Comma-separated variant names")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    paths = sorted(glob.glob(os.path.join(TRADES_DIR, "*.trades.jsonl.gz")))
    names = [os.path.basename(p).replace(".trades.jsonl.gz", "") for p in paths]
    if args.only:
        keep = set(args.only.split(","))
        names = [n for n in names if n in keep]
    
    filtered = []
    for n in names:
        p = os.path.join(TRADES_DIR, f"{n}.trades.jsonl.gz")
        if os.path.getsize(p) > 1000:
            filtered.append(n)
    names = filtered
    
    configs = [
        {"mode": "fixed", "label": "Baseline (fixed 0.5%)"},
        {"mode": "equity_curve", "sma_period": 100, "label": "Equity Curve SMA-100"},
        {"mode": "equity_curve", "sma_period": 50, "label": "Equity Curve SMA-50"},
        {"mode": "drawdown_gate", "dd_threshold": 10.0, "label": "DD Gate 10%"},
        {"mode": "drawdown_gate", "dd_threshold": 15.0, "label": "DD Gate 15%"},
        {"mode": "combined", "sma_period": 100, "dd_threshold": 10.0, "label": "Combined (SMA-100 + DD-10%)"},
    ]
    
    jobs = []
    for n in names:
        path = os.path.join(TRADES_DIR, f"{n}.trades.jsonl.gz")
        jobs.append((n, path, configs))
    
    print(f"drawdown_study: {len(names)} strategies × {len(configs)} configs = {len(jobs)} jobs", flush=True)
    
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one_strategy, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.extend(fut.result())
            except Exception as e:
                j = futs[fut]
                for cfg in j[2]:
                    results.append({"strategy": j[0], "mode": cfg["mode"], "error": str(e)})
            if i % 10 == 0:
                print(f"  [{i}/{len(jobs)} done, {time.time()-t0:.0f}s]", flush=True)
    
    # Write per-strategy JSON
    from collections import defaultdict
    by_strat = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_strat[r["strategy"]].append(r)
    
    for strat, cfgs in by_strat.items():
        with open(os.path.join(OUTPUT_DIR, f"{strat}.drawdown.json"), "w") as f:
            json.dump(cfgs, f, indent=1)
    
    with open(os.path.join(OUTPUT_DIR, "drawdown_study.json"), "w") as f:
        json.dump(results, f, indent=1)
    
    # Build comparison table
    lines = ["# Drawdown Study: Equity Curve + Drawdown Gate\n",
             "## Per-Strategy Comparison\n"]
    
    for strat, cfgs in sorted(by_strat.items()):
        lines.append(f"### {strat}\n")
        lines.append("| Mode | PnL | Return% | MaxDD% | Sharpe | WR | PF | Trades | Skipped% |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(cfgs, key=lambda x: x["mode"]):
            pf = r.get("profit_factor") or "inf"
            lines.append(
                f"| {r['mode']} | ${r['total_pnl']:,.0f} | {r['return_pct']:.1f}% | "
                f"{r['max_dd_pct']:.1f}% | {r['sharpe_per_trade']:.4f} | "
                f"{100*r['win_rate']:.1f}% | {pf} | {r['n_trades_taken']:,} | {r.get('skip_pct',0):.1f}% |")
        lines.append("")
    
    # Average improvement summary
    lines.append("## Average Improvement vs Baseline\n")
    lines.append("| Mode | Avg ΔPnL% | Avg ΔMaxDD% | Avg ΔSharpe | Median ΔMaxDD% |")
    lines.append("|---|---|---|---|---|")
    
    for cfg in configs[1:]:
        dp_list, dd_list, ds_list = [], [], []
        for strat, cfgs in by_strat.items():
            bl = next((r for r in cfgs if r["mode"] == "fixed"), None)
            tr = next((r for r in cfgs if r["mode"] == cfg["mode"]), None)
            if bl and tr:
                dp_list.append(tr["return_pct"] - bl["return_pct"])
                dd_list.append(tr["max_dd_pct"] - bl["max_dd_pct"])
                ds_list.append(tr["sharpe_per_trade"] - bl["sharpe_per_trade"])
        if dp_list:
            lines.append(f"| {cfg['label']} | {sum(dp_list)/len(dp_list):+.1f}% | {sum(dd_list)/len(dd_list):+.1f}% | {sum(ds_list)/len(ds_list):+.4f} | {sorted(dd_list)[len(dd_list)//2]:+.1f}% |")
    
    lines.append("\n## Recommendations\n")
    fixed_results = [r for r in results if r.get("mode") == "fixed" and "error" not in r]
    if fixed_results:
        best_fixed = max(fixed_results, key=lambda r: r.get("total_pnl", 0))
        lines.append(f"- **Best baseline**: {best_fixed['strategy']} — PnL=${best_fixed['total_pnl']:,.0f}, MaxDD={best_fixed['max_dd_pct']:.1f}%")
    else:
        best_fixed = None
        lines.append("- No valid baseline results found")
    
    combined_results = [r for r in results if r.get("mode") == "combined" and "error" not in r]
    if combined_results and best_fixed:
        best_c = max(combined_results, key=lambda r: r.get("total_pnl", 0))
        dd_red = best_fixed["max_dd_pct"] - best_c["max_dd_pct"]
        pnl_ret = 100 * best_c["total_pnl"] / best_fixed["total_pnl"] if best_fixed["total_pnl"] > 0 else 0
        lines.append(f"- **Best combined**: {best_c['strategy']} — PnL=${best_c['total_pnl']:,.0f} ({pnl_ret:.0f}% of baseline), MaxDD={best_c['max_dd_pct']:.1f}%")
        lines.append(f"- **Drawdown reduction**: {dd_red:+.1f}% points")
    
    with open(os.path.join(OUTPUT_DIR, "drawdown_study.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(open(os.path.join(OUTPUT_DIR, "drawdown_study.md")).read())
    print(f"{'='*80}")
    print(f"\nTotal: {elapsed:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
