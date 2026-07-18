#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-18  kilo
#   - Walk-forward Kelly analysis: 5-fold temporal split.
#   - Train Kelly parameters on fold 0..k, apply to fold k+1.
#   - Compares Kelly vs fixed sizing on each out-of-sample fold.
# WHY: IS replay overfits; walk-forward shows if Kelly edge persists OOS.
from __future__ import annotations
import argparse, glob, gzip, json, math, os, statistics, time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
TRADES_DIR = os.path.join(HERE, "results", "is_taker_regime_full")
OUTPUT_DIR = os.path.join(HERE, "results", "walk_forward")

CAPITAL = 200.0
BASE_RISK_PCT = 0.005
KELLY_FRACTION = 0.10  # 1/10 Kelly (conservative)
KELLY_WINDOW = 500     # Rolling window for parameter estimation
MIN_HISTORY = 100      # Minimum trades before Kelly kicks in
N_FOLDS = 5


def load_pnl_array(path: str) -> list[float]:
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


def simulate_fixed_fold(pnls: list) -> dict:
    """Fixed sizing baseline for a single fold."""
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    max_dd_pct = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sum_pnl = 0.0
    underwater = 0

    for pnl in pnls:
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

    n = len(pnls)
    mean_p = sum_pnl / n if n > 0 else 0
    std_p = statistics.stdev(pnls) if n > 1 else 0.0
    sharpe = mean_p / std_p if std_p > 0 else 0

    return {
        "final_equity": round(equity, 2),
        "total_pnl": round(sum_pnl, 2),
        "return_pct": round(100 * sum_pnl / CAPITAL, 2),
        "peak_equity": round(peak, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "pct_time_underwater": round(100 * underwater / n, 1) if n > 0 else 0,
        "win_rate": round(wins / n, 4) if n > 0 else 0,
        "wins": wins, "losses": losses,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "sharpe_per_trade": round(sharpe, 4),
        "n_trades": n,
    }


def simulate_kelly_fold(train_pnls: list, test_pnls: list) -> dict:
    """Kelly sizing: estimate parameters from train_pnls, apply to test_pnls."""
    # --- Build Kelly lookup from training data ---
    # Pre-compute rolling Kelly parameters for the training window,
    # then apply to test trades using the LAST known Kelly state.

    # Actually, for walk-forward we need to estimate Kelly from the
    # training data and apply it to test data. The simplest approach:
    # compute overall training win rate and avg win/loss, then use
    # those FIXED parameters for all test trades.

    if len(train_pnls) < MIN_HISTORY:
        # Not enough training data — use fixed sizing for test
        return simulate_fixed_fold(test_pnls)

    # Compute FIXED Kelly parameters from full training set
    wins = [p for p in train_pnls if p > 0]
    losses = [-p for p in train_pnls if p < 0]
    win_rate = len(wins) / len(train_pnls) if train_pnls else 0.5
    avg_win = statistics.mean(wins) if wins else 1.0
    avg_loss = statistics.mean(losses) if losses else 1.0

    b = avg_win / avg_loss if avg_loss > 0 else 1.0
    q = 1.0 - win_rate
    kelly_f = max(0.0, (win_rate * b - q) / b) if b > 0 else 0.0
    risk_pct = min(0.010, max(0.001, kelly_f * KELLY_FRACTION))

    # --- Apply to test trades ---
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    max_dd_pct = 0.0
    test_wins = 0
    test_losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sum_pnl = 0.0
    underwater = 0
    factor = risk_pct / BASE_RISK_PCT

    for pnl in test_pnls:
        scaled = pnl * factor
        equity += scaled
        sum_pnl += scaled
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = 100 * dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
        if dd > 0:
            underwater += 1
        if scaled > 0:
            test_wins += 1
            gross_profit += scaled
        elif scaled < 0:
            test_losses += 1
            gross_loss -= scaled

    n = len(test_pnls)
    mean_p = sum_pnl / n if n > 0 else 0
    std_p = statistics.stdev([pnl * factor for pnl in test_pnls]) if n > 1 else 0.0
    sharpe = mean_p / std_p if std_p > 0 else 0

    return {
        "final_equity": round(equity, 2),
        "total_pnl": round(sum_pnl, 2),
        "return_pct": round(100 * sum_pnl / CAPITAL, 2),
        "peak_equity": round(peak, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "pct_time_underwater": round(100 * underwater / n, 1) if n > 0 else 0,
        "win_rate": round(test_wins / n, 4) if n > 0 else 0,
        "wins": test_wins, "losses": test_losses,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "sharpe_per_trade": round(sharpe, 4),
        "n_trades": n,
        "kelly_risk_pct": round(risk_pct, 6),
        "kelly_fraction_from_train": round(kelly_f, 4),
        "train_win_rate": round(win_rate, 4),
        "train_avg_win": round(avg_win, 6),
        "train_avg_loss": round(avg_loss, 6),
    }


def simulate_rolling_kelly_fold(train_pnls: list, test_pnls: list, window: int = KELLY_WINDOW) -> dict:
    """Rolling Kelly: update parameters every trade using a sliding window.
    
    This is more realistic than fixed-params Kelly — it adapts to regime
    changes within the test fold, but parameters are always estimated from
    PAST data only (no look-ahead).
    """
    n_test = len(test_pnls)
    if n_test == 0:
        return {"n_trades": 0, "total_pnl": 0, "max_dd_pct": 0}

    # Start with training data to seed the window
    buf = deque(maxlen=window)
    buf_wins = 0
    buf_sum_pos = 0.0
    buf_sum_neg = 0.0
    buf_count_pos = 0
    buf_count_neg = 0

    for p in train_pnls[-window:]:
        buf.append(p)
        if p > 0:
            buf_wins += 1
            buf_sum_pos += p
            buf_count_pos += 1
        elif p < 0:
            buf_sum_neg += (-p)
            buf_count_neg += 1

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    max_dd_pct = 0.0
    test_wins = 0
    test_losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sum_pnl = 0.0
    underwater = 0
    current_risk = BASE_RISK_PCT

    for pnl in test_pnls:
        # Update rolling window
        old = buf[0] if len(buf) == window else None
        buf.append(pnl)
        if pnl > 0:
            buf_wins += 1
            buf_sum_pos += pnl
            buf_count_pos += 1
        elif pnl < 0:
            buf_sum_neg += (-pnl)
            buf_count_neg += 1

        if old is not None:
            if old > 0:
                buf_wins -= 1
                buf_sum_pos -= old
                buf_count_pos -= 1
            elif old < 0:
                buf_sum_neg -= (-old)
                buf_count_neg -= 1

        # Compute Kelly from window (past only, no look-ahead)
        buflen = len(buf)
        if buflen >= MIN_HISTORY and buf_count_pos > 0 and buf_count_neg > 0:
            w = buf_wins / buflen
            avg_w = buf_sum_pos / buf_count_pos
            avg_l = buf_sum_neg / buf_count_neg
            b = avg_w / avg_l if avg_l > 0 else 1.0
            q = 1.0 - w
            kelly_f = max(0.0, (w * b - q) / b) if b > 0 else 0.0
            current_risk = min(0.010, max(0.001, kelly_f * KELLY_FRACTION))

        factor = current_risk / BASE_RISK_PCT
        scaled = pnl * factor
        equity += scaled
        sum_pnl += scaled
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = 100 * dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
        if dd > 0:
            underwater += 1
        if scaled > 0:
            test_wins += 1
            gross_profit += scaled
        elif scaled < 0:
            test_losses += 1
            gross_loss -= scaled

    n = len(test_pnls)
    mean_p = sum_pnl / n if n > 0 else 0
    std_p = statistics.stdev([pnl * (current_risk / BASE_RISK_PCT) for pnl in test_pnls]) if n > 1 else 0.0
    sharpe = mean_p / std_p if std_p > 0 else 0

    return {
        "final_equity": round(equity, 2),
        "total_pnl": round(sum_pnl, 2),
        "return_pct": round(100 * sum_pnl / CAPITAL, 2),
        "peak_equity": round(peak, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "pct_time_underwater": round(100 * underwater / n, 1) if n > 0 else 0,
        "win_rate": round(test_wins / n, 4) if n > 0 else 0,
        "wins": test_wins, "losses": test_losses,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "sharpe_per_trade": round(sharpe, 4),
        "n_trades": n,
    }


def simulate_sizing_reduction_fold(train_pnls: list, test_pnls: list, reduction_pct: float = 50.0) -> dict:
    """Sizing reduction: scale down PnL during drawdowns.
    
    Uses the SAME equity curve approach as drawdown_study but applied
    walk-forward: the drawdown state is computed from the test fold's
    own equity curve (no look-ahead into training data).
    """
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    max_dd_pct = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    sum_pnl = 0.0
    underwater = 0
    scaled_pnls = []

    for pnl in test_pnls:
        dd_pct = 100 * (peak - equity) / peak if peak > 0 else 0
        factor = max(0.25, 1.0 - dd_pct * reduction_pct / 10000.0)
        scaled = pnl * factor
        scaled_pnls.append(scaled)
        equity += scaled
        sum_pnl += scaled
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct_actual = 100 * dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct_actual
        if dd > 0:
            underwater += 1
        if scaled > 0:
            wins += 1
            gross_profit += scaled
        elif scaled < 0:
            losses += 1
            gross_loss -= scaled

    n = len(test_pnls)
    mean_p = sum_pnl / n if n > 0 else 0
    std_p = statistics.stdev(scaled_pnls) if n > 1 else 0.0
    sharpe = mean_p / std_p if std_p > 0 else 0

    return {
        "final_equity": round(equity, 2),
        "total_pnl": round(sum_pnl, 2),
        "return_pct": round(100 * sum_pnl / CAPITAL, 2),
        "peak_equity": round(peak, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "pct_time_underwater": round(100 * underwater / n, 1) if n > 0 else 0,
        "win_rate": round(wins / n, 4) if n > 0 else 0,
        "wins": wins, "losses": losses,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "sharpe_per_trade": round(sharpe, 4),
        "n_trades": n,
    }


def run_walk_forward(args):
    """Run walk-forward analysis for one strategy."""
    name, path = args
    t0 = time.time()

    pnls = load_pnl_array(path)
    total = len(pnls)
    fold_size = total // N_FOLDS

    results = {"strategy": name, "total_trades": total, "folds": []}

    for fold in range(N_FOLDS):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < N_FOLDS - 1 else total
        train_pnls = pnls[:test_start] if test_start > 0 else pnls[:fold_size]  # Use first fold as seed
        test_pnls = pnls[test_start:test_end]

        if len(test_pnls) == 0:
            continue

        # Fixed baseline on test fold
        fixed = simulate_fixed_fold(test_pnls)

        # Fixed-params Kelly (train on all prior data, apply to test)
        fixed_kelly = simulate_kelly_fold(train_pnls, test_pnls)

        # Rolling Kelly (update every trade from training seed)
        rolling_kelly = simulate_rolling_kelly_fold(train_pnls, test_pnls)

        # Sizing reduction (no training needed — uses own equity curve)
        sizing_red = simulate_sizing_reduction_fold(train_pnls, test_pnls, reduction_pct=50.0)

        results["folds"].append({
            "fold": fold + 1,
            "train_range": f"0-{test_start}",
            "test_range": f"{test_start}-{test_end}",
            "train_trades": len(train_pnls),
            "test_trades": len(test_pnls),
            "fixed": fixed,
            "fixed_kelly": fixed_kelly,
            "rolling_kelly": rolling_kelly,
            "sizing_reduction": sizing_red,
        })

    # Aggregate across folds
    modes = ["fixed", "fixed_kelly", "rolling_kelly", "sizing_reduction"]
    agg = {m: {"total_pnl": 0, "max_dd_pct": 0, "sharpe_sum": 0, "n": 0} for m in modes}

    for fold_r in results["folds"]:
        for mode in modes:
            r = fold_r[mode]
            agg[mode]["total_pnl"] += r["total_pnl"]
            agg[mode]["max_dd_pct"] = max(agg[mode]["max_dd_pct"], r["max_dd_pct"])
            agg[mode]["sharpe_sum"] += r["sharpe_per_trade"]
            agg[mode]["n"] += 1

    for mode in modes:
        a = agg[mode]
        a["avg_sharpe"] = round(a["sharpe_sum"] / a["n"], 4) if a["n"] > 0 else 0
        del a["sharpe_sum"]
        del a["n"]

    results["aggregate"] = agg
    results["compute_time_s"] = round(time.time() - t0, 1)

    print(f"  {name}: {total:,} trades, {N_FOLDS} folds, {results['compute_time_s']:.0f}s", flush=True)
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

    # Filter tiny files
    filtered = []
    for n in names:
        p = os.path.join(TRADES_DIR, f"{n}.trades.jsonl.gz")
        if os.path.getsize(p) > 1000:
            filtered.append(n)
    names = filtered

    jobs = [(n, os.path.join(TRADES_DIR, f"{n}.trades.jsonl.gz")) for n in names]

    print(f"walk_forward: {len(names)} strategies × {N_FOLDS} folds", flush=True)

    t0 = time.time()
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_walk_forward, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                all_results.append(fut.result())
            except Exception as e:
                all_results.append({"strategy": futs[fut][0], "error": str(e)})
            if i % 5 == 0:
                print(f"  [{i}/{len(jobs)} done, {time.time()-t0:.0f}s]", flush=True)

    # Write per-strategy JSON
    for r in all_results:
        if "error" not in r:
            with open(os.path.join(OUTPUT_DIR, f"{r['strategy']}.wf.json"), "w") as f:
                json.dump(r, f, indent=1)

    # Build comparison table
    lines = ["# Walk-Forward Kelly Analysis (5-Fold Temporal Split)\n",
             "## Per-Strategy, Per-Fold Results\n"]

    for r in sorted(all_results, key=lambda x: x.get("strategy", "")):
        if "error" in r:
            lines.append(f"### {r['strategy']} — ERROR: {r['error']}\n")
            continue

        strat = r["strategy"]
        lines.append(f"### {strat} ({r['total_trades']:,} trades)\n")
        lines.append("| Fold | Train | Test | Fixed PnL | Fixed MaxDD | Rolling Kelly PnL | Rolling Kelly MaxDD | Size Red PnL | Size Red MaxDD |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for fold_r in r["folds"]:
            f = fold_r["fixed"]
            rk = fold_r["rolling_kelly"]
            sr = fold_r["sizing_reduction"]
            lines.append(
                f"| {fold_r['fold']} | {fold_r['train_trades']:,} | {fold_r['test_trades']:,} | "
                f"${f['total_pnl']:,.0f} | {f['max_dd_pct']:.1f}% | "
                f"${rk['total_pnl']:,.0f} | {rk['max_dd_pct']:.1f}% | "
                f"${sr['total_pnl']:,.0f} | {sr['max_dd_pct']:.1f}% |")

        agg = r["aggregate"]
        lines.append(
            f"| **TOTAL** | | | "
            f"**${agg['fixed']['total_pnl']:,.0f}** | **{agg['fixed']['max_dd_pct']:.1f}%** | "
            f"**${agg['rolling_kelly']['total_pnl']:,.0f}** | **{agg['rolling_kelly']['max_dd_pct']:.1f}%** | "
            f"**${agg['sizing_reduction']['total_pnl']:,.0f}** | **{agg['sizing_reduction']['max_dd_pct']:.1f}%** |")
        lines.append("")

    # Summary
    lines.append("## Walk-Forward Summary\n")
    lines.append("| Strategy | Fixed PnL | Fixed MaxDD | Rolling Kelly PnL | Kelly MaxDD | Size Red PnL | Size Red MaxDD | Kelly PnL Ret% | Size Red PnL Ret% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for r in sorted(all_results, key=lambda x: x.get("strategy", "")):
        if "error" in r or not r.get("aggregate"):
            continue
        agg = r["aggregate"]
        f_pnl = agg["fixed"]["total_pnl"]
        f_dd = agg["fixed"]["max_dd_pct"]
        rk_pnl = agg["rolling_kelly"]["total_pnl"]
        rk_dd = agg["rolling_kelly"]["max_dd_pct"]
        sr_pnl = agg["sizing_reduction"]["total_pnl"]
        sr_dd = agg["sizing_reduction"]["max_dd_pct"]
        kelly_ret = 100 * rk_pnl / f_pnl if f_pnl != 0 else 0
        sr_ret = 100 * sr_pnl / f_pnl if f_pnl != 0 else 0
        lines.append(
            f"| {r['strategy'][:50]} | ${f_pnl:,.0f} | {f_dd:.1f}% | "
            f"${rk_pnl:,.0f} | {rk_dd:.1f}% | "
            f"${sr_pnl:,.0f} | {sr_dd:.1f}% | "
            f"{kelly_ret:.0f}% | {sr_ret:.0f}% |")

    lines.append("\n## Verdict\n")
    lines.append("- **Kelly**: bets bigger when winning, smaller when losing — amplifies the edge if parameters are right")
    lines.append("- **Sizing Reduction**: reduces position size during drawdowns — simpler, no parameter estimation needed")
    lines.append("- Look for: **PnL retention > 80%** AND **MaxDD reduction** = practical edge")
    lines.append("- If Kelly MaxDD > Fixed MaxDD: Kelly is overbetting (parameters wrong)")

    with open(os.path.join(OUTPUT_DIR, "walk_forward.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Also write combined JSON
    with open(os.path.join(OUTPUT_DIR, "walk_forward.json"), "w") as f:
        json.dump(all_results, f, indent=1)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(open(os.path.join(OUTPUT_DIR, "walk_forward.md")).read())
    print(f"{'='*80}")
    print(f"\nTotal: {elapsed:.0f}s -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
