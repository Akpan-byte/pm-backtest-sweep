#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-18  kilo
#   - Entry timing sweep: tests different opening_window_sec values.
#   - Loads 2k sample once, runs all window configs for one strategy.
#   - Outputs JSON results for aggregation.
# WHY: Test whether waiting N seconds after market open improves fills.
"""Entry timing sweep. Usage:
  python3 sweep_entry_timing.py --strategy tf_dema_lb20_dev002_emax85_alp0001 --subsample 5
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import driver  # noqa: E402
from engine.portfolio import Portfolio  # noqa: E402

# Opening window durations to test (seconds)
WINDOWS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]

# Load registry entries
SWEEP_REGISTRY = {}
for reg_path in ("trend_sweep_registry.json", "trend_regime_registry.json",
                 os.environ.get("BT_EXTRA_STRATEGIES", "")):
    if reg_path and os.path.exists(reg_path):
        with open(reg_path) as fh:
            SWEEP_REGISTRY.update(json.load(fh))


def load_2k_files() -> list[str]:
    """Load the is_files_2k.txt list."""
    path = os.path.join(HERE, "is_files_2k.txt")
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def load_compact(path: str):
    """Load a compact .pkl.gz file."""
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def oos_skip(arr: dict) -> bool:
    """Skip markets that start in the OOS window."""
    from datetime import date, timezone
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    OOS_START = date(2026, 7, 1)
    t_arr = arr.get("t")
    if t_arr is None or len(t_arr) == 0:
        return True
    from datetime import datetime
    try:
        dt = datetime.fromtimestamp(t_arr[0] / 1000.0, tz=ET)
        if dt.date() >= OOS_START:
            return True
    except Exception:
        pass
    return False


def run_one_window(name: str, reg: dict, files: list[str], window_sec: float,
                   subsample: int = 1) -> dict:
    """Run one strategy with one opening_window_sec setting."""
    import copy
    reg_copy = copy.deepcopy(reg)
    reg_copy["opening_window_sec"] = window_sec

    fn = driver.load_signal_fn(reg_copy)
    pf = Portfolio(name=f"sweep:{name}_w{window_sec}", capital=driver.CAPITAL)
    n_markets = 0
    n_signals = 0
    n_triggered = 0
    t0 = time.time()

    for i, f in enumerate(files):
        if pf.cash < 0.25 and not pf.active_trades:
            break
        snaps = load_compact(f)
        if oos_skip(snaps):
            del snaps
            continue
        n_markets += 1
        # Subsample: keep every Nth snapshot
        if subsample > 1:
            t_arr = snaps["t"]
            n = len(t_arr)
            keep_idx = list(range(0, n, subsample))
            # Always keep last snapshot for expiry
            if (n - 1) not in keep_idx:
                keep_idx.append(n - 1)
            for key in ("t", "btc", "pu", "pd", "ua", "ub", "da", "db"):
                if key in snaps and isinstance(snaps[key], list):
                    snaps[key] = [snaps[key][j] for j in keep_idx]
                elif key in snaps and hasattr(snaps[key], '__getitem__'):
                    try:
                        snaps[key] = [snaps[key][j] for j in keep_idx]
                    except (IndexError, TypeError):
                        pass
        result = driver.run_market_taker_arr(snaps, reg_copy, fn, pf=pf)
        n_signals += result.get("n_signals", 0)
        n_triggered += result.get("n_triggered", 0)
        del snaps

    closed = pf.closed_trades
    total_pnl = sum(t.pnl for t in closed)
    committed = sum(t.entry_notional for t in pf.active_trades.values())

    # Win/loss stats
    wins = sum(1 for t in closed if t.pnl > 0)
    losses = sum(1 for t in closed if t.pnl <= 0)
    win_rate = wins / len(closed) if closed else 0

    # Entry price stats
    entry_prices = [t.entry_price for t in closed if t.entry_price > 0]
    avg_entry = sum(entry_prices) / len(entry_prices) if entry_prices else 0

    # Max drawdown
    equity = driver.CAPITAL
    peak = driver.CAPITAL
    max_dd = 0
    for t in closed:
        equity += t.pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        "strategy": name,
        "opening_window_sec": window_sec,
        "subsample": subsample,
        "n_markets": n_markets,
        "n_trades": len(closed),
        "n_signals": n_signals,
        "n_triggered": n_triggered,
        "total_pnl": round(total_pnl, 4),
        "cash": round(pf.cash, 4),
        "equity": round(pf.cash + committed, 4),
        "win_rate": round(win_rate, 4),
        "wins": wins,
        "losses": losses,
        "avg_entry_price": round(avg_entry, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "runtime_s": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Entry timing sweep")
    parser.add_argument("--strategy", required=True, help="Strategy name from registry")
    parser.add_argument("--subsample", type=int, default=5, help="Snapshot subsample (1=full)")
    parser.add_argument("--windows", default=None, help="Comma-separated windows (default: all)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    windows = WINDOWS
    if args.windows:
        windows = [float(x) for x in args.windows.split(",")]

    # Load registry
    if args.strategy not in SWEEP_REGISTRY:
        print(f"ERROR: strategy '{args.strategy}' not in registry", file=sys.stderr)
        print(f"Available: {sorted(SWEEP_REGISTRY.keys())[:10]}...", file=sys.stderr)
        sys.exit(1)

    reg = SWEEP_REGISTRY[args.strategy]
    print(f"Strategy: {args.strategy}")
    print(f"Windows: {windows}")
    print(f"Subsample: {args.subsample}x")
    print()

    # Load file list
    files = load_2k_files()
    print(f"Loaded {len(files)} market files")

    # Run all windows
    results = []
    for w in windows:
        print(f"  Window {w:>4.0f}s ... ", end="", flush=True)
        r = run_one_window(args.strategy, reg, files, w, args.subsample)
        results.append(r)
        print(f"PnL=${r['total_pnl']:>8.2f}  WR={r['win_rate']*100:.1f}%  "
              f"DD={r['max_dd_pct']:.1f}%  trades={r['n_trades']}  "
              f"entry={r['avg_entry_price']:.4f}  {r['runtime_s']:.1f}s")

    # Save results
    out_path = args.output or os.path.join(HERE, f"sweep_{args.strategy}.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'Window':>8} {'PnL':>10} {'WinRate':>8} {'MaxDD':>8} {'Trades':>7} {'AvgEntry':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['opening_window_sec']:>7.0f}s ${r['total_pnl']:>9.2f} {r['win_rate']*100:>7.1f}% "
              f"{r['max_dd_pct']:>7.1f}% {r['n_trades']:>7} {r['avg_entry_price']:>10.4f}")


if __name__ == "__main__":
    main()
