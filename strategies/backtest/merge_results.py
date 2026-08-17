# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created strategies/backtest/merge_results.py: post-processing that scans a
#     results dir for per-(strategy,symbol) trades CSV + metrics JSON from the
#     12 GHA backtests, builds a combined manifest, and recomputes metrics for
#     the 4 instrument combos by concatenating symbol-tagged trades.
# WHY: Produce the 4 combo reports (all, equity, index, micro) without
#      re-running signals; only the metric engine is re-invoked on merged trades.
"""Merge 12 backtest outputs into a manifest + 4 instrument-combo reports."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import OrderedDict
from pathlib import Path

import os

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
sys.path.insert(0, os.environ.get("TOPSTEP_STRATS_DIR", "/config/topstep-strats"))
from topstep_strats.backtest import run_backtest  # noqa: E402
from topstep_strats.metrics import calculate_metrics  # noqa: E402

POINT_VALUES = {
    "NQ": 20.0, "ES": 50.0, "YM": 5.0,
    "GC": 10.0, "SI": 25.0,
    "BTC": 1.0, "ETH": 1.0, "SOL": 1.0,
}

COMBOS = {
    "all": ["NQ", "ES", "YM", "GC", "SI", "BTC", "ETH", "SOL"],
    "equity_futures": ["NQ", "ES", "YM"],
    "metals": ["GC", "SI"],
    "crypto": ["BTC", "ETH", "SOL"],
    "nq_es": ["NQ", "ES"],
    "nq_ym": ["NQ", "YM"],
    "es_ym": ["ES", "YM"],
}


def load_metrics(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _window_suffix(tag: str) -> str:
    for suffix in ("_IS", "_OOS", "_FULL"):
        if tag.endswith(suffix):
            return suffix.lstrip("_")
    return "SINGLE"


def compute_combo(df: pd.DataFrame, n_mc: int, n_boot: int) -> dict:
    if df.empty:
        return {"basic": {"win_rate": 0.0, "cagr": 0.0, "sharpe_ratio": 0.0,
                          "max_drawdown": 0.0, "total_return": 0.0}}
    # Pooled point value: use a per-trade point value column (trades carry the
    # symbol; metrics need a single point_value).  For cross-symbol combos we
    # value each symbol in dollars per contract via its own point value and
    # aggregate as a dollar portfolio (not a single-instrument curve).
    out = {}
    for name, symbols in COMBOS.items():
        sub = df[df["symbol"].isin(symbols)]
        if sub.empty:
            continue
        frames = []
        for sym, grp in sub.groupby("symbol"):
            f = grp.copy()
            f["_point_value"] = POINT_VALUES[sym]
            frames.append(f)
        merged = pd.concat(frames, ignore_index=True)
        # Normalize: convert each symbol's price-space pnl into points at its
        # own point value, then value the whole pool with a reference point
        # value of 1.0 (dollar PnL per unit).
        merged["pnl_dollar"] = merged["pnl"] * merged["_point_value"]
        sig = merged[["entry_time", "direction", "entry_price", "stop_loss",
                      "take_profit", "exit_time", "exit_price", "pnl_dollar",
                      "exit_reason"]].rename(columns={"pnl_dollar": "pnl"})
        params = {
            "initial_capital": 100_000.0,
            "point_value": 1.0,
            "slippage": 0.0,
            "commission": 0.0,
            "topstep": {"enabled": False},
        }
        res = run_backtest(sig, params)
        out[name] = calculate_metrics(res, n_mc=n_mc, n_boot=n_boot, random_state=42)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir with *_trades.csv + *_metrics.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-mc", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_files = sorted(glob.glob(str(results / "*_metrics.json")))
    # Skip the *_rth_* / *_overnight_* variants for the 12 core backtests.
    core = [p for p in metrics_files
            if not any(x in Path(p).stem for x in ("_rth_", "_overnight_"))]
    manifest = OrderedDict()
    for p in core:
        m = load_metrics(p)
        tag = Path(p).stem.replace("_metrics", "")
        manifest[tag] = {
            "metrics_path": p,
            "n_trades": m["basic"].get("n_trades", 0),
            "win_rate": m["basic"].get("win_rate"),
            "cagr": m["basic"].get("cagr"),
            "sharpe": m["basic"].get("sharpe_ratio"),
            "max_drawdown": m["basic"].get("max_drawdown"),
            "total_return": m["basic"].get("total_return"),
        }

    # Load all core trades for combo recomputation, grouped by window suffix.
    trades_files = sorted(glob.glob(str(results / "*_trades.csv")))
    core_trades = [p for p in trades_files
                   if not any(x in Path(p).stem for x in ("_rth_", "_overnight_"))]
    trades_by_window: dict[str, list[pd.DataFrame]] = {}
    for p in core_trades:
        tag = Path(p).stem.replace("_trades", "")
        window = _window_suffix(tag)
        try:
            trades_by_window.setdefault(window, []).append(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            continue

    combos_by_window = {}
    for window, frames in trades_by_window.items():
        all_trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combos_by_window[window] = compute_combo(all_trades, args.n_mc, args.n_boot)

    report = {"manifest": manifest, "combos_by_window": combos_by_window}
    with open(out / "star_backtest_results.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    lines = ["# StarTrading YouTube Strategies Backtest Results\n"]
    lines.append("| tag | trades | win_rate | cagr | sharpe | maxDD | total_ret |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag, m in manifest.items():
        lines.append(f"| {tag} | {m['n_trades']} | {m['win_rate']:.3f} | {m['cagr']:.3f} "
                     f"| {m['sharpe']:.2f} | {m['max_drawdown']:.3f} | {m['total_return']:.3f} |")
    for window, combos in combos_by_window.items():
        lines.append(f"\n## Instrument Combos — {window}\n")
        lines.append("| combo | symbols | trades | win_rate | cagr | sharpe | maxDD | total_ret |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, m in combos.items():
            b = m["basic"]
            lines.append(f"| {name} | {','.join(COMBOS[name])} | {b.get('n_trades', 0)} "
                         f"| {b.get('win_rate', 0):.3f} | {b.get('cagr', 0):.3f} "
                         f"| {b.get('sharpe_ratio', 0):.2f} | {b.get('max_drawdown', 0):.3f} "
                         f"| {b.get('total_return', 0):.3f} |")
    with open(out / "star_backtest_results.md", "w") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote {out}/star_backtest_results.{{json,md}}")


if __name__ == "__main__":
    main()
