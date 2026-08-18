# CHANGE_SUMMARY
# 2026-08-18  kilo
#   - Created strategies/backtest/dmc_sweep.py: parameter sweep runner for
#     dumb_money_concepts. Iterates a grid of tunables, runs each (config,
#     symbol) through the harness on the in-sample window, and writes a
#     summary CSV plus per-config trade/metrics files.
# WHY: Find DMC configurations that push win rate toward 90%+ while keeping
#      expectancy positive for prop-firm payout modelling.

"""In-sample parameter sweep for Dumb Money Concepts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # /config
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
sys.path.insert(0, "/config/topstep-strats")

from strategies.backtest.engine import StrategyHarness
from strategies.signals import dumb_money_concepts as dmc
from topstep_strats.backtest import run_backtest
from topstep_strats.metrics import calculate_metrics

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dmc_sweep")

POINT_VALUES = {"NQ": 20.0, "ES": 50.0, "YM": 5.0}
DEFAULT_PIP_VALUES = {"NQ": 1.0, "ES": 0.2, "YM": 2.0}

# Tunable grid. RR targets and dollar targets are included so we can push
# win rate toward 90%+ while measuring expectancy across a wide target space.
PARAM_GRID = {
    "level_test_atr_frac": [0.10, 0.20, 0.30],
    "retest_atr_frac": [0.05, 0.10, 0.15],
    "sl_buffer_frac": [0.05, 0.10, 0.20],
    "swing_window": [2, 3],
    "rejection_strictness": ["body", "wick"],
    "target_type": [
        "origin", "level_mid",
        "fixed_0.25r", "fixed_0.33r", "fixed_0.5r", "fixed_0.75r",
        "fixed_1r", "fixed_1.5r", "fixed_2r", "fixed_3r",
        "dollar_25", "dollar_50", "dollar_75", "dollar_100", "dollar_150",
        "dollar_200", "dollar_300", "dollar_500", "dollar_750", "dollar_1000",
    ],
    "one_test_rule": [True, False],
}


def load_1m(csv: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    ts_col = "timestamp" if "timestamp" in df.columns else "ts"
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.loc[start:end]
    df = df[~df.index.duplicated(keep="last")]
    return df


def _config_id(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def to_signals_df(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df[["entry_time", "direction", "entry_price", "stop_loss", "take_profit",
               "exit_time", "exit_price", "pnl", "exit_reason"]]


def compute_metrics(trades: list[dict], point_value: float, n_mc: int = 2000, n_boot: int = 2000) -> dict:
    if not trades:
        return {
            "basic": {
                "start_equity": 100_000.0, "end_equity": 100_000.0,
                "total_return": 0.0, "cagr": 0.0, "sharpe_ratio": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "n_trades": 0,
            }
        }
    signals = to_signals_df(trades)
    params = {
        "initial_capital": 100_000.0,
        "point_value": point_value,
        "slippage": 0.0,
        "commission": 0.0,
        "topstep": {"enabled": False},
    }
    result = run_backtest(signals, params)
    return calculate_metrics(result, n_mc=n_mc, n_boot=n_boot, random_state=42)


def run_config(args: tuple) -> dict:
    symbol, csv, params, start, end, outdir, n_mc, n_boot = args
    cfg_id = _config_id(params)
    tag = f"{symbol}_dmc_{cfg_id}"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dmc.set_params(params)
    dmc.reset_state()

    point_value = POINT_VALUES[symbol.upper()]
    pip_value = DEFAULT_PIP_VALUES[symbol.upper()]
    df = load_1m(csv, start, end)

    scratch = Path(tempfile.mkdtemp(prefix=f"dmc_{cfg_id}_{symbol}_"))
    harness = StrategyHarness(
        strategy="dumb_money_concepts",
        symbol=symbol,
        point_value=point_value,
        pip_value=pip_value,
        max_reentries=0,
        scratch_root=scratch,
    )
    t0 = time.time()
    trades = harness.run(df)
    elapsed = time.time() - t0

    metrics = compute_metrics(trades, point_value, n_mc, n_boot)
    basic = metrics["basic"]

    # Save trades.
    trades_path = outdir / f"{tag}_trades.csv"
    if trades:
        pd.DataFrame(trades).to_csv(trades_path, index=False)
    else:
        trades_path.write_text("")

    # Save metrics.
    metrics_path = outdir / f"{tag}_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    # Save params.
    params_path = outdir / f"{tag}_params.json"
    with open(params_path, "w") as fh:
        json.dump(params, fh, indent=2, default=str)

    return {
        "tag": tag,
        "symbol": symbol,
        "config_id": cfg_id,
        "params": params,
        "n_trades": len(trades),
        "win_rate": basic["win_rate"],
        "cagr": basic["cagr"],
        "sharpe": basic["sharpe_ratio"],
        "maxDD": basic["max_drawdown"],
        "total_ret": basic["total_return"],
        "elapsed": elapsed,
        "trades_path": str(trades_path),
        "metrics_path": str(metrics_path),
    }


def build_jobs(symbols, csv_map, grid, start, end, outdir, n_mc, n_boot):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    jobs = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        for sym in symbols:
            jobs.append((sym, csv_map[sym], params, start, end, outdir, n_mc, n_boot))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="NQ,ES,YM", help="comma-separated futures symbols")
    ap.add_argument("--csv-dir", default="/config/projects/trading/v5_orb_nq_backtest/market_data",
                    help="directory containing SYM_1min.csv files")
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--outdir", default="/tmp/dmc_sweep")
    ap.add_argument("--workers", type=int, default=0, help="0=auto")
    ap.add_argument("--n-mc", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--top-n", type=int, default=20, help="top configs to print")
    ap.add_argument("--min-win-rate", type=float, default=0.0, help="filter report to configs >= this WR")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    csv_map = {s: str(Path(args.csv_dir) / f"{s}_1min.csv") for s in symbols}

    jobs = build_jobs(symbols, csv_map, PARAM_GRID, args.start, args.end, args.outdir, args.n_mc, args.n_boot)
    print(f"DMC sweep: {len(jobs)} jobs ({len(PARAM_GRID)} param combos x {len(symbols)} symbols)")

    workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)
    t0 = time.time()
    if workers == 1:
        results = [run_config(j) for j in jobs]
    else:
        with mp.Pool(workers) as pool:
            results = pool.map(run_config, jobs)
    print(f"Sweep finished in {time.time() - t0:.1f}s")

    outdir = Path(args.outdir)
    df = pd.DataFrame(results)
    summary = outdir / "summary.csv"
    df.to_csv(summary, index=False)
    print(f"Summary written to {summary}")

    # Report top configs by a prop-payout-friendly score: positive return,
    # high win rate, controlled drawdown.
    df["score"] = (
        df["win_rate"].clip(lower=0) * 2.0
        + df["total_ret"].clip(lower=-1, upper=2)
        - df["maxDD"].abs() * 2.0
        + df["sharpe"].clip(lower=-2, upper=2) * 0.5
    )
    filtered = df[df["win_rate"] >= args.min_win_rate]
    top = filtered.sort_values("score", ascending=False).head(args.top_n)
    print(f"\nTop {args.top_n} configs (min WR {args.min_win_rate:.0%}):")
    print(top[["tag", "n_trades", "win_rate", "cagr", "sharpe", "maxDD", "total_ret", "score"]].to_string(index=False))

    # Best by win rate (for the 90%+ goal).
    best_wr = df.sort_values("win_rate", ascending=False).head(args.top_n)
    print(f"\nTop {args.top_n} by win rate:")
    print(best_wr[["tag", "n_trades", "win_rate", "cagr", "sharpe", "maxDD", "total_ret", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
