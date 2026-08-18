# CHANGE_SUMMARY
# 2026-08-18  kilo
#   - Created strategies/backtest/dmc_sweep_single.py: runs one DMC parameter
#     config on one symbol. Used by the GitHub Actions sweep so each worker can
#     call a simple CLI per job instead of importing the full sweep runner.
# WHY: Simplify per-job execution in the distributed GHA sweep.

"""Run a single DMC config/symbol backtest."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
sys.path.insert(0, "/config/topstep-strats")

from strategies.backtest.engine import StrategyHarness
from strategies.signals import dumb_money_concepts as dmc
from topstep_strats.backtest import run_backtest
from topstep_strats.metrics import calculate_metrics


def load_1m(csv: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    ts_col = "timestamp" if "timestamp" in df.columns else "ts"
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.loc[start:end]
    df = df[~df.index.duplicated(keep="last")]
    return df


def to_signals_df(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df[["entry_time", "direction", "entry_price", "stop_loss", "take_profit",
               "exit_time", "exit_price", "pnl", "exit_reason"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--params", required=True, help="JSON dict of DMC parameters")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-mc", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    params = json.loads(args.params)
    dmc.set_params(params)
    dmc.reset_state()

    point_values = {"NQ": 20.0, "ES": 50.0, "YM": 5.0}
    point_value = point_values[args.symbol.upper()]

    df = load_1m(args.csv, args.start, args.end)
    tag = f"{args.symbol.upper()}_dmc_{args.config_id}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scratch = Path(tempfile.mkdtemp(prefix=f"dmc_{args.config_id}_{args.symbol}_"))
    harness = StrategyHarness(
        strategy="dumb_money_concepts",
        symbol=args.symbol,
        point_value=point_value,
        max_reentries=0,
        scratch_root=scratch,
    )
    t0 = time.time()
    trades = harness.run(df)
    elapsed = time.time() - t0

    if trades:
        signals = to_signals_df(trades)
        bt_params = {
            "initial_capital": 100_000.0,
            "point_value": point_value,
            "slippage": 0.0,
            "commission": 0.0,
            "topstep": {"enabled": False},
        }
        result = run_backtest(signals, bt_params)
        metrics = calculate_metrics(result, n_mc=args.n_mc, n_boot=args.n_boot, random_state=42)
    else:
        metrics = {
            "basic": {
                "start_equity": 100_000.0, "end_equity": 100_000.0,
                "total_return": 0.0, "cagr": 0.0, "sharpe_ratio": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "n_trades": 0,
            }
        }

    if trades:
        pd.DataFrame(trades).to_csv(outdir / f"{tag}_trades.csv", index=False)
    else:
        (outdir / f"{tag}_trades.csv").write_text("")
    with open(outdir / f"{tag}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    with open(outdir / f"{tag}_params.json", "w") as fh:
        json.dump(params, fh, indent=2, default=str)

    basic = metrics["basic"]
    print(f"[{tag}] trades={len(trades)} wr={basic['win_rate']:.3f} cagr={basic['cagr']:.3f} "
          f"sharpe={basic['sharpe_ratio']:.2f} maxdd={basic['max_drawdown']:.3f} "
          f"ret={basic['total_return']:.3f} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
