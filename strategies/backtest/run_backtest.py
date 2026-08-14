# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created strategies/backtest/run_backtest.py: CLI that loads 1m OHLCV,
#     runs one or all StarTrading strategies over an in-sample window, feeds the
#     tagged trades into the topstep-strats backtest engine + full metrics suite
#     (20k MC/bootstrap), and writes trades CSV + metrics JSON per (strategy,
#     symbol). Trades carry a `symbol` column so instrument combos can be built
#     by merging trade files without re-running signals.
# WHY: Production entry point for the 12 individual backtests (4 strategies x
#      3 instruments) run on GHA + Akpan laptop, plus small-slice validation.
"""CLI for running the StarTrading futures backtest harness."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import os

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # /config
sys.path.insert(0, os.environ.get("TOPSTEP_STRATS_DIR", "/config/topstep-strats"))

from strategies.backtest.engine import StrategyHarness  # noqa: E402
from topstep_strats.backtest import run_backtest  # noqa: E402
from topstep_strats.metrics import calculate_metrics  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_backtest")

POINT_VALUES = {"NQ": 20.0, "ES": 50.0, "YM": 5.0}


def load_1m(csv: str, start: str, end: str) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(csv)
    ts_col = "timestamp" if "timestamp" in df.columns else "ts"
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.loc[start:end]
    df = df[~df.index.duplicated(keep="last")]
    log.info("loaded %s rows in %.1fs (%s..%s)", len(df), time.time() - t0, start, end)
    return df


def to_signals_df(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df[["entry_time", "direction", "entry_price", "stop_loss", "take_profit",
               "exit_time", "exit_price", "pnl", "exit_reason"]]


def save_result(outdir: Path, tag: str, trades: list[dict], metrics: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    trades_path = outdir / f"{tag}_trades.csv"
    metrics_path = outdir / f"{tag}_metrics.json"
    if trades:
        pd.DataFrame(trades).to_csv(trades_path, index=False)
    else:
        trades_path.write_text("")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    return trades_path, metrics_path


def et_hour(ts) -> int | None:
    try:
        return pd.to_datetime(ts, utc=True).tz_convert("America/New_York").hour
    except Exception:
        return None


def split_rth(trades: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition trades into NY RTH (09:30-16:00 ET) vs the rest (overnight).

    The signals run 24/7 as coded; this split lets us report both the
    as-written behavior and the intended NY intraday behavior side by side.
    """
    rth, over = [], []
    for t in trades:
        h = et_hour(t["entry_time"])
        (rth if h is not None and 9 <= h < 16 else over).append(t)
    return rth, over


def run_one(
    strategy: str,
    symbol: str,
    csv: str,
    start: str,
    end: str,
    outdir: Path,
    n_mc: int,
    n_boot: int,
    pip_value: float,
    max_reentries: int | None,
    scratch_root: Path,
) -> dict:
    point_value = POINT_VALUES[symbol.upper()]
    df = load_1m(csv, start, end)
    t0 = time.time()
    harness = StrategyHarness(
        strategy=strategy,
        symbol=symbol,
        point_value=point_value,
        pip_value=pip_value,
        max_reentries=max_reentries,
        scratch_root=scratch_root,
    )
    trades = harness.run(df)
    log.info("%s %s: %d trades in %.1fs", strategy, symbol, len(trades), time.time() - t0)

    def compute(trades_subset: list[dict]) -> dict:
        signals = to_signals_df(trades_subset)
        params = {
            "initial_capital": 100_000.0,
            "point_value": point_value,
            "slippage": 0.0,
            "commission": 0.0,
            "topstep": {"enabled": False},
        }
        result = run_backtest(signals, params) if not signals.empty else {
            "trades": signals, "equity_curve": pd.Series(dtype=float),
            "daily_returns": pd.Series(dtype=float),
            "start_of_day_to_trough_drawdown": [], "summary": {},
        }
        return calculate_metrics(result, n_mc=n_mc, n_boot=n_boot, random_state=42)

    tag = f"{symbol.upper()}_{strategy}"
    full_metrics = compute(trades)
    trades_path, metrics_path = save_result(outdir, tag, trades, full_metrics)

    # RTH vs overnight split reports (as-coded 24/7 run, filtered at output).
    rth_trades, over_trades = split_rth(trades)
    sub = {}
    for label, subset in (("rth", rth_trades), ("overnight", over_trades)):
        if not subset:
            continue
        m = compute(subset)
        t_path, m_path = save_result(outdir, f"{tag}_{label}", subset, m)
        sub[label] = {
            "trades": str(t_path), "metrics": str(m_path), "n_trades": len(subset),
            "win_rate": m["basic"]["win_rate"], "cagr": m["basic"]["cagr"],
            "sharpe": m["basic"]["sharpe_ratio"], "max_drawdown": m["basic"]["max_drawdown"],
        }

    return {
        "tag": tag,
        "strategy": strategy,
        "symbol": symbol.upper(),
        "point_value": point_value,
        "pip_value": pip_value,
        "trades": str(trades_path),
        "metrics": str(metrics_path),
        "n_trades": len(trades),
        "n_rth": len(rth_trades),
        "n_overnight": len(over_trades),
        "win_rate": full_metrics["basic"]["win_rate"],
        "cagr": full_metrics["basic"]["cagr"],
        "sharpe": full_metrics["basic"]["sharpe_ratio"],
        "max_drawdown": full_metrics["basic"]["max_drawdown"],
        "total_return": full_metrics["basic"]["total_return"],
        "rth": sub.get("rth"),
        "overnight": sub.get("overnight"),
    }


def main():
    ap = argparse.ArgumentParser(description="StarTrading futures backtest harness")
    ap.add_argument("--strategy", choices=list(StrategyHarness.SIGNALS), help="strategy; omit for all")
    ap.add_argument("--symbol", required=True, choices=["NQ", "ES", "YM"])
    ap.add_argument("--csv", required=True, help="path to 1m OHLCV csv")
    ap.add_argument("--start", default="2016-06-01", help="in-sample start (UTC)")
    ap.add_argument("--end", default="2023-12-31", help="in-sample end (UTC)")
    ap.add_argument("--outdir", default="/tmp/opencode/star_bt", help="output dir")
    ap.add_argument("--n-mc", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--pip-value", type=float, default=1.0)
    ap.add_argument("--max-reentries", type=int, default=None)
    ap.add_argument("--scratch", default="/tmp/strategies_state", help="signal state isolation root")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    scratch = Path(args.scratch)
    strategies = [args.strategy] if args.strategy else list(StrategyHarness.SIGNALS)

    manifest = []
    t_total = time.time()
    for strategy in strategies:
        res = run_one(
            strategy, args.symbol, args.csv, args.start, args.end, outdir,
            args.n_mc, args.n_boot, args.pip_value, args.max_reentries, scratch,
        )
        manifest.append(res)
        print(f"[{res['symbol']}/{res['strategy']}] trades={res['n_trades']} "
              f"(rth={res['n_rth']} over={res['n_overnight']}) wr={res['win_rate']:.3f} "
              f"cagr={res['cagr']:.3f} sharpe={res['sharpe']:.2f} maxdd={res['max_drawdown']:.3f}")
    with open(outdir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"done in {time.time() - t_total:.1f}s -> {outdir}")


if __name__ == "__main__":
    main()
