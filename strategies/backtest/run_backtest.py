# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created strategies/backtest/run_backtest.py: CLI that loads 1m OHLCV,
#     runs one or all StarTrading strategies over an in-sample window, feeds the
#     tagged trades into the topstep-strats backtest engine + full metrics suite
#     (20k MC/bootstrap), and writes trades CSV + metrics JSON per (strategy,
#     symbol). Trades carry a `symbol` column so instrument combos can be built
#     by merging trade files without re-running signals.
# 2026-08-17  kilo
#   - Expanded symbol universe and CLI to support the 7 YouTube strategy signals
#     across 8 instruments (NQ, ES, YM, GC, SI, BTC, ETH, SOL).
#   - Added --oos-start / --oos-end arguments (default 2024-01-01..2025-12-31).
#     When --oos-start is supplied, each (strategy, symbol) is run three times:
#     IS, OOS, and FULL, producing tags {SYM}_{strategy}_IS/_OOS/_FULL.
#     When omitted, only the IS window is run for backward compatibility.
#   - Confirmed risk_pct defaults to None so the harness trades one contract
#     per signal unless explicit fractional sizing is requested.
# WHY: Production entry point for the 56 individual backtests (7 strategies x
#      8 instruments) with IS/OOS/FULL walk-forward splits.
"""CLI for running the StarTrading futures backtest harness."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import os

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # /config
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
sys.path.insert(0, os.environ.get("TOPSTEP_STRATS_DIR", "/config/topstep-strats"))

from strategies.backtest.engine import StrategyHarness  # noqa: E402
from topstep_strats.backtest import run_backtest  # noqa: E402
from topstep_strats.metrics import calculate_metrics  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_backtest")

POINT_VALUES = {
    "NQ": 20.0, "ES": 50.0, "YM": 5.0,
    "GC": 10.0, "SI": 25.0,
    "BTC": 1.0, "ETH": 1.0, "SOL": 1.0,
}

DEFAULT_PIP_VALUES = {
    "NQ": 1.0, "ES": 0.2, "YM": 2.0,
    "GC": 10.0, "SI": 0.5,
    "BTC": 100.0, "ETH": 10.0, "SOL": 1.0,
}


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
    dll: float | None = None,
    risk_pct: float | None = None,
    initial_capital: float = 100_000.0,
    max_drawdown: float | None = None,
    eod_drawdown: float | None = None,
    trail_at_tp: bool = False,
    trail_distance: float | None = None,
    session_entry_hour_utc: int = 0,
    tag_suffix: str | None = None,
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
        dll=dll,
        risk_pct=risk_pct,
        initial_capital=initial_capital,
        max_drawdown=max_drawdown,
        eod_drawdown=eod_drawdown,
        trail_at_tp=trail_at_tp,
        trail_distance=trail_distance,
        session_entry_hour_utc=session_entry_hour_utc,
    )
    trades = harness.run(df)
    log.info("%s %s: %d trades in %.1fs", strategy, symbol, len(trades), time.time() - t0)

    def zero_metrics() -> dict:
        """Return a valid metrics dict when no trades or empty data exist."""
        return {
            "basic": {
                "start_equity": initial_capital,
                "end_equity": initial_capital,
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_start_index": None,
                "max_drawdown_end_index": None,
                "win_rate": 0.0,
                "n_trades": 0,
                "n_days": 0,
            },
            "probabilistic_sharpe_ratio": 0.0,
            "deflated_sharpe_ratio": 0.0,
            "start_of_day_to_trough_drawdown": {"values": [], "max": 0.0},
            "markov_transition_strength": {
                "transition_matrix": [[0.0, 0.0], [0.0, 0.0]],
                "counts": [[0, 0], [0, 0]],
                "chi2": 0.0,
                "pvalue": 1.0,
                "strength": 0.0,
            },
            "brownian_motion_test": {"variance_ratio": 0.0, "z_stat": 0.0, "pvalue": 1.0, "q": 5},
            "bayesian_sharpe": {"mean": 0.0, "median": 0.0, "ci_95": [0.0, 0.0], "samples": []},
        }

    def compute(trades_subset: list[dict]) -> dict:
        if not trades_subset:
            return zero_metrics()
        signals = to_signals_df(trades_subset)
        params = {
            "initial_capital": 100_000.0,
            "point_value": point_value,
            "slippage": 0.0,
            "commission": 0.0,
            "topstep": {"enabled": False},
        }
        result = run_backtest(signals, params)
        return calculate_metrics(result, n_mc=n_mc, n_boot=n_boot, random_state=42)

    tag = f"{symbol.upper()}_{strategy}"
    if tag_suffix:
        tag = f"{tag}_{tag_suffix}"
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
        "window": tag_suffix or "IS",
        "start": start,
        "end": end,
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
    ap.add_argument("--symbol", required=True,
                     choices=["NQ", "ES", "YM", "GC", "SI", "BTC", "ETH", "SOL"])
    ap.add_argument("--csv", required=True, help="path to 1m OHLCV csv")
    ap.add_argument("--start", default="2016-06-01", help="in-sample start (UTC)")
    ap.add_argument("--end", default="2023-12-31", help="in-sample end (UTC)")
    ap.add_argument("--oos-start", default=None, help="out-of-sample start (UTC); triggers IS+OOS+FULL runs")
    ap.add_argument("--oos-end", default="2025-12-31", help="out-of-sample end (UTC)")
    ap.add_argument("--outdir", default="/tmp/opencode/star_bt", help="output dir")
    ap.add_argument("--n-mc", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--pip-value", type=float, default=None,
                     help="price units per pip; default varies by symbol")
    ap.add_argument("--max-reentries", type=int, default=None)
    ap.add_argument("--dll", type=float, default=None, help="daily loss limit in $ (hard mid-trade cut, then halt for the day)")
    ap.add_argument("--risk-pct", type=float, default=None, help="fixed-fractional risk per trade (qty from stop distance); None=1 contract")
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--max-drawdown", type=float, default=None, help="intra-day trailing drawdown limit in $ (Apex/E2T style)")
    ap.add_argument("--eod-drawdown", type=float, default=None, help="end-of-day trailing drawdown limit in $ (Topstep style)")
    ap.add_argument("--trail-at-tp", action="store_true", default=False, help="when TP hit, move SL to breakeven and trail instead of closing")
    ap.add_argument("--trail-distance", type=float, default=None, help="trail distance in points after TP hit (default: TP distance)")
    ap.add_argument("--session-entry-hour-utc", type=int, default=0, help="entry hour in UTC for mos_session_daily_draw (0=MOS, 5=Asian, 7=London)")
    ap.add_argument(
        "--scratch",
        default=tempfile.mkdtemp(prefix="strategies_run_"),
        help="signal state isolation root (default: fresh temp dir per run, so no stale"
        " date-keyed state leaks between runs and alters entries)",
    )
    args = ap.parse_args()
    if args.pip_value is None:
        args.pip_value = DEFAULT_PIP_VALUES.get(args.symbol, 1.0)

    outdir = Path(args.outdir)
    scratch = Path(args.scratch)
    strategies = [args.strategy] if args.strategy else list(StrategyHarness.SIGNALS)

    if args.oos_start:
        windows = [
            (args.start, args.end, "IS"),
            (args.oos_start, args.oos_end, "OOS"),
            (args.start, args.oos_end, "FULL"),
        ]
    else:
        windows = [(args.start, args.end, None)]

    manifest = []
    t_total = time.time()
    for strategy in strategies:
        for start, end, suffix in windows:
            res = run_one(
                strategy, args.symbol, args.csv, start, end, outdir,
                args.n_mc, args.n_boot, args.pip_value, args.max_reentries, scratch,
                dll=args.dll, risk_pct=args.risk_pct, initial_capital=args.initial_capital,
                max_drawdown=args.max_drawdown, eod_drawdown=args.eod_drawdown,
                trail_at_tp=args.trail_at_tp, trail_distance=args.trail_distance,
                session_entry_hour_utc=args.session_entry_hour_utc,
                tag_suffix=suffix,
            )
            manifest.append(res)
            print(f"[{res['tag']}] {res['start']}..{res['end']} "
                  f"trades={res['n_trades']} (rth={res['n_rth']} over={res['n_overnight']}) "
                  f"wr={res['win_rate']:.3f} cagr={res['cagr']:.3f} "
                  f"sharpe={res['sharpe']:.2f} maxdd={res['max_drawdown']:.3f}")
    with open(outdir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"done in {time.time() - t_total:.1f}s -> {outdir}")


if __name__ == "__main__":
    main()
