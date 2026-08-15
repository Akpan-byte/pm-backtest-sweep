# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created strategies/backtest/run_portfolio.py: CLI for portfolio-level DLL
#     backtests of the winners book (mos_session_daily_draw + fifteen_min_range_
#     scalp over NQ/ES/YM).  Runs PortfolioHarness (one shared daily loss bucket
#     across all six instruments, bar-level cuts + rest-of-day halt), then feeds
#     the merged trades through the topstep metrics suite like run_backtest.py.
# WHY: Portfolio-level (not per-instrument) DLL per user's decision; per-symbol
#      engines alone understate the worst-day tail.

"""CLI for portfolio-level DLL backtests of the StarTrading winners book."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # /config
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
sys.path.insert(0, os.environ.get("TOPSTEP_STRATS_DIR", "/config/topstep-strats"))

from strategies.backtest.portfolio_harness import (  # noqa: E402
    PortfolioHarness,
    load_1m_fast,
)
from topstep_strats.backtest import run_backtest  # noqa: E402
from topstep_strats.metrics import calculate_metrics  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_portfolio")

POINT_VALUES = {"NQ": 20.0, "ES": 50.0, "YM": 5.0}


def to_signals_df(trades: list[dict]) -> pd.DataFrame:
    """Convert engine trades to the metrics engine's signal table.

    The engine's ``pnl`` is in points x qty per trade, but the winners book
    mixes NQ ($20/pt) / ES ($50/pt) / YM ($5/pt).  Normalize to DOLLARS
    (pnl * point_value per symbol) and use point_value=1.0 downstream so the
    portfolio equity curve and drawdown are computed in real dollars.
    """
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["pnl"] = df["pnl"] * df["symbol"].map(POINT_VALUES)
    cols = [c for c in ["entry_time", "direction", "entry_price", "stop_loss",
                        "take_profit", "exit_time", "exit_price", "pnl", "exit_reason"]
            if c in df.columns]
    return df[cols]


def compute_metrics(trades: list[dict], n_mc: int, n_boot: int) -> dict:
    signals = to_signals_df(trades)
    params = {
        "initial_capital": 100_000.0,
        "point_value": 1.0,
        "slippage": 0.0,
        "commission": 0.0,
        "topstep": {"enabled": False},
    }
    if signals.empty:
        return {"basic": {}}
    result = run_backtest(signals, params)
    return calculate_metrics(result, n_mc=n_mc, n_boot=n_boot, random_state=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", type=float, default=None)
    ap.add_argument("--risk-pct", type=float, default=None)
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    ap.add_argument("--n-mc", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--data-root", default="/tmp/opencode/fvg-market-data")
    ap.add_argument("--outdir", default="/tmp/opencode/star_portfolio")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--strategy", default=None,
                    help="Single strategy (e.g. mos_session_daily_draw); omit for winners book")
    ap.add_argument("--instruments", default=None,
                    help="Comma-separated instrument list (e.g. NQ,ES); omit for all three")
    args = ap.parse_args()

    instruments = [s.strip().upper() for s in args.instruments.split(",")] if args.instruments else ["NQ", "ES", "YM"]
    data_root = Path(args.data_root)
    data = {}
    for sym in instruments:
        df = load_1m_fast(data_root / f"{sym}_1min.csv")
        data[sym] = df.loc[args.start:args.end]

    if args.strategy:
        combos = [(args.strategy, sym) for sym in instruments]
    else:
        from strategies.backtest.portfolio_harness import WINNERS_COMBOS
        combos = [(s, sym) for s, sym in WINNERS_COMBOS if sym in instruments]

    t0 = time.time()
    ph = PortfolioHarness(
        data=data,
        combos=combos,
        dll=args.dll,
        risk_pct=args.risk_pct,
        initial_capital=args.initial_capital,
        scratch_root=Path(tempfile.mkdtemp(prefix="portfolio_run_")),
    )
    trades = ph.run()
    log.info("portfolio dll=%s risk=%s: %d trades in %.1fs",
             args.dll, args.risk_pct, len(trades), time.time() - t0)

    tag = args.tag or f"portfolio_dll_{int(args.dll) if args.dll is not None else 'none'}"
    outdir = Path(args.outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    if trades:
        df_t = pd.DataFrame(trades)
        df_t["pnl_dollar"] = df_t["pnl"] * df_t["symbol"].map(POINT_VALUES)
        df_t.to_csv(outdir / "trades.csv", index=False)
    else:
        (outdir / "trades.csv").write_text("")

    # Composite point value irrelevant for return-based metrics (equity from
    # dollar pnl); metrics engine uses point_value for drawdown on dollar pnl.
    metrics = compute_metrics(trades, n_mc=args.n_mc, n_boot=args.n_boot)

    # Day-level realized PnL from the trade stream (DLL-clamped exits carry the
    # exact realized dollars the engine assigned).
    day_stats = {}
    if trades:
        dt = pd.DataFrame(trades)
        dt["day"] = pd.to_datetime(dt["exit_time"], utc=True).dt.date
        dt["pnl_dollar"] = dt["pnl"] * dt["symbol"].map(POINT_VALUES)
        dg = dt.groupby("day")["pnl_dollar"].agg(["sum", "size"])
        day_stats = {
            "n_days": int(len(dg)),
            "worst_day": float(dg["sum"].min()),
            "best_day": float(dg["sum"].max()),
            "mean_day": float(dg["sum"].mean()),
            "median_day": float(dg["sum"].median()),
            "days_at_bucket": int((dg["sum"] <= -(args.dll or 0)).sum()),
            "std_day": float(dg["sum"].std()),
        }

    payload = {
        "tag": tag,
        "dll": args.dll,
        "risk_pct": args.risk_pct,
        "initial_capital": args.initial_capital,
        "start": args.start,
        "end": args.end,
        "n_trades": len(trades),
        "metrics": metrics,
        "day_stats": day_stats,
    }
    with open(outdir / "report.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    b = metrics.get("basic", {})
    print(f"[{tag}] trades={len(trades)} wr={b.get('win_rate'):.3f} "
          f"cagr={b.get('cagr'):.3f} sharpe={b.get('sharpe_ratio'):.2f} "
          f"maxdd={b.get('max_drawdown'):.3f} totret={b.get('total_return'):.3f} "
          f"worst_day={day_stats.get('worst_day'):,.0f} best_day={day_stats.get('best_day'):,.0f} "
          f"mean_day={day_stats.get('mean_day'):,.0f}")
    print(f"done in {time.time() - t0:.1f}s -> {outdir}")


if __name__ == "__main__":
    main()