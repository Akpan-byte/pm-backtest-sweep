#!/usr/bin/env python3
"""Run one No-Wick filter configuration and write metrics."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

from common import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    profit_factor,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
)
from data import load_ohlcv
from filters import compute_filter_columns
from no_wick import detect_a_setups, detect_no_wick, detect_structure, simulate_no_wick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--filters", default="", help="Comma-separated filter names, or empty/none for baseline")
    parser.add_argument("--session-start", default="09:30")
    parser.add_argument("--session-end", default="16:00")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-sl-pips", type=float, default=5.0)
    parser.add_argument("--breather-pct", type=float, default=0.10)
    parser.add_argument("--omar-offset-pips", type=float, default=2.0)
    parser.add_argument("--omar-threshold-pips", type=float, default=10.0)
    parser.add_argument("--pip-size", type=float, default=1.0)
    args = parser.parse_args()

    df = load_ohlcv(args.data_path)
    df = detect_structure(df)
    df = detect_a_setups(df)
    df = detect_no_wick(df, pip_tol=0.0)
    df = compute_filter_columns(df)

    filter_names = [f.strip() for f in args.filters.split(",") if f.strip() and f.strip().lower() != "none"]
    mask = pd.Series(True, index=df.index)
    for name in filter_names:
        col = f"f_{name}"
        if col not in df.columns:
            raise ValueError(f"Unknown filter: {name}")
        mask &= df[col].fillna(False)

    session_start = args.session_start if args.session_start else None
    session_end = args.session_end if args.session_end else None

    eq, trades = simulate_no_wick(
        df,
        min_sl_pips=args.min_sl_pips,
        breather_pct=args.breather_pct,
        omar_offset_pips=args.omar_offset_pips,
        omar_threshold_pips=args.omar_threshold_pips,
        pip_size=args.pip_size,
        session_start=session_start,
        session_end=session_end,
        close_at_session_end=True,
        signal_mask=mask,
    )

    trade_pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in trade_pnls if p > 0)
    periods_per_year = 252 * 78
    rets = eq.pct_change().dropna()

    metrics = {
        "filters": args.filters,
        "session": f"{args.session_start}-{args.session_end}",
        "total_return": float((eq.iloc[-1] / 100000.0) - 1.0),
        "net_profit": float(eq.iloc[-1] - 100000.0),
        "final_equity": float(eq.iloc[-1]),
        "sharpe": float(sharpe_ratio(rets, periods_per_year)),
        "sortino": float(sortino_ratio(rets, periods_per_year)),
        "max_drawdown": float(max_drawdown(eq)),
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "profit_factor": float(profit_factor(trade_pnls)) if trade_pnls else 0.0,
        "psr": float(probabilistic_sharpe_ratio(rets, 0.0, periods_per_year)),
        "dsr": float(deflated_sharpe_ratio(rets, 1, periods_per_year)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
