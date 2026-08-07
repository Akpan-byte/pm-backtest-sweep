#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import load_ohlcv
from orb_1h_5min_dollar import backtest_1h_5min_dollar


def resample_ohlcv(df_1min: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV to a higher timeframe."""
    df = df_1min.copy()
    # Localize to NY if not already
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    resampled = df.resample(freq, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-1min", required=True, help="Path to NQ_1min.csv")
    parser.add_argument("--dollar-stop", type=float, required=True)
    parser.add_argument("--dollar-target", type=float, required=True)
    parser.add_argument("--or-minutes", type=int, default=240)
    parser.add_argument("--contract-value", type=float, default=20.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df_1min = load_ohlcv(args.input_1min)
    df_1h = resample_ohlcv(df_1min, "1h")
    df_5min = resample_ohlcv(df_1min, "5min")

    result = backtest_1h_5min_dollar(
        df_1h,
        df_5min,
        or_minutes=args.or_minutes,
        dollar_stop=args.dollar_stop,
        dollar_target=args.dollar_target,
        contract_value=args.contract_value,
    )

    out = {
        "signal_bars": "1h",
        "exec_bars": "5min",
        "dollar_stop": args.dollar_stop,
        "dollar_target": args.dollar_target,
        "or_minutes": args.or_minutes,
        "contract_value": args.contract_value,
        **result,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"stop=${args.dollar_stop} target=${args.dollar_target}: ret={result['total_return']:.2%} sharpe={result['sharpe']:.3f} wr={result['win_rate']:.1%} dd={result['max_drawdown']:.2%} total=${result['total_dollars']:,.0f}")


if __name__ == "__main__":
    main()
