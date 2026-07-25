#!/usr/bin/env python3
"""
Run a single JJ Simon Fair-Price backtest from the siloed project.

Examples:
    python3 scripts/run_single_backtest.py --profile 50k --instrument NQ
    python3 scripts/run_single_backtest.py --profile 150k --instrument ES --pm false
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jj_simon_fair_price import FairPriceConfig, load_data, run_backtest, trades_to_dataframe


def load_profile(profile_name: str) -> dict:
    """Load a profile from config/profiles.json."""
    config_path = PROJECT_ROOT / "config" / "profiles.json"
    with open(config_path) as f:
        data = json.load(f)
    profiles = data["profiles"]
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile {profile_name}. Available: {list(profiles.keys())}")
    return profiles[profile_name]


def build_config(profile: dict, instrument: str, pm: bool, bos_lookback: int) -> FairPriceConfig:
    """Build a FairPriceConfig from profile JSON and CLI overrides."""
    config_path = PROJECT_ROOT / "config" / "profiles.json"
    with open(config_path) as f:
        data = json.load(f)
    inst_info = data["instruments"][instrument]

    return FairPriceConfig(
        profile=profile["label"],
        starting_balance=profile["starting_balance"],
        sl_pts=profile["sl_pts"],
        tp_pts=profile["tp_pts"],
        risk_pct=profile["risk_pct"],
        bos_lookback=bos_lookback,
        mean_reversion_distance=profile["mean_reversion_distance"],
        news_spike_threshold=profile["news_spike_threshold"],
        dynamic_candle_trigger=profile["dynamic_candle_trigger"],
        dynamic_sl_pts=profile["dynamic_sl_pts"],
        dynamic_tp_pts=profile["dynamic_tp_pts"],
        dynamic_size_reduction=profile["dynamic_size_reduction"],
        enable_pm_session=pm,
        point_value=inst_info["point_value"],
        tick_size=inst_info["tick_size"],
        max_morning_trades=profile["max_morning_trades"],
        max_consecutive_losses=profile["max_consecutive_losses"],
    )


def summarize(trades: list, final_balance: float, starting_balance: float) -> None:
    """Print a concise summary of the backtest."""
    df = trades_to_dataframe(trades)
    if df.empty:
        print("No trades generated.")
        return

    pnl = df["net"].values
    wins = pnl > 0
    losses = pnl < 0
    win_rate = wins.sum() / len(pnl)
    gross_profit = pnl[wins].sum() if wins.any() else 0.0
    gross_loss = abs(pnl[losses].sum()) if losses.any() else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = starting_balance + np.cumsum(pnl)
    peaks = np.maximum.accumulate(equity)
    dd = (peaks - equity) / peaks * 100.0

    print("\n" + "=" * 60)
    print("  JJ SIMON NQ FAIR-PRICE — SINGLE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Starting balance:  ${starting_balance:,.2f}")
    print(f"  Final balance:     ${final_balance:,.2f}")
    print(f"  Net P&L:           ${final_balance - starting_balance:,.2f}")
    print(f"  Total return:      {(final_balance / starting_balance - 1) * 100:.2f}%")
    print(f"  Total trades:      {len(trades)}")
    print(f"  Win rate:          {win_rate:.2%}")
    print(f"  Profit factor:     {profit_factor:.2f}")
    print(f"  Max drawdown:      {dd.max():.2f}%")
    print(f"  Avg trade:         ${pnl.mean():.2f}")
    print("=" * 60)

    # Exit reason breakdown
    print("\n  Exit reason breakdown:")
    print(df["exit_reason"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="Run a single JJ Simon backtest")
    parser.add_argument("--profile", default="50k", choices=["50k", "150k"], help="Prop-firm profile")
    parser.add_argument("--instrument", default="NQ", choices=["NQ", "ES", "YM"], help="Futures symbol")
    parser.add_argument("--pm", default="true", choices=["true", "false"], help="Enable PM session")
    parser.add_argument("--bos_lookback", type=int, default=5, help="BOS lookback candles")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    config = build_config(profile, args.instrument, args.pm == "true", args.bos_lookback)

    print(f"Loading {args.instrument} data...")
    df = load_data(args.instrument)
    print(f"Loaded {len(df)} bars from {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")

    print(f"Running backtest: profile={args.profile}, instrument={args.instrument}, PM={args.pm}, BOS={args.bos_lookback}...")
    trades, final_balance = run_backtest(df, config)

    summarize(trades, final_balance, config.starting_balance)

    # Save trades CSV
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    trades_df = trades_to_dataframe(trades)
    out_path = out_dir / f"{args.instrument}_{args.profile}_pm{args.pm}_bos{args.bos_lookback}_trades.csv"
    trades_df.to_csv(out_path, index=False)
    print(f"\nTrades saved to {out_path}")


if __name__ == "__main__":
    main()
