#!/usr/bin/env python3
"""Tiny validation across NQ/ES/YM and all variants (1 month, no MC/bootstrap)."""
from __future__ import annotations

from pathlib import Path

from flexing_joe_orb.backtest import run_backtest
from flexing_joe_orb.models import StrategyConfig

DATA_ROOT = Path("/config/projects/trading/v5_orb_nq_backtest/market_data")
START_DATE = "2016-06-01"
END_DATE = "2016-06-30"
VARIANTS = ["one_trade_per_day", "one_per_direction", "reentries"]
INSTRUMENTS = [
    ("NQ", 20.0, 0.25),
    ("ES", 50.0, 0.25),
    ("YM", 5.0, 1.0),
]

print(f"{'Symbol':<6} {'Variant':<20} {'Trades':>7} {'Net PnL':>10} {'Win%':>7} {'Sharpe':>8} {'MaxDD$':>10} {'50kStd':>8} {'50kCon':>8} {'100kStd':>9} {'100kCon':>9}")
print("-" * 110)

for symbol, point_value, tick_size in INSTRUMENTS:
    data_path = DATA_ROOT / f"{symbol}_1min.csv"
    for variant in VARIANTS:
        flags = {}
        if variant == "one_trade_per_day":
            flags = {"one_trade_per_day": True, "one_trade_per_direction": False}
        elif variant == "one_per_direction":
            flags = {"one_trade_per_day": False, "one_trade_per_direction": True}
        else:
            flags = {"one_trade_per_day": False, "one_trade_per_direction": False}

        cfg = StrategyConfig(
            symbol=symbol,
            data_path=str(data_path),
            start_date=START_DATE,
            end_date=END_DATE,
            point_value=point_value,
            tick_size=tick_size,
            mc_runs=0,
            bootstrap_samples=0,
            prop_mc_runs=0,
            prop_bootstrap_samples=0,
            **flags,
        )
        result = run_backtest(cfg)
        m = result["metrics"]
        pf = result["prop_firm_payouts"]
        print(
            f"{symbol:<6} {variant:<20} {m['total_trades']:>7} {m['net_pnl']:>+10.0f} "
            f"{m['win_rate']:>7.1f} {m['sharpe']:>8.2f} {m['max_drawdown_dollars']:>10.0f} "
            f"{pf['50k_standard']['total_payouts']:>8} {pf['50k_consistency']['total_payouts']:>8} "
            f"{pf['100k_standard']['total_payouts']:>9} {pf['100k_consistency']['total_payouts']:>9}"
        )

print("\nTiny validation complete.")
