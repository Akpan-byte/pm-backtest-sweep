#!/usr/bin/env python3
"""Combined main ORB + counter-strategy backtest with prop firm metrics."""

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/config/projects/trading")

from flexing_joe_orb.backtest import run_backtest
from flexing_joe_orb.data import load_ohlcv_csv
from flexing_joe_orb.models import StrategyConfig, Trade
from flexing_joe_orb.metrics import summarize_metrics
from flexing_joe_orb.prop_firm import attach_prop_firm_analysis


INSTRUMENTS = {
    "NQ": {"point_value": 20.0, "tick_size": 0.25},
    "ES": {"point_value": 50.0, "tick_size": 0.25},
    "YM": {"point_value": 5.0, "tick_size": 1.0},
}

DATA_ROOT = Path("/config/projects/trading/v5_orb_nq_backtest/market_data")


def orb_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_min"] = df.index.hour * 60 + df.index.minute
    filtered = df[(df["time_min"] >= 570) & (df["time_min"] < 630)].copy()
    filtered["date"] = filtered.index.date
    orb = filtered.groupby("date").agg(
        orb_high=("high", "max"),
        orb_low=("low", "min"),
    )
    orb["orb_range"] = orb["orb_high"] - orb["orb_low"]
    return orb.reset_index()


def simulate_counter_trades(main_trades: list, df_1m: pd.DataFrame, config: StrategyConfig) -> list:
    """Generate counter-strategy trades for first-trade loss days."""
    orb = orb_levels(df_1m)
    orb["date"] = pd.to_datetime(orb["date"]).dt.date

    # First trade per day
    trades_df = pd.DataFrame(main_trades)
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_time"], utc=True).dt.tz_convert("America/New_York").dt.date
    first = trades_df.sort_values("entry_time").groupby("entry_date").first().reset_index()
    first = first[first["net_pnl"] <= 0].copy()
    first["date"] = first["entry_date"]

    merged = first.merge(orb, on="date", how="left")
    grouped = dict(list(df_1m.groupby(df_1m.index.date)))

    counter_trades = []
    for row in merged.itertuples(index=False):
        day_bars = grouped.get(row.date)
        if day_bars is None or len(day_bars) < 10:
            continue

        counter_dir = -row.direction
        entry_price = row.exit_price
        entry_time = pd.Timestamp(row.exit_time).tz_convert("America/New_York")
        orb_range = row.orb_range

        if counter_dir == 1:
            stop_price = row.orb_low
            target = entry_price + orb_range
        else:
            stop_price = row.orb_high
            target = entry_price - orb_range

        after = day_bars[day_bars.index >= entry_time]
        if after.empty:
            continue

        locked = False
        exit_price = exit_reason = None
        for ts, b in after.iterrows():
            # Stop
            if counter_dir == 1 and b["low"] <= stop_price:
                exit_price = stop_price
                exit_reason = "COUNTER_STOP"
                break
            if counter_dir == -1 and b["high"] >= stop_price:
                exit_price = stop_price
                exit_reason = "COUNTER_STOP"
                break

            # Lock level (1x orb range profit) -> move stop to breakeven
            if not locked:
                if counter_dir == 1 and b["high"] >= target:
                    stop_price = entry_price
                    locked = True
                if counter_dir == -1 and b["low"] <= target:
                    stop_price = entry_price
                    locked = True

            # EOD
            if ts.hour == 16 and ts.minute == 0:
                exit_price = b["close"]
                exit_reason = "COUNTER_EOD"
                break

        if exit_price is None:
            exit_price = after.iloc[-1]["close"]
            exit_reason = "COUNTER_EOD"

        gross = counter_dir * (exit_price - entry_price) * config.point_value
        slippage_cost = config.slippage_points * config.point_value / config.tick_size
        net = gross - (2 * config.commission_per_contract + slippage_cost)

        counter_trades.append({
            "entry_time": entry_time.isoformat(),
            "exit_time": ts.isoformat(),
            "direction": counter_dir,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "contracts": config.contracts_per_trade,
            "gross_pnl": round(gross, 4),
            "commission": round(2 * config.commission_per_contract, 4),
            "slippage": round(slippage_cost, 4),
            "net_pnl": round(net, 4),
            "exit_reason": exit_reason,
        })

    return counter_trades


def compute_eval_pass_times(daily_pnl: dict, eval_profit_target: float, daily_loss_limit: float,
                            max_loss_limit: float, min_days: int) -> dict:
    """Compute distribution of eval pass times starting from each possible day."""
    dates = sorted(daily_pnl.keys())
    pnl_values = [daily_pnl[d] for d in dates]
    pass_days = []

    for start_idx in range(len(dates)):
        cum = 0.0
        max_dd = 0.0
        peak = 0.0
        blown = False
        days_used = 0
        for i in range(start_idx, len(dates)):
            days_used += 1
            cum += pnl_values[i]
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

            if pnl_values[i] < -daily_loss_limit:
                blown = True
                break
            if max_dd > max_loss_limit:
                blown = True
                break
            if cum >= eval_profit_target and days_used >= min_days:
                pass_days.append(days_used)
                break
        else:
            # Reached end without passing
            pass_days.append(None)

    passes = [d for d in pass_days if d is not None]
    return {
        "starts": len(pass_days),
        "passes": len(passes),
        "pass_rate": len(passes) / len(pass_days) if pass_days else 0,
        "avg_days_to_pass": float(np.mean(passes)) if passes else None,
        "median_days_to_pass": float(np.median(passes)) if passes else None,
        "max_days_to_pass": int(max(passes)) if passes else None,
        "distribution": pd.Series(passes).value_counts().sort_index().to_dict() if passes else {},
    }


def run_combined(symbol: str, data_root: Path = DATA_ROOT,
                 prop_mc_runs: int = 20_000,
                 prop_bootstrap_samples: int = 20_000,
                 contracts_per_trade: int = 1,
                 variant: str = "one_trade_per_day",
                 max_entries_per_day: int = 999,
                 start_date: str = "2016-06-01",
                 end_date: str = "2026-05-29",
                 min_orb_range_multiple: Optional[float] = None) -> dict:
    flags = {"one_trade_per_day": False, "one_trade_per_direction": False}
    if variant == "one_trade_per_day":
        flags = {"one_trade_per_day": True, "one_trade_per_direction": False}
    elif variant == "one_per_direction":
        flags = {"one_trade_per_day": False, "one_trade_per_direction": True}
    elif variant == "reentries":
        flags = {"one_trade_per_day": False, "one_trade_per_direction": False}
    else:
        raise ValueError(f"Unknown variant: {variant}")

    inst = INSTRUMENTS[symbol]
    config = StrategyConfig(
        symbol=symbol,
        data_path=str(data_root / f"{symbol}_1min.csv"),
        start_date=start_date,
        end_date=end_date,
        point_value=inst["point_value"],
        tick_size=inst["tick_size"],
        commission_per_contract=2.5,
        slippage_points=0.25,
        initial_account_size=50_000.0,
        contracts_per_trade=contracts_per_trade,
        daily_loss_limit=900.0,
        trailing_drawdown_limit=2_000.0,
        max_entries_per_day=max_entries_per_day,
        min_orb_range_multiple=min_orb_range_multiple,
        mc_runs=0,
        bootstrap_samples=0,
        prop_mc_runs=0,
        prop_bootstrap_samples=0,
        **flags,
    )

    print(f"\nRunning main strategy for {symbol}...")
    main_result = run_backtest(config)
    main_trades = main_result["trades"]
    main_metrics = main_result["metrics"]

    print(f"Loading 1m data for {symbol}...")
    df_1m = load_ohlcv_csv(config.data_path)
    df_1m = df_1m[(df_1m.index >= pd.Timestamp(start_date, tz="UTC")) &
                  (df_1m.index < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1))]

    print(f"Simulating counter-strategy for {symbol}...")
    counter_trades = simulate_counter_trades(main_trades, df_1m, config)

    combined_trades = main_trades + counter_trades

    # Recompute daily PnL
    daily_pnl = {}
    for t in combined_trades:
        d = pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d")
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t["net_pnl"]

    combined_trade_objects = [
        Trade(
            entry_time=pd.Timestamp(t["entry_time"]),
            exit_time=pd.Timestamp(t["exit_time"]),
            direction=t["direction"],
            entry_price=t["entry_price"],
            exit_price=t["exit_price"],
            contracts=t["contracts"],
            gross_pnl=t["gross_pnl"],
            commission=t["commission"],
            slippage=t["slippage"],
            net_pnl=t["net_pnl"],
            exit_reason=t["exit_reason"],
        )
        for t in combined_trades
    ]
    metrics = summarize_metrics(combined_trade_objects, daily_pnl, initial_equity=config.initial_account_size)

    result = {
        "symbol": symbol,
        "parameters": asdict(config),
        "main_trades": len(main_trades),
        "counter_trades": len(counter_trades),
        "combined_trades": len(combined_trades),
        "main_only_metrics": main_metrics,
        "daily_pnl": {k: round(v, 2) for k, v in sorted(daily_pnl.items())},
        "metrics": metrics,
        "trades": combined_trades,
    }

    # Prop firm analysis
    result = attach_prop_firm_analysis(
        result, prop_mc_runs=prop_mc_runs, prop_bootstrap_samples=prop_bootstrap_samples
    )

    # Eval pass times
    print(f"Computing eval pass times for {symbol}...")
    result["eval_pass_times"] = {
        "50k_eval": compute_eval_pass_times(
            result["daily_pnl"], eval_profit_target=2_000.0,
            daily_loss_limit=900.0, max_loss_limit=2_000.0, min_days=2
        ),
        "150k_eval": compute_eval_pass_times(
            result["daily_pnl"], eval_profit_target=4_000.0,
            daily_loss_limit=3_500.0, max_loss_limit=4_500.0, min_days=2
        ),
    }

    return result


def main():
    import argparse
    from functools import partial
    parser = argparse.ArgumentParser(description="Combined main ORB + counter strategy backtest")
    parser.add_argument("--workers", type=int, default=3, help="Max parallel workers (default 3)")
    parser.add_argument("--output", default="/tmp/combined_main_counter_results.json", help="Output JSON path")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Directory containing {SYMBOL}_1min.csv files")
    parser.add_argument("--contracts-per-trade", type=int, default=1, help="Number of contracts per trade")
    parser.add_argument("--variant", default="one_trade_per_day", choices=["one_trade_per_day", "reentries", "one_per_direction"], help="Backtest variant")
    parser.add_argument("--max-entries-per-day", type=int, default=999, help="Max entries per day (reentries variant)")
    parser.add_argument("--symbols", default="NQ,ES,YM", help="Comma-separated symbols to run")
    parser.add_argument("--prop-mc-runs", type=int, default=20_000, help="Prop-firm Monte Carlo runs")
    parser.add_argument("--prop-bootstrap-samples", type=int, default=20_000, help="Prop-firm bootstrap samples")
    parser.add_argument("--start-date", default="2016-06-01", help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-05-29", help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--min-orb-range-multiple", type=float, default=None, help="Skip sessions with ORB range below this multiple of the median lookback range")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    runner = partial(
        run_combined,
        data_root=args.data_root,
        prop_mc_runs=args.prop_mc_runs,
        prop_bootstrap_samples=args.prop_bootstrap_samples,
        contracts_per_trade=args.contracts_per_trade,
        variant=args.variant,
        max_entries_per_day=args.max_entries_per_day,
        start_date=args.start_date,
        end_date=args.end_date,
        min_orb_range_multiple=args.min_orb_range_multiple,
    )
    if args.workers <= 1:
        results = {sym: runner(sym) for sym in symbols}
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = dict(zip(symbols, executor.map(runner, symbols)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # Print summary
    for sym, r in results.items():
        m = r["metrics"]
        print(f"\n=== {sym} COMBINED ===")
        print(f"Main trades: {r['main_trades']}, Counter trades: {r['counter_trades']}")
        print(f"Net PnL: ${m['net_pnl']:+.0f}  Win rate: {m['win_rate']:.1f}%  PF: {m.get('profit_factor', 0):.2f}")
        print(f"Max DD: ${m['max_drawdown_dollars']:.0f} ({m['max_drawdown_pct']:.1f}%)")
        print("Eval pass times:")
        for eval_name, stats in r["eval_pass_times"].items():
            print(f"  {eval_name}: pass rate {stats['pass_rate']:.1%}, avg {stats['avg_days_to_pass']:.1f} days, median {stats['median_days_to_pass']:.1f} days")


if __name__ == "__main__":
    main()
