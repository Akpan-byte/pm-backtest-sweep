"""End-to-end backtest runner for the Flexing Joe ORB strategy."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data import load_ohlcv_csv, load_optional_csv
from .execution import FuturesExecutionEngine
from .mc_bootstrap import attach_mc_and_bootstrap
from .metrics import summarize_metrics
from .models import Signal, StrategyConfig, Trade
from .prop_firm import attach_prop_firm_analysis
from .signals import generate_all_signals


def _variant_to_config_flags(variant: str) -> Dict[str, Any]:
    """Convert CLI variant name to strategy flag overrides."""
    if variant == "one_trade_per_day":
        return {"one_trade_per_day": True, "one_trade_per_direction": False}
    if variant == "one_per_direction":
        return {"one_trade_per_day": False, "one_trade_per_direction": True}
    if variant == "reentries":
        return {"one_trade_per_day": False, "one_trade_per_direction": False}
    raise ValueError(f"Unknown variant: {variant}")


def _filter_date_range(
    df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]
) -> pd.DataFrame:
    """Filter DataFrame index to [start_date, end_date] inclusive (UTC)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex on the input DataFrame")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        # Include the full end date.
        df = df[df.index < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)]
    return df


def _serialize_signal(s: Signal) -> Dict[str, Any]:
    return {
        "timestamp": s.timestamp.isoformat(),
        "direction": s.direction,
        "entry_price": s.entry_price,
        "stop_price": s.stop_price,
        "target_price": s.target_price,
        "contracts": s.contracts,
        "reason": s.reason,
    }


def _serialize_trade(t: Trade) -> Dict[str, Any]:
    return {
        "entry_time": t.entry_time.isoformat(),
        "exit_time": t.exit_time.isoformat(),
        "direction": t.direction,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "contracts": t.contracts,
        "gross_pnl": t.gross_pnl,
        "commission": t.commission,
        "slippage": t.slippage,
        "net_pnl": t.net_pnl,
        "exit_reason": t.exit_reason,
    }


def _compute_daily_pnl(trades: List[Trade]) -> Dict[str, float]:
    """Aggregate net PnL by entry date."""
    daily: Dict[str, float] = {}
    for t in trades:
        d = t.entry_time.strftime("%Y-%m-%d")
        daily[d] = daily.get(d, 0.0) + t.net_pnl
    return {k: round(v, 2) for k, v in sorted(daily.items())}


def run_backtest(config: StrategyConfig) -> Dict[str, Any]:
    """Load data, generate signals, execute, and summarize a full backtest."""
    # Load primary 1-minute data.
    df_1m = load_ohlcv_csv(config.data_path)
    df_1m = _filter_date_range(df_1m, config.start_date, config.end_date)
    if df_1m.empty:
        raise ValueError("No data loaded after date filtering")

    # Optional cross-instrument data.
    optional_dfs: Dict[str, Optional[pd.DataFrame]] = {
        "vix": load_optional_csv(config.vix_path),
        "es": load_optional_csv(config.es_path),
        "nq": load_optional_csv(config.nq_path),
    }
    optional_dfs = {k: v for k, v in optional_dfs.items() if v is not None}

    # Generate signals.
    signals = generate_all_signals(df_1m, config, optional_dfs)

    # Execute.
    engine = FuturesExecutionEngine(config)
    trades, exec_summary = engine.execute_signals(df_1m, signals)

    # Daily PnL map.
    daily_pnl = _compute_daily_pnl(trades)

    # Metrics.
    metrics = summarize_metrics(
        trades, daily_pnl, initial_equity=config.initial_account_size
    )

    result = {
        "parameters": asdict(config),
        "signals": [_serialize_signal(s) for s in signals],
        "trades": [_serialize_trade(t) for t in trades],
        "daily_pnl": daily_pnl,
        "execution_summary": exec_summary,
        "metrics": metrics,
    }
    result = attach_mc_and_bootstrap(result, config)
    result = attach_prop_firm_analysis(
        result,
        prop_mc_runs=config.prop_mc_runs,
        prop_bootstrap_samples=config.prop_bootstrap_samples,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Flexing Joe ORB backtest")
    parser.add_argument("--symbol", default="NQ", help="Futures symbol")
    parser.add_argument("--data-path", required=True, help="Path to 1-min OHLCV CSV")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--variant",
        default="reentries",
        choices=["one_trade_per_day", "reentries", "one_per_direction"],
        help="Backtest variant",
    )
    parser.add_argument("--point-value", type=float, default=None)
    parser.add_argument("--tick-size", type=float, default=None)
    parser.add_argument("--commission-per-contract", type=float, default=None)
    parser.add_argument("--slippage-points", type=float, default=None)
    parser.add_argument("--initial-account-size", type=float, default=None)
    parser.add_argument("--contracts-per-trade", type=int, default=None)
    parser.add_argument("--daily-loss-limit", type=float, default=None)
    parser.add_argument("--trailing-drawdown-limit", type=float, default=None)
    parser.add_argument("--session-start-time", default=None)
    parser.add_argument("--session-end-time", default=None)
    parser.add_argument("--orb-minutes", type=int, default=None)
    parser.add_argument("--ema-period", type=int, default=None)
    parser.add_argument("--target-multiple", type=float, default=None)
    parser.add_argument("--max-entries-per-day", type=int, default=None)
    parser.add_argument("--vix-path", default=None)
    parser.add_argument("--es-path", default=None)
    parser.add_argument("--nq-path", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--mc-runs", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    args = parser.parse_args()

    overrides = _variant_to_config_flags(args.variant)
    for arg_name, config_name in {
        "point_value": "point_value",
        "tick_size": "tick_size",
        "commission_per_contract": "commission_per_contract",
        "slippage_points": "slippage_points",
        "initial_account_size": "initial_account_size",
        "contracts_per_trade": "contracts_per_trade",
        "daily_loss_limit": "daily_loss_limit",
        "trailing_drawdown_limit": "trailing_drawdown_limit",
        "session_start_time": "session_start_time",
        "session_end_time": "session_end_time",
        "orb_minutes": "orb_minutes",
        "ema_period": "ema_period",
        "target_multiple": "target_multiple",
        "max_entries_per_day": "max_entries_per_day",
        "vix_path": "vix_path",
        "es_path": "es_path",
        "nq_path": "nq_path",
        "random_seed": "random_seed",
        "mc_runs": "mc_runs",
        "bootstrap_samples": "bootstrap_samples",
    }.items():
        val = getattr(args, arg_name)
        if val is not None:
            overrides[config_name] = val

    config = StrategyConfig(
        symbol=args.symbol,
        data_path=args.data_path,
        start_date=args.start_date,
        end_date=args.end_date,
        **overrides,
    )

    result = run_backtest(config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    m = result["metrics"]
    print(f"Trades: {m['total_trades']}, Net PnL: ${m['net_pnl']:+.2f}")
    print(f"Win Rate: {m['win_rate']:.1f}%, Max DD: ${m['max_drawdown_dollars']:.2f}")
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
