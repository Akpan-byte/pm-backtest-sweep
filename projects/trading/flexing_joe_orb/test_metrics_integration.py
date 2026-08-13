"""Integration test: run a tiny real-data backtest and compute metrics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flexing_joe_orb.metrics import summarize_metrics
from flexing_joe_orb.models import Trade

sys.path.insert(0, "/config/worktrees/opencode-orb/topstep_strats")

import pandas as pd

from futures_production_engine import (
    FuturesCLOBEngine,
    FuturesConfig,
    FuturesORBSignalGenerator,
    load_ohlcv_csv,
)


def convert_trade(ft) -> Trade:
    """Convert reference engine FuturesTrade to models.Trade."""
    return Trade(
        entry_time=ft.entry_time,
        exit_time=ft.exit_time,
        direction=ft.direction,
        entry_price=ft.entry_price,
        exit_price=ft.exit_price,
        contracts=ft.contracts,
        gross_pnl=ft.gross_pnl_dollars,
        commission=ft.commission_dollars,
        slippage=ft.slippage_dollars,
        net_pnl=ft.net_pnl_dollars,
        exit_reason=ft.exit_reason,
    )


def main():
    csv_path = "/config/projects/trading/v5_orb_nq_backtest/market_data/NQ_1min.csv"
    df = load_ohlcv_csv(csv_path)
    df = df.iloc[:10_000].copy()
    print(f"Loaded {len(df)} rows from {csv_path}")

    config = FuturesConfig(symbol="NQ")
    signal_gen = FuturesORBSignalGenerator(orb_minutes=15)
    engine = FuturesCLOBEngine(config)

    signals = signal_gen.generate_signals(df)
    print(f"Generated {len(signals)} signals")

    raw_trades, exec_summary = engine.execute_signals(df, signals)
    print(f"Executed {len(raw_trades)} trades")

    trades = [convert_trade(t) for t in raw_trades]

    # Build daily_pnl map from trades.
    daily_pnl: dict[str, float] = {}
    for t in trades:
        d = t.entry_time.strftime("%Y-%m-%d")
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t.net_pnl

    summary = summarize_metrics(trades, daily_pnl, initial_equity=config.initial_account_size)
    print("Metrics summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    assert summary["total_trades"] == len(trades)
    print("Integration test completed successfully.")


if __name__ == "__main__":
    main()
