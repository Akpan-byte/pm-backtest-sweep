"""Quick smoke test: run backtest on first 10k rows of NQ data."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable as `flexing_joe_orb`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from flexing_joe_orb.backtest import run_backtest
from flexing_joe_orb.models import StrategyConfig


def main():
    config = StrategyConfig(
        symbol="NQ",
        data_path="/config/projects/trading/v5_orb_nq_backtest/market_data/NQ_1min.csv",
        start_date="2016-06-01",
        end_date="2016-07-31",
        point_value=20.0,
        tick_size=0.25,
        commission_per_contract=2.50,
        slippage_points=0.25,
        initial_account_size=50_000.0,
        contracts_per_trade=1,
        daily_loss_limit=900.0,
        trailing_drawdown_limit=2_000.0,
        max_entries_per_day=3,
        one_trade_per_day=False,
        one_trade_per_direction=False,
        mc_runs=1_000,
        bootstrap_samples=1_000,
    )

    result = run_backtest(config)

    print("Execution summary:")
    for k, v in result["execution_summary"].items():
        print(f"  {k}: {v}")

    print("\nMetrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    print("\nMonte Carlo (1k):")
    for k, v in result["monte_carlo_50k"].items():
        print(f"  {k}: {v}")

    print("\nProp firm 50k standard:")
    print(result["prop_firm_payouts"]["50k_standard"])

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
