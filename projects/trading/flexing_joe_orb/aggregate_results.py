#!/usr/bin/env python3
"""Aggregate chunked backtest results into a final 10-year report.

Recomputes metrics, Monte Carlo, bootstrap CI, and prop-firm payouts on the
combined trade list and daily PnL series.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from flexing_joe_orb.mc_bootstrap import attach_mc_and_bootstrap
from flexing_joe_orb.metrics import summarize_metrics
from flexing_joe_orb.models import StrategyConfig, Trade
from flexing_joe_orb.prop_firm import attach_prop_firm_analysis


def _load_chunk_results(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Group chunk JSONs by (instrument, variant)."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        key = f"{data['parameters']['symbol']}_{data['parameters'].get('variant', 'unknown')}"
        grouped.setdefault(key, []).append(data)
    return grouped


def _combine(
    group: List[Dict[str, Any]],
    mc_runs: int = 50_000,
    bootstrap_samples: int = 50_000,
    prop_mc_runs: int = 20_000,
    prop_bootstrap_samples: int = 20_000,
) -> Dict[str, Any]:
    """Combine trades and daily PnL from a group of chunks."""
    all_trades: List[Trade] = []
    daily_pnl: Dict[str, float] = {}
    parameters: Dict[str, Any] = {}

    for chunk in group:
        parameters = chunk["parameters"]
        for t in chunk["trades"]:
            trade = Trade(
                entry_time=pd.to_datetime(t["entry_time"]),
                exit_time=pd.to_datetime(t["exit_time"]),
                direction=int(t["direction"]),
                entry_price=float(t["entry_price"]),
                exit_price=float(t["exit_price"]),
                contracts=int(t["contracts"]),
                gross_pnl=float(t["gross_pnl"]),
                commission=float(t["commission"]),
                slippage=float(t["slippage"]),
                net_pnl=float(t["net_pnl"]),
                exit_reason=t["exit_reason"],
            )
            all_trades.append(trade)
            d = trade.entry_time.strftime("%Y-%m-%d")
            daily_pnl[d] = daily_pnl.get(d, 0.0) + trade.net_pnl

    all_trades.sort(key=lambda t: t.entry_time)
    daily_pnl = {k: round(v, 2) for k, v in sorted(daily_pnl.items())}

    initial_equity = float(parameters.get("initial_account_size", 50_000.0))
    metrics = summarize_metrics(all_trades, daily_pnl, initial_equity=initial_equity)

    result = {
        "parameters": parameters,
        "trades": [
            {
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
            for t in all_trades
        ],
        "daily_pnl": daily_pnl,
        "metrics": metrics,
    }

    # Reconstruct a minimal StrategyConfig for MC/bootstrap seed/count.
    cfg = StrategyConfig(
        symbol=parameters.get("symbol", "NQ"),
        data_path=parameters.get("data_path", ""),
        start_date=parameters.get("start_date"),
        end_date=parameters.get("end_date"),
        point_value=float(parameters.get("point_value", 20.0)),
        tick_size=float(parameters.get("tick_size", 0.25)),
        commission_per_contract=float(parameters.get("commission_per_contract", 2.5)),
        slippage_points=float(parameters.get("slippage_points", 0.25)),
        initial_account_size=initial_equity,
        contracts_per_trade=int(parameters.get("contracts_per_trade", 1)),
        daily_loss_limit=float(parameters.get("daily_loss_limit", 900.0)),
        trailing_drawdown_limit=float(parameters.get("trailing_drawdown_limit", 2000.0)),
        session_start_time=parameters.get("session_start_time", "09:30"),
        session_end_time=parameters.get("session_end_time", "16:00"),
        orb_minutes=int(parameters.get("orb_minutes", 30)),
        ema_period=int(parameters.get("ema_period", 20)),
        target_multiple=float(parameters.get("target_multiple", 2.0)),
        max_entries_per_day=int(parameters.get("max_entries_per_day", 999)),
        one_trade_per_day=bool(parameters.get("one_trade_per_day", False)),
        one_trade_per_direction=bool(parameters.get("one_trade_per_direction", False)),
        mc_runs=mc_runs,
        bootstrap_samples=bootstrap_samples,
        prop_mc_runs=prop_mc_runs,
        prop_bootstrap_samples=prop_bootstrap_samples,
        random_seed=int(parameters.get("random_seed", 42)),
    )

    result = attach_mc_and_bootstrap(result, cfg)
    result = attach_prop_firm_analysis(
        result,
        prop_mc_runs=prop_mc_runs,
        prop_bootstrap_samples=prop_bootstrap_samples,
    )
    return result


def _combine_and_write(
    args: Tuple[str, List[Dict[str, Any]], Path, int, int, int, int]
) -> Dict[str, Any]:
    key, group, output_dir, mc_runs, bootstrap_samples, prop_mc_runs, prop_bootstrap_samples = args
    combined = _combine(
        group,
        mc_runs=mc_runs,
        bootstrap_samples=bootstrap_samples,
        prop_mc_runs=prop_mc_runs,
        prop_bootstrap_samples=prop_bootstrap_samples,
    )
    out_path = output_dir / f"final_{key}.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    m = combined["metrics"]
    pf = combined["prop_firm_payouts"]
    return {
        "instrument_variant": key,
        "total_trades": m["total_trades"],
        "net_pnl": m["net_pnl"],
        "win_rate": m["win_rate"],
        "sharpe": m["sharpe"],
        "max_drawdown_dollars": m["max_drawdown_dollars"],
        "max_drawdown_pct": m["max_drawdown_pct"],
        "50k_standard_payouts": pf["50k_standard"]["total_payouts"],
        "50k_consistency_payouts": pf["50k_consistency"]["total_payouts"],
        "100k_standard_payouts": pf["100k_standard"]["total_payouts"],
        "100k_consistency_payouts": pf["100k_consistency"]["total_payouts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Flexing Joe ORB chunks")
    parser.add_argument("--results-dir", required=True, help="Directory with chunk JSONs")
    parser.add_argument("--output-dir", default="results", help="Directory for final reports")
    parser.add_argument("--mc-runs", type=int, default=50_000, help="Monte Carlo runs on combined data")
    parser.add_argument("--bootstrap-samples", type=int, default=50_000, help="Bootstrap samples on combined data")
    parser.add_argument("--prop-mc-runs", type=int, default=20_000, help="Prop-firm Monte Carlo runs on combined data")
    parser.add_argument("--prop-bootstrap-samples", type=int, default=20_000, help="Prop-firm bootstrap samples on combined data")
    parser.add_argument("--workers", type=int, default=8, help="Parallel aggregation workers")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _load_chunk_results(results_dir)

    tasks: List[Tuple[str, List[Dict[str, Any]], Path, int, int, int, int]] = [
        (
            key,
            group,
            output_dir,
            args.mc_runs,
            args.bootstrap_samples,
            args.prop_mc_runs,
            args.prop_bootstrap_samples,
        )
        for key, group in grouped.items()
        if group
    ]

    summary_rows: List[Dict[str, Any]] = []
    print(f"Aggregating {len(tasks)} result sets with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_combine_and_write, t): t[0] for t in tasks}
        for future in as_completed(futures):
            key = futures[future]
            row = future.result()
            summary_rows.append(row)
            print(f"Wrote final_{key}.json")

    summary_rows.sort(key=lambda r: r["instrument_variant"])

    summary_path = output_dir / "final_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)
    print(f"Wrote summary {summary_path}")

    # Print a readable table.
    print("\nFinal Summary")
    print("-" * 140)
    print(
        f"{'Instrument/Variant':<30} {'Trades':>8} {'Net PnL':>12} {'Win%':>7} "
        f"{'Sharpe':>8} {'MaxDD$':>10} {'50kStd':>8} {'50kCon':>8} {'100kStd':>9} {'100kCon':>9}"
    )
    print("-" * 140)
    for row in summary_rows:
        print(
            f"{row['instrument_variant']:<30} {row['total_trades']:>8} {row['net_pnl']:>+12.0f} "
            f"{row['win_rate']:>7.1f} {row['sharpe']:>8.2f} {row['max_drawdown_dollars']:>10.0f} "
            f"{row['50k_standard_payouts']:>8} {row['50k_consistency_payouts']:>8} "
            f"{row['100k_standard_payouts']:>9} {row['100k_consistency_payouts']:>9}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
