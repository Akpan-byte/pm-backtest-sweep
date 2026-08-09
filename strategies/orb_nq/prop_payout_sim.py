#!/usr/bin/env python3
"""Prop-firm payout simulator for ORB strategies (Topstep-style rules)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def simulate_payouts(daily_pnl: dict,
                     account_start: float,
                     profit_target: float,
                     daily_loss_limit: float,
                     payout_rate: float = 0.40,
                     min_active_days: int = 5,
                     consistency_max_day_pct: float = 0.50,
                     max_payout: float | None = None) -> dict:
    """Simulate Topstep-style payouts from a daily PnL series.

    Rules modelled:
      - Daily loss limit: if a day's PnL <= -daily_loss_limit, that day's loss is
        capped at -daily_loss_limit (trading halts for the day).
      - Payout window: a payout is triggered when cumulative profit >= profit_target,
        active days >= min_active_days, and consistency rule is met.
      - Consistency: no single day's profit exceeds consistency_max_day_pct of the
        window's cumulative profit.
      - Payout amount: payout_rate * cumulative_profit, capped at max_payout.
      - After payout, cumulative profit resets to 0 and active days reset.
    """
    series = pd.Series(daily_pnl).sort_index()
    if series.empty:
        return {}

    if max_payout is None:
        max_payout = profit_target * payout_rate

    balance = account_start
    cum_profit = 0.0
    active_days = 0
    window_pnls: list[float] = []

    payouts: list[dict] = []
    first_payout_days: int | None = None

    for i, (date, raw_pnl) in enumerate(series.items()):
        # Apply daily loss limit
        pnl = max(raw_pnl, -daily_loss_limit)
        balance += pnl
        cum_profit += pnl
        active_days += 1
        window_pnls.append(pnl)

        if cum_profit >= profit_target and active_days >= min_active_days:
            window_profit = sum(window_pnls)
            if window_profit <= 0:
                continue
            max_day = max(window_pnls)
            if max_day / window_profit <= consistency_max_day_pct:
                payout = min(cum_profit * payout_rate, max_payout)
                payouts.append({
                    "date": str(date),
                    "active_days": active_days,
                    "cum_profit": float(cum_profit),
                    "payout": float(payout),
                    "max_day_pct": float(max_day / window_profit),
                })
                if first_payout_days is None:
                    first_payout_days = i + 1
                # Reset for next payout cycle
                cum_profit = 0.0
                active_days = 0
                window_pnls = []

    total_payout = sum(p["payout"] for p in payouts)
    avg_payout = total_payout / len(payouts) if payouts else 0.0
    return {
        "account_start": account_start,
        "profit_target": profit_target,
        "daily_loss_limit": daily_loss_limit,
        "payout_rate": payout_rate,
        "min_active_days": min_active_days,
        "consistency_max_day_pct": consistency_max_day_pct,
        "max_payout": max_payout,
        "total_payouts": len(payouts),
        "total_payout_dollars": float(total_payout),
        "avg_payout_dollars": float(avg_payout),
        "first_payout_days": first_payout_days,
        "final_balance": float(balance),
        "first_few_payouts": payouts[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-pnl-json", required=True,
                        help="JSON file mapping date -> daily PnL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    daily_pnl = json.loads(Path(args.daily_pnl_json).read_text())
    # Convert string keys to date objects
    daily_pnl = {pd.to_datetime(k).date(): float(v) for k, v in daily_pnl.items()}

    results = {}
    for label, account_start, profit_target, daily_loss_limit in [
        ("topstep_50k", 50000.0, 3000.0, 900.0),
        ("topstep_150k", 150000.0, 10000.0, 3000.0),
    ]:
        results[label] = simulate_payouts(
            daily_pnl,
            account_start=account_start,
            profit_target=profit_target,
            daily_loss_limit=daily_loss_limit,
            payout_rate=0.40,
            min_active_days=5,
            consistency_max_day_pct=0.50,
            max_payout=profit_target * 0.40,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
