#!/usr/bin/env python3
"""
Topstep prop-farm simulator v4 — with XFA Scaling Plan.

The XFA Scaling Plan limits max contracts based on current balance:
  Balance      50K XFA   100K XFA   150K XFA
  <$1,500         2         3          3
  $1,500-$2,000   3         4          4
  $2,000-$3,000   5         5          5
  $3,000-$4,500   5        10         10
  >$4,500         5        10         15

Max contracts never increase mid-session. Changes take effect next session.
A payout that drops balance into a lower band lowers the limit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ── XFA Scaling Plan Tiers ──────────────────────────────────────────────────
# (balance_threshold, max_contracts_for_50k, max_contracts_for_100k, max_contracts_for_150k)
# Checked in order; first match wins.
XFA_SCALING_TIERS = [
    (4_500, 5, 10, 15),
    (3_000, 5, 10, 10),
    (2_000, 5,  5,  5),
    (1_500, 3,  4,  4),
    (0,     2,  3,  3),
]


def xfa_max_contracts(balance: float, account_size_label: str) -> int:
    """Return the max contracts allowed by the XFA Scaling Plan for current balance."""
    idx = {"50K": 1, "100K": 2, "150K": 3}.get(account_size_label, 1)
    for threshold, c50, c100, c150 in XFA_SCALING_TIERS:
        if balance >= threshold:
            return [c50, c100, c150][idx - 1]
    return 2  # fallback: lowest tier


@dataclass
class CostAssumptions:
    combine_purchase_50k: float = 149.0
    combine_purchase_150k: float = 299.0
    reset_fee: float = 99.0
    data_fee_monthly: float = 105.0
    commission_per_contract_roundtrip: float = 5.0


@dataclass
class TopstepRules:
    account_size_label: str
    starting_balance: float
    profit_target: float
    consistency_target: float
    mll_offset: float
    dll: float
    max_contracts: int
    payout_cap_standard: float
    payout_cap_consistency: float


@dataclass
class PayoutEvent:
    date: date
    amount: float
    balance_before: float
    balance_after: float
    path: str


@dataclass
class AccountAttempt:
    start_date: date
    mode: str  # "combine" or "xfa"
    end_date: Optional[date] = None
    outcome: str = ""  # "passed", "combine_blow", "xfa_blow"
    payouts: List[PayoutEvent] = field(default_factory=list)
    costs: float = 0.0


@dataclass
class SimResult:
    variant: str
    rules: TopstepRules
    costs: CostAssumptions
    total_payouts: float
    net_profit: float
    total_costs: float
    n_combines_purchased: int
    n_resets: int
    n_passes: int
    n_xfa_blows: int
    n_combine_blows: int
    first_payout_day: Optional[int]
    total_trading_days: int
    active_trading_days: int
    attempts: List[AccountAttempt]
    daily_log: pd.DataFrame
    reset_after_payout: bool = False

    def summary(self) -> dict:
        return {
            "variant": self.variant,
            "account_size": self.rules.account_size_label,
            "total_payouts": round(self.total_payouts, 2),
            "total_costs": round(self.total_costs, 2),
            "net_profit": round(self.net_profit, 2),
            "n_combines_purchased": self.n_combines_purchased,
            "n_resets": self.n_resets,
            "n_passes": self.n_passes,
            "n_xfa_blows": self.n_xfa_blows,
            "n_combine_blows": self.n_combine_blows,
            "first_payout_day": self.first_payout_day,
            "total_trading_days": self.total_trading_days,
            "active_trading_days": self.active_trading_days,
            "reset_after_payout": self.reset_after_payout,
        }


def load_daily_pnl(path: str, commission_per_contract: float, pnl_scale: float = 1.0) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)

    trades = data.get("trades", [])
    comm_by_date: Dict[date, float] = {}
    for t in trades:
        d = pd.to_datetime(t.get("entry_time") or t.get("exit_time")).date()
        qty = int(t.get("qty", 1))
        comm_by_date[d] = comm_by_date.get(d, 0.0) + commission_per_contract * qty * pnl_scale

    rows = []
    for r in data["day_results"]:
        d = pd.to_datetime(r["date"]).date()
        gross = float(r.get("day_pnl", 0.0)) * pnl_scale
        comm = comm_by_date.get(d, 0.0)
        rows.append({"date": d, "gross_pnl": gross, "commission": comm, "net_pnl": gross - comm})
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def make_rules(account_size: str) -> TopstepRules:
    if account_size == "50K":
        return TopstepRules(
            account_size_label="50K",
            starting_balance=50_000.0,
            profit_target=3_000.0,
            consistency_target=0.50,
            mll_offset=2_000.0,
            dll=1_000.0,
            max_contracts=5,
            payout_cap_standard=4_000.0,
            payout_cap_consistency=6_000.0,
        )
    else:
        return TopstepRules(
            account_size_label="150K",
            starting_balance=150_000.0,
            profit_target=9_000.0,
            consistency_target=0.50,
            mll_offset=4_500.0,
            dll=3_000.0,
            max_contracts=15,
            payout_cap_standard=10_000.0,
            payout_cap_consistency=12_000.0,
        )


def simulate_prop_farm(
    daily_pnl: pd.DataFrame,
    variant: str,
    rules: TopstepRules,
    costs: CostAssumptions,
    path: str,
    reset_after_payout: bool = False,
    desired_contracts: int = 1,
) -> SimResult:
    """Replay daily PnL through repeated Combine/XFA lifecycle.

    desired_contracts: how many contracts the strategy wants to trade.
    The XFA Scaling Plan caps this based on current balance.
    """
    df = daily_pnl.copy()
    df["net_pnl"] = df["net_pnl"].fillna(0.0)

    # State
    mode = "combine"
    combine_balance = rules.starting_balance
    xfa_balance = 0.0
    xfa_mll = -rules.mll_offset
    xfa_mll_locked = False
    combine_high = combine_balance
    combine_mll = combine_balance - rules.mll_offset
    best_day_combine = 0.0
    total_profit_combine = 0.0

    # XFA scaling: track effective contracts this session
    xfa_session_max_contracts = 2  # start of XFA = lowest tier

    # Counters
    n_combines_purchased = 0
    n_resets = 0
    n_passes = 0
    n_xfa_blows = 0
    n_combine_blows = 0
    total_payouts = 0.0
    total_costs = 0.0
    first_payout_day = None

    # XFA path counters
    winning_days_since_payout = 0
    trading_days_since_payout = 0
    daily_pnls_since_payout: List[float] = []

    attempts: List[AccountAttempt] = []
    current_attempt: Optional[AccountAttempt] = None
    cooldown_until: Optional[date] = None

    daily_log_rows: List[dict] = []
    trading_day_index = 0
    active_trading_days = 0

    def buy_combine(start_d: date) -> None:
        nonlocal mode, combine_balance, combine_high, combine_mll, best_day_combine, total_profit_combine
        nonlocal current_attempt, n_combines_purchased, total_costs
        mode = "combine"
        combine_balance = rules.starting_balance
        combine_high = combine_balance
        combine_mll = combine_balance - rules.mll_offset
        best_day_combine = 0.0
        total_profit_combine = 0.0
        fee = costs.combine_purchase_50k if rules.account_size_label == "50K" else costs.combine_purchase_150k
        n_combines_purchased += 1
        total_costs += fee
        current_attempt = AccountAttempt(start_date=start_d, mode="combine", costs=fee)
        attempts.append(current_attempt)

    def reset_or_buy_new(start_d: date) -> None:
        nonlocal n_resets, total_costs
        if current_attempt and current_attempt.mode == "combine" and current_attempt.outcome == "combine_blow":
            n_resets += 1
            total_costs += costs.reset_fee
            current_attempt.costs += costs.reset_fee
            current_attempt.end_date = start_d
            buy_combine(start_d)
        else:
            buy_combine(start_d)

    def start_xfa(start_d: date) -> None:
        nonlocal mode, xfa_balance, xfa_mll, xfa_mll_locked
        nonlocal winning_days_since_payout, trading_days_since_payout, daily_pnls_since_payout
        nonlocal xfa_session_max_contracts, current_attempt
        mode = "xfa"
        xfa_balance = 0.0
        xfa_mll = -rules.mll_offset
        xfa_mll_locked = False
        winning_days_since_payout = 0
        trading_days_since_payout = 0
        daily_pnls_since_payout = []
        # XFA starts at $0 balance → lowest scaling tier
        xfa_session_max_contracts = xfa_max_contracts(0.0, rules.account_size_label)
        current_attempt = AccountAttempt(start_date=start_d, mode="xfa")
        attempts.append(current_attempt)

    def update_xfa_scaling() -> None:
        """Recalculate max contracts for next session based on current balance."""
        nonlocal xfa_session_max_contracts
        xfa_session_max_contracts = xfa_max_contracts(xfa_balance, rules.account_size_label)

    # Start first Combine
    first_date = df.index[0]
    buy_combine(first_date)

    for ts, row in df.iterrows():
        d = ts if isinstance(ts, date) else ts.date()
        trading_day_index += 1

        # Cooldown check
        if cooldown_until is not None and d <= cooldown_until:
            daily_log_rows.append({
                "date": d, "mode": "cooldown", "pnl": 0.0,
                "balance": 0.0, "mll": 0.0, "event": "cooldown",
            })
            continue

        cooldown_until = None
        active_trading_days += 1

        # Daily data fee prorated
        total_costs += costs.data_fee_monthly / 21.0
        if current_attempt:
            current_attempt.costs += costs.data_fee_monthly / 21.0

        raw_pnl = float(row["net_pnl"])

        # Daily stop
        if path == "standard":
            daily_stop = 900.0 if rules.account_size_label == "50K" else 3000.0
        else:
            daily_stop = rules.payout_cap_consistency * 0.50

        if mode == "combine":
            # DLL cap
            dll_hit = False
            if raw_pnl < -rules.dll:
                effective_pnl = -rules.dll
                dll_hit = True
            else:
                effective_pnl = raw_pnl

            # Daily stop cap
            if daily_stop > 0 and effective_pnl > daily_stop:
                effective_pnl = daily_stop

            combine_balance += effective_pnl

            # Update trailing MLL
            if combine_balance > combine_high:
                combine_high = combine_balance
                combine_mll = combine_high - rules.mll_offset

            # Consistency tracking
            if effective_pnl > 0:
                total_profit_combine += effective_pnl
                if effective_pnl > best_day_combine:
                    best_day_combine = effective_pnl

            consistency = best_day_combine / total_profit_combine if total_profit_combine > 0 else 1.0
            net_profit = combine_balance - rules.starting_balance
            passed = net_profit >= rules.profit_target and consistency <= rules.consistency_target
            blown = combine_balance <= combine_mll

            daily_log_rows.append({
                "date": d, "mode": "combine", "pnl": round(effective_pnl, 2),
                "balance": round(combine_balance, 2), "mll": round(combine_mll, 2),
                "dll_hit": dll_hit, "event": "pass" if passed else ("blow" if blown else ""),
            })

            if passed:
                n_passes += 1
                if current_attempt:
                    current_attempt.outcome = "passed"
                    current_attempt.end_date = d
                start_xfa(d)
                continue

            if blown:
                n_combine_blows += 1
                if current_attempt:
                    current_attempt.outcome = "combine_blow"
                    current_attempt.end_date = d
                cooldown_until = d + timedelta(days=1)
                reset_or_buy_new(cooldown_until + timedelta(days=1))
                continue

        else:  # xfa
            # ── XFA Scaling Plan: determine effective contracts ──
            max_contracts_allowed = xfa_session_max_contracts
            effective_contracts = min(desired_contracts, max_contracts_allowed)
            # Scale factor: what fraction of desired size we can actually trade
            scale_factor = effective_contracts / desired_contracts if desired_contracts > 0 else 1.0

            # Apply scale to PnL
            scaled_pnl = raw_pnl * scale_factor

            # DLL cap
            dll_hit = False
            if scaled_pnl < -rules.dll:
                effective_pnl = -rules.dll
                dll_hit = True
            else:
                effective_pnl = scaled_pnl

            # Daily stop cap
            if daily_stop > 0 and effective_pnl > daily_stop:
                effective_pnl = daily_stop

            xfa_balance += effective_pnl

            # Update trailing MLL
            if not xfa_mll_locked:
                candidate_mll = xfa_balance - rules.mll_offset
                if candidate_mll > xfa_mll:
                    xfa_mll = candidate_mll
                if xfa_mll >= 0:
                    xfa_mll = 0.0
                    xfa_mll_locked = True
            else:
                xfa_mll = 0.0

            blown = xfa_balance <= xfa_mll

            daily_log_rows.append({
                "date": d, "mode": "xfa", "pnl": round(effective_pnl, 2),
                "balance": round(xfa_balance, 2), "mll": round(xfa_mll, 2),
                "dll_hit": dll_hit, "event": "blow" if blown else "",
                "scaling_contracts": effective_contracts,
                "scaling_max": max_contracts_allowed,
            })

            if blown:
                n_xfa_blows += 1
                if current_attempt:
                    current_attempt.outcome = "xfa_blow"
                    current_attempt.end_date = d
                cooldown_until = d + timedelta(days=1)
                reset_or_buy_new(cooldown_until + timedelta(days=1))
                continue

            # Update scaling for NEXT session
            update_xfa_scaling()

            # Payout eligibility
            if effective_pnl > 150:
                winning_days_since_payout += 1
            trading_days_since_payout += 1
            daily_pnls_since_payout.append(effective_pnl)

            eligible = False
            if path == "standard":
                if winning_days_since_payout >= 5:
                    eligible = True
            else:
                if trading_days_since_payout >= 3:
                    total_profit_since = sum(p for p in daily_pnls_since_payout if p > 0)
                    if total_profit_since > 0:
                        largest_day = max(daily_pnls_since_payout)
                        consistency_since = largest_day / total_profit_since
                        if consistency_since <= 0.40:
                            eligible = True

            if eligible and xfa_balance > 0:
                cap = rules.payout_cap_standard if path == "standard" else rules.payout_cap_consistency
                payout_amount = min(xfa_balance * 0.5, cap)
                if payout_amount >= 125:
                    balance_before = xfa_balance
                    xfa_balance -= payout_amount
                    total_payouts += payout_amount
                    if current_attempt:
                        ev = PayoutEvent(d, payout_amount, balance_before, xfa_balance, path)
                        current_attempt.payouts.append(ev)
                    winning_days_since_payout = 0
                    trading_days_since_payout = 0
                    daily_pnls_since_payout = []
                    if first_payout_day is None:
                        first_payout_day = trading_day_index

                    # Recalc scaling after payout drops balance
                    update_xfa_scaling()

                    if reset_after_payout:
                        total_payouts += xfa_balance
                        if current_attempt:
                            ev = PayoutEvent(d, xfa_balance, xfa_balance, 0.0, path + "_withdrawal")
                            current_attempt.payouts.append(ev)
                            current_attempt.outcome = "reset_after_payout"
                            current_attempt.end_date = d
                        xfa_balance = 0.0
                        cooldown_until = d + timedelta(days=1)
                        buy_combine(cooldown_until + timedelta(days=1))
                        continue
                    else:
                        xfa_mll = 0.0
                        xfa_mll_locked = True

    net_profit = total_payouts - total_costs

    daily_log = pd.DataFrame(daily_log_rows)
    if not daily_log.empty:
        daily_log = daily_log.set_index("date")

    return SimResult(
        variant=variant,
        rules=rules,
        costs=costs,
        total_payouts=total_payouts,
        net_profit=net_profit,
        total_costs=total_costs,
        n_combines_purchased=n_combines_purchased,
        n_resets=n_resets,
        n_passes=n_passes,
        n_xfa_blows=n_xfa_blows,
        n_combine_blows=n_combine_blows,
        first_payout_day=first_payout_day,
        total_trading_days=trading_day_index,
        active_trading_days=active_trading_days,
        attempts=attempts,
        daily_log=daily_log,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Topstep prop-farm simulator v4 with XFA Scaling Plan")
    parser.add_argument("--input", required=True, help="Backtest result JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--account-size", choices=["50K", "150K"], default="50K")
    parser.add_argument("--path", choices=["standard", "consistency"], default="standard")
    parser.add_argument("--variant", required=True, help="Variant label for summary")
    parser.add_argument("--combine-purchase-50k", type=float, default=149.0)
    parser.add_argument("--combine-purchase-150k", type=float, default=299.0)
    parser.add_argument("--reset-fee", type=float, default=99.0)
    parser.add_argument("--data-fee-monthly", type=float, default=105.0)
    parser.add_argument("--commission-per-contract", type=float, default=5.0)
    parser.add_argument("--reset-after-payout", action="store_true")
    parser.add_argument("--pnl-scale", type=float, default=1.0,
                        help="Scale daily PnL by this factor (position-size multiplier)")
    parser.add_argument("--desired-contracts", type=int, default=1,
                        help="How many contracts the strategy wants to trade (capped by XFA scaling)")
    args = parser.parse_args()

    costs = CostAssumptions(
        combine_purchase_50k=args.combine_purchase_50k,
        combine_purchase_150k=args.combine_purchase_150k,
        reset_fee=args.reset_fee,
        data_fee_monthly=args.data_fee_monthly,
        commission_per_contract_roundtrip=args.commission_per_contract,
    )
    rules = make_rules(args.account_size)
    df = load_daily_pnl(args.input, costs.commission_per_contract_roundtrip, pnl_scale=args.pnl_scale)
    result = simulate_prop_farm(
        df, args.variant, rules, costs, args.path,
        reset_after_payout=args.reset_after_payout,
        desired_contracts=args.desired_contracts,
    )

    summary = result.summary()
    summary["desired_contracts"] = args.desired_contracts
    summary["payouts"] = [
        {"date": str(ev.date), "amount": ev.amount, "balance_before": ev.balance_before, "balance_after": ev.balance_after, "path": ev.path}
        for attempt in result.attempts
        for ev in attempt.payouts
    ]
    summary["attempts"] = [
        {
            "start_date": str(a.start_date),
            "end_date": str(a.end_date) if a.end_date else None,
            "mode": a.mode,
            "outcome": a.outcome,
            "costs": round(a.costs, 2),
            "n_payouts": len(a.payouts),
            "payout_total": round(sum(ev.amount for ev in a.payouts), 2),
        }
        for a in result.attempts
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
