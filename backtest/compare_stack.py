#!/usr/bin/env python3
"""Compare 6-leg orb_fade stack (with and without DEMA)."""
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LEGS = ["b1m_liqpull40", "b1m_imbslope30", "b1m_chand40", "l5m_mom30", "l5m_liqpull60", "l3m_bbpct40"]

leg_results = {}
for leg in LEGS:
    path = os.path.join(HERE, "results", "quant_bn", f"btc_{leg}.quant.json")
    if os.path.exists(path):
        with open(path) as f:
            leg_results[leg] = json.load(f)

if not leg_results:
    print("ERROR: No quant results found")
    exit(1)

total_pnl = sum(r.get("core", {}).get("total_pnl", 0) for r in leg_results.values())
total_trades = sum(r.get("n_trades", 0) for r in leg_results.values())
avg_wr = sum(r.get("core", {}).get("win_rate", 0) * r.get("n_trades", 0) for r in leg_results.values()) / total_trades if total_trades else 0
avg_pf = sum(r.get("core", {}).get("profit_factor", 0) for r in leg_results.values()) / len(leg_results)
avg_sharpe = sum(r.get("risk", {}).get("sharpe_per_trade", 0) for r in leg_results.values()) / len(leg_results)

print("=" * 60)
print("6-LEG ORB FADE STACK (IS, Binance)")
print("=" * 60)
print(f"  Legs: {len(LEGS)}")
print(f"  Total trades: {total_trades:,}")
print(f"  Total PnL: ${total_pnl:,.2f}")
print(f"  Avg WR: {avg_wr:.4f}")
print(f"  Avg PF: {avg_pf:.4f}")
print(f"  Avg Sharpe: {avg_sharpe:.4f}")

# Load DEMA trades
dema_path = os.path.join(HERE, "dema_trades.csv")
dema_exists = os.path.exists(dema_path)
dema_stats = None
if dema_exists:
    dema_pnls = []
    with open(dema_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dema_pnls.append(float(row["pnl"]))
            except (ValueError, KeyError):
                pass
    if dema_pnls:
        dema_n = len(dema_pnls)
        dema_total = sum(dema_pnls)
        dema_wr = sum(1 for p in dema_pnls if p > 0) / dema_n
        wins = [p for p in dema_pnls if p > 0]
        losses = [p for p in dema_pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        dema_pf = gross_profit / gross_loss
        avg = dema_total / dema_n
        var = sum((p - avg) ** 2 for p in dema_pnls) / dema_n
        dema_sharpe = avg / math.sqrt(var) if var > 0 else 0
        dema_stats = {
            "trades": dema_n, "pnl": round(dema_total, 2),
            "wr": round(dema_wr, 4), "pf": round(dema_pf, 4),
            "sharpe": round(dema_sharpe, 4),
        }
        print(f"\n  DEMA: {dema_n:,} trades, PnL=${dema_total:,.2f}, WR={dema_pf:.4f}, PF={dema_pf:.4f}")

# Combined stack + DEMA (shared wallet)
if dema_stats:
    combined_pnl = total_pnl + dema_stats["pnl"]
    combined_trades = total_trades + dema_stats["trades"]
    # Weighted avg WR
    stack_wins = avg_wr * total_trades
    dema_wins = dema_stats["wr"] * dema_stats["trades"]
    combined_wr = (stack_wins + dema_wins) / combined_trades if combined_trades else 0
    # Avg PF and Sharpe
    combined_pf = (avg_pf + dema_stats["pf"]) / 2
    combined_sharpe = (avg_sharpe + dema_stats["sharpe"]) / 2

    print(f"\n{'=' * 60}")
    print("STACK + DEMA COMBINED")
    print("=" * 60)
    print(f"  Total trades: {combined_trades:,}")
    print(f"  Total PnL: ${combined_pnl:,.2f}")
    print(f"  Avg WR: {combined_wr:.4f}")
    print(f"  Avg PF: {combined_pf:.4f}")
    print(f"  Avg Sharpe: {combined_sharpe:.4f}")

report = {
    "stack_only": {
        "legs": LEGS,
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_win_rate": round(avg_wr, 4),
        "avg_pf": round(avg_pf, 4),
        "avg_sharpe": round(avg_sharpe, 4),
        "per_leg": {
            k: {
                "trades": v.get("n_trades", 0),
                "pnl": v.get("core", {}).get("total_pnl", 0),
                "wr": v.get("core", {}).get("win_rate", 0),
                "pf": v.get("core", {}).get("profit_factor", 0),
                "sharpe": v.get("risk", {}).get("sharpe_per_trade", 0),
            }
            for k, v in leg_results.items()
        },
    }
}

if dema_stats:
    report["dema_only"] = dema_stats
    report["stack_plus_dema"] = {
        "total_trades": combined_trades,
        "total_pnl": round(combined_pnl, 2),
        "avg_win_rate": round(combined_wr, 4),
        "avg_pf": round(combined_pf, 4),
        "avg_sharpe": round(combined_sharpe, 4),
    }

with open(os.path.join(HERE, "..", "stack_comparison.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\nWrote stack_comparison.json")
