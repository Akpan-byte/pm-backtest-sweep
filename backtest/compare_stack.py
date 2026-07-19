#!/usr/bin/env python3
"""Compare 6-leg orb_fade stack (with and without DEMA)."""
import json
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

dema_path = os.path.join(HERE, "dema_trades.csv")
dema_exists = os.path.exists(dema_path)
print(f"  DEMA trades available: {dema_exists}")

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
with open(os.path.join(HERE, "..", "stack_comparison.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\nWrote stack_comparison.json")
