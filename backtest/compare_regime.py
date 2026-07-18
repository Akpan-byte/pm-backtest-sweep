#!/usr/bin/env python3
"""Parse regime-filtered quant results and emit baseline-vs-regime comparison table."""
import json, os, glob, re, sys
from collections import defaultdict

BASELINES = {
    "dema_lb20_alp0001": {"pnl": 556.17, "dd_pct": 161.37, "name": "tf_dema_lb20_dev002_emax85_alp0001"},
    "vwap_ticks_lb50": {"pnl": 500.98, "dd_pct": 181.54, "name": "tf_vwap_ticks_lb50_dev002_emax80"},
    "dema_lb20_alp0002": {"pnl": 277.03, "dd_pct": 97.50, "name": "tf_dema_lb20_dev002_emax85_alp0002"},
    "alma_lb50": {"pnl": 265.24, "dd_pct": 179.21, "name": "tf_alma_lb50_dev002_emax80_alm0075_alm6"},
    "holt_lb50": {"pnl": 189.66, "dd_pct": 134.42, "name": "tf_holt_lb50_dev002_emax85_alp0002_hol0005"},
}

LEG_RE = re.compile(r"^(tf_[^_]+(?:_[^_]+){4,}?)_rV\d+")

def leg_of(name):
    m = LEG_RE.match(name)
    return m.group(1) if m else name

def load_quant(qdir):
    out = {}
    for path in glob.glob(os.path.join(qdir, "*.quant.json")):
        name = os.path.basename(path).replace(".quant.json", "")
        try:
            with open(path) as fh:
                out[name] = json.load(fh)
        except Exception as e:
            print(f"WARN: failed to load {path}: {e}", file=sys.stderr)
    return out

def pick_best(variants, baseline):
    """Pick variant maximizing PnL while reducing DD vs baseline."""
    best = None
    best_score = -1e9
    for v in variants:
        c = v.get("core", {})
        dd = v.get("drawdown", {})
        pnl = c.get("total_pnl", 0.0)
        dd_pct = dd.get("max_dd_pct", 0.0)
        # Score: PnL improvement minus penalty for DD increase.
        # Strong preference for positive PnL; smaller DD is better.
        pnl_delta = pnl - baseline["pnl"]
        dd_delta = dd_pct - baseline["dd_pct"]
        score = pnl_delta - 2.0 * max(0.0, dd_delta) + 0.01 * pnl
        if score > best_score:
            best_score = score
            best = v
    return best

def main(qdir):
    quant = load_quant(qdir)
    by_leg = defaultdict(list)
    for name, r in quant.items():
        by_leg[leg_of(name)].append(r)

    rows = []
    for leg_key, base in BASELINES.items():
        variants = by_leg.get(base["name"], [])
        best = pick_best(variants, base) if variants else None
        if best:
            bc = best.get("core", {})
            bd = best.get("drawdown", {})
            br = best.get("risk", {})
            rows.append({
                "leg": leg_key,
                "base_name": base["name"],
                "base_pnl": base["pnl"],
                "base_dd": base["dd_pct"],
                "regime_name": best["strategy"],
                "regime_pnl": bc.get("total_pnl", 0.0),
                "regime_dd": bd.get("max_dd_pct", 0.0),
                "regime_trades": best.get("n_trades", 0),
                "regime_sharpe": br.get("sharpe_per_trade", 0.0),
                "regime_psr": br.get("psr", 0.0),
            })
        else:
            rows.append({
                "leg": leg_key,
                "base_name": base["name"],
                "base_pnl": base["pnl"],
                "base_dd": base["dd_pct"],
                "regime_name": "N/A",
                "regime_pnl": 0.0,
                "regime_dd": 0.0,
                "regime_trades": 0,
                "regime_sharpe": 0.0,
                "regime_psr": 0.0,
            })

    print("| leg | baseline | base PnL$ | base DD% | regime variant | regime PnL$ | regime DD% | ΔPnL$ | ΔDD% | trades | Sharpe | PSR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dp = r["regime_pnl"] - r["base_pnl"]
        ddd = r["regime_dd"] - r["base_dd"]
        print(f"| {r['leg']} | {r['base_name']} | {r['base_pnl']:.2f} | {r['base_dd']:.1f} | "
              f"{r['regime_name']} | {r['regime_pnl']:.2f} | {r['regime_dd']:.1f} | "
              f"{dp:+.2f} | {ddd:+.1f} | {r['regime_trades']} | {r['regime_sharpe']:.3f} | {r['regime_psr']:.3f} |")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/quant_taker_trend_regime")
