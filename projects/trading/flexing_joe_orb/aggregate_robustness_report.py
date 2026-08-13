#!/usr/bin/env python3
"""Aggregate robustness validation reports into a single markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def fmt_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.2f}"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    return str(v)


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n\n"


def build_report(files: List[str], real_regime_files: List[str] = None) -> str:
    reports = [load_json(f) for f in files]
    real_regime_files = real_regime_files or []

    out = "# Robustness, Statistical Edge, Regime & Walk-Forward Validation Report\n\n"
    out += "**Generated:** from existing backtest outputs using the Gemini Spark / `futures_production_engine` checklist.\n\n"
    out += "**Pass thresholds:** DSR > 0.40 | WRC p < 0.15 | FDR q < 0.15 | MC p5 > -$50 | Param stability ≥ 50.\n\n"

    # Summary table
    rows = []
    for r in reports:
        p = r.get("parameters", {})
        v = r.get("validation", {})
        label = f"{p.get('symbol','?')} c{p.get('contracts_per_trade','?')} me{p.get('max_entries_per_day','?')}"
        rows.append({
            "Config": label,
            "Trades": v.get("total_trades", 0),
            "Win%": v.get("win_rate", 0),
            "Net PnL": v.get("net_pnl", 0),
            "Sharpe": v.get("sharpe_ratio", 0),
            "Sortino": v.get("sortino_ratio", 0),
            "DSR": v.get("dsr", 0),
            "WRC p": v.get("wrc_pvalue", 0),
            "FDR q": v.get("fdr_qvalue", 0),
            "MC p5": v.get("mc_p5", 0),
            "Stability": v.get("param_stability_score", 0),
            "Pass": v.get("pass", False),
            "Fail reasons": ", ".join(v.get("fail_reasons", [])),
            "Max DD%": v.get("max_drawdown_pct", 0),
        })

    out += "## 1. Validation Summary\n\n"
    out += "| Config | Trades | Win% | Net PnL | Sharpe | Sortino | DSR | WRC p | FDR q | MC p5 | Stability | Pass | Fail reasons | Max DD% |\n"
    out += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|\n"
    for row in rows:
        out += f"| {row['Config']} | {row['Trades']:,} | {row['Win%']:.2f} | ${row['Net PnL']:,.2f} | {row['Sharpe']:.3f} | {row['Sortino']:.3f} | {row['DSR']:.4f} | {row['WRC p']:.4f} | {row['FDR q']:.4f} | ${row['MC p5']:,.2f} | {row['Stability']:.1f} | {row['Pass']} | {row['Fail reasons']} | {row['Max DD%']:.2f} |\n"
    out += "\n"

    # Per-config details
    out += "## 2. Per-Config Validation Details\n\n"
    for r in reports:
        p = r.get("parameters", {})
        v = r.get("validation", {})
        label = f"{p.get('symbol','?')} contracts={p.get('contracts_per_trade','?')} max_entries={p.get('max_entries_per_day','?')}"
        out += f"### {label}\n\n"
        out += f"- **Source:** `{r.get('source_file')}`\n"
        out += f"- **Period:** {p.get('start_date')} → {p.get('end_date')}\n"
        out += f"- **Trades:** {v.get('total_trades'):,} | Win rate: {v.get('win_rate')}% | Net PnL: ${v.get('net_pnl'):,.2f}\n"
        out += f"- **Sharpe:** {v.get('sharpe_ratio')} | **Sortino:** {v.get('sortino_ratio')}\n"
        out += f"- **Skew:** {v.get('skew')} | **Kurtosis:** {v.get('kurtosis')}\n"
        out += f"- **DSR:** {v.get('dsr')} | **WRC p:** {v.get('wrc_pvalue')} | **FDR q:** {v.get('fdr_qvalue')}\n"
        out += f"- **Monte Carlo (50k, 5% noise):** p5=${v.get('mc_p5'):,.2f}, p50=${v.get('mc_p50'):,.2f}, mean=${v.get('mc_mean'):,.2f}, p95=${v.get('mc_p95'):,.2f}\n"
        out += f"- **Param stability score:** {v.get('param_stability_score')} (threshold 50)\n"
        out += f"- **Max drawdown:** {v.get('max_drawdown_pct')}%\n"
        out += f"- **Pass:** {v.get('pass')} | **Fail reasons:** {', '.join(v.get('fail_reasons', []))}\n\n"

    # Walk-forward
    out += "## 3. Walk-Forward / Out-of-Sample Stability\n\n"
    for r in reports:
        p = r.get("parameters", {})
        label = f"{p.get('symbol','?')} c{p.get('contracts_per_trade','?')}"
        wf = r.get("walk_forward", [])
        if not wf:
            out += f"No walk-forward windows for {label}.\n\n"
            continue
        out += f"### {label} — {len(wf)} rolling windows (2yr train / 6mo test)\n\n"
        out += "| Train | Test | Train PnL | Test PnL | Train Sharpe | Test Sharpe | Train WR | Test WR |\n"
        out += "|---|---|---:|---:|---:|---:|---:|---:|\n"
        for w in wf:
            out += (
                f"| {w['train_start']}→{w['train_end']} | {w['test_start']}→{w['test_end']} | "
                f"${w['train_net_pnl']:,.2f} | ${w['test_net_pnl']:,.2f} | "
                f"{w['train_sharpe']:.3f} | {w['test_sharpe']:.3f} | "
                f"{w['train_win_rate']:.1f}% | {w['test_win_rate']:.1f}% |\n"
            )
        # OOS stability stats
        train_sharpes = [w['train_sharpe'] for w in wf]
        test_sharpes = [w['test_sharpe'] for w in wf]
        train_pnl = sum(w['train_net_pnl'] for w in wf)
        test_pnl = sum(w['test_net_pnl'] for w in wf)
        out += f"\n**Aggregate OOS:** Train PnL ${train_pnl:,.2f} | Test PnL ${test_pnl:,.2f} | Avg train Sharpe {sum(train_sharpes)/len(train_sharpes):.3f} | Avg test Sharpe {sum(test_sharpes)/len(test_sharpes):.3f}\n\n"

    # Regime analysis
    out += "## 4. Regime Analysis (Daily-PnL Sign Persistence Proxy)\n\n"
    out += "> **Caution:** The regime labels below are defined by consecutive winning/losing daily PnL streaks (`up_trend` = two consecutive winning days, `down_trend` = two consecutive losing days, `chop` = mixed). This is a coarse proxy, not a true market regime filter based on price action, volatility, or trend indicators. Results should be interpreted as descriptive only.\n\n"
    for r in reports:
        p = r.get("parameters", {})
        label = f"{p.get('symbol','?')} c{p.get('contracts_per_trade','?')}"
        ra = r.get("regime_analysis", {})
        regimes = ra.get("regimes", {})
        if not regimes:
            continue
        out += f"### {label}\n\n"
        out += "| Regime | Days | Win Days | Win% | Avg Daily PnL | Total PnL | Trades/Day |\n"
        out += "|---|---:|---:|---:|---:|---:|---:|\n"
        for name, g in regimes.items():
            out += f"| {name} | {g['days']:,} | {g['win_days']:,} | {g['win_rate_pct']}% | ${g['avg_daily_pnl']:,.2f} | ${g['total_pnl']:,.2f} | {g['trades_per_day']:.2f} |\n"
        out += f"\nStreaks: avg win streak {ra.get('avg_win_streak_len')} (max {ra.get('max_win_streak_len')}), avg loss streak {ra.get('avg_loss_streak_len')} (max {ra.get('max_loss_streak_len')}).\n\n"

    # Real regime analysis
    if real_regime_files:
        out += "## 5. Real Market Regime Analysis (from 1-min OHLCV)\n\n"
        out += "Regimes computed from daily price action: gap size, 20-day ATR-relative range, 5-day trend, close vs EMA20, day-of-week, and month.\n\n"
        for rr in real_regime_files:
            data = load_json(rr)
            source = Path(rr).stem.replace("real_regime_", "")
            out += f"### {source}\n\n"
            regimes = data.get("regime_analysis", {})
            for regime_type, groups in regimes.items():
                if not groups:
                    continue
                out += f"**{regime_type}**\n\n"
                out += "| Regime | Days | Traded | Win% | Avg Daily PnL | Total PnL |\n"
                out += "|---|---:|---:|---:|---:|---:|\n"
                for name, g in groups.items():
                    out += (
                        f"| {name} | {g['days']:,} | {g['traded_days']:,} | "
                        f"{g['win_rate_pct']}% | ${g['avg_daily_pnl']:,.2f} | ${g['total_pnl']:,.2f} |\n"
                    )
                out += "\n"

    # Interpretation
    out += "## 6. Interpretation & Red Flags\n\n"
    out += "- **Profitability is extreme.** Net PnL figures in the millions over 10 years on a $50k account imply returns far above any realistic futures strategy. This is the strongest warning sign of overfitting, lookahead bias, or a bug in the simulation.\n"
    out += "- **Sharpe ratios are implausibly high.** Sharpe values of 8–25 on per-trade PnL are not characteristic of any real-world intraday futures strategy.\n"
    out += "- **Statistical tests pass with p-values at the floor.** WRC and FDR p-values hit the minimum resolvable value (0.0005), driven by the enormous sample mean rather than by genuine edge.\n"
    out += "- **Param stability fails for every config.** The stability score compares observed Sharpe to the expected maximum Sharpe under 50,000 random trials. Failing this gate across all configs suggests the reported Sharpe is not stable relative to the search space.\n"
    out += "- **Real regime analysis shows volatility dependence.** High-volatility days are strongly profitable; low-volatility days are flat or negative. This is a plausible edge but also implies the strategy will underperform in calm markets.\n"
    out += "- **Recommendation:** Do not trade live until (a) the simulation is independently audited for lookahead bias/slippage assumptions, (b) the strategy is validated on true hold-out data and alternative market regimes, and (c) the Sharpe/PnL magnitudes are reconciled to realistic futures returns.\n\n"

    # Next steps
    out += "## 7. Required Next Steps Before Live Trading\n\n"
    out += "1. **Audit execution for lookahead bias.** Verify entries use only bars that have closed and that slippage/commission assumptions are conservative.\n"
    out += "2. **Test on true out-of-sample data.** Hold back the most recent 1–2 years entirely from any optimization.\n"
    out += "3. **Implement real regime filters.** Test volatility (ATR), trend (EMA slope), gap size, prior-day structure, VIX, and cross-instrument alignment.\n"
    out += "4. **Reconcile PnL magnitudes.** If results remain in the millions, the strategy is almost certainly overfit or contains a simulation error.\n"
    out += "5. **Paper trade on Topstep.** Only after the above steps; use the smallest account size and strict risk limits.\n"
    out += "6. **Re-run this validation suite** on the audited simulation and compare pass/fail outcomes.\n\n"

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Robustness JSON files")
    parser.add_argument("--regime-files", nargs="+", default=None, help="Optional real_regime JSON files")
    parser.add_argument("--output", required=True, help="Output markdown path")
    args = parser.parse_args()

    md = build_report(args.inputs, real_regime_files=args.regime_files)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
