#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-24  coder
#   - Created results aggregator for chunked backtest output.
# WHY: After all chunks complete, merge into a single report with rankings.

"""
Aggregate chunked backtest results into a final report.

Usage:
  python aggregate_results.py --results_dir results/
  python aggregate_results.py --results_dir all_results/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aggregate")


def load_chunks(results_dir: Path) -> list[dict]:
    """Load all chunk JSON files."""
    chunks = []
    for path in sorted(results_dir.rglob("*.json")):
        if "runner_summary" in path.name or "final_report" in path.name:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if "results" in data:
                chunks.append(data)
                logger.info("Loaded %s: %d results", path.name, len(data["results"]))
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
    return chunks


def aggregate(chunks: list[dict]) -> dict:
    """Merge all chunks into one report."""
    all_results = []
    all_errors = []

    for chunk in chunks:
        all_results.extend(chunk.get("results", []))
        all_errors.extend(chunk.get("errors", []))

    logger.info("Total results: %d, Total errors: %d", len(all_results), len(all_errors))

    # Deduplicate by config_id (keep best if reruns exist)
    seen = {}
    for r in all_results:
        cid = r.get("config_id", "")
        if cid not in seen or r.get("trade_count", 0) > seen[cid].get("trade_count", 0):
            seen[cid] = r
    deduped = list(seen.values())
    logger.info("Unique configs: %d", len(deduped))

    # Rankings
    def safe_metric(r, *path, default=0.0):
        v = r
        for p in path:
            if isinstance(v, dict):
                v = v.get(p, default)
            else:
                return default
        return v if isinstance(v, (int, float)) else default

    # Filter to configs with trades
    traded = [r for r in deduped if r.get("trade_count", 0) > 0]
    logger.info("Configs with trades: %d / %d", len(traded), len(deduped))

    # Sort by net profit
    by_profit = sorted(traded, key=lambda r: r.get("final_balance", 0), reverse=True)

    # Sort by Sharpe
    by_sharpe = sorted(
        traded,
        key=lambda r: safe_metric(r, "quant_suite", "metrics", "sharpe"),
        reverse=True,
    )

    # Sort by profit factor
    by_pf = sorted(
        traded,
        key=lambda r: safe_metric(r, "quant_suite", "metrics", "profit_factor"),
        reverse=True,
    )

    # Sort by DSR
    by_dsr = sorted(
        traded,
        key=lambda r: safe_metric(r, "quant_suite", "dsr"),
        reverse=True,
    )

    # Sort by lowest max drawdown
    by_dd = sorted(
        traded,
        key=lambda r: safe_metric(r, "quant_suite", "metrics", "max_drawdown"),
    )

    def top_n(rankings, n=10):
        return [
            {
                "config_id": r["config_id"],
                "instrument": r["instrument"],
                "profile": r["profile"],
                "trades": r["trade_count"],
                "final_balance": r["final_balance"],
                "sharpe": safe_metric(r, "quant_suite", "metrics", "sharpe"),
                "profit_factor": safe_metric(r, "quant_suite", "metrics", "profit_factor"),
                "max_dd": safe_metric(r, "quant_suite", "metrics", "max_drawdown"),
                "dsr": safe_metric(r, "quant_suite", "dsr"),
                "bayesian_wr": safe_metric(r, "quant_suite", "bayesian_winrate"),
                "mc_p50_bal": safe_metric(r, "quant_suite", "monte_carlo", "P50_balance"),
                "mc_p95_dd": safe_metric(r, "quant_suite", "monte_carlo", "P95_max_dd"),
                "walk_forward": safe_metric(r, "quant_suite", "walk_forward"),
            }
            for r in rankings[:n]
        ]

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_configs": len(deduped),
        "configs_with_trades": len(traded),
        "total_errors": len(all_errors),
        "errors": all_errors[:20],
        "top_10_by_profit": top_n(by_profit),
        "top_10_by_sharpe": top_n(by_sharpe),
        "top_10_by_profit_factor": top_n(by_pf),
        "top_10_by_dsr": top_n(by_dsr),
        "top_10_by_lowest_drawdown": top_n(by_dd),
        "all_results": deduped,
    }

    return report


def print_summary(report: dict):
    """Print a readable summary."""
    print("\n" + "=" * 70)
    print("  JJ SIMON NQ FAIR-PRICE — QUANT SUITE RESULTS")
    print("=" * 70)
    print(f"  Total configs:  {report['total_configs']}")
    print(f"  With trades:    {report['configs_with_trades']}")
    print(f"  Errors:         {report['total_errors']}")
    print()

    for label, key in [
        ("TOP 10 BY PROFIT", "top_10_by_profit"),
        ("TOP 10 BY SHARPE", "top_10_by_sharpe"),
        ("TOP 10 BY PROFIT FACTOR", "top_10_by_profit_factor"),
        ("TOP 10 BY DSR", "top_10_by_dsr"),
        ("TOP 10 BY LOWEST DRAWDOWN", "top_10_by_lowest_drawdown"),
    ]:
        print(f"\n  {label}")
        print("  " + "-" * 66)
        for i, r in enumerate(report[key], 1):
            print(
                f"  {i:2d}. {r['instrument']:3s} {r['config_id'][:50]:<50s} "
                f"trades={r['trades']:4d}  bal=${r['final_balance']:>8.2f}  "
                f"sharpe={r['sharpe']:6.2f}  PF={r['profit_factor']:5.2f}  "
                f"DD={r['max_dd']:6.2f}"
            )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Aggregate backtest results")
    parser.add_argument("--results_dir", required=True, help="Directory with chunk JSONs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    chunks = load_chunks(results_dir)
    if not chunks:
        logger.error("No chunks found in %s", results_dir)
        sys.exit(1)

    report = aggregate(chunks)
    print_summary(report)

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full report (all results)
    full_path = out_dir / f"final_report_full_{ts}.json"
    with open(full_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Full report saved to %s", full_path)

    # Summary only (no all_results)
    summary = {k: v for k, v in report.items() if k != "all_results"}
    summary_path = out_dir / f"final_report_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
