#!/usr/bin/env python3
"""Incrementally download GHA artifacts and append per-variant trades to master files.

Avoids storing all partials at once by processing one worker artifact at a time.
"""
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
BACKTEST = os.path.dirname(HERE)
REPO = "Akpan-byte/lead-sites"
MASTER_DIR = os.path.join(BACKTEST, "results", "is_taker_regime_full")

RUNS = [
    ("29575222994", [f"regime-partial-{i}" for i in range(19, 39)]),
    ("29586079492", [f"regime-partial-{i}" for i in range(0, 19)]),
]


def run(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def append_worker(worker_merged_dir: str):
    os.makedirs(MASTER_DIR, exist_ok=True)
    for trades_path in glob.glob(os.path.join(worker_merged_dir, "*.trades.jsonl.gz")):
        variant = os.path.basename(trades_path).replace(".trades.jsonl.gz", "")
        master_path = os.path.join(MASTER_DIR, f"{variant}.trades.jsonl.gz")
        # append worker trades to master
        with gzip.open(master_path, "at", encoding="utf-8") as out:
            with gzip.open(trades_path, "rt", encoding="utf-8") as fh:
                out.writelines(fh)
        # overwrite summary with worker summary (will recompute later)
        worker_summ = trades_path.replace(".trades.jsonl.gz", ".summary.json")
        master_summ = os.path.join(MASTER_DIR, f"{variant}.summary.json")
        if os.path.exists(worker_summ):
            shutil.copy2(worker_summ, master_summ)


def process_artifact(run_id: str, artifact: str):
    # gh run download extracts artifact files directly into --dir, not into a subdir.
    with tempfile.TemporaryDirectory(prefix=f"worker_{artifact}_") as tmp:
        merged_dir = os.path.join(tmp, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        # download artifact (files land directly in tmp)
        run([
            "gh", "run", "download", run_id,
            "--repo", REPO,
            "--name", artifact,
            "--dir", tmp,
        ], cwd=BACKTEST)
        # merge this worker's chunks
        run([
            sys.executable, os.path.join(HERE, "merge.py"),
            "--partial-dir", tmp,
            "--out-dir", merged_dir,
        ], cwd=BACKTEST)
        # append to master
        append_worker(merged_dir)
        # temp dir auto-deleted
    print(f"  appended {artifact}")


def recompute_summaries():
    """Recompute summary JSONs from the final master trade streams."""
    for trades_path in glob.glob(os.path.join(MASTER_DIR, "*.trades.jsonl.gz")):
        variant = os.path.basename(trades_path).replace(".trades.jsonl.gz", "")
        summ_path = os.path.join(MASTER_DIR, f"{variant}.summary.json")
        pnls = []
        with gzip.open(trades_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    pnls.append(float(json.loads(line)["pnl"]))
        total_pnl = sum(pnls)
        equity = 200.0 + total_pnl
        peak = 200.0
        max_dd = 0.0
        running = 200.0
        for p in pnls:
            running += p
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)
        summary = {
            "strategy": variant,
            "n_closed": len(pnls),
            "total_pnl": round(total_pnl, 4),
            "equity": round(equity, 4),
            "start_capital": 200.0,
            "pnl_pct": round(100 * total_pnl / 200.0, 2),
            "max_dd_usd": round(max_dd, 4),
            "max_dd_pct": round(100 * max_dd / 200.0, 2),
        }
        with open(summ_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)


def main():
    os.makedirs(MASTER_DIR, exist_ok=True)
    for run_id, artifacts in RUNS:
        print(f"\n=== run {run_id} ===")
        for artifact in artifacts:
            process_artifact(run_id, artifact)
    print("\n=== recomputing summaries ===")
    recompute_summaries()
    print(f"\nmaster results in {MASTER_DIR}")
    print("next: run quant_suite.py then build_comparison.py")


if __name__ == "__main__":
    main()
