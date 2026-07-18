#!/usr/bin/env python3
"""Fast incremental GHA artifact -> master trade files.

Streams each worker chunk directly into the master variant file without an
intermediate per-worker merge or drawdown recomputation.
"""
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKTEST = os.path.dirname(HERE)
REPO = "Akpan-byte/lead-sites"
MASTER_DIR = os.path.join(BACKTEST, "results", "is_taker_regime_full")

RUNS = [
    ("29575222994", [f"regime-partial-{i}" for i in range(23, 39)]),
    ("29586079492", [f"regime-partial-{i}" for i in range(0, 19)]),
]


def download(run_id: str, artifact: str, dest: str):
    subprocess.run([
        "gh", "run", "download", run_id,
        "--repo", REPO,
        "--name", artifact,
        "--dir", dest,
    ], cwd=BACKTEST, check=True)


def chunk_variants(partial_dir: str):
    """Yield (variant, trades_path, summary_path) for chunk files."""
    for trades_path in glob.glob(os.path.join(partial_dir, "*_chunk*.trades.jsonl.gz")):
        basename = os.path.basename(trades_path).replace(".trades.jsonl.gz", "")
        m = re.match(r"^(.+)_chunk\d+$", basename)
        if not m:
            continue
        variant = m.group(1)
        summ_path = trades_path.replace(".trades.jsonl.gz", ".summary.json")
        yield variant, trades_path, summ_path


def process_artifact(run_id: str, artifact: str):
    with tempfile.TemporaryDirectory(prefix=f"wk_{artifact}_") as tmp:
        download(run_id, artifact, tmp)
        for variant, trades_path, _ in chunk_variants(tmp):
            master_path = os.path.join(MASTER_DIR, f"{variant}.trades.jsonl.gz")
            os.makedirs(MASTER_DIR, exist_ok=True)
            with gzip.open(master_path, "at", encoding="utf-8") as out:
                with gzip.open(trades_path, "rt", encoding="utf-8") as fh:
                    # stream line by line to keep memory low
                    for line in fh:
                        if line.strip():
                            out.write(line)
    print(f"  appended {artifact}", flush=True)


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
        print(f"\n=== run {run_id} ===", flush=True)
        for artifact in artifacts:
            process_artifact(run_id, artifact)
    print("\n=== recomputing summaries ===", flush=True)
    recompute_summaries()
    print(f"\nmaster results in {MASTER_DIR}", flush=True)
    print("next: run quant_suite.py then build_comparison.py", flush=True)


if __name__ == "__main__":
    main()
