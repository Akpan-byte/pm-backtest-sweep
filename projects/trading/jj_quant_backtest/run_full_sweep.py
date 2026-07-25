#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-25  kilo
#   - Created local parallel sweep runner for jj_quant_backtest chunks.
#   - Launches up to --workers chunks concurrently using subprocess.
#   - Supports --quick mode for fast iteration.
# WHY: The GitHub Actions workflow is useful for CI, but local iteration needs
#      a simple way to run all 20 chunks on the VM without manually queueing
#      background tasks.

"""
Run all chunks for one instrument locally in parallel.

Usage:
  python run_full_sweep.py --instrument NQ --workers 4 --quick
  python run_full_sweep.py --instrument NQ --workers 4 --quick --aggregate
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_full_sweep")

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "jj_quant_backtest.py"


def run_chunk(args: tuple[str, int, int, bool]) -> dict:
    """Run one chunk in a subprocess."""
    instrument, chunk_id, total_chunks, quick = args
    cmd = [
        sys.executable,
        str(RUNNER),
        "--instrument", instrument,
        "--chunk_id", str(chunk_id),
        "--total_chunks", str(total_chunks),
    ]
    if quick:
        cmd.append("--quick")

    logger.info("Starting chunk %d/%d for %s", chunk_id + 1, total_chunks, instrument)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, check=False)
        elapsed = time.time() - t0
        if result.returncode != 0:
            logger.error("Chunk %d/%d failed (%.1fs):\n%s", chunk_id + 1, total_chunks, elapsed, result.stderr[-2000:])
            return {"chunk_id": chunk_id, "status": "failed", "elapsed": elapsed, "stderr": result.stderr}
        logger.info("Chunk %d/%d finished in %.1fs", chunk_id + 1, total_chunks, elapsed)
        return {"chunk_id": chunk_id, "status": "ok", "elapsed": elapsed}
    except Exception as e:
        logger.error("Chunk %d/%d exception: %s", chunk_id + 1, total_chunks, e)
        return {"chunk_id": chunk_id, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run all backtest chunks locally in parallel")
    parser.add_argument("--instrument", required=True, choices=["NQ", "ES", "YM"])
    parser.add_argument("--total_chunks", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results after sweep")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("LOCAL SWEEP: %s | chunks=%d | workers=%d | quick=%s", args.instrument, args.total_chunks, args.workers, args.quick)
    logger.info("=" * 60)

    tasks = [(args.instrument, i, args.total_chunks, args.quick) for i in range(args.total_chunks)]

    t_start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(run_chunk, tasks):
            results.append(res)

    elapsed = time.time() - t_start
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - ok

    logger.info("=" * 60)
    logger.info("SWEEP COMPLETE: %d ok, %d failed, %.1fs total", ok, failed, elapsed)
    logger.info("=" * 60)

    if failed > 0:
        for r in results:
            if r["status"] != "ok":
                logger.error("  chunk %d: %s", r["chunk_id"], r.get("error", r.get("stderr", ""))[:200])

    if args.aggregate and ok > 0:
        logger.info("Aggregating results...")
        agg_cmd = [sys.executable, str(SCRIPT_DIR / "aggregate_results.py"), "--results_dir", str(SCRIPT_DIR / "results")]
        subprocess.run(agg_cmd, cwd=SCRIPT_DIR, check=False)

    # Save sweep metadata
    meta_path = SCRIPT_DIR / "results" / f"{args.instrument}_sweep_meta.json"
    meta_path.parent.mkdir(exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump({
            "instrument": args.instrument,
            "total_chunks": args.total_chunks,
            "workers": args.workers,
            "quick": args.quick,
            "ok": ok,
            "failed": failed,
            "elapsed_sec": round(elapsed, 2),
            "chunks": results,
        }, f, indent=2, default=str)
    logger.info("Sweep metadata saved to %s", meta_path)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
