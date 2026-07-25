#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-24  coder
#   - Created laptop parallel runner using all 16 threads via SSH.
# WHY: Laptop has 16 threads + 27GB RAM — adds significant parallel capacity.

"""
Laptop parallel runner for JJ Simon Quant Suite Backtest.

Runs all 3 instruments x 20 chunks in parallel using multiprocessing.

Usage:
  python laptop_runner.py
  python laptop_runner.py --chunks 16 --instruments NQ ES YM
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
RESULTS_DIR = SCRIPT_DIR / "results"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("laptop_runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / f"laptop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)

INSTRUMENTS = ["NQ", "ES", "YM"]


def run_chunk_remote(args: tuple) -> dict:
    """Run a chunk on the laptop via SSH."""
    instrument, chunk_id, total_chunks, laptop_host = args
    remote_dir = "/c/Users/akpan/jj_quant_backtest"  # Adjust as needed

    cmd = [
        "sudo", "tailscale", "ssh", f"akpan@{laptop_host}",
        f"cd {remote_dir} && python jj_quant_backtest.py "
        f"--instrument {instrument} --chunk_id {chunk_id} --total_chunks {total_chunks}"
    ]

    logger.info("Starting %s chunk %d/%d on laptop", instrument, chunk_id + 1, total_chunks)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.error(
                "FAILED: %s chunk %d: returncode=%d\nstderr: %s",
                instrument, chunk_id, result.returncode, result.stderr[:500],
            )
            return {
                "instrument": instrument,
                "chunk_id": chunk_id,
                "status": "failed",
                "error": result.stderr[:2000],
                "elapsed": elapsed,
            }

        logger.info(
            "Completed %s chunk %d/%d in %.1fs",
            instrument, chunk_id + 1, total_chunks, elapsed,
        )
        return {
            "instrument": instrument,
            "chunk_id": chunk_id,
            "status": "completed",
            "elapsed": elapsed,
        }

    except subprocess.TimeoutExpired:
        logger.error("TIMEOUT: %s chunk %d after 3600s", instrument, chunk_id)
        return {
            "instrument": instrument,
            "chunk_id": chunk_id,
            "status": "timeout",
            "elapsed": 3600,
        }
    except Exception as e:
        logger.error("ERROR: %s chunk %d: %s", instrument, chunk_id, e)
        return {
            "instrument": instrument,
            "chunk_id": chunk_id,
            "status": "error",
            "error": str(e),
            "elapsed": time.time() - t0,
        }


def run_local_chunk(args: tuple) -> dict:
    """Run a chunk locally."""
    instrument, chunk_id, total_chunks = args

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "jj_quant_backtest.py"),
        "--instrument", instrument,
        "--chunk_id", str(chunk_id),
        "--total_chunks", str(total_chunks),
    ]

    logger.info("Starting local %s chunk %d/%d", instrument, chunk_id + 1, total_chunks)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
            cwd=str(SCRIPT_DIR),
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.error(
                "FAILED local %s chunk %d: %s",
                instrument, chunk_id, result.stderr[:500],
            )
            return {
                "instrument": instrument,
                "chunk_id": chunk_id,
                "status": "failed",
                "error": result.stderr[:2000],
                "elapsed": elapsed,
            }

        logger.info("Local %s chunk %d/%d done in %.1fs", instrument, chunk_id + 1, total_chunks, elapsed)
        return {
            "instrument": instrument,
            "chunk_id": chunk_id,
            "status": "completed",
            "elapsed": elapsed,
        }

    except subprocess.TimeoutExpired:
        logger.error("TIMEOUT local %s chunk %d", instrument, chunk_id)
        return {"instrument": instrument, "chunk_id": chunk_id, "status": "timeout", "elapsed": 3600}
    except Exception as e:
        logger.error("ERROR local %s chunk %d: %s", instrument, chunk_id, e)
        return {"instrument": instrument, "chunk_id": chunk_id, "status": "error", "error": str(e), "elapsed": time.time() - t0}


def main():
    parser = argparse.ArgumentParser(description="Laptop parallel runner")
    parser.add_argument("--chunks", type=int, default=20, help="Total chunks per instrument")
    parser.add_argument("--instruments", nargs="+", default=INSTRUMENTS)
    parser.add_argument("--laptop_host", default="win-c2anevbhn6q")
    parser.add_argument("--local_workers", type=int, default=16, help="Local parallel workers")
    parser.add_argument("--remote", action="store_true", help="Run on laptop via SSH")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("JJ SIMON QUANT SUITE — PARALLEL RUNNER")
    logger.info("Instruments: %s", args.instruments)
    logger.info("Chunks per instrument: %d", args.chunks)
    logger.info("Mode: %s", "remote (laptop)" if args.remote else f"local ({args.local_workers} workers)")
    logger.info("=" * 60)

    t_start = time.time()

    # Build task list
    tasks = []
    for instrument in args.instruments:
        for chunk_id in range(args.chunks):
            if args.remote:
                tasks.append((instrument, chunk_id, args.chunks, args.laptop_host))
            else:
                tasks.append((instrument, chunk_id, args.chunks))

    total_tasks = len(tasks)
    logger.info("Total tasks: %d", total_tasks)

    # Run in parallel
    results = []
    worker_fn = run_chunk_remote if args.remote else run_local_chunk
    max_workers = args.local_workers if not args.remote else min(16, total_tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                status = result.get("status", "unknown")
                inst = result.get("instrument", "?")
                chunk = result.get("chunk_id", "?")
                elapsed = result.get("elapsed", 0)

                done = len(results)
                pct = done / total_tasks * 100
                logger.info(
                    "[%d/%d %.0f%%] %s chunk %s: %s (%.1fs)",
                    done, total_tasks, pct, inst, chunk, status, elapsed,
                )
            except Exception as e:
                logger.error("Future failed: %s", e)

    # Summary
    elapsed = time.time() - t_start
    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") != "completed")

    logger.info("=" * 60)
    logger.info("COMPLETE: %d/%d succeeded, %d failed, %.1fs total wall time",
                completed, total_tasks, failed, elapsed)
    logger.info("=" * 60)

    # Save summary
    summary = {
        "total_tasks": total_tasks,
        "completed": completed,
        "failed": failed,
        "wall_time_sec": round(elapsed, 1),
        "results": results,
    }
    summary_path = RESULTS_DIR / f"runner_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
