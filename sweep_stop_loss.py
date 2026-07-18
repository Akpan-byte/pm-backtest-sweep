#!/usr/bin/env python3
"""Sweep stop_loss_pct values across VWAP and HOLT strategies on the 2k sample.

Usage:
    python3 sweep_stop_loss.py <strategy_name> <stop_loss_pct>

Examples:
    python3 sweep_stop_loss.py tf_vwap_ticks_lb20_dev002_emax80 0.30
    python3 sweep_stop_loss.py tf_holt_lb20_dev002_emax80_alp0001_hol0005 0.40
"""
import json
import subprocess
import sys
import os

STRATEGY_NAME = sys.argv[1] if len(sys.argv) > 1 else "tf_vwap_ticks_lb20_dev002_emax80"
STOP_LOSS_PCT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30

REGISTRY = "trend_sweep_registry.json"
IS_LIST = "is_files_2k.txt"
SAMPLE_TAR = "sample_2k.tar.gz"
RESULTS_DIR = "/tmp/stop_loss_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

with open(REGISTRY) as f:
    registry = json.load(f)

if STRATEGY_NAME not in registry:
    print(f"Strategy {STRATEGY_NAME} not in registry")
    sys.exit(1)

cfg = dict(registry[STRATEGY_NAME])
cfg["stop_loss_pct"] = STOP_LOSS_PCT

with open(IS_LIST) as f:
    is_files = [l.strip() for l in f if l.strip()]

cmd = [
    sys.executable, "driver.py",
    "--registry-json", json.dumps({STRATEGY_NAME: cfg}),
    "--is-list", IS_LIST,
    "--sample-tar", SAMPLE_TAR,
    "--oos-days", "0",
    "--workers", "8",
    "--compact",
]

print(f"Running stop-loss sweep: {STRATEGY_NAME} stop_loss_pct={STOP_LOSS_PCT}")
result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)
