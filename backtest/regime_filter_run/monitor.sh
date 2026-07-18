#!/bin/bash
# CHANGE_SUMMARY
# 2026-07-17  kilo
#   - 11-minute status monitor for the regime-filter pipeline.
#   - Logs merge/finalize progress on laptop plus VM resource headroom.
# WHY: User asked to check everything every 11 minutes.

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/logs/monitor.log"
mkdir -p "$HERE/logs"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
{
  echo "=== $(ts) regime-filter monitor ==="

  # VM partials (may be cleared after merge moved to laptop)
  vm_done=$(ls "$HERE/partials"/*.summary.json 2>/dev/null | wc -l)
  echo "VM partials summary files: $vm_done"

  # Laptop partials / merge status
  if laptop_out=$(tailscale ssh akpan@100.93.22.56 "cd /home/akpan/kilo_regime_filter/backtest/regime_filter_run && python3 laptop_status.py" 2>/dev/null); then
    echo "laptop: $laptop_out"
  else
    echo "laptop: status query failed"
  fi

  echo "laptop merge log tail:"
  tailscale ssh akpan@100.93.22.56 "tail -n 5 /home/akpan/kilo_regime_filter/backtest/regime_filter_run/logs/merge.log" 2>/dev/null || echo "  (no merge log)"

  echo "laptop merged results:"
  tailscale ssh akpan@100.93.22.56 "ls /home/akpan/kilo_regime_filter/backtest/results/is_taker_regime_full/ 2>/dev/null | wc -l; du -sh /home/akpan/kilo_regime_filter/backtest/results/is_taker_regime_full/ 2>/dev/null" 2>/dev/null || echo "  (no results dir)"

  echo "laptop finalize log tail:"
  tailscale ssh akpan@100.93.22.56 "tail -n 5 /home/akpan/kilo_regime_filter/backtest/regime_filter_run/logs/finalize.log" 2>/dev/null || echo "  (no finalize log)"

  # GHA run
  gha_run="29575222994"
  if gha_status=$(gh run view "$gha_run" --repo Akpan-byte/lead-sites --json status,conclusion 2>/dev/null); then
    echo "GHA $gha_run: $gha_status"
  else
    echo "GHA $gha_run: status query failed"
  fi

  # VM resources
  echo "VM load/cpu:"
  top -bn1 | head -2 | tail -1
  echo "VM disk:"
  df -h / | tail -1

  # Orchestrator last line
  echo "orchestrator last log:"
  tail -n 1 "$HERE/logs/orchestrate.log" 2>/dev/null || echo "  (no log)"

  echo ""
} >> "$LOG"
