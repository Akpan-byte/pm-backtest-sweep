#!/usr/bin/env python3
"""Combine chunked trade files into a single file, dedup by (opened_at, direction)."""
import argparse
import gzip
import json
import os
import glob

FIELDS = ["opened_at", "closed_at", "entry_price", "shares", "pnl", "direction",
          "condition_id", "reason", "source"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, help="leg name (e.g. l5m_mom30)")
    ap.add_argument("--chunks-dir", required=True, help="artifacts root directory")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--chunks", type=int, default=5, help="number of chunks")
    args = ap.parse_args()

    seen = set()
    all_trades = []

    for i in range(args.chunks):
        # GHA artifact structure: artifacts/l5m-{leg}-chunk-{i}/results/is_bn/btc_{leg}.trades.jsonl.gz
        pattern = os.path.join(args.chunks_dir, f"l5m-*-{args.leg}-*chunk-{i}", "results", "is_bn",
                              f"btc_{args.leg}.trades.jsonl.gz")
        matches = glob.glob(pattern)
        if not matches:
            # Try alternate naming
            pattern = os.path.join(args.chunks_dir, f"*{args.leg}*chunk*{i}*", "results", "is_bn",
                                  f"btc_{args.leg}.trades.jsonl.gz")
            matches = glob.glob(pattern)
        if not matches:
            print(f"  WARN: no trades file found for chunk {i} (pattern: {pattern})")
            continue

        chunk_path = matches[0]
        with gzip.open(chunk_path, "rt") as f:
            for line in f:
                try:
                    t = json.loads(line)
                    key = (t.get("opened_at"), t.get("direction"), t.get("condition_id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    all_trades.append(t)
                except Exception:
                    pass

        print(f"  Chunk {i}: loaded {os.path.basename(os.path.dirname(os.path.dirname(chunk_path)))} ({len(all_trades)} unique so far)")

    # Sort by opened_at
    all_trades.sort(key=lambda t: t.get("opened_at", ""))

    # Write combined file
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"btc_{args.leg}.trades.jsonl.gz")
    with gzip.open(out_path, "wt") as f:
        for t in all_trades:
            f.write(json.dumps(t) + "\n")

    if all_trades:
        total_pnl = sum(t.get("pnl", 0) for t in all_trades)
        wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
        print(f"\n  Combined: {len(all_trades)} unique trades, PnL=${total_pnl:.2f}, WR={wins/len(all_trades)*100:.1f}%")
    else:
        print(f"\n  No trades found for {args.leg}")
    print(f"  Written to {out_path}")

if __name__ == "__main__":
    main()
