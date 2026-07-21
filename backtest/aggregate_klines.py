#!/usr/bin/env python3
"""Aggregate reference 1m klines into daily + 8h bars for the signal cache.

Reads zipped CSV files from REF_BN_DIR (Binance 1m klines from Google Drive),
aggregates into daily and 8h bars, writes JSON to OUTDIR.
"""
import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path


def load_1m_bars(ref_dir: str, max_files: int = 100) -> list[dict]:
    """Load 1m bars from zipped CSVs. Returns list of {open_ms, open, high, low, close, volume}."""
    bars = []
    ref_path = Path(ref_dir)
    csv_files = sorted(ref_path.glob("*.csv.gz"))[:max_files]
    if not csv_files:
        csv_files = sorted(ref_path.glob("*.csv"))[:max_files]
    for f in csv_files:
        opener = gzip.open if f.suffix == ".gz" else open
        mode = "rt" if f.suffix == ".gz" else "r"
        with opener(f, mode) as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    open_ms = int(row[0])
                    o = float(row[1])
                    h = float(row[2])
                    l = float(row[3])
                    c = float(row[4])
                    v = float(row[5])
                    bars.append({"open_ms": open_ms, "open": o, "high": h,
                                 "low": l, "close": c, "volume": v})
                except (ValueError, IndexError):
                    continue
    bars.sort(key=lambda b: b["open_ms"])
    return bars


def aggregate_daily(bars: list[dict]) -> list[list]:
    """Aggregate 1m bars into daily bars. Returns Binance kline format."""
    days = defaultdict(lambda: {"open": None, "high": float("-inf"),
                                 "low": float("inf"), "close": None, "volume": 0.0,
                                 "open_ms": None})
    for b in bars:
        day_ms = (b["open_ms"] // 86400000) * 86400000
        d = days[day_ms]
        if d["open"] is None:
            d["open"] = b["open"]
            d["open_ms"] = day_ms
        d["high"] = max(d["high"], b["high"])
        d["low"] = min(d["low"], b["low"])
        d["close"] = b["close"]
        d["volume"] += b["volume"]

    result = []
    for day_ms in sorted(days.keys()):
        d = days[day_ms]
        if d["open"] is not None:
            # Binance kline format: [open_ms, open, high, low, close, volume, ...]
            result.append([d["open_ms"], d["open"], d["high"], d["low"],
                           d["close"], d["volume"]])
    return result


def aggregate_8h(bars: list[dict]) -> list[list]:
    """Aggregate 1m bars into 8h bars. Returns Binance kline format."""
    slots = defaultdict(lambda: {"open": None, "high": float("-inf"),
                                  "low": float("inf"), "close": None, "volume": 0.0,
                                  "open_ms": None})
    for b in bars:
        # 8h slot: 3 slots per day (0, 8, 16)
        slot_ms = (b["open_ms"] // (8 * 3600000)) * (8 * 3600000)
        s = slots[slot_ms]
        if s["open"] is None:
            s["open"] = b["open"]
            s["open_ms"] = slot_ms
        s["high"] = max(s["high"], b["high"])
        s["low"] = min(s["low"], b["low"])
        s["close"] = b["close"]
        s["volume"] += b["volume"]

    result = []
    for slot_ms in sorted(slots.keys()):
        s = slots[slot_ms]
        if s["open"] is not None:
            result.append([s["open_ms"], s["open"], s["high"], s["low"],
                           s["close"], s["volume"]])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True, help="Directory with 1m kline CSVs")
    ap.add_argument("--outdir", default="/tmp/klines_cache", help="Output directory")
    ap.add_argument("--max-files", type=int, default=200, help="Max CSV files to read")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading 1m bars from {args.ref_dir}...")
    bars = load_1m_bars(args.ref_dir, max_files=args.max_files)
    print(f"Loaded {len(bars)} 1m bars")

    if not bars:
        print("ERROR: No bars loaded")
        return

    date_min = bars[0]["open_ms"]
    date_max = bars[-1]["open_ms"]
    from datetime import datetime, timezone
    print(f"Date range: {datetime.fromtimestamp(date_min/1000, tz=timezone.utc).date()} to "
          f"{datetime.fromtimestamp(date_max/1000, tz=timezone.utc).date()}")

    daily = aggregate_daily(bars)
    with open(f"{args.outdir}/daily_klines.json", "w") as f:
        json.dump(daily, f)
    print(f"Daily: {len(daily)} bars")

    h8 = aggregate_8h(bars)
    with open(f"{args.outdir}/8h_klines.json", "w") as f:
        json.dump(h8, f)
    print(f"8h: {len(h8)} bars")


if __name__ == "__main__":
    main()
