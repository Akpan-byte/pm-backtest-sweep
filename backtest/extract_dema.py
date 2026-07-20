#!/usr/bin/env python3
"""Extract DEMA trades from the tarball — streams directly to CSV, deduped."""
import csv
import gzip
import json
import os
import tarfile

TARBALL = os.path.join(os.path.dirname(__file__), "..", "dl", "trades-part4.tar.gz")
OUT = os.path.join(os.path.dirname(__file__), "dema_trades.csv")
TARGET = "tf_dema_lb20_dev002_emax85_alp0001.trades.jsonl.gz"

FIELDS = ["opened_at", "closed_at", "entry_price", "shares", "pnl", "direction", "condition_id"]
seen = set()
n = 0
n_dup = 0
with tarfile.open(TARBALL, "r:gz") as tar:
    for m in tar:
        if m.name.endswith(TARGET) and "rV" not in m.name:
            f = tar.extractfile(m)
            with gzip.open(f, "rt", errors="replace") as gz, \
                 open(OUT, "w", newline="") as csvf:
                w = csv.DictWriter(csvf, fieldnames=FIELDS)
                w.writeheader()
                for line in gz:
                    try:
                        t = json.loads(line)
                        key = (t.get("opened_at"), t.get("condition_id"), t.get("direction"))
                        if key in seen:
                            n_dup += 1
                            continue
                        seen.add(key)
                        w.writerow({k: t.get(k, "") for k in FIELDS})
                        n += 1
                    except Exception:
                        pass
            break

print(f"DEMA: {n} unique trades extracted ({n_dup} duplicates skipped) to {OUT}")
