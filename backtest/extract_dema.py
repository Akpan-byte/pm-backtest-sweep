#!/usr/bin/env python3
"""Extract DEMA trades from the tarball downloaded in ../dl/."""
import csv
import gzip
import json
import os
import tarfile

TARBALL = os.path.join(os.path.dirname(__file__), "..", "dl", "trades-part4.tar.gz")
OUT = os.path.join(os.path.dirname(__file__), "dema_trades.csv")
TARGET = "tf_dema_lb20_dev002_emax85_alp0001.trades.jsonl.gz"

pnls = []
with tarfile.open(TARBALL, "r:gz") as tar:
    for m in tar:
        if m.name.endswith(TARGET) and "rV" not in m.name:
            f = tar.extractfile(m)
            with gzip.open(f, "rt", errors="replace") as gz:
                for line in gz:
                    try:
                        pnls.append(json.loads(line))
                    except Exception:
                        pass
            break

FIELDS = ["opened_at", "closed_at", "entry_price", "shares", "pnl", "direction", "condition_id"]
with open(OUT, "w", newline="") as csvf:
    w = csv.DictWriter(csvf, fieldnames=FIELDS)
    w.writeheader()
    for t in pnls:
        w.writerow({k: t.get(k, "") for k in FIELDS})

print(f"DEMA: {len(pnls)} trades extracted to {OUT}")
