#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-25  kilo
#   - Created backtest_orb_fade_2d.py: a self-contained 2-day replay of the
#     fixed Luxembourg ORB-fade signals against Polymarket 5m BTC markets.
#   - Reads PM snapshots directly from /root/polybacktest_data/btc_5m/.json.gz
#     (no ticks.csv/orderbook.csv dependency).
#   - Builds spot feed from the live Dublin collector CSVs for Binance and
#     Hyperliquid; the Google Drive archive snapshots were only ~20 min/day
#     near midnight and were too sparse for a 2-day replay.
#   - Uses the backtest-specific signal module (host copy) and a private
#     backtest state directory so production stores are not touched.
#   - Patches IND percentiles and vwapside_8h gate to the snapshot time so the
#     backtest does not use today-only values for Jul 23/24 markets.
# 2026-08-09  kilo
#   - Added CLI overrides: --pm-dir, --spot-csv, --cap, --chunk, --nchunks so the
#     script can run on GitHub Actions with downloaded data and arbitrary caps.
#   - Added PROJECT_ROOT fallback to a vendored repo copy when the host path
#     /root/projects/trading/paper-trading is not available.
#   - Per-chunk output filenames include _chunk{N}_of{M} to avoid collisions.
# WHY: Enable parallel historical backtest sweep on GitHub Actions for caps
#      0.70 vs 0.85 across Hyperliquid and Binance spot feeds.
"""2-day ORB-fade backtest using Polymarket 5m BTC data + spot feed."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import importlib.util
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Determine project root: host path when available, else vendored repo copy.
_HOST_PROJECT_ROOT = "/root/projects/trading/paper-trading"
_REPO_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects", "trading", "paper-trading")
if os.path.isdir(os.path.join(_HOST_PROJECT_ROOT, "backtest_data")):
    PROJECT_ROOT = _HOST_PROJECT_ROOT
else:
    PROJECT_ROOT = _REPO_PROJECT_ROOT

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the backtest-specific signal module (host copy with l5m gate + or_high_pre)
SIGNAL_PATH = os.path.join(PROJECT_ROOT, "backtest_data", "orb_fade_signal_backtest.py")
if not os.path.exists(SIGNAL_PATH):
    raise RuntimeError(f"orb_fade signal module not found at {SIGNAL_PATH}")
spec = importlib.util.spec_from_file_location("orb_fade_backtest", SIGNAL_PATH)
orb_fade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orb_fade)

from engine.execution import position_size  # noqa: E402

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
CAPITAL = 247.0
RISK_PCT = 0.005
MIN_CONTRACTS = 5
MAX_ENTRY_PRICE = 0.85
EXIT_SNIPE_PRICE = 0.97
FEE_PCT = 0.01

PM_DATA_DIR = "/root/polybacktest_data/btc_5m"
BINANCE_CSV = "/root/collector/data/binance/bin_spot_btcusdt.csv"
HL_CSV = "/root/collector/data/hyperliquid/hl_spot_BTC.csv"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "orb_fade_full_stack.json")
REPORT_DIR = os.path.join(PROJECT_ROOT, "backtest_data")
STATE_DIR = os.path.join(REPORT_DIR, "orb_fade_backtest_state")


# ---------------------------------------------------------------------------
# Spot feed helpers
# ---------------------------------------------------------------------------
def load_spot_csv(path: str, use_mid: bool = True, start_ts: float = None, end_ts: float = None):
    """Return (timestamps, prices) arrays from collector CSV, optionally filtered."""
    ts, prices = [], []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                t = datetime.fromisoformat(row["timestamp"]).timestamp()
            except Exception:
                continue
            if end_ts is not None and t > end_ts:
                continue
            if start_ts is not None and t < start_ts:
                # File is roughly chronological; keep scanning in case of out-of-order rows
                continue
            if use_mid:
                bid = float(row.get("bid") or 0)
                ask = float(row.get("ask") or 0)
                if bid > 0 and ask > 0:
                    p = (bid + ask) / 2.0
                else:
                    p = float(row.get("last_price") or row.get("mid") or 0)
            else:
                p = float(row.get("last_price") or row.get("mid") or 0)
            if p > 0:
                ts.append(t)
                prices.append(p)
    if not ts:
        return None, None
    # Ensure ascending (some collectors may append out of order)
    pairs = sorted(zip(ts, prices))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def spot_at(ts, prices, t):
    """Nearest-neighbor spot price."""
    if not ts:
        return None
    i = bisect.bisect_left(ts, t)
    if i == 0:
        return prices[0]
    if i >= len(ts):
        return prices[-1]
    if abs(ts[i] - t) < abs(ts[i - 1] - t):
        return prices[i]
    return prices[i - 1]


# ---------------------------------------------------------------------------
# PM snapshot loading
# ---------------------------------------------------------------------------
def _manifest_path(pm_data_dir: str):
    return os.path.join(pm_data_dir, "btc_5m_manifest.json")


def _manifest_one(fpath: Path):
    """Extract cid, first_time, last_time from a PM json.gz file by reading only
    the head and tail.  Much faster than json.load() for large archives."""
    decoder = json.JSONDecoder()
    try:
        # Open in binary mode so end-relative seeks work.
        with gzip.open(fpath, "rb") as fh:
            # First object should be within first 16 KB of decompressed text
            head = fh.read(16384).decode("utf-8", errors="replace")
            start = head.find("{")
            if start == -1:
                return None
            first_obj, _ = decoder.raw_decode(head, start)

            # Last object: jump near end and try each '{' in reverse
            try:
                fh.seek(-24576, 2)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", errors="replace")
            end_bracket = tail.rfind("]")
            # If file is tiny, just parse everything
            if len(tail) <= 16384:
                data, _ = decoder.raw_decode(tail)
                if not data:
                    return None
                return {
                    "cid": str(data[0]["market_id"]),
                    "file": str(fpath),
                    "first_time": data[0]["time"],
                    "last_time": data[-1]["time"],
                    "n": len(data),
                }
            positions = [m.start() for m in re.finditer(r"{", tail)]
            last_obj = None
            for pos in reversed(positions):
                try:
                    obj, obj_end = decoder.raw_decode(tail, pos)
                except Exception:
                    continue
                if end_bracket == -1:
                    last_obj = obj
                    break
                # The last top-level object ends immediately before the closing ']'.
                between = tail[obj_end:end_bracket]
                if between.strip() == "":
                    last_obj = obj
                    break
            if first_obj is None or last_obj is None or not isinstance(last_obj, dict):
                return None
            if "time" not in last_obj or "market_id" not in first_obj:
                return None
            return {
                "cid": str(first_obj["market_id"]),
                "file": str(fpath),
                "first_time": first_obj["time"],
                "last_time": last_obj["time"],
                "n": None,
            }
    except Exception as exc:
        print(f"WARN: could not read {fpath}: {exc}", file=sys.stderr)
        return None


def build_manifest(pm_data_dir: str, force: bool = False):
    """Build a lightweight manifest of all PM files in parallel."""
    manifest_path = _manifest_path(pm_data_dir)
    if not force and os.path.exists(manifest_path):
        return manifest_path
    print(f"Building manifest {manifest_path} ...")
    files = sorted(Path(pm_data_dir).glob("*.json.gz"))
    entries = []
    # Parallel build
    import multiprocessing

    with multiprocessing.Pool(processes=4) as pool:
        for i, entry in enumerate(pool.imap_unordered(_manifest_one, files), 1):
            if entry:
                entries.append(entry)
            if i % 1000 == 0:
                print(f"  manifest {i}/{len(files)}")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh)
    print(f"  manifest done: {len(entries)} entries")
    return manifest_path


def load_markets_in_window(window_start: datetime, window_end: datetime, pm_data_dir: str):
    """Return market metadata for markets whose first snapshot falls inside the
    backtest window.  Snapshots are loaded lazily during the replay."""
    manifest_path = build_manifest(pm_data_dir)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        entries = json.load(fh)
    meta = []
    for e in entries:
        first_t = datetime.fromisoformat(e["first_time"].replace("Z", "+00:00"))
        if not (window_start <= first_t <= window_end):
            continue
        cid = e["cid"]
        first_ts = first_t.timestamp()
        expiry_ts = math.ceil(first_ts / 300.0) * 300.0
        meta.append((cid, first_t, expiry_ts, Path(e["file"]), None))
    # Sort by market start time for approximate chronological replay
    meta.sort(key=lambda x: x[1])
    return meta


def fetch_market_outcomes(markets_meta, cache_path=None, max_workers=20):
    """Query Polymarket CLOB API for the final resolved outcome of each market.

    Returns a dict mapping condition_id -> 'YES' | 'NO' | None.
    The CLOB result is the Chainlink TWAP-based settlement outcome, so using it
    for expiry resolution makes the backtest match the live market resolution.
    """
    cids = sorted({m[0] for m in markets_meta})
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if set(cids).issubset(set(cached.keys())):
                print(f"Loaded outcomes for {len(cached)} markets from {cache_path}")
                return cached
        except Exception as exc:
            print(f"WARN: could not load outcomes cache: {exc}")

    outcomes = {}
    lock = __import__("threading").Lock()

    def _fetch(cid):
        url = f"https://clob.polymarket.com/markets/{cid}"
        req = urllib.request.Request(url, headers={"User-Agent": "orb-fade-backtest/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            print(f"WARN: outcome fetch failed for {cid}: {exc}")
            return
        tokens = data.get("tokens", [])
        winner = None
        for tok in tokens:
            if tok.get("winner") is True:
                outcome = tok.get("outcome", "").lower()
                if outcome in ("up", "yes"):
                    winner = "YES"
                elif outcome in ("down", "no"):
                    winner = "NO"
                break
        with lock:
            outcomes[cid] = winner

    print(f"Fetching outcomes for {len(cids)} markets from Polymarket CLOB ...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_fetch, cids))
    found = sum(1 for v in outcomes.values() if v is not None)
    print(f"  resolved {found}/{len(cids)} markets")

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(outcomes, fh)
            print(f"  cached outcomes to {cache_path}")
        except Exception as exc:
            print(f"WARN: could not write outcomes cache: {exc}")
    return outcomes


def convert_book(pm_book):
    """Convert PM book dict-format to the signal module's tuple format."""
    if not pm_book:
        return None
    bids = [[float(b["price"]), float(b["size"])] for b in pm_book.get("bids", [])]
    asks = [[float(a["price"]), float(a["size"])] for a in pm_book.get("asks", [])]
    return {"bids": bids, "asks": asks}


# ---------------------------------------------------------------------------
# Historical IND / gate helpers
# ---------------------------------------------------------------------------
def fetch_binance_bars(symbol: str, interval: str, limit: int):
    """Fetch klines from Binance public API and drop the forming current day."""
    url = (
        f"https://api.binance.com/api/v3/klines?symbol={symbol}"
        f"&interval={interval}&limit={limit}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "orb-fade-backtest/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = json.loads(r.read().decode("utf-8"))
    today_open_ms = int(
        datetime.now(timezone.utc)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .timestamp()
        * 1000
    )
    bars = []
    for k in raw:
        open_ms = int(k[0])
        if open_ms >= today_open_ms:
            continue  # forming day -> drop (no lookahead)
        bars.append(
            {
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "open_ms": open_ms,
            }
        )
    return bars


def build_ind_cache(daily_bars, legs_cfg):
    """Pre-compute indicator features once; return a getter for a given time."""
    all_feats = orb_fade._compute_ind_features(daily_bars)
    needed = {leg["filter"]["feat"] for leg in legs_cfg if leg["filter"]["feat"].startswith("IND_")}
    open_ms_list = [b["open_ms"] for b in daily_bars]
    cache = {}

    def get_ind_pct(t: float):
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        today_ms = int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp() * 1000)
        key = (today_ms, dt.hour * 60 + dt.minute)
        if key in cache:
            orb_fade._IND_PCT.update(cache[key])
            return
        idx = bisect.bisect_left(open_ms_list, today_ms) - 1
        minutes_today = dt.hour * 60 + dt.minute
        pct = {}
        for feat in needed:
            series = all_feats.get(feat, [])
            if idx >= 0:
                completed = [v for v in series[: idx + 1] if not math.isnan(v)]
            else:
                completed = []
            if len(completed) >= 2:
                pct[feat] = orb_fade._daily_pct_rank(completed, minutes_today)
            else:
                pct[feat] = float("nan")
        cache[key] = pct
        orb_fade._IND_PCT.update(pct)
        orb_fade._IND_READY["ok"] = True

    return get_ind_pct


def build_gate_vwap(eight_h_bars):
    """Pre-load the full historical vwapside series so _gate_vwapside_now()
    can answer for any snapshot timestamp."""
    pub, side = orb_fade._build_vwapside_series(eight_h_bars)
    orb_fade._GATE_VWAP["pub"] = pub
    orb_fade._GATE_VWAP["side"] = side
    # Set ts far in the future so _refresh_gate_vwap never tries to re-fetch
    orb_fade._GATE_VWAP["ts"] = time.time() + 86400 * 365


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def fmt_pnl(x):
    return f"{x:+.2f}"


def summarize_trades(trades, legs_cfg):
    per_leg = defaultdict(
        lambda: {
            "signals_evaluated": 0,
            "signals_triggered": 0,
            "entries": 0,
            "exits_snipe": 0,
            "exits_expiry": 0,
            "wins": 0,
            "losses": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "fees": 0.0,
        }
    )
    for tr in trades:
        leg = tr["leg_id"]
        per_leg[leg]["entries"] += 1
        per_leg[leg]["exits_snipe"] += 1 if tr["reason"].startswith("snipe") else 0
        per_leg[leg]["exits_expiry"] += 1 if tr["reason"] == "expiry_resolve" else 0
        if tr["net_pnl"] > 0:
            per_leg[leg]["wins"] += 1
        else:
            per_leg[leg]["losses"] += 1
        per_leg[leg]["gross_pnl"] += tr["gross_pnl"]
        per_leg[leg]["net_pnl"] += tr["net_pnl"]
        per_leg[leg]["fees"] += tr["entry_fee"] + tr["exit_fee"]

    total = defaultdict(float)
    total["signals_evaluated"] = 0
    total["signals_triggered"] = 0
    total["entries"] = 0
    total["exits_snipe"] = 0
    total["exits_expiry"] = 0
    total["wins"] = 0
    total["losses"] = 0
    total["gross_pnl"] = 0.0
    total["net_pnl"] = 0.0
    total["fees"] = 0.0
    for leg in legs_cfg:
        lid = leg["leg_id"]
        d = per_leg[lid]
        for k in total:
            total[k] += d[k]
    return per_leg, total


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------
def run_backtest(
    spot_feed: str,
    start_iso: str,
    end_iso: str,
    pm_data_dir: str = None,
    spot_csv: str = None,
    max_entry_price: float = None,
    chunk: int = None,
    nchunks: int = None,
    outcomes_cache: str = None,
    outcomes_only: bool = False,
):
    if max_entry_price is None:
        max_entry_price = MAX_ENTRY_PRICE
    pm_data_dir = pm_data_dir or PM_DATA_DIR

    window_start = datetime.fromisoformat(start_iso)
    window_end = datetime.fromisoformat(end_iso)

    # Load config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    legs_cfg = cfg["legs"]

    # Discover PM markets first so we can fetch outcomes even without a spot feed
    print(f"Discovering PM markets in {pm_data_dir} ...")
    markets_meta = load_markets_in_window(window_start, window_end, pm_data_dir)
    print(f"  {len(markets_meta)} markets in window")

    # Fetch / cache resolved market outcomes (Chainlink TWAP settlement)
    outcomes = fetch_market_outcomes(markets_meta, cache_path=outcomes_cache)
    if outcomes_only:
        print("Outcomes fetched; exiting.")
        return None, None, []

    # Optional worker chunking by sorted market index
    if nchunks is not None and chunk is not None:
        if nchunks <= 0:
            raise ValueError("--nchunks must be > 0")
        if not (0 <= chunk < nchunks):
            raise ValueError(f"--chunk must be in [0, {nchunks})")
        total_markets = len(markets_meta)
        markets_meta = [m for i, m in enumerate(markets_meta) if i % nchunks == chunk]
        print(f"  chunk {chunk}/{nchunks}: processing {len(markets_meta)} of {total_markets} markets")

    # Spot feed
    if spot_csv:
        spot_path = spot_csv
    elif spot_feed == "binance":
        spot_path = BINANCE_CSV
    else:
        spot_path = HL_CSV
    window_start_ts = window_start.timestamp()
    window_end_ts = window_end.timestamp()
    print(f"Loading {spot_feed} spot feed from {spot_path} ...")
    spot_ts, spot_prices = load_spot_csv(
        spot_path, use_mid=True, start_ts=window_start_ts - 3600, end_ts=window_end_ts + 3600
    )
    if spot_ts is None:
        raise RuntimeError(f"Could not load spot feed from {spot_path}")
    print(f"  {len(spot_ts)} spot rows, {min(spot_ts)} to {max(spot_ts)}")

    # Setup isolated signal state
    os.makedirs(STATE_DIR, exist_ok=True)
    orb_fade._PM_STORE_DIR = STATE_DIR
    orb_fade._PM_SEED_PATH = os.path.join(STATE_DIR, "pm_seed.json")
    orb_fade._GATE_SEED_PATH = os.path.join(STATE_DIR, "vwapside_gate_seed.json")

    # Clear any stale module state from a previous run in the same process
    orb_fade._ORB_STATE.clear()
    orb_fade._BOOK_BUF.clear()
    orb_fade._PM_STORE.clear()
    orb_fade._GATE_STORE.clear()
    orb_fade._IND_PCT.clear()
    orb_fade._IND_READY.update({"ok": False, "ts": 0.0, "source": None})
    orb_fade._GATE_VWAP.update({"pub": [], "side": [], "ts": 0.0})

    # Register legs / warm stores with historical anchor time
    orb_fade._LEGS = list(legs_cfg)
    for leg in legs_cfg:
        if leg["filter"]["feat"].startswith("PM_"):
            orb_fade._load_pm_store(leg["leg_id"])
        if leg["leg_id"] in orb_fade._GATED_LEGS:
            orb_fade._load_gate_store(leg["leg_id"])
    seed_info = orb_fade.seed_pm_stores(orb_fade._LEGS, now=window_start_ts)
    gate_seed_info = orb_fade.seed_gate_stores(orb_fade._LEGS, now=window_start_ts)
    print(f"PM seed info: {json.dumps(seed_info, default=str)}")
    print(f"Gate seed info: {json.dumps(gate_seed_info, default=str)}")

    # Fetch historical bars and build IND/gate caches
    print("Fetching historical daily bars for IND features ...")
    daily_bars = fetch_binance_bars("BTCUSDT", "1d", 120)
    print(f"  got {len(daily_bars)} daily bars")
    get_ind_pct = build_ind_cache(daily_bars, legs_cfg)

    print("Fetching historical 8h bars for vwapside gate ...")
    eight_h_bars = fetch_binance_bars("BTCUSDT", "8h", 120)
    print(f"  got {len(eight_h_bars)} 8h bars")
    build_gate_vwap(eight_h_bars)

    # Tracking
    positions = {}  # (leg_id, cid) -> dict
    trades = []  # closed trades
    signals_evaluated = {leg["leg_id"]: 0 for leg in legs_cfg}
    signals_triggered = {leg["leg_id"]: 0 for leg in legs_cfg}

    n_markets = len(markets_meta)
    for mi, (cid, first_t, expiry_ts, fpath, _) in enumerate(markets_meta, 1):
        if mi % 50 == 0 or mi == 1:
            print(f"Processing market {mi}/{n_markets} ({cid}) ...")

        with gzip.open(fpath, "rt") as fh:
            snapshots = json.load(fh)

        last_spot = None
        for snap in snapshots:
            t_str = snap["time"]
            t = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
            ts = t.timestamp()
            rem_sec = expiry_ts - ts

            spot = spot_at(spot_ts, spot_prices, ts)
            if spot is None or spot <= 0:
                continue
            last_spot = spot

            # Update IND percentiles to snapshot time
            get_ind_pct(ts)

            # Update book buffer (the fixed-runner requirement)
            yes_book = convert_book(snap.get("orderbook_up"))
            no_book = convert_book(snap.get("orderbook_down"))
            orb_fade.update_book(cid, yes_book, no_book)

            yp = float(snap.get("price_up") or 0)
            np_val = float(snap.get("price_down") or 0)

            yes_ask = (
                min(yes_book["asks"], key=lambda x: x[0])[0]
                if yes_book and yes_book.get("asks")
                else None
            )
            no_ask = (
                min(no_book["asks"], key=lambda x: x[0])[0]
                if no_book and no_book.get("asks")
                else None
            )

            for leg in legs_cfg:
                leg_id = leg["leg_id"]
                pos_key = (leg_id, cid)

                # ---- exits ----
                if pos_key in positions:
                    pos = positions[pos_key]
                    direction = pos["direction"]
                    exit_price = None
                    reason = None
                    if direction == "YES" and yp >= EXIT_SNIPE_PRICE:
                        exit_price = EXIT_SNIPE_PRICE
                        reason = "snipe_yes_0.97"
                    elif direction == "NO" and np_val >= EXIT_SNIPE_PRICE:
                        exit_price = EXIT_SNIPE_PRICE
                        reason = "snipe_no_0.97"
                    elif rem_sec <= 0:
                        winner = outcomes.get(cid)
                        if winner is not None:
                            exit_price = 1.0 if direction == winner else 0.0
                            reason = "expiry_resolve_twap"
                        else:
                            # Fallback only if CLOB outcome is missing
                            if direction == "YES":
                                exit_price = 1.0 if spot >= pos["entry_spot"] else 0.0
                            else:
                                exit_price = 1.0 if spot < pos["entry_spot"] else 0.0
                            reason = "expiry_resolve_spot_fallback"

                    if exit_price is not None:
                        gross_pnl = (exit_price - pos["entry_price"]) * pos["size"]
                        entry_fee = pos["entry_price"] * pos["size"] * FEE_PCT
                        exit_fee = exit_price * pos["size"] * FEE_PCT
                        net_pnl = gross_pnl - entry_fee - exit_fee
                        trades.append(
                            {
                                "leg_id": leg_id,
                                "market_id": cid,
                                "direction": direction,
                                "entry_time": pos["entry_time"],
                                "entry_price": pos["entry_price"],
                                "entry_spot": pos["entry_spot"],
                                "exit_time": t_str,
                                "exit_price": exit_price,
                                "exit_spot": spot,
                                "size": pos["size"],
                                "gross_pnl": gross_pnl,
                                "entry_fee": entry_fee,
                                "exit_fee": exit_fee,
                                "net_pnl": net_pnl,
                                "reason": reason,
                            }
                        )
                        del positions[pos_key]
                    continue

                # ---- entries ----
                if rem_sec <= 5:
                    continue

                signals_evaluated[leg_id] += 1
                sig = orb_fade.orb_fade_signal(
                    leg_id=leg_id,
                    filter_cfg=leg["filter"],
                    spot_price=spot,
                    rem_sec=rem_sec,
                    yp=yp,
                    np_val=np_val,
                    yes_ask=yes_ask,
                    no_ask=no_ask,
                    tf_hint="5m",
                    market_id=cid,
                    or_window_seconds=leg["or_window_seconds"],
                    max_reentries=leg["max_reentries"],
                    max_entry_price=max_entry_price,
                    now=ts,
                )
                if not sig or not sig.get("triggered"):
                    continue
                signals_triggered[leg_id] += 1

                direction = sig["direction"]
                book = yes_book if direction == "YES" else no_book
                if not book or not book.get("asks"):
                    continue
                best_ask = min(book["asks"], key=lambda x: x[0])[0]
                if best_ask <= 0 or best_ask > max_entry_price:
                    continue

                size = position_size(
                    capital=CAPITAL,
                    entry_price=best_ask,
                    risk_pct=RISK_PCT,
                    min_contracts=MIN_CONTRACTS,
                )
                if size < MIN_CONTRACTS:
                    continue

                # Persist raw PM feature and gate value for future percentiles
                feat = leg["filter"]["feat"]
                if feat.startswith("PM_"):
                    raw_val = sig.get("filter_raw")
                    if raw_val is not None:
                        orb_fade.record_pm_trade(leg_id, feat, raw_val, ts=ts)
                if leg_id in orb_fade._GATED_LEGS:
                    gate_side = sig.get("gate_vwapside")
                    if gate_side is not None:
                        orb_fade.record_gate_trade(leg_id, gate_side, ts=ts)

                positions[pos_key] = {
                    "direction": direction,
                    "entry_price": best_ask,
                    "size": size,
                    "entry_spot": spot,
                    "entry_time": t_str,
                }

        # Resolve any position still open at end of this market's snapshots
        for pos_key, pos in list(positions.items()):
            if not pos_key[1] == cid:
                continue
            direction = pos["direction"]
            winner = outcomes.get(cid)
            spot = last_spot if last_spot is not None else spot_at(spot_ts, spot_prices, expiry_ts)
            if spot is None:
                spot = pos["entry_spot"]
            if winner is not None:
                exit_price = 1.0 if direction == winner else 0.0
                reason = "expiry_resolve_twap_end_of_data"
            else:
                if direction == "YES":
                    exit_price = 1.0 if spot >= pos["entry_spot"] else 0.0
                else:
                    exit_price = 1.0 if spot < pos["entry_spot"] else 0.0
                reason = "expiry_resolve_spot_fallback_end_of_data"
            gross_pnl = (exit_price - pos["entry_price"]) * pos["size"]
            entry_fee = pos["entry_price"] * pos["size"] * FEE_PCT
            exit_fee = exit_price * pos["size"] * FEE_PCT
            net_pnl = gross_pnl - entry_fee - exit_fee
            trades.append(
                {
                    "leg_id": pos_key[0],
                    "market_id": cid,
                    "direction": direction,
                    "entry_time": pos["entry_time"],
                    "entry_price": pos["entry_price"],
                    "entry_spot": pos["entry_spot"],
                    "exit_time": f"end_of_data_{first_t.isoformat()}",
                    "exit_price": exit_price,
                    "exit_spot": spot,
                    "size": pos["size"],
                    "gross_pnl": gross_pnl,
                    "entry_fee": entry_fee,
                    "exit_fee": exit_fee,
                    "net_pnl": net_pnl,
                    "reason": reason,
                }
            )
            del positions[pos_key]

    # Aggregate
    per_leg, total = summarize_trades(trades, legs_cfg)
    for leg in legs_cfg:
        lid = leg["leg_id"]
        per_leg[lid]["signals_evaluated"] = signals_evaluated[lid]
        per_leg[lid]["signals_triggered"] = signals_triggered[lid]
    total["signals_evaluated"] = sum(signals_evaluated.values())
    total["signals_triggered"] = sum(signals_triggered.values())

    return per_leg, total, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-feed", choices=["binance", "hyperliquid"], default="binance")
    ap.add_argument("--start", default="2026-07-23T00:00:00+00:00")
    ap.add_argument("--end", default="2026-07-25T23:59:59+00:00")
    ap.add_argument("--report-dir", default=REPORT_DIR)
    ap.add_argument("--pm-dir", default=PM_DATA_DIR, help="Directory containing Polymarket BTC 5m .json.gz files")
    ap.add_argument("--spot-csv", default=None, help="Override spot CSV path")
    ap.add_argument("--cap", type=float, default=MAX_ENTRY_PRICE, help="Max entry price cap")
    ap.add_argument("--chunk", type=int, default=None, help="Worker chunk index (0-based)")
    ap.add_argument("--nchunks", type=int, default=None, help="Total worker chunks")
    ap.add_argument("--outcomes-cache", default=None, help="Path to cache fetched market outcomes")
    ap.add_argument("--outcomes-only", action="store_true", help="Fetch/cache outcomes and exit")
    args = ap.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)

    t0 = time.time()
    per_leg, total, trades = run_backtest(
        args.spot_feed,
        args.start,
        args.end,
        pm_data_dir=args.pm_dir,
        spot_csv=args.spot_csv,
        max_entry_price=args.cap,
        chunk=args.chunk,
        nchunks=args.nchunks,
        outcomes_cache=args.outcomes_cache,
        outcomes_only=args.outcomes_only,
    )
    elapsed = time.time() - t0
    if per_leg is None and total is None:
        return 0

    # Unique filename suffix including chunk and cap
    chunk_suffix = ""
    if args.chunk is not None and args.nchunks is not None:
        chunk_suffix = f"_chunk{args.chunk}_of{args.nchunks}"
    suffix = f"{args.spot_feed}_cap{args.cap:.2f}{chunk_suffix}_2d"
    csv_path = os.path.join(args.report_dir, f"{suffix}_trades.csv")
    if trades:
        keys = trades[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(trades)
        print(f"Wrote {len(trades)} trades to {csv_path}")
    else:
        Path(csv_path).touch()
        print(f"No trades; touched {csv_path}")

    # Write report
    report_path = os.path.join(args.report_dir, f"orb_fade_backtest_{suffix}_report.txt")
    lines = []
    lines.append("ORB-fade 2-day backtest report")
    lines.append(f"Spot feed: {args.spot_feed}")
    lines.append(f"Window: {args.start} -> {args.end}")
    lines.append(f"Capital per trade: ${CAPITAL}, risk/trade: {RISK_PCT*100:.2f}%, min contracts: {MIN_CONTRACTS}")
    lines.append(f"Max entry price: {args.cap}, exit snipe: {EXIT_SNIPE_PRICE}, fees: {FEE_PCT*100:.2f}% entry+exit")
    lines.append(f"Run time: {elapsed:.1f}s")
    lines.append("")
    lines.append("Per-leg metrics:")
    lines.append(
        f"{'leg_id':<16} {'eval':>8} {'trig':>8} {'fills':>8} {'snipe':>8} "
        f"{'expiry':>8} {'wins':>8} {'losses':>8} {'win%':>8} {'gross':>12} {'fees':>12} {'net':>12}"
    )
    for leg in orb_fade._LEGS:
        lid = leg["leg_id"]
        d = per_leg[lid]
        entries = d["entries"]
        win_pct = (d["wins"] / entries * 100.0) if entries else 0.0
        lines.append(
            f"{lid:<16} {d['signals_evaluated']:>8} {d['signals_triggered']:>8} {entries:>8} "
            f"{d['exits_snipe']:>8} {d['exits_expiry']:>8} {d['wins']:>8} {d['losses']:>8} "
            f"{win_pct:>7.1f}% {d['gross_pnl']:>+11.2f} {d['fees']:>+11.2f} {d['net_pnl']:>+11.2f}"
        )
    lines.append("")
    lines.append("Total:")
    entries = total["entries"]
    win_pct = (total["wins"] / entries * 100.0) if entries else 0.0
    lines.append(
        f"{'TOTAL':<16} {total['signals_evaluated']:>8} {total['signals_triggered']:>8} {entries:>8} "
        f"{total['exits_snipe']:>8} {total['exits_expiry']:>8} {total['wins']:>8} {total['losses']:>8} "
        f"{win_pct:>7.1f}% {total['gross_pnl']:>+11.2f} {total['fees']:>+11.2f} {total['net_pnl']:>+11.2f}"
    )
    lines.append("")
    lines.append("Notes:")
    lines.append(f"- PM snapshots read directly from {args.pm_dir}/*.json.gz.")
    lines.append("- Google Drive Binance archive snapshots were only ~20 min/day near midnight;")
    lines.append("  this replay uses the provided spot CSV for full spot coverage.")
    if args.spot_csv:
        lines.append(f"- Spot feed CSV: {args.spot_csv}")
    elif args.spot_feed == "hyperliquid":
        lines.append("- Hyperliquid spot feed.")
    else:
        lines.append("- Binance spot feed.")
    if args.chunk is not None and args.nchunks is not None:
        lines.append(f"- Worker chunk {args.chunk} of {args.nchunks}.")
    lines.append("- Expiry resolution uses the actual Polymarket/Chainlink TWAP outcome from the CLOB API.")
    lines.append("  This matches live market settlement and is no longer derived from the spot snapshot.")
    lines.append("- PM-percentile seed copied from host pm_seed.json; vwapside gate seed is absent,")
    lines.append("  so gate flips may be sparse until enough trades accrue.")
    lines.append("- IND percentiles and vwapside gate are pinned to each snapshot's timestamp.")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote report to {report_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
