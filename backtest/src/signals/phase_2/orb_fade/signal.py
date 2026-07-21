# CHANGE_SUMMARY
# 2026-07-13  deploy-agent
#   - Created signals/phase_2/orb_fade/signal.py: a LIVE port of the
#     /config/backtest/filters_v2 ORB fade strategies.
#   - Replicates the base btc_orb opening-range-breakout detection inline
#     (per-leg, per-market state) then applies a percentile-gated FADE flip:
#     when the configured filter's trailing percentile meets its threshold the
#     breakout direction is mirrored (YES<->NO), exactly like the backtest's
#     "fade" action (mirror trade / buy the opposite binary side).
#   - IND_* daily features (mom30_1d, bbpct20_1d, chand22_1d) computed from
#     Binance daily klines (Coinbase fallback) with the engine's no-lookahead
#     publish convention (use last COMPLETED daily bar) and a faithful
#     forward-filled trailing-10080-1m-bar percentile rank.
#   - PM_* orderbook-flow features (liq_pull20, imb_slope20) computed from a
#     rolling full-depth book buffer (entry side), percentile-ranked over the
#     strategy's OWN trailing-7d trade sequence (min 20 -> NaN -> breakout),
#     persisted to disk for restart continuity.
# WHY: The backtest validated these as post-hoc trade replays. To run them as
#      real paper traders we must reproduce the exact percentile-gated fade
#      decision live. This module keeps that math faithful and dependency-free
#      (stdlib only; the VPS python has no pandas/numpy).
#
# 2026-07-13  kimi (seed-agent)
#   - Added AUTOMATIC PM-percentile seeding at startup (seed_pm_stores), driven
#     by a durable seed artifact at state/orb_fade/_seed/pm_seed.json holding the
#     backtest's 7-day multi-regime RAW PM-feature distributions (PM_liq_pull20,
#     PM_imb_slope20 per base-ORB variant).
#   - warm_start() now seeds each PM leg's percentile reference BEFORE trading,
#     so the fade filter fires from trade #1 instead of warming up over ~20
#     self-trades (cold start -> pct NaN -> raw breakouts, a losing strategy).
#   - IDEMPOTENT + MERGE: prior seed entries (flagged "seed") are replaced by a
#     fresh spread; live-collected entries are preserved and augmented. Seed
#     timestamps are spread across the trailing-7d window (kept inside the cutoff
#     edge) so the window never immediately drops them; they age out gradually and
#     hand off to live data. Re-seeding at every boot re-baselines (self-healing).
#   - The store still holds RAW feature values; percentiles are computed FROM them
#     (raw-vs-percentile bug NOT reintroduced).
# WHY: On a cold start the empty store made every PM leg run raw breakouts for
#      hours, then rank against a thin ~20-30-sample single-regime reference
#      instead of the 7-day multi-regime distribution the backtest validated.
# 2026-07-13  kimi/deploy (prefixa Fix B: late-seen skip)
#   - When a market is first processed AFTER its OR window closed and no real range
#     was ever built (or_high<=0 / or_low==inf), STOP seeding or_high/or_low from the
#     current spot. Instead mark state["skip"]=True and permanently skip the market
#     for this leg, returning "late-seen market skipped". An early guard right after
#     setdefault returns the same no_signal whenever state["skip"] is set.
#   - Only fires on genuine late arrival (mid-contract restart). Fresh markets and
#     properly-built ORs are untouched; warm_start/seed_pm_stores/record_pm_trade
#     and the PM/IND fade logic are unchanged. Does not affect the backtest (which
#     replays from market start and always builds a real OR).
# WHY: The old seed-from-spot path fired a degenerate breakout on the next tick for
#      any 5m market the runner first saw after its 60s OR closed. Skipping removes
#      that broader-than-backtest behavior. Verified low-risk subtractive change.
# 2026-07-13  kimi/deploy (orlog: OR lifecycle visibility)
#   - skip_late: the genuine first late-detection branch now returns
#     {"or_event": "skip_late"} (once per market). The early guard above still
#     returns the plain no-event skip on subsequent ticks, so it logs exactly once.
#   - or_built_fresh: a once-per-market guard inserted right after the skip branch
#     logs the first tick a REAL opening range exists (or_high>0, or_low!=inf),
#     returning {"or_event":"or_built_fresh","or_high":..,"or_low":..}; it returns
#     early ONLY on that single transition tick and normal breakout evaluation
#     resumes on the next tick. warm_start/seed_pm_stores/record_pm_trade and the
#     PM+IND fade logic are untouched.
# WHY: Make the OR lifecycle visible in logs to confirm the bots catch each new 5m
#      market fresh (real OR built) vs skip it (late-seen). Only *triggered* trades
#      were logged before; OR-built and skip events were silent.
# 2026-07-14  kimi (vwapside_8h regime gate, 5m legs ONLY)
#   - Wired the validated IND_vwapside_8h regime gate into the two 5m legs
#     (5m_mom30, 5m_liqpull) as a FLIPPER: when the fade filter fires AND the
#     trailing trade-pct of vwapside_8h over the leg's OWN trade sequence is >= 50,
#     the fade is flipped back to a breakout (ride it) instead of fading.
#   - vwapside_8h = sign(8h_bar_close - daily-anchored VWAP), ties->+1; VWAP is
#     cumulative over the UTC day on Binance 8h klines, published at the 8h bar
#     close. Verified 988/988 identical to the backtest's own IND_vwapside_8h.
#   - Trailing percentile reuses the EXACT engine.trailing_trade_percentile_ranks
#     semantics (7d window, strictly-less, min 20 -> NaN -> no flip). Per-leg gate
#     store persisted to state/orb_fade/<leg>_gate.json and warm-seeded at boot
#     from state/orb_fade/_seed/vwapside_gate_seed.json (PM-seed pattern).
#   - gate_flip log event + gate_vwapside/gate_pct/gate_flip fields on the signal.
#   - 1m/3m/all other legs: UNCHANGED (every gate path guarded by
#     leg_id in _GATED_LEGS = {5m_mom30, 5m_liqpull}).
# WHY: regime_gate_eval/wfa/flip + gate_compound_replay validated vwapside_8h thr50
#      as the sole robust gate; it helps the 5m legs, wrecks 1m, dormant on 3m,
#      so it is wired into the two 5m legs only.
# 2026-07-14  kimi (extend vwapside_8h gate to full_stack's two 5m legs)
#   - Added full_stack's two 5m leg_ids (l5m_mom30, l5m_liqpull60) to _GATED_LEGS,
#     alongside the standalone 5m legs. full_stack runs the IDENTICAL signals
#     (IND_mom30_1d<80 fade; PM_liq_pull20<60 fade) under l5m_* leg_ids, so the
#     exact same gate now applies to them. Every gate path was already keyed by
#     `leg_id in _GATED_LEGS` (seed_gate_stores, _load/_save_gate_store ->
#     <leg_id>_gate.json, record_gate_trade, _gate_pct_rank, the flip), so adding
#     the leg_ids is the ONLY change needed: each leg seeds from the shared
#     vwapside_gate_seed.json artifact AND its own persisted <leg_id>_gate.json at
#     startup, and accrues its own trailing vwapside sequence per fill.
#   - bn_full_stack (bn_l5m_* leg_ids) is intentionally NOT gated; l3m_bbpct40 and
#     the 1m legs stay ungated byte-for-byte. No other leg_id was added.
# WHY: full_stack's 5m legs are the same validated strategies; they should get the
#      same regime-gate protection as the standalone 5m legs. Per-leg state is fully
#      leg_id-scoped, so this cannot leak into any other leg or unit.
"""Live ORB-fade signal: base opening-range breakout + percentile-gated fade.

Faithful live port of the filters_v2 backtest fade strategies. One module-level
state serves any number of "legs" (each leg = one base-ORB variant + one fade
filter). The runner drives it:
  warm_start(legs)            -> fetch daily klines, compute IND percentiles,
                                 load persisted PM trade-feature stores.
  refresh_indicators()        -> re-fetch + recompute IND percentiles (periodic).
  update_book(mid, yb, nb)    -> push one full-depth book snapshot per market.
  orb_fade_signal(...)        -> base breakout, then maybe flip to fade.
  record_pm_trade(leg, ...)   -> persist a taken trade's raw PM feature value.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.request
from datetime import datetime, timezone

log = logging.getLogger("orb_fade")

# ---------------------------------------------------------------------------
# Constants (mirror the backtest engine where applicable)
# ---------------------------------------------------------------------------
_BUFFER_PCT = 0.0005          # btc_orb breakout buffer (matches signals btc_orb)
_MIN_COOLDOWN_TICKS = 3
_REENTRY_SIZE_SCALE = [1.0, 0.75, 0.50, 0.33]

# Percentile windows (engine.py): IND -> trailing 10080 1m bars (7 days);
# PM  -> trailing 7d of the strategy's own trades, min 20 valid.
_PCT_WINDOW_1M_BARS = 10080
_TRADE_PCT_WINDOW_SECS = 7 * 86400.0
_TRADE_PCT_MIN_COUNT = 20

_BOOK_BUF_LEN = 25            # rolling full-depth book snapshots kept per side
_PM_LOOKBACK = 20            # PM features use snapshot[i] vs snapshot[i-20]

# Persistence for PM per-trade percentile stores (restart continuity).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_PM_STORE_DIR = os.path.join(_PROJECT_ROOT, "state", "orb_fade")

# PM-percentile seed artifact: durable historical 7d raw-feature distribution,
# auto-merged into each PM leg's store at boot (see seed_pm_stores).
_PM_SEED_PATH = os.path.join(_PM_STORE_DIR, "_seed", "pm_seed.json")
# Keep the oldest seeded sample this far inside the trailing-7d cutoff edge so
# freshly-spread seed timestamps are never immediately dropped by the window.
_SEED_EDGE_BUFFER_SECS = 7200.0

_DURATION_MAP = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}

_OPS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_ORB_STATE = {}        # (leg_id, market_id) -> per-market ORB state dict
_BOOK_BUF = {}         # market_id -> {"up": [(imb5, depth10)], "down": [...]}
_IND_PCT = {}          # feature name -> current percentile rank (0..100) or NaN
_IND_READY = {"ok": False, "ts": 0.0, "source": None}
_PM_STORE = {}         # leg_id -> list of {"ts": float, "feat": str, "value": float}
_LEGS = []             # registered leg configs

# --- vwapside_8h regime gate (5m legs ONLY) -----------------------------------
# The gate is validated ONLY on btc_orb_5m_5re and is wired into exactly these
# deployed 5m legs. Every gate code path is guarded by `leg_id in _GATED_LEGS`, so
# the 1m/3m legs (different leg_ids) are untouched byte-for-byte. The set holds
# BOTH the standalone 5m legs (5m_mom30/5m_liqpull) and full_stack's two 5m legs
# (l5m_mom30/l5m_liqpull60), which run the IDENTICAL signals (IND_mom30_1d<80 fade,
# PM_liq_pull20<60 fade) and so get the IDENTICAL gate. bn_full_stack's bn_l5m_*
# leg_ids are deliberately NOT included, and l3m_bbpct40 / the 1m legs stay ungated.
_GATED_LEGS = {"5m_mom30", "5m_liqpull", "l5m_mom30", "l5m_liqpull60"}
_GATE_THR = 50.0                 # trailing trade-pct(vwapside_8h) >= thr -> flip fade->brk
_GATE_8H_SECS = 8 * 3600.0
_GATE_KLINE_REFRESH_SECS = 300.0   # re-fetch 8h klines at most this often
_GATE_STORE = {}       # leg_id -> list of {"ts": float, "value": +1/-1, "seed"?: bool}
_GATE_VWAP = {"pub": [], "side": [], "ts": 0.0}   # cached 8h vwapside series (sorted)
_GATE_SEED_PATH = os.path.join(_PM_STORE_DIR, "_seed", "vwapside_gate_seed.json")


def _make_orb_state():
    return {
        "or_high": float("-inf"),
        "or_low": float("inf"),
        "or_closed": False,
        "entry_count": {"YES": 0, "NO": 0},
        "inside_after_break": {"YES": False, "NO": False},
        "cooldown": {"YES": 0, "NO": 0},
    }


def _no_signal(reason, extra=None):
    d = {
        "triggered": False,
        "direction": None,
        "confidence": 0.0,
        "entry_price": 0.0,
        "signal_price": 0.0,
        "source": "ORB_FADE",
        "faded": False,
        "reason": reason,
    }
    if extra:
        d.update(extra)
    return d


# ===========================================================================
# IND daily features (mom30_1d, bbpct20_1d, chand22_1d) + percentile ranks
# ===========================================================================

def _http_get_json(url, timeout=12):
    req = urllib.request.Request(url, headers={"User-Agent": "orb-fade/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_daily_klines(limit=70):
    """Return list of completed-day dicts {open,high,low,close} ascending.

    Primary: Binance BTCUSDT 1d (matches the backtest's reference series).
    Fallback: Coinbase BTC-USD daily candles. The current (forming) day bar is
    dropped so every returned bar is fully closed (no lookahead).
    """
    today_open_ms = int(
        datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
    )
    # --- Local cache (GHA pre-download to bypass geo-blocks) ---
    cache_path = os.environ.get("BINANCE_DAILY_CACHE", "")
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                raw = json.load(f)
            bars = []
            for k in raw:
                ot = int(k[0])
                if ot >= today_open_ms:
                    continue
                bars.append({"open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                             "close": float(k[4]), "open_ms": ot})
            if len(bars) >= 25:
                return bars, "cache"
        except Exception as exc:
            log.warning("cached daily klines failed: %s", exc)
    # --- Binance ---
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=%d" % limit
        raw = _http_get_json(url)
        bars = []
        for k in raw:
            ot = int(k[0])
            if ot >= today_open_ms:
                continue  # forming day -> drop (no lookahead)
            bars.append({"open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                         "close": float(k[4]), "open_ms": ot})
        if len(bars) >= 25:
            return bars, "binance"
    except Exception as exc:  # noqa: BLE001
        log.warning("binance daily klines failed: %s", exc)
    # --- Coinbase fallback: [time, low, high, open, close, volume], newest first ---
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400"
        raw = _http_get_json(url)
        bars = []
        today_open_s = today_open_ms / 1000.0
        for c in raw:
            t = int(c[0])
            if t >= today_open_s:
                continue
            bars.append({"open": float(c[3]), "high": float(c[2]), "low": float(c[1]),
                         "close": float(c[4]), "open_ms": t * 1000})
        bars.sort(key=lambda b: b["open_ms"])
        if len(bars) >= 25:
            return bars, "coinbase"
    except Exception as exc:  # noqa: BLE001
        log.warning("coinbase daily candles failed: %s", exc)
    return [], None


def _wilder(vals, period):
    """Wilder's smoothing == pandas ewm(alpha=1/period, adjust=False,
    min_periods=period).mean(). Recursion starts from the FIRST valid value
    (y0 = x0; y_i = y_{i-1} + alpha*(x_i - y_{i-1})); output is NaN until
    `period` observations have been seen. Matches the backtest exactly."""
    out = [float("nan")] * len(vals)
    alpha = 1.0 / period
    ewma = None
    count = 0
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        count += 1
        if ewma is None:
            ewma = v
        else:
            ewma = ewma + alpha * (v - ewma)
        if count >= period:
            out[i] = ewma
    return out


def _true_range(high, low, close):
    tr = []
    for i in range(len(close)):
        if i == 0:
            tr.append(high[i] - low[i])
        else:
            pc = close[i - 1]
            tr.append(max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc)))
    return tr


def _rolling_mean(vals, period):
    out = [float("nan")] * len(vals)
    for i in range(period - 1, len(vals)):
        out[i] = sum(vals[i - period + 1:i + 1]) / period
    return out


def _rolling_std_pop(vals, period):
    out = [float("nan")] * len(vals)
    for i in range(period - 1, len(vals)):
        w = vals[i - period + 1:i + 1]
        m = sum(w) / period
        out[i] = math.sqrt(sum((x - m) ** 2 for x in w) / period)
    return out


def _rolling_max(vals, period):
    out = [float("nan")] * len(vals)
    for i in range(period - 1, len(vals)):
        out[i] = max(vals[i - period + 1:i + 1])
    return out


def _compute_ind_features(bars):
    """Compute mom30_1d, bbpct20_1d, chand22_1d on completed daily bars.

    Returns dict name -> list aligned to bars (NaN during warmup). Definitions
    mirror features_indicators._indicators on the 1d timeframe.
    """
    close = [b["close"] for b in bars]
    high = [b["high"] for b in bars]
    low = [b["low"] for b in bars]
    n = len(close)
    feats = {}

    # mom30 = close/close.shift(30) - 1
    mom30 = [float("nan")] * n
    for i in range(30, n):
        if close[i - 30] > 0:
            mom30[i] = close[i] / close[i - 30] - 1.0
    feats["IND_mom30_1d"] = mom30

    # bbpct20 = (close - lower) / (upper - lower), BB(20, 2 sigma, pop std)
    mid = _rolling_mean(close, 20)
    sd = _rolling_std_pop(close, 20)
    bbpct = [float("nan")] * n
    for i in range(n):
        if not math.isnan(mid[i]) and not math.isnan(sd[i]):
            upper = mid[i] + 2.0 * sd[i]
            lower = mid[i] - 2.0 * sd[i]
            if upper != lower:
                bbpct[i] = (close[i] - lower) / (upper - lower)
    feats["IND_bbpct20_1d"] = bbpct

    # chand22 = (close - (max(high,22) - 3*ATR22)) / ATR22  (ATR Wilder)
    tr = _true_range(high, low, close)
    atr22 = _wilder(tr, 22)
    hi22 = _rolling_max(high, 22)
    chand = [float("nan")] * n
    for i in range(n):
        if not math.isnan(hi22[i]) and not math.isnan(atr22[i]) and atr22[i] != 0:
            chand[i] = (close[i] - (hi22[i] - 3.0 * atr22[i])) / atr22[i]
    feats["IND_chand22_1d"] = chand

    return feats


def _minutes_since_utc_midnight():
    now = datetime.now(timezone.utc)
    return now.hour * 60 + now.minute


def _daily_pct_rank(completed_vals, minutes_today, window=_PCT_WINDOW_1M_BARS):
    """Faithful forward-filled trailing-window percentile of the current value.

    completed_vals: ascending completed daily indicator values (tail = current,
    i.e. yesterday's indicator, forward-filled across today). Reproduces
    engine.rolling_percentile_ranks over the forward-filled 1m grid: today's
    `minutes_today` bars carry the current value, each prior day 1440 bars,
    truncated to `window` total. rank = 100 * (weight of values strictly less
    than current) / (total weight). NaN if fewer than 2 usable values.
    """
    cur = completed_vals[-1]
    if cur is None or (isinstance(cur, float) and math.isnan(cur)):
        return float("nan")
    segs = [(cur, max(1, min(minutes_today, window)))]
    rem = window - segs[0][1]
    i = len(completed_vals) - 2
    while rem > 0 and i >= 0:
        v = completed_vals[i]
        w = min(1440, rem)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            segs.append((v, w))
        rem -= w
        i -= 1
    total = sum(w for _, w in segs)
    if total <= 0:
        return float("nan")
    less = sum(w for v, w in segs if v < cur)
    return 100.0 * less / total


def refresh_indicators(force=False):
    """(Re)fetch daily klines and recompute IND percentiles. Throttled to ~10 min."""
    now = time.time()
    if not force and (now - _IND_READY["ts"]) < 600 and _IND_READY["ok"]:
        return _IND_READY["ok"]
    bars, source = _fetch_daily_klines()
    if not bars:
        _IND_READY["ok"] = False
        return False
    feats = _compute_ind_features(bars)
    minutes_today = _minutes_since_utc_midnight()
    needed = {leg["filter"]["feat"] for leg in _LEGS if leg["filter"]["feat"].startswith("IND_")}
    for feat in needed:
        series = feats.get(feat)
        if not series:
            continue
        # completed (non-NaN) values, ascending; tail is the current (yesterday) value
        completed = [v for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if len(completed) < 2:
            _IND_PCT[feat] = float("nan")
            continue
        _IND_PCT[feat] = _daily_pct_rank(completed, minutes_today)
    _IND_READY["ok"] = True
    _IND_READY["ts"] = now
    _IND_READY["source"] = source
    log.info("IND refresh ok source=%s pct=%s", source,
             {k: (round(v, 1) if not math.isnan(v) else None) for k, v in _IND_PCT.items()})
    return True


# ===========================================================================
# PM orderbook-flow features (liq_pull20, imb_slope20) + trade-sequence pct
# ===========================================================================

def _side_metrics(book):
    """(imb5, depth10) for one side's full-depth book {bids,asks} best-first."""
    if not book:
        return float("nan"), float("nan")
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    b5 = sum(s for _, s in bids[:5])
    a5 = sum(s for _, s in asks[:5])
    imb5 = (b5 - a5) / (b5 + a5) if (b5 + a5) > 0 else float("nan")
    depth10 = sum(s for _, s in bids[:10]) + sum(s for _, s in asks[:10])
    return imb5, depth10


def update_book(market_id, yes_book, no_book):
    """Push one full-depth snapshot for a market into the rolling buffers."""
    buf = _BOOK_BUF.setdefault(market_id, {"up": [], "down": []})
    buf["up"].append(_side_metrics(yes_book))
    buf["down"].append(_side_metrics(no_book))
    if len(buf["up"]) > _BOOK_BUF_LEN:
        del buf["up"][:-_BOOK_BUF_LEN]
    if len(buf["down"]) > _BOOK_BUF_LEN:
        del buf["down"][:-_BOOK_BUF_LEN]


def _pm_raw(market_id, side, feat):
    """Raw PM feature value from the rolling buffer for `side` ('up'/'down').

    Uses snapshot[i] (latest) vs snapshot[i-20]. Returns NaN if insufficient
    history. Mirrors features_pm_book._fill_trade family (c).
    """
    buf = _BOOK_BUF.get(market_id, {}).get(side) or []
    if len(buf) < _PM_LOOKBACK + 1:
        return float("nan")
    imb_now, depth_now = buf[-1]
    imb_old, depth_old = buf[-1 - _PM_LOOKBACK]
    if feat == "PM_imb_slope20":
        if math.isnan(imb_now) or math.isnan(imb_old):
            return float("nan")
        return (imb_now - imb_old) / _PM_LOOKBACK
    if feat == "PM_liq_pull20":
        if math.isnan(depth_now) or math.isnan(depth_old) or depth_old <= 0:
            return float("nan")
        return (depth_old - depth_now) / depth_old
    return float("nan")


def _pm_store_path(leg_id):
    return os.path.join(_PM_STORE_DIR, "%s_pm.json" % leg_id)


def _load_pm_store(leg_id):
    path = _pm_store_path(leg_id)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                _PM_STORE[leg_id] = data
                return
    except Exception as exc:  # noqa: BLE001
        log.warning("pm store load failed %s: %s", leg_id, exc)
    _PM_STORE[leg_id] = []


def _save_pm_store(leg_id):
    try:
        os.makedirs(_PM_STORE_DIR, exist_ok=True)
        path = _pm_store_path(leg_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_PM_STORE.get(leg_id, []), f)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("pm store save failed %s: %s", leg_id, exc)


def _load_seed_artifact():
    """Load the durable PM-percentile seed artifact; None if absent/unreadable."""
    try:
        if os.path.exists(_PM_SEED_PATH):
            with open(_PM_SEED_PATH, "r", encoding="utf-8") as f:
                art = json.load(f)
            if isinstance(art, dict) and isinstance(art.get("distributions"), dict):
                return art
    except Exception as exc:  # noqa: BLE001
        log.warning("pm seed artifact load failed: %s", exc)
    return None


def _seed_variant_for_leg(leg_id, artifact):
    """Which backtest base-ORB variant's PM-feature distribution seeds this leg.

    Prefers the artifact's explicit leg_variant map; falls back to inferring the
    base timeframe from the leg_id substring (1m/3m/5m) so future legs still seed.
    """
    lv = artifact.get("leg_variant") or {}
    if leg_id in lv:
        return lv[leg_id]
    if "1m" in leg_id:
        return "btc_orb_1m_5re"
    if "3m" in leg_id:
        return "btc_orb_3m_5re"
    if "5m" in leg_id:
        return "btc_orb_5m_5re"
    return None


def seed_pm_stores(legs, now=None):
    """Seed each PM leg's percentile reference from the historical 7d distribution.

    IDEMPOTENT + MERGE (never clobber): prior seed entries (flagged "seed") are
    replaced by a fresh spread, while live-collected entries (no "seed" flag) are
    preserved and augmented. On an already-warm store this is a benign enrichment,
    not a wipe. Seeded timestamps are spread across the trailing-7d window (kept
    _SEED_EDGE_BUFFER_SECS inside the cutoff edge) so the window never immediately
    drops them, and they age out gradually, handing off to live data. Re-running at
    every boot re-baselines the 7d reference (self-healing). The store still holds
    RAW values; percentiles are computed from them downstream.
    """
    now = now or time.time()
    artifact = _load_seed_artifact()
    if not artifact:
        log.info("pm seed: no artifact at %s; PM legs warm up from live trades", _PM_SEED_PATH)
        return {}
    dists = artifact.get("distributions", {})
    out = {}
    for leg in legs:
        leg_id = leg.get("leg_id")
        feat = (leg.get("filter") or {}).get("feat", "")
        if not feat.startswith("PM_"):
            continue
        variant = _seed_variant_for_leg(leg_id, artifact)
        vals = dists.get(variant, {}).get(feat) if variant else None
        if not vals:
            log.warning("pm seed: no distribution for leg=%s feat=%s variant=%s",
                        leg_id, feat, variant)
            continue
        store = _PM_STORE.get(leg_id, [])
        live = [r for r in store if not r.get("seed")]  # preserve live learning
        clean = [float(v) for v in vals
                 if v is not None and not (isinstance(v, float) and math.isnan(v))]
        n = len(clean)
        span = _TRADE_PCT_WINDOW_SECS - _SEED_EDGE_BUFFER_SECS
        start = now - span
        seeded = []
        for i, v in enumerate(clean):
            ts = start if n == 1 else start + span * (i / (n - 1))
            seeded.append({"ts": ts, "feat": feat, "value": v, "seed": True})
        _PM_STORE[leg_id] = live + seeded
        _save_pm_store(leg_id)
        out[leg_id] = {"variant": variant, "feat": feat,
                       "seed_n": len(seeded), "live_n": len(live)}
        log.info("pm seed: leg=%s feat=%s variant=%s seed_n=%d live_n=%d",
                 leg_id, feat, variant, len(seeded), len(live))
    return out


def record_pm_trade(leg_id, feat, value, ts=None):
    """Persist a taken trade's raw PM feature value (for future percentiles)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return
    ts = ts or time.time()
    store = _PM_STORE.setdefault(leg_id, [])
    store.append({"ts": ts, "feat": feat, "value": float(value)})
    cutoff = ts - _TRADE_PCT_WINDOW_SECS - 3600
    _PM_STORE[leg_id] = [r for r in store if r["ts"] >= cutoff]
    _save_pm_store(leg_id)


def _pm_pct_rank(leg_id, feat, value, now=None):
    """Percentile rank of `value` among this leg's trailing-7d trade values.

    NaN (-> breakout) when value is NaN or fewer than _TRADE_PCT_MIN_COUNT valid
    values exist in the window. Mirrors engine.trailing_trade_percentile_ranks.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    now = now or time.time()
    cutoff = now - _TRADE_PCT_WINDOW_SECS
    vals = [r["value"] for r in _PM_STORE.get(leg_id, [])
            if r["feat"] == feat and r["ts"] >= cutoff
            and not (isinstance(r["value"], float) and math.isnan(r["value"]))]
    # include the current value in its own window (as the backtest does)
    vals.append(value)
    if len(vals) < _TRADE_PCT_MIN_COUNT:
        return float("nan")
    less = sum(1 for v in vals if v < value)
    return 100.0 * less / len(vals)


# ===========================================================================
# vwapside_8h regime gate (5m legs ONLY) — flip a fade to a breakout when the
# trailing trade-pct of vwapside_8h over the leg's own trade sequence >= _GATE_THR.
# Faithful to features_indicators (daily-anchored VWAP on 8h bars) and to
# engine.trailing_trade_percentile_ranks (strictly-less, min 20, current included).
# ===========================================================================
import bisect  # local to the gate block; module-level, imported once


def _fetch_8h_klines(limit=90):
    """Ascending list of 8h bar dicts {open_ms,high,low,close,volume} from Binance
    (the backtest's reference series). The forming bar is kept; the publish-time
    lookup only ever selects CLOSED bars (close <= now). Empty list on failure ->
    the gate stays dormant (no flips), a safe degradation."""
    # --- Local cache (GHA pre-download to bypass geo-blocks) ---
    cache_path = os.environ.get("BINANCE_8H_CACHE", "")
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                raw = json.load(f)
            bars = [{"open_ms": int(k[0]), "high": float(k[2]), "low": float(k[3]),
                     "close": float(k[4]), "volume": float(k[5])} for k in raw]
            bars.sort(key=lambda b: b["open_ms"])
            return bars
        except Exception as exc:
            log.warning("cached 8h klines failed: %s", exc)
    try:
        url = ("https://api.binance.com/api/v3/klines?symbol=BTCUSDT"
               "&interval=8h&limit=%d" % limit)
        raw = _http_get_json(url)
        bars = [{"open_ms": int(k[0]), "high": float(k[2]), "low": float(k[3]),
                 "close": float(k[4]), "volume": float(k[5])} for k in raw]
        bars.sort(key=lambda b: b["open_ms"])
        return bars
    except Exception as exc:  # noqa: BLE001
        log.warning("gate 8h klines failed: %s", exc)
        return []


def _build_vwapside_series(bars):
    """Compute (publish_s, vwapside) per 8h bar. Faithful to features_indicators:
    tp=(h+l+c)/3, VWAP = cumsum(tp*vol)/cumsum(vol) reset each UTC day (grouped by
    the bar's START day), vwapside = +1 if close>=vwap else -1, published at the 8h
    bar close (open + 8h). Verified 988/988 identical to the backtest feature."""
    pub, side = [], []
    cur_day = None
    cum_pv = 0.0
    cum_v = 0.0
    for b in bars:
        day = b["open_ms"] // 86400000  # UTC day of the bar's START
        if day != cur_day:
            cur_day, cum_pv, cum_v = day, 0.0, 0.0
        tp = (b["high"] + b["low"] + b["close"]) / 3.0
        cum_pv += tp * b["volume"]
        cum_v += b["volume"]
        if cum_v <= 0:
            continue
        vwap = cum_pv / cum_v
        pub.append(b["open_ms"] / 1000.0 + _GATE_8H_SECS)
        side.append(1.0 if b["close"] >= vwap else -1.0)
    return pub, side


def _refresh_gate_vwap(now=None):
    """(Re)fetch 8h klines and rebuild the cached vwapside series (throttled)."""
    now = now or time.time()
    if _GATE_VWAP["pub"] and (now - _GATE_VWAP["ts"]) < _GATE_KLINE_REFRESH_SECS:
        return True
    bars = _fetch_8h_klines()
    if bars:
        _GATE_VWAP["pub"], _GATE_VWAP["side"] = _build_vwapside_series(bars)
        _GATE_VWAP["ts"] = now
        return True
    return bool(_GATE_VWAP["pub"])


def _gate_vwapside_now(now=None):
    """vwapside_8h of the last 8h bar closed <= now (NaN if unavailable)."""
    now = now or time.time()
    _refresh_gate_vwap(now)
    pub = _GATE_VWAP["pub"]
    if not pub:
        return float("nan")
    i = bisect.bisect_right(pub, now) - 1
    if i < 0:
        return float("nan")
    return _GATE_VWAP["side"][i]


def _gate_store_path(leg_id):
    return os.path.join(_PM_STORE_DIR, "%s_gate.json" % leg_id)


def _load_gate_store(leg_id):
    path = _gate_store_path(leg_id)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                _GATE_STORE[leg_id] = data
                return
    except Exception as exc:  # noqa: BLE001
        log.warning("gate store load failed %s: %s", leg_id, exc)
    _GATE_STORE[leg_id] = []


def _save_gate_store(leg_id):
    try:
        os.makedirs(_PM_STORE_DIR, exist_ok=True)
        path = _gate_store_path(leg_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_GATE_STORE.get(leg_id, []), f)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("gate store save failed %s: %s", leg_id, exc)


def _load_gate_seed_artifact():
    """Load the durable vwapside-gate seed artifact; None if absent/unreadable."""
    try:
        if os.path.exists(_GATE_SEED_PATH):
            with open(_GATE_SEED_PATH, "r", encoding="utf-8") as f:
                art = json.load(f)
            if isinstance(art, dict) and isinstance(art.get("values"), list):
                return art
    except Exception as exc:  # noqa: BLE001
        log.warning("gate seed artifact load failed: %s", exc)
    return None


def seed_gate_stores(legs, now=None):
    """Seed each gated leg's vwapside-percentile reference from the historical
    (backtest btc_orb_5m_5re last-7d) vwapside_8h sequence so the gate is warm from
    trade #1. IDEMPOTENT + MERGE (PM-seed pattern): prior seed entries (flagged
    "seed") are replaced by a fresh spread; live-collected entries are preserved and
    augmented. Seed timestamps are spread across the trailing-7d window (kept
    _SEED_EDGE_BUFFER_SECS inside the cutoff edge) and age out gradually, handing off
    to live data. The store holds raw ±1 values; percentiles are computed from them.
    """
    now = now or time.time()
    artifact = _load_gate_seed_artifact()
    if not artifact:
        log.info("gate seed: no artifact at %s; gate warms from live trades", _GATE_SEED_PATH)
        return {}
    clean = [float(v) for v in artifact.get("values", []) if v in (-1, 1, -1.0, 1.0)]
    out = {}
    for leg in legs:
        leg_id = leg.get("leg_id")
        if leg_id not in _GATED_LEGS:
            continue
        store = _GATE_STORE.get(leg_id, [])
        live = [r for r in store if not r.get("seed")]  # preserve live learning
        n = len(clean)
        if n == 0:
            continue
        span = _TRADE_PCT_WINDOW_SECS - _SEED_EDGE_BUFFER_SECS
        start = now - span
        seeded = []
        for i, v in enumerate(clean):
            ts = start if n == 1 else start + span * (i / (n - 1))
            seeded.append({"ts": ts, "value": v, "seed": True})
        _GATE_STORE[leg_id] = live + seeded
        _save_gate_store(leg_id)
        out[leg_id] = {"seed_n": len(seeded), "live_n": len(live)}
        log.info("gate seed: leg=%s seed_n=%d live_n=%d", leg_id, len(seeded), len(live))
    return out


def record_gate_trade(leg_id, value, ts=None):
    """Persist a taken trade's vwapside_8h value (feeds future gate percentiles).
    Called by the runner on EVERY gated-leg fill (fade or breakout), matching the
    backtest's percentile over the strategy's full trade sequence."""
    if leg_id not in _GATED_LEGS:
        return
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return
    ts = ts or time.time()
    store = _GATE_STORE.setdefault(leg_id, [])
    store.append({"ts": ts, "value": float(value)})
    cutoff = ts - _TRADE_PCT_WINDOW_SECS - 3600
    _GATE_STORE[leg_id] = [r for r in store if r["ts"] >= cutoff]
    _save_gate_store(leg_id)


def _gate_pct_rank(leg_id, value, now=None):
    """Percentile rank of vwapside `value` among this leg's trailing-7d gate values.

    Mirrors engine.trailing_trade_percentile_ranks EXACTLY (same math as
    _pm_pct_rank but over the gate store): window = trailing 7d, current value
    included in its own window, strictly-less count, min 20 valid -> NaN -> no flip.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    now = now or time.time()
    cutoff = now - _TRADE_PCT_WINDOW_SECS
    vals = [r["value"] for r in _GATE_STORE.get(leg_id, [])
            if r["ts"] >= cutoff
            and not (isinstance(r["value"], float) and math.isnan(r["value"]))]
    vals.append(value)  # include the current value in its own window
    if len(vals) < _TRADE_PCT_MIN_COUNT:
        return float("nan")
    less = sum(1 for v in vals if v < value)
    return 100.0 * less / len(vals)


# ===========================================================================
# Warm start
# ===========================================================================

def warm_start(legs):
    """Register legs, compute IND percentiles, load PM stores. Call once at boot."""
    global _LEGS
    _LEGS = list(legs)
    for leg in _LEGS:
        if leg["filter"]["feat"].startswith("PM_"):
            _load_pm_store(leg["leg_id"])
        if leg["leg_id"] in _GATED_LEGS:
            _load_gate_store(leg["leg_id"])
    # Seed PM-percentile references from the historical 7d distribution BEFORE
    # trading begins, so the fade filter fires from trade #1 (idempotent + merge).
    seed_info = seed_pm_stores(_LEGS)
    # Seed the vwapside_8h regime-gate stores (5m legs) and prime the 8h vwapside
    # series, so the gate is warm from trade #1 (idempotent + merge, PM-seed pattern).
    gate_seed_info = seed_gate_stores(_LEGS)
    _refresh_gate_vwap(time.time())
    ok = refresh_indicators(force=True)
    return {"ind_ok": ok, "source": _IND_READY.get("source"),
            "ind_pct": {k: (None if math.isnan(v) else round(v, 2)) for k, v in _IND_PCT.items()},
            "pm_seed": seed_info, "gate_seed": gate_seed_info}


# ===========================================================================
# Signal
# ===========================================================================

def orb_fade_signal(
    leg_id,
    filter_cfg,
    spot_price=None,
    rem_sec=0,
    yp=None,
    np_val=None,
    yes_ask=None,
    no_ask=None,
    tf_hint=None,
    market_id=None,
    or_window_seconds=60,
    max_reentries=5,
    max_entry_price=0.85,
    now=None,
    or_high_pre=None,
    or_low_pre=None,
    **kwargs,
):
    """Base ORB breakout for one leg, then percentile-gated fade flip.

    Returns a signal dict. When the fade filter fires, `direction` is mirrored
    to the opposite side and `faded` is True (entry_price is the opposite ask).

    or_high_pre / or_low_pre: optional pre-seeded OR from Binance 1m klines
    (backtest mode). When provided, the OR is finalized immediately on first
    tick instead of waiting for the live tick-driven OR window to close.
    """
    if spot_price is None or spot_price <= 0:
        return _no_signal("no spot")

    duration = _DURATION_MAP.get(tf_hint)
    if duration is None:
        try:
            duration = int(tf_hint)
        except Exception:
            duration = 300
    if or_window_seconds >= duration:
        return _no_signal("or_window >= duration")

    state_key = (leg_id, market_id)
    state = _ORB_STATE.setdefault(state_key, _make_orb_state())
    if state.get("skip"):
        return _no_signal("late-seen market skipped")

    for d in ("YES", "NO"):
        if state["cooldown"][d] > 0:
            state["cooldown"][d] -= 1

    # Backtest mode: if pre-seeded OR is provided, finalize immediately on first tick.
    if or_high_pre is not None and or_low_pre is not None and not state.get("or_closed"):
        state["or_high"] = or_high_pre
        state["or_low"] = or_low_pre
        state["or_closed"] = True

    elapsed = duration - rem_sec
    if elapsed <= or_window_seconds and not state.get("or_closed"):
        if spot_price > state["or_high"]:
            state["or_high"] = spot_price
        if spot_price < state["or_low"]:
            state["or_low"] = spot_price
        return _no_signal("OR window active")

    state["or_closed"] = True
    if state["or_high"] <= 0 or state["or_low"] == float("inf"):
        state["skip"] = True
        return _no_signal("late-seen market skipped (no real OR built)", {"or_event": "skip_late"})

    # orlog: once-per-market visibility that a REAL opening range was built (fresh
    # catch). Fires ONLY on the single transition tick where a real OR first exists;
    # normal breakout evaluation resumes on the next tick (~0.25s later, re-evaluated
    # every tick), so no breakout opportunity is lost.
    if (not state.get("or_built_logged")) and state["or_high"] > 0 and state["or_low"] != float("inf"):
        state["or_built_logged"] = True
        return _no_signal("OR built fresh", {"or_event": "or_built_fresh",
                                             "or_high": state["or_high"], "or_low": state["or_low"]})

    if rem_sec < 5:
        return _no_signal("time guard")

    buf = spot_price * _BUFFER_PCT
    buy_trigger = state["or_high"] + buf
    sell_trigger = state["or_low"] - buf

    if spot_price < state["or_high"] and state["entry_count"]["YES"] > 0:
        state["inside_after_break"]["YES"] = True
    if spot_price > state["or_low"] and state["entry_count"]["NO"] > 0:
        state["inside_after_break"]["NO"] = True

    direction = None
    entry_index = 0
    if spot_price >= buy_trigger:
        direction = "YES"
    elif spot_price <= sell_trigger:
        direction = "NO"
    if direction is None:
        return _no_signal("spot=%.2f inside [%.2f, %.2f]" % (spot_price, sell_trigger, buy_trigger))

    n = state["entry_count"][direction]
    is_first = n == 0
    is_reentry = n > 0 and n <= max_reentries and state["inside_after_break"][direction]
    if not (is_first or is_reentry):
        return _no_signal("max_reentries_or_no_pullback")
    if state["cooldown"][direction] > 0:
        return _no_signal("cooldown")

    # ---- fade filter evaluation (percentile-gated) -------------------------
    feat = filter_cfg.get("feat")
    op = filter_cfg.get("op", "<")
    thr = float(filter_cfg.get("thr", 0))
    pct = float("nan")
    raw_val = None
    if feat.startswith("IND_"):
        pct = _IND_PCT.get(feat, float("nan"))
    elif feat.startswith("PM_"):
        # entry side = base breakout side (YES -> up book, NO -> down book)
        side = "up" if direction == "YES" else "down"
        raw_val = _pm_raw(market_id, side, feat)
        pct = _pm_pct_rank(leg_id, feat, raw_val, now=now)

    faded = False
    if not math.isnan(pct) and _OPS[op](pct, thr):
        faded = True

    # ---- vwapside_8h regime gate (5m legs ONLY): flip a fade to a breakout ----
    # The gate value/percentile is computed on EVERY gated-leg signal (fade or
    # breakout) so the runner can persist the per-trade vwapside for future
    # percentiles (the backtest ranks over the strategy's FULL trade sequence).
    # The FLIP itself only applies when the fade filter actually fired.
    gate_side = None
    gate_pct = float("nan")
    gate_flip = False
    if leg_id in _GATED_LEGS:
        gnow = now if now is not None else time.time()
        gate_side = _gate_vwapside_now(gnow)
        gate_pct = _gate_pct_rank(leg_id, gate_side, now=gnow)
        if faded and not math.isnan(gate_pct) and gate_pct >= _GATE_THR:
            gate_flip = True
            faded = False   # ride the breakout instead of fading
            log.info("gate_flip leg=%s market=%s vwap_side=%+.0f gate_pct=%.1f base=%s",
                     leg_id, market_id, gate_side, gate_pct, direction)

    final_dir = direction
    if faded:
        final_dir = "NO" if direction == "YES" else "YES"

    # entry price = ask of the FINAL side (we lift the offer as a taker)
    entry_price = yes_ask if final_dir == "YES" else no_ask
    if entry_price is None or entry_price <= 0:
        entry_price = yp if final_dir == "YES" else np_val
    if entry_price is None or entry_price <= 0:
        return _no_signal("no price")
    if entry_price > max_entry_price:
        return _no_signal("price_cap", {"faded": faded, "pct": pct})

    # commit base ORB entry state (per leg, per market)
    state["entry_count"][direction] += 1
    state["inside_after_break"][direction] = False
    state["cooldown"][direction] = _MIN_COOLDOWN_TICKS

    size_scale = _REENTRY_SIZE_SCALE[min(n, len(_REENTRY_SIZE_SCALE) - 1)]

    # gate annotation for the reason string (empty for non-gated legs)
    gate_suffix = ""
    if leg_id in _GATED_LEGS:
        gate_suffix = " gate=%s gpct=%s%s" % (
            "NaN" if gate_side is None else "%+.0f" % gate_side,
            "NaN" if math.isnan(gate_pct) else "%.0f" % gate_pct,
            " FLIP" if gate_flip else "")

    return {
        "triggered": True,
        "direction": final_dir,
        "base_direction": direction,
        "confidence": size_scale,
        "entry_price": float(entry_price),
        "signal_price": float(entry_price),
        "source": "ORB_FADE",
        "faded": faded,
        "gate_vwapside": gate_side,
        "gate_pct": (None if math.isnan(gate_pct) else round(gate_pct, 2)),
        "gate_flip": gate_flip,
        "filter_feat": feat,
        "filter_pct": (None if math.isnan(pct) else round(pct, 2)),
        "filter_raw": (None if raw_val is None or math.isnan(raw_val) else raw_val),
        "entry_index": n,
        "size_scale": size_scale,
        "is_reentry": n > 0,
        "or_window_seconds": or_window_seconds,
        "reason": (
            "%s%s #%d base=%s final=%s spot=%.2f trig=%s price=%.3f feat=%s pct=%s thr=%s%s%s"
            % (
                "BRK(gateflip)" if gate_flip else ("FADE" if faded else "BRK"),
                "(RE)" if n > 0 else "",
                n, direction, final_dir, spot_price,
                ">=%.2f" % buy_trigger if direction == "YES" else "<=%.2f" % sell_trigger,
                entry_price, feat,
                "NaN" if math.isnan(pct) else "%.1f" % pct,
                "%s%.0f" % (op, thr),
                "" if not math.isnan(pct) else " (pct NaN->breakout)",
                gate_suffix,
            )
        ),
    }
