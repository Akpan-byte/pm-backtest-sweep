# CHANGE_SUMMARY
# 2026-08-17  kilo
#   - Created signals/brandontrades_supply_demand.py (Strategy 7).
#   - Implements institutional supply/demand imbalance trading:
#     * Zone plotting across 30m/45m/1h/2h/3h/4h built from the engine's
#       completed 15m/1h/4h bars (no lookahead — only closed bars).
#     * 5m execution timeframe with candlestick rejection confirmation,
#       volume spike, and absorption check.
#     * SL on the opposite side of the confirmation candle; TP at the next
#       structural opposing zone.
#   - Uses StateStore keyed by (asset, date) to persist active zones, entry
#     counts, and cooldowns across ticks.
# WHY: Complete the StarTrading futures signal suite with the brandontrades
#      supply/demand blueprint.

"""Strategy 7 (FUTURES): BrandonTrades Supply & Demand Imbalance.

Documentation / master blueprint: docs/brandontrades_supply_demand.md

Core logic:
  Plot supply/demand zones from large momentum impulses on higher timeframes
  (30m-4h), then wait for price to return to a zone and print a 5m rejection
  candle (wick, engulfing, doji) on a volume spike.  Enter at the confirmation
  close, stop beyond the confirmation candle, target the next opposing zone.

Signal kwargs:
  five_m_bars   : list of 5m OHLCV dicts (execution / confirmation)
  fifteen_m_bars: list of 15m OHLCV dicts (used to build 30m/45m zones)
  one_h_bars    : list of 1h OHLCV dicts (used as-is and to build 2h/3h zones)
  four_h_bars   : list of 4h OHLCV dicts (used as-is for 4h zones)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..core import time_utils as tu
from ..core import candle_utils as cu
from ..core.state_store import StateStore
from .common import (
    no_signal,
    validate_signal_inputs,
    reentry_scale,
    MIN_COOLDOWN_TICKS,
)

log = logging.getLogger("brandontrades_supply_demand")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "BRANDONTRADES_SUPPLY_DEMAND"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("brandontrades_supply_demand", _PROJECT_ROOT)

# Zone lookback configuration.
HTF_ZONE_TFS = ("30m", "45m", "1h", "2h", "3h", "4h")
MAX_ZONE_AGE_BARS = {
    "30m": 48,
    "45m": 32,
    "1h": 24,
    "2h": 16,
    "3h": 12,
    "4h": 8,
}

# Detection thresholds.
IMPULSE_BODY_RATIO = 0.65          # body / range
IMPULSE_RELATIVE_SIZE = 2.0        # impulse body >= N * median HTF body
BASEING_OPPOSITE_COLOR = True      # basing candle must be opposite color to impulse
VOLUME_SMA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.0            # current volume > SMA(volume, 20)
WICK_REJECTION_MULT = 2.0          # rejection wick >= N * body
MIN_CONFIRMATION_BODY_RATIO = 0.30 # avoid doji-only confirmation
CONFIRMATION_LOOKBACK = 5          # how many recent 5m bars to scan
ZONE_EXPIRY_DAYS = 5               # zones older than this are purged


def _make_state():
    return {
        "today": None,
        "zones": [],                   # list of active {tf, type, high, low, timestamp}
        "pending_zone": None,          # zone price currently entered but not confirmed
        "entry_count": {"LONG": 0, "SHORT": 0},
        "cooldown": {"LONG": 0, "SHORT": 0},
        "last_entry_time": None,
    }


# ----- HTF bar builders (no lookahead) --------------------------------------

def _build_htf_from_15m(bars_15m: list[dict], n: int) -> list[dict]:
    """Aggregate consecutive 15m bars into HTF bars of n*15m."""
    if len(bars_15m) < n:
        return []
    out = []
    for i in range(0, len(bars_15m) - len(bars_15m) % n, n):
        chunk = bars_15m[i:i + n]
        out.append({
            "timestamp": chunk[-1]["timestamp"],
            "open": chunk[0]["open"],
            "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(c.get("volume", 0) for c in chunk),
        })
    return out


def _build_htf_from_1h(bars_1h: list[dict], n: int) -> list[dict]:
    """Aggregate consecutive 1h bars into HTF bars of n*1h."""
    if len(bars_1h) < n:
        return []
    out = []
    for i in range(0, len(bars_1h) - len(bars_1h) % n, n):
        chunk = bars_1h[i:i + n]
        out.append({
            "timestamp": chunk[-1]["timestamp"],
            "open": chunk[0]["open"],
            "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(c.get("volume", 0) for c in chunk),
        })
    return out


def _get_htf_bars(tf: str, bars_15m: list[dict], bars_1h: list[dict], bars_4h: list[dict]) -> list[dict]:
    if tf == "30m":
        return _build_htf_from_15m(bars_15m, 2)
    if tf == "45m":
        return _build_htf_from_15m(bars_15m, 3)
    if tf == "1h":
        return bars_1h
    if tf == "2h":
        return _build_htf_from_1h(bars_1h, 2)
    if tf == "3h":
        return _build_htf_from_1h(bars_1h, 3)
    if tf == "4h":
        return bars_4h
    return []


# ----- zone detection -------------------------------------------------------

def _is_impulse(candle: dict, median_body: float) -> bool:
    """Large momentum candle with strong body and size above recent median."""
    if cu.candle_body_ratio(candle) < IMPULSE_BODY_RATIO:
        return False
    body = cu.candle_body_size(candle)
    if median_body > 0 and body < IMPULSE_RELATIVE_SIZE * median_body:
        return False
    return body > 0


def _detect_zones(htf_bars: list[dict], tf: str) -> list[dict]:
    """Scan [basing, impulse] pairs and return supply/demand zones."""
    zones = []
    if len(htf_bars) < 3:
        return zones
    bodies = [cu.candle_body_size(c) for c in htf_bars if cu.candle_body_size(c) > 0]
    median_body = sorted(bodies)[len(bodies) // 2] if bodies else 0.0

    for i in range(1, len(htf_bars)):
        base = htf_bars[i - 1]
        impulse = htf_bars[i]
        base_body = cu.candle_body_size(base)
        impulse_body = cu.candle_body_size(impulse)

        if impulse_body <= base_body:
            continue
        if BASEING_OPPOSITE_COLOR:
            base_bull = base["close"] > base["open"]
            imp_bull = impulse["close"] > impulse["open"]
            if base_bull == imp_bull:
                continue
        if not _is_impulse(impulse, median_body):
            continue

        if impulse["close"] > impulse["open"]:
            # Demand zone: full basing range, extended to impulse wick low.
            z_low = min(base["low"], impulse["low"])
            z_high = base["high"]
            z_type = "DEMAND"
        else:
            # Supply zone: full basing range, extended to impulse wick high.
            z_high = max(base["high"], impulse["high"])
            z_low = base["low"]
            z_type = "SUPPLY"

        if z_high <= z_low:
            continue
        zones.append({
            "tf": tf,
            "type": z_type,
            "high": float(z_high),
            "low": float(z_low),
            "timestamp": impulse["timestamp"],
            "impulse_idx": i,
        })
    return zones


def _prune_zones(zones: list[dict], now: datetime) -> list[dict]:
    """Drop zones older than ZONE_EXPIRY_DAYS."""
    cutoff = now - timedelta(days=ZONE_EXPIRY_DAYS)
    kept = []
    for z in zones:
        ts = z["timestamp"]
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                continue
        if ts >= cutoff:
            kept.append(z)
    return kept


# ----- confirmation logic ---------------------------------------------------

def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _is_rejection_candle(candle: dict, direction: str) -> bool:
    """Check for wick rejection, engulfing, or hammer/gravestone/dragonfly."""
    body = cu.candle_body_size(candle)
    rng = cu.candle_range(candle)
    if rng <= 0 or body <= 0:
        return False
    body_ratio = body / rng
    if body_ratio < MIN_CONFIRMATION_BODY_RATIO:
        # Doji variants: dragonfly (long lower wick) for longs,
        # gravestone (long upper wick) for shorts.
        upper = candle["high"] - max(candle["open"], candle["close"])
        lower = min(candle["open"], candle["close"]) - candle["low"]
        if direction == "LONG" and lower >= WICK_REJECTION_MULT * upper and lower > 0:
            return True
        if direction == "SHORT" and upper >= WICK_REJECTION_MULT * lower and upper > 0:
            return True
        return False

    upper = candle["high"] - max(candle["open"], candle["close"])
    lower = min(candle["open"], candle["close"]) - candle["low"]

    if direction == "LONG":
        # Bullish candle with long lower wick.
        if candle["close"] > candle["open"] and lower >= WICK_REJECTION_MULT * body:
            return True
    else:
        # Bearish candle with long upper wick.
        if candle["close"] < candle["open"] and upper >= WICK_REJECTION_MULT * body:
            return True

    return False


def _is_engulfing(curr: dict, prev: dict, direction: str) -> bool:
    """True if curr body fully engulfs prev body in the trade direction."""
    curr_body_top = max(curr["open"], curr["close"])
    curr_body_bot = min(curr["open"], curr["close"])
    prev_body_top = max(prev["open"], prev["close"])
    prev_body_bot = min(prev["open"], prev["close"])
    if direction == "LONG":
        return curr["close"] > curr["open"] and curr_body_bot <= prev_body_bot and curr_body_top >= prev_body_top
    return curr["close"] < curr["open"] and curr_body_top >= prev_body_top and curr_body_bot <= prev_body_bot


def _find_confirmation(
    direction: str,
    zone: dict,
    five_m_bars: list[dict],
) -> dict | None:
    """Return the most recent confirmation candle if price entered the zone."""
    if len(five_m_bars) < 2:
        return None
    window = five_m_bars[-CONFIRMATION_LOOKBACK:]
    volumes = [c.get("volume", 0) for c in five_m_bars if c.get("volume")]
    vol_sma = _sma(volumes, VOLUME_SMA_PERIOD)

    # Iterate newest-to-oldest so the freshest confirmation wins.
    for i in range(len(window) - 1, 0, -1):
        curr = window[i]
        prev = window[i - 1]
        z_low, z_high = zone["low"], zone["high"]

        # Price must have entered the zone on this or the previous candle.
        entered = (
            (z_low <= curr["low"] <= z_high)
            or (z_low <= curr["high"] <= z_high)
            or (curr["low"] <= z_low <= curr["high"])
            or (curr["low"] <= z_high <= curr["high"])
        )
        if not entered:
            continue

        # Absorption: price failed to break through the zone boundary.
        if direction == "LONG" and curr["close"] < z_low:
            continue
        if direction == "SHORT" and curr["close"] > z_high:
            continue

        # Rejection or engulfing pattern.
        rejected = _is_rejection_candle(curr, direction)
        engulfing = _is_engulfing(curr, prev, direction)
        if not (rejected or engulfing):
            continue

        # Volume spike.
        if vol_sma is not None and vol_sma > 0:
            if curr.get("volume", 0) <= vol_sma * VOLUME_SPIKE_MULT:
                continue

        return curr
    return None


# ----- target selection -----------------------------------------------------

def _next_opposing_zone(zones: list[dict], direction: str, entry_price: float) -> dict | None:
    """Return the nearest active zone of the opposite type beyond entry_price."""
    opp = "SUPPLY" if direction == "LONG" else "DEMAND"
    candidates = [z for z in zones if z["type"] == opp]
    if direction == "LONG":
        beyond = [z for z in candidates if z["low"] > entry_price]
        if not beyond:
            return None
        return min(beyond, key=lambda z: z["low"])
    beyond = [z for z in candidates if z["high"] < entry_price]
    if not beyond:
        return None
    return max(beyond, key=lambda z: z["high"])


# ----- signal ---------------------------------------------------------------

def brandontrades_supply_demand(
    spot_price=None,
    asset="NQ",
    max_reentries=3,
    five_m_bars=None,
    fifteen_m_bars=None,
    one_h_bars=None,
    four_h_bars=None,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_et_now()
    today = now.date()
    if not (five_m_bars and fifteen_m_bars and one_h_bars and four_h_bars):
        return no_signal("missing_bars", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)

    # Refresh HTF supply/demand zones from fully closed bars.
    new_zones = []
    for tf in HTF_ZONE_TFS:
        htf = _get_htf_bars(tf, fifteen_m_bars, one_h_bars, four_h_bars)
        if not htf:
            continue
        max_age = MAX_ZONE_AGE_BARS.get(tf, 24)
        lookback = htf[-max_age:] if len(htf) > max_age else htf
        for z in _detect_zones(lookback, tf):
            # Avoid duplicates from overlapping timeframe aggregations.
            if not any(
                existing["type"] == z["type"]
                and abs(existing["high"] - z["high"]) < 1e-9
                and abs(existing["low"] - z["low"]) < 1e-9
                for existing in state["zones"]
            ):
                new_zones.append(z)

    if new_zones:
        state["zones"].extend(new_zones)
    state["zones"] = _prune_zones(state["zones"], now)

    if not state["zones"]:
        store.save(key, state)
        return no_signal("no_zones", SOURCE)

    # Find the zone that spot_price currently occupies.
    active_zone = None
    for z in state["zones"]:
        if z["low"] <= spot_price <= z["high"]:
            active_zone = z
            break
    if active_zone is None:
        state["pending_zone"] = None
        store.save(key, state)
        return no_signal("price_not_in_zone", SOURCE)

    direction = "LONG" if active_zone["type"] == "DEMAND" else "SHORT"

    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        store.save(key, state)
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        store.save(key, state)
        return no_signal("cooldown", SOURCE)

    confirmation = _find_confirmation(direction, active_zone, five_m_bars)
    if confirmation is None:
        state["pending_zone"] = active_zone
        store.save(key, state)
        return no_signal("no_confirmation", SOURCE)

    entry_price = confirmation["close"]
    if direction == "LONG":
        sl = confirmation["low"]
        # Target the next supply zone; default to 1:2 RR if none found.
        target = _next_opposing_zone(state["zones"], direction, entry_price)
        tp = target["low"] if target else entry_price + 2 * (entry_price - sl)
    else:
        sl = confirmation["high"]
        target = _next_opposing_zone(state["zones"], direction, entry_price)
        tp = target["high"] if target else entry_price - 2 * (sl - entry_price)

    if sl == entry_price:
        store.save(key, state)
        return no_signal("zero_stop_distance", SOURCE)

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
    state["last_entry_time"] = confirmation["timestamp"]
    state["pending_zone"] = None
    store.save(key, state)

    return {
        "triggered": True,
        "direction": direction,
        "confidence": reentry_scale(n),
        "entry_price": float(entry_price),
        "signal_price": float(entry_price),
        "source": SOURCE,
        "sl": float(sl),
        "tp": float(tp),
        "zone": active_zone,
        "confirmation_timestamp": confirmation["timestamp"],
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} asset={asset.upper()} "
            f"zone={active_zone['tf']}:{active_zone['type']} "
            f"entry={entry_price:.3f} sl={sl:.3f} tp={tp:.3f}"
        ),
    }


# ----- sanity check ---------------------------------------------------------

def _bar(ts, o, h, l, c, v=1000):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


if __name__ == "__main__":
    import sys

    base = datetime(2026, 8, 17, 10, 0, 0, tzinfo=UTC)

    # Reset any persisted state from prior runs so this sanity check is isolated.
    _test_asset = "NQ"
    _test_date = base.astimezone(EST).date()
    _test_mx = 3
    _test_key = store.make_key(_test_asset, _test_date, _test_mx)
    store._mem.pop(_test_key, None)
    _test_path = store._path(_test_key)
    if _test_path.exists():
        _test_path.unlink()

    # 4h bars: small basing -> large bearish impulse -> continuation.
    four_h = [
        _bar(base - timedelta(hours=12), 4500.0, 4500.5, 4499.5, 4500.2, 500),
        _bar(base - timedelta(hours=8), 4500.2, 4500.5, 4485.0, 4485.5, 5000),  # supply impulse
        _bar(base - timedelta(hours=4), 4485.5, 4490.0, 4480.0, 4482.0, 500),
    ]

    # 1h bars: same supply structure with enough history for 1h zone detection.
    one_h = [
        _bar(base - timedelta(hours=4), 4500.0, 4502.0, 4498.0, 4501.0, 400),
        _bar(base - timedelta(hours=3), 4501.0, 4502.0, 4499.0, 4500.0, 400),
        _bar(base - timedelta(hours=2), 4500.0, 4501.0, 4499.0, 4500.5, 400),   # basing
        _bar(base - timedelta(hours=1), 4500.5, 4501.0, 4485.0, 4486.0, 4000),  # impulse
    ]

    # 15m bars: enough to build 30m/45m bars and produce a supply zone.
    fifteen_m = [
        _bar(base - timedelta(hours=2), 4500.0, 4501.0, 4499.0, 4500.5, 100),
        _bar(base - timedelta(hours=1, minutes=45), 4500.5, 4501.0, 4499.5, 4500.8, 100),
        _bar(base - timedelta(hours=1, minutes=30), 4500.8, 4501.0, 4500.0, 4500.7, 100),
        _bar(base - timedelta(hours=1, minutes=15), 4500.7, 4501.0, 4500.2, 4500.9, 100),
        _bar(base - timedelta(hours=1), 4500.9, 4501.1, 4500.5, 4501.0, 100),
        _bar(base - timedelta(minutes=45), 4501.0, 4501.2, 4500.8, 4501.0, 100),
        _bar(base - timedelta(minutes=30), 4501.0, 4501.1, 4500.9, 4501.0, 100),  # 30m basing
        _bar(base - timedelta(minutes=15), 4501.0, 4501.0, 4485.0, 4486.0, 2000), # 30m impulse
    ]

    # 5m execution bars: price returns into the supply zone and prints rejection.
    five_m = [
        _bar(base - timedelta(minutes=10), 4490.0, 4495.0, 4488.0, 4492.0, 800),
        _bar(base - timedelta(minutes=5), 4495.0, 4501.0, 4494.0, 4500.8, 1200),
        # Confirmation candle: bearish with long upper wick inside the zone.
        _bar(base, 4501.0, 4503.0, 4500.8, 4500.8, 2500),
    ]

    # Patch wall-clock time so the session gate is satisfied.
    orig_et = tu.get_et_now
    orig_utc = tu.get_utc_now
    try:
        tu.get_et_now = lambda: base.astimezone(EST)
        tu.get_utc_now = lambda: base
        sig = brandontrades_supply_demand(
            spot_price=4500.8,
            asset="NQ",
            max_reentries=3,
            five_m_bars=five_m,
            fifteen_m_bars=fifteen_m,
            one_h_bars=one_h,
            four_h_bars=four_h,
        )
        assert sig["triggered"] is True, f"expected trigger, got {sig}"
        assert sig["direction"] == "SHORT", f"expected SHORT, got {sig['direction']}"
        assert sig["entry_price"] == 4500.8
        assert sig["sl"] == 4503.0
        assert "tp" in sig
        print("SHORT supply-demand signal OK", sig["reason"])

        # No signal when price is far from any zone.
        no_sig = brandontrades_supply_demand(
            spot_price=4400.0,
            asset="NQ",
            max_reentries=3,
            five_m_bars=five_m,
            fifteen_m_bars=fifteen_m,
            one_h_bars=one_h,
            four_h_bars=four_h,
        )
        assert no_sig["triggered"] is False, f"expected no signal, got {no_sig}"
        print("No-signal case OK")

        # Defensive: empty / minimal windows must not crash.
        empty_sig = brandontrades_supply_demand(
            spot_price=None,
            asset="NQ",
            max_reentries=3,
            five_m_bars=[],
            fifteen_m_bars=[],
            one_h_bars=[],
            four_h_bars=[],
        )
        assert empty_sig["triggered"] is False, f"expected no signal for empty input, got {empty_sig}"
        print("Empty input case OK")

        minimal_sig = brandontrades_supply_demand(
            spot_price=4500.0,
            asset="NQ",
            max_reentries=3,
            five_m_bars=[_bar(base, 4500.0, 4501.0, 4499.0, 4500.5, 100)],
            fifteen_m_bars=[_bar(base, 4500.0, 4501.0, 4499.0, 4500.5, 100)],
            one_h_bars=[_bar(base, 4500.0, 4501.0, 4499.0, 4500.5, 100)],
            four_h_bars=[_bar(base, 4500.0, 4501.0, 4499.0, 4500.5, 100)],
        )
        assert minimal_sig["triggered"] is False, f"expected no signal for minimal bars, got {minimal_sig}"
        print("Minimal bars case OK")
    finally:
        tu.get_et_now = orig_et
        tu.get_utc_now = orig_utc

    sys.exit(0)


# QA_REPORT: passed
# Date: 2026-08-17
# Reviewer: QA subagent
# Issues found and fixes applied:
#   1. Sanity check failed on first run because StateStore persisted cooldown/entry
#      state from a prior execution. Fixed by resetting the on-disk and in-memory
#      state for the test key before the sanity-check block in __main__.
#   2. Cooldown/zone state was not persisted on several no-signal return paths
#      (no_zones, max_reentries, cooldown, zero_stop_distance). Added store.save()
#      before those returns so decremented cooldown counters and updated zones are
#      durable across ticks/process restarts.
#   3. Added defensive __main__ cases for empty inputs and minimal bar windows to
#      confirm the signal returns a non-triggered dict without crashing.
# Lookahead bias: No future data is referenced. HTF bars are built from the supplied
# bar lists and the signal treats the last element of each list as the latest closed
# bar (the engine is responsible for not feeding partially-formed bars).
# Return contract: When triggered the dict contains triggered, direction, confidence,
# entry_price, signal_price, source, sl, tp, and reason; no_signal returns the standard
# non-triggered dict. StateStore keys are (asset, date, variant, max_reentries) so
# state does not leak across assets or dates.
