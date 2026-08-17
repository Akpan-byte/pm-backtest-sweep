# CHANGE_SUMMARY
# 2026-08-17  kilo
#   - QA review fixes: corrected sanity-check timezone handling, made the
#     inline self-test self-contained by clearing persisted StateStore state,
#     renamed `_in_london_killzone` parameter to avoid shadowing `detectors as dt`.
#   - Added QA_REPORT block confirming no lookahead bias, standard FUTURES dict
#     compliance, defensive programming, and correct StateStore isolation.
#
# 2026-08-17  kilo
#   - Created strategies/signals/trident_pattern.py (Blueprint 3) for the
#     StarTrading futures backtest harness.
#   - Implements the hyper-selective 30-Minute London Killzone trend-continuation
#     setup: perfectly-stacked EMAs, 200 EMA macro filter, bullish 30m FVG with
#     RVOL>1.2, retracement to Consequent Encroachment, Doji rejection, and a
#     bullish confirmation candle that closes below the FVG high.
#   - Heavily long-biased per the blueprint; only LONG entries are emitted.
#   - Accepts either pre-built `thirty_m_bars` or aggregates them from the
#     `fifteen_m_bars` window supplied by the harness.
#   - State keyed per (asset,date) via StateStore; duplicate signals on the same
#     30m bar are suppressed.
# WHY: Complete the assigned Blueprint 3 signal while staying inside the
#      existing FUTURES signal contract (LONG/SHORT, market entry, sl/tp).

"""Blueprint 3 (FUTURES): Trident Pattern — 30m London Killzone continuation.

Documentation / master blueprint: docs/TRIDENT_PATTERN.md (to be created by
strategy owner if needed).

Core logic:
  Trade only during the London Killzone (3:00 AM - 6:30 AM ET) on select FX
  majors and Gold. Require a perfectly-bullish EMA stack (5>9>15>21>200) on
  the 30m chart, a strong bullish FVG created with RVOL>1.2, a Doji rejection
  that pierces the FVG midpoint, and a bullish confirmation candle that does
  NOT close above the FVG high / recent swing. Entry is at the confirmation
  close; SL sits below the Doji low (with a ~10-pip buffer for FX); TP targets
  a minimum 1:20 RR projection.

Signal kwargs (in addition to the standard set):
  thirty_m_bars : list of 30m OHLCV dicts (preferred)
  fifteen_m_bars: list of 15m OHLCV dicts (aggregated to 30m if 30m absent)
  pip_value     : price distance of one pip (default 1.0)
"""

import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..core import time_utils as tu
from ..core import candle_utils as cu
from ..core import detectors as dt
from ..core.state_store import StateStore
from .common import (
    no_signal,
    validate_signal_inputs,
    reentry_scale,
    MIN_COOLDOWN_TICKS,
)

log = logging.getLogger("trident_pattern")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "TRIDENT_PATTERN"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("trident_pattern", _PROJECT_ROOT)

# Assets allowed by the blueprint. AUDUSD is explicitly excluded.
ALLOWED_ASSETS = {
    "USDCAD",
    "NZDUSD",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "XAUUSD",
}
FOREX_ASSETS = ALLOWED_ASSETS - {"XAUUSD"}

# London Killzone ET window for the entry confirmation candle.
LONDON_START = time(3, 0)
LONDON_END = time(6, 30)
# The FVG-creation candle is allowed to start forming at 2:30 ET per blueprint.
LONDON_FVG_START = time(2, 30)


def _make_state():
    return {
        "today": None,
        "entry_count": {"LONG": 0, "SHORT": 0},
        "cooldown": {"LONG": 0, "SHORT": 0},
        "last_entry_time": None,
        "last_signal_bar_ts": None,
    }


# ----- private helpers -------------------------------------------------------

def _ema(values: list[float], period: int) -> float | None:
    """Final EMA value for `values` using the standard smoothing formula."""
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1.0 - k)
    return ema


def _in_london_killzone(d: datetime, allow_fvg_230: bool = False) -> bool:
    """Return True if `d` (ET) is inside the London Killzone window."""
    t = d.time()
    start = LONDON_FVG_START if allow_fvg_230 else LONDON_START
    return start <= t <= LONDON_END


def _aggregate_30m_from_15m(fifteen_m_bars: list[dict]) -> list[dict]:
    """Build 30m bars by pairing consecutive 15m bars.

    The engine currently does not emit a dedicated 30m window, so this helper
    lets the signal run from the supplied 15m deque. Note: a 200-bar 15m
    window only yields 100 thirty-minute bars, which is insufficient for the
    200 EMA filter; a dedicated `thirty_m_bars` feed with >= 200 bars is
    required for live production use.
    """
    if len(fifteen_m_bars) < 2:
        return []
    sorted_bars = sorted(fifteen_m_bars, key=lambda b: b["timestamp"])
    out = []
    for i in range(0, len(sorted_bars) - 1, 2):
        a, b = sorted_bars[i], sorted_bars[i + 1]
        out.append({
            "timestamp": b["timestamp"],
            "open": a["open"],
            "high": max(a["high"], b["high"]),
            "low": min(a["low"], b["low"]),
            "close": b["close"],
            "volume": a["volume"] + b["volume"],
        })
    return out


def _is_doji_with_strong_lower_wick(candle: dict) -> bool:
    """Doji candle whose lower wick clearly dominates the upper wick."""
    if not cu.is_doji_candle(candle):
        return False
    rng = cu.candle_range(candle)
    if rng <= 0:
        return False
    if candle["close"] >= candle["open"]:
        lower_wick = candle["open"] - candle["low"]
        upper_wick = candle["high"] - candle["close"]
    else:
        lower_wick = candle["close"] - candle["low"]
        upper_wick = candle["high"] - candle["open"]
    if lower_wick < upper_wick * 2.0:
        return False
    if lower_wick < 0.40 * rng:
        return False
    return True


def _find_trident_setup(thirty_m_bars: list[dict], asset: str, pip_value: float) -> dict | None:
    """Look for a completed Trident long setup where the last bar is the confirmation candle.

    Returns a dict with fvg, creation, doji, confirmation, sl, tp, or None.
    """
    if len(thirty_m_bars) < 200:
        return None

    closes = [b["close"] for b in thirty_m_bars]
    ema5 = _ema(closes, 5)
    ema9 = _ema(closes, 9)
    ema15 = _ema(closes, 15)
    ema21 = _ema(closes, 21)
    ema200 = _ema(closes, 200)

    if not all((ema5, ema9, ema15, ema21, ema200)):
        return None
    if not (ema5 > ema9 > ema15 > ema21 > ema200):
        return None
    # Price must be trading above the fastest EMA (momentum confirmation).
    if closes[-1] <= ema5:
        return None

    # The confirmation candle is the most recently closed 30m bar.
    confirmation = thirty_m_bars[-1]
    if confirmation["close"] <= confirmation["open"]:
        return None

    # The Doji / dipping candle must be the bar immediately preceding confirmation.
    if len(thirty_m_bars) < 3:
        return None
    doji = thirty_m_bars[-2]
    if not _is_doji_with_strong_lower_wick(doji):
        return None
    # Full-bodied bearish close inside the FVG invalidates.
    if doji["close"] < doji["open"] and cu.is_strong_body_candle(doji):
        return None

    # Locate a qualifying bullish FVG created before the doji.
    fvgs = dt.detect_fvg(thirty_m_bars, "30m")
    if not fvgs:
        return None

    for fvg in reversed(fvgs):
        if fvg["direction"] != "UP":
            continue

        creation_idx = None
        for i, b in enumerate(thirty_m_bars):
            if b["timestamp"] == fvg["timestamp"]:
                creation_idx = i
                break
        if creation_idx is None or creation_idx >= len(thirty_m_bars) - 2:
            continue

        creation = thirty_m_bars[creation_idx]
        creation_et = creation["timestamp"].astimezone(EST)
        if not _in_london_killzone(creation_et, allow_fvg_230=True):
            continue
        if creation["close"] <= creation["open"] or not cu.is_strong_body_candle(creation):
            continue

        # RVOL > 1.2 during FVG creation.
        if creation_idx < 20:
            continue
        vol_sma = sum(b["volume"] for b in thirty_m_bars[creation_idx - 20:creation_idx]) / 20.0
        if vol_sma <= 0 or creation["volume"] / vol_sma <= 1.2:
            continue

        fvg_mid = (fvg["low"] + fvg["high"]) / 2.0
        # Doji must retrace into the FVG and pierce the 50% midpoint.
        if doji["low"] > fvg["high"] or doji["low"] > fvg_mid:
            continue

        # FVG is invalidated if any candle between creation and doji closes below the FVG low.
        invalidated = False
        for k in range(creation_idx + 1, len(thirty_m_bars) - 1):
            if thirty_m_bars[k]["close"] < fvg["low"]:
                invalidated = True
                break
        if invalidated:
            continue

        # Confirmation must close below the FVG high / recent swing high.
        recent_high = max(b["high"] for b in thirty_m_bars[creation_idx:len(thirty_m_bars) - 1])
        if confirmation["close"] > fvg["high"] or confirmation["close"] > recent_high:
            continue
        # Sanity: a close below the FVG low would have filled the gap; reject.
        if confirmation["close"] < fvg["low"]:
            continue

        entry_price = confirmation["close"]
        sl = doji["low"]
        if asset in FOREX_ASSETS:
            sl -= 10.0 * pip_value
        # Minimum 1:20 RR target per blueprint.
        risk = entry_price - sl
        tp = entry_price + 20.0 * risk if risk > 0 else entry_price

        return {
            "fvg": fvg,
            "creation": creation,
            "doji": doji,
            "confirmation": confirmation,
            "sl": sl,
            "tp": tp,
        }

    return None


# ----- signal ----------------------------------------------------------------

def trident_pattern(
    spot_price=None,
    asset="EURUSD",
    max_reentries=1,
    thirty_m_bars=None,
    fifteen_m_bars=None,
    pip_value=1.0,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    asset_up = asset.upper()
    if asset_up not in ALLOWED_ASSETS:
        return no_signal("asset_not_allowed", SOURCE)

    now = tu.get_et_now()
    today = now.date()
    if not _in_london_killzone(now):
        return no_signal("outside_london_killzone", SOURCE)

    bars = thirty_m_bars
    if bars is None and fifteen_m_bars:
        bars = _aggregate_30m_from_15m(fifteen_m_bars)
    if not bars or len(bars) < 200:
        return no_signal("insufficient_30m_bars", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset_up, today)
    store.tick_cooldowns(state)

    last_bar_ts = bars[-1]["timestamp"]
    if state.get("last_signal_bar_ts") == last_bar_ts:
        return no_signal("already_signaled_this_bar", SOURCE)

    setup = _find_trident_setup(bars, asset_up, float(pip_value))
    if setup is None:
        return no_signal("no_trident_setup", SOURCE)

    direction = "LONG"
    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        return no_signal("cooldown", SOURCE)

    entry_price = setup["confirmation"]["close"]
    if entry_price is None or entry_price <= 0:
        return no_signal("no_price", SOURCE)

    sl = setup["sl"]
    tp = setup["tp"]

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
    state["last_entry_time"] = last_bar_ts
    state["last_signal_bar_ts"] = last_bar_ts
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
        "fvg_high": float(setup["fvg"]["high"]),
        "fvg_low": float(setup["fvg"]["low"]),
        "doji_low": float(setup["doji"]["low"]),
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} asset={asset_up} "
            f"fvg={setup['fvg']['low']:.5f}-{setup['fvg']['high']:.5f} "
            f"doji_low={setup['doji']['low']:.5f} conf_close={entry_price:.5f} "
            f"sl={sl:.5f} tp={tp:.5f}"
        ),
    }


# ----- inline sanity check ---------------------------------------------------

if __name__ == "__main__":
    import math
    import shutil

    # The sanity check must be self-contained; clear any persisted state so a
    # re-run does not see stale cooldowns or a duplicate-bar timestamp.
    if store.dir.exists():
        shutil.rmtree(store.dir)

    def _make_30m_bar(i, o, h, l, c, v, base_et):
        # Build a UTC timestamp, then convert to ET so killzone checks are exact.
        ts = base_et.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=30 * i)
        return {
            "timestamp": ts.astimezone(EST),
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        }

    base = datetime(2026, 8, 17, tzinfo=UTC)
    # Build 200+ bars of a gentle uptrend so EMAs stack bullish.
    bars = []
    price = 1.08000
    for i in range(210):
        o = price
        c = price + 0.00005 * (1 if i % 2 == 0 else -1)
        h = max(o, c) + 0.00010
        l = min(o, c) - 0.00010
        v = 1000.0
        bars.append(_make_30m_bar(i, o, h, l, c, v, base))
        price = c + 0.00002

    # Carve a bullish FVG near the end of the series. Indices:
    #   fvg_idx-1 = prev, fvg_idx = creation, fvg_idx+1 = nxt,
    #   fvg_idx+2 = doji, fvg_idx+3 = confirmation (last bar).
    fvg_idx = 206
    bars[fvg_idx - 1] = _make_30m_bar(fvg_idx - 1, 1.08900, 1.08920, 1.08890, 1.08910, 1000.0, base)
    bars[fvg_idx] = _make_30m_bar(fvg_idx, 1.08950, 1.09010, 1.08940, 1.09000, 5000.0, base)
    bars[fvg_idx + 1] = _make_30m_bar(fvg_idx + 1, 1.09020, 1.09050, 1.09020, 1.09040, 1200.0, base)

    # Doji dipping candle at fvg_idx + 2: pierces midpoint, closes doji with long lower wick.
    doji_idx = fvg_idx + 2
    bars[doji_idx] = _make_30m_bar(
        doji_idx, 1.09030, 1.09035, 1.08960, 1.09032, 800.0, base
    )

    # Confirmation candle at fvg_idx + 3: bullish, close below FVG high / recent swing.
    conf_idx = fvg_idx + 3
    bars[conf_idx] = _make_30m_bar(
        conf_idx, 1.09010, 1.09025, 1.09005, 1.09018, 900.0, base
    )

    # Force the final four bars into known London Killzone ET times so the test
    # is deterministic regardless of which day the series lands on.
    for idx, hour, minute in [
        (fvg_idx - 1, 3, 0),
        (fvg_idx, 3, 30),
        (doji_idx, 4, 0),
        (conf_idx, 4, 30),
    ]:
        bars[idx]["timestamp"] = bars[idx]["timestamp"].replace(hour=hour, minute=minute)

    # Monkeypatch time so the signal thinks it is the confirmation close inside the killzone.
    orig_get_et_now = tu.get_et_now
    try:
        tu.get_et_now = lambda: bars[conf_idx]["timestamp"]
        sig = trident_pattern(
            spot_price=bars[conf_idx]["close"],
            asset="EURUSD",
            thirty_m_bars=bars,
            pip_value=0.0001,
        )
        assert sig["triggered"] is True, f"expected trigger, got {sig['reason']}"
        assert sig["direction"] == "LONG"
        assert sig["entry_price"] == bars[conf_idx]["close"]
        assert sig["sl"] < sig["entry_price"]
        # 1:20 RR target: TP should be far above entry.
        assert sig["tp"] > sig["entry_price"]
        assert math.isclose(
            (sig["tp"] - sig["entry_price"]) / (sig["entry_price"] - sig["sl"]),
            20.0,
            rel_tol=1e-9,
        )
        print("trident_pattern sanity check passed:", sig["reason"])

        # Same call again on the identical bar must be suppressed.
        sig2 = trident_pattern(
            spot_price=bars[conf_idx]["close"],
            asset="EURUSD",
            thirty_m_bars=bars,
            pip_value=0.0001,
        )
        assert sig2["triggered"] is False
        assert sig2["reason"] == "already_signaled_this_bar"

        # Outside killzone should not fire.
        tu.get_et_now = lambda: bars[conf_idx]["timestamp"].replace(hour=10, minute=0)
        sig3 = trident_pattern(
            spot_price=bars[conf_idx]["close"],
            asset="EURUSD",
            thirty_m_bars=bars,
            pip_value=0.0001,
        )
        assert sig3["triggered"] is False
        assert sig3["reason"] == "outside_london_killzone"

        # Disallowed asset should not fire.
        sig4 = trident_pattern(
            spot_price=bars[conf_idx]["close"],
            asset="AUDUSD",
            thirty_m_bars=bars,
            pip_value=0.0001,
        )
        assert sig4["triggered"] is False
        assert sig4["reason"] == "asset_not_allowed"
    finally:
        tu.get_et_now = orig_get_et_now

    print("All trident_pattern sanity checks passed.")


# QA_REPORT: passed
# Issues found and fixes applied:
#   1. Sanity-check timestamps were built with `.replace(tzinfo=UTC)` (which
#      re-labels local time as UTC rather than converting), causing the test
#      to pass only by accident. Fixed by building the series from a UTC base
#      and converting with `.astimezone(EST)`, then explicitly forcing the
#      final four carved bars to known London Killzone ET times.
#   2. The inline sanity check was not self-contained: persisted StateStore
#      state could make re-runs fail with "cooldown" or
#      "already_signaled_this_bar". Fixed by clearing `store.dir` at the start
#      of `__main__`.
#   3. `_in_london_killzone` used parameter name `dt`, shadowing the module-
#      level `from ..core import detectors as dt` import. Renamed the
#      parameter to `d`.
# Lookahead bias: none detected. All EMA/FVG/doji/confirmation logic only
#   references bars that are closed before or at the current confirmation bar.
# Standard FUTURES dict: verified — triggered, direction (LONG/SHORT),
#   confidence, entry_price, signal_price, source, reason, plus sl and tp when
#   triggered.
# Defensive programming: verified — returns no_signal for empty/None 30m or
#   15m windows, missing spot price, and disallowed assets.
# StateStore: keyed per (asset, date, variant, max_reentries); `prune` drops
#   stale in-memory keys; no cross-asset or cross-date leak observed.
