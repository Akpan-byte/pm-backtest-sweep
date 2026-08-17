# CHANGE_SUMMARY
# 2026-08-17  coder
#   - Created strategies/signals/trade_ats_ma_master.py (Strategy 5 / Master
#     Pattern).  Implements a multi-timeframe mean-reversion-to-trend signal:
#     daily HTF bias + phase filter, 1H LTF execution against a 48-period
#     calibrated MA, LONG/SHORT market entry, dynamic SL/TP.
# WHY: FUTURES signal contract for the StarTrading backtest harness.

"""Blueprint 5 (FUTURES): Multi-Timeframe Mean-Reversion to Trend (Master Pattern).

Documentation / master blueprint: docs/trade_ats_ma_master.md (to be created)

Core logic:
  Use daily bars to establish HTF trend direction and confirm we are in a
  "Phase 3" trending regime (sustained deviation from value).  On the 1H LTF,
  enter when price pushes inefficiently through the calibrated 48-period MA
  (length = 2 * 24 one-hour candles in a trading day) in the opposite direction
  of the HTF trend, expecting mean reversion toward the MA / continuation of
  the HTF move.

Signal kwargs (in addition to the standard set):
  daily_bars : list of daily OHLCV dicts (HTF trend + phase)
  one_h_bars : list of 1h OHLCV dicts (calibrated MA + execution)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from ..core import time_utils as tu
    from ..core import candle_utils as cu
    from ..core.state_store import StateStore
    from .common import (
        no_signal,
        validate_signal_inputs,
        reentry_scale,
        MIN_COOLDOWN_TICKS,
    )
except ImportError:
    # Allow direct execution for local sanity checks.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from strategies.core import time_utils as tu
    from strategies.core import candle_utils as cu
    from strategies.core.state_store import StateStore
    from strategies.signals.common import (
        no_signal,
        validate_signal_inputs,
        reentry_scale,
        MIN_COOLDOWN_TICKS,
    )

log = logging.getLogger("trade_ats_ma_master")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "ATS_MA_MASTER"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("trade_ats_ma_master", _PROJECT_ROOT)

# Strategy constants
HTF_TREND_EMA_LEN = 21          # daily EMA for HTF direction
HTF_TREND_BARS_OUTSIDE = 2      # consecutive closes on inefficient side of EMA
LTF_MA_LEN = 48                 # 2 * 24 one-hour candles in a day
LTF_ATR_LEN = 14
LTF_SWING_LOOKBACK = 12         # bars used for structural SL
MIN_DEVIATION_ATR = 0.5         # price must extend at least 0.5 ATR beyond MA


def _make_state():
    return {
        "today": None,
        "htf_bias": None,            # "LONG" / "SHORT"
        "phase": None,               # "CONTRACTION" / "EXPANSION" / "TREND"
        "entry_count": {"LONG": 0, "SHORT": 0},
        "cooldown": {"LONG": 0, "SHORT": 0},
        "last_entry_time": None,
    }


# ----- indicator helpers ------------------------------------------------------

def _ema(values: list[float], period: int) -> float | None:
    """Return the EMA of the last `period` values."""
    if len(values) < period:
        return None
    k = 2.0 / (period + 1.0)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1.0 - k)
    return ema


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _std(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    avg = sum(values[-period:]) / period
    var = sum((v - avg) ** 2 for v in values[-period:]) / period
    return var ** 0.5


def _htf_trend_and_phase(daily_bars: list[dict]) -> tuple[str | None, str | None]:
    """Return (bias, phase) from daily bars.

    bias  : "LONG" / "SHORT" based on close vs 21 EMA and slope direction.
    phase : "TREND" only when price has closed on the inefficient side of the
            21 EMA for HTF_TREND_BARS_OUTSIDE consecutive bars AND the EMA is
            rising (LONG) or falling (SHORT).  This captures "Phase 3"
            sustained deviation from value without requiring price to close
            outside the Bollinger Bands, which is uncommon in clean trends.
    """
    if len(daily_bars) < HTF_TREND_EMA_LEN + HTF_TREND_BARS_OUTSIDE + 1:
        return None, None

    closes = [c["close"] for c in daily_bars]
    ema21 = _ema(closes, HTF_TREND_EMA_LEN)
    if ema21 is None:
        return None, None

    # EMA slope: last EMA must be higher/lower than a few bars back.
    ema_values = []
    k = 2.0 / (HTF_TREND_EMA_LEN + 1.0)
    ema = closes[0]
    for v in closes:
        ema = v * k + ema * (1.0 - k)
        ema_values.append(ema)

    last_close = closes[-1]
    trend_up = (
        all(daily_bars[-i]["close"] > ema_values[-i] for i in range(1, HTF_TREND_BARS_OUTSIDE + 1))
        and ema_values[-1] > ema_values[-HTF_TREND_BARS_OUTSIDE - 1]
    )
    trend_down = (
        all(daily_bars[-i]["close"] < ema_values[-i] for i in range(1, HTF_TREND_BARS_OUTSIDE + 1))
        and ema_values[-1] < ema_values[-HTF_TREND_BARS_OUTSIDE - 1]
    )

    if trend_up:
        phase = "TREND"
        bias = "LONG"
    elif trend_down:
        phase = "TREND"
        bias = "SHORT"
    else:
        # Not in a sustained trend phase; do not trade.
        phase = "SKIP"
        if last_close > ema21:
            bias = "LONG"
        else:
            bias = "SHORT"

    return bias, phase


def _ltf_ma(one_h_bars: list[dict]) -> float | None:
    """Calibrated 48-period SMA on 1H closes."""
    if len(one_h_bars) < LTF_MA_LEN:
        return None
    return _sma([c["close"] for c in one_h_bars], LTF_MA_LEN)


def _structural_extreme(one_h_bars: list[dict], direction: str) -> float | None:
    """Recent swing low (LONG) or high (SHORT) for structural stop placement."""
    window = one_h_bars[-LTF_SWING_LOOKBACK:]
    if not window:
        return None
    if direction == "LONG":
        return min(c["low"] for c in window)
    return max(c["high"] for c in window)


# ----- signal -----------------------------------------------------------------

def trade_ats_ma_master(
    spot_price=None,
    asset="NQ",
    max_reentries=3,
    daily_bars=None,
    one_h_bars=None,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    now = tu.get_et_now()
    today = now.date()

    if not (daily_bars and one_h_bars):
        return no_signal("missing_bars", SOURCE)

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)

    # Step 1: HTF trend + phase.
    bias, phase = _htf_trend_and_phase(daily_bars)
    state["htf_bias"] = bias
    state["phase"] = phase
    if bias is None or phase != "TREND":
        return no_signal("no_htf_trend_phase", SOURCE)

    # Step 2: LTF calibrated MA.
    ma = _ltf_ma(one_h_bars)
    if ma is None:
        return no_signal("no_ltf_ma", SOURCE)

    # Step 3: price must be inefficiently extended beyond the MA on the LTF.
    atr = cu.calculate_atr(one_h_bars[-(LTF_ATR_LEN + 1):], LTF_ATR_LEN)
    if atr is None or atr <= 0:
        return no_signal("no_ltf_atr", SOURCE)

    last = one_h_bars[-1]
    close = last["close"]
    deviation = close - ma

    if bias == "LONG":
        # In a bullish HTF trend, buy when 1H price is pushed below value (MA).
        if deviation >= 0 or abs(deviation) < MIN_DEVIATION_ATR * atr:
            return no_signal("no_ltf_deviation_long", SOURCE)
        direction = "LONG"
    else:
        # In a bearish HTF trend, sell when 1H price is pushed above value.
        if deviation <= 0 or abs(deviation) < MIN_DEVIATION_ATR * atr:
            return no_signal("no_ltf_deviation_short", SOURCE)
        direction = "SHORT"

    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        return no_signal("cooldown", SOURCE)

    entry_price = close
    if entry_price is None or entry_price <= 0:
        return no_signal("no_price", SOURCE)

    # Step 4: dynamic stop loss.
    # Method 1: structural extreme in the LTF lookback.
    structural = _structural_extreme(one_h_bars, direction)
    # Method 2: 1:1 return-to-value distance (entry -> MA -> SL).
    one_to_one_sl = 2.0 * entry_price - ma

    if direction == "LONG":
        # SL is below entry; pick the level closest to entry (tightest risk).
        candidates = [one_to_one_sl]
        if structural is not None:
            candidates.append(structural)
        sl = max(candidates)
        # TP: return to value (MA), then let the harness trail/continue if price
        # cuts through and keeps running.
        tp = ma
    else:
        candidates = [one_to_one_sl]
        if structural is not None:
            candidates.append(structural)
        sl = min(candidates)
        tp = ma

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
    state["last_entry_time"] = last["timestamp"]
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
        "htf_bias": bias,
        "ltf_ma": float(ma),
        "deviation_atr": float(deviation / atr),
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} bias={bias} "
            f"asset={asset.upper()} ma={ma:.2f} close={close:.2f} "
            f"sl={sl:.2f} tp={tp:.2f} price={entry_price:.3f}"
        ),
    }


# ----- inline sanity check ----------------------------------------------------

def _make_bar(ts, o, h, l, c, v=1):
    return {
        "timestamp": ts,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
    }


if __name__ == "__main__":
    from datetime import datetime, timedelta
    import shutil

    base = datetime(2026, 1, 1, tzinfo=UTC)

    # Ensure a clean state store for deterministic sanity checks.
    store._mem.clear()
    shutil.rmtree(store.dir, ignore_errors=True)

    # Build 30 daily bars: tight low-volatility base, then a sharp breakout so
    # the last two daily closes sit well above the rising 21 EMA (Phase 3).
    daily = []
    price = 100.0
    for i in range(30):
        if i < 20:
            o = price
            c = price + (0.02 if i % 2 == 0 else -0.02)
            h = max(o, c) + 0.02
            l = min(o, c) - 0.02
            price = c
        elif i == 20:
            o = price
            c = price + 10.0
            h = c + 0.1
            l = o - 0.05
            price = c
        else:
            o = price
            c = price + 1.5
            h = c + 0.1
            l = o - 0.05
            price = c
        daily.append(_make_bar(base + timedelta(days=i), o, h, l, c))

    # Build 60 one-hour bars: first 48 flat around 100, then a sharp dip that
    # closes far below the 48-period SMA while the daily HTF is bullish.
    one_h = []
    price = 100.0
    for i in range(60):
        if i < 48:
            o = price
            c = price + (0.02 if i % 2 == 0 else -0.02)
            h = max(o, c) + 0.02
            l = min(o, c) - 0.02
            price = c
        elif i < 59:
            o = price
            c = price + 0.1
            h = max(o, c) + 0.05
            l = min(o, c) - 0.05
            price = c
        else:
            # trigger candle: close well below the 48 SMA
            o = price + 0.5
            c = price - 5.0
            h = max(o, c) + 0.1
            l = min(o, c) - 0.1
            price = c
        one_h.append(_make_bar(base + timedelta(hours=i), o, h, l, c))

    sig = trade_ats_ma_master(
        spot_price=one_h[-1]["close"],
        asset="NQ",
        max_reentries=3,
        daily_bars=daily,
        one_h_bars=one_h,
    )
    assert isinstance(sig, dict), "signal must return a dict"
    assert "triggered" in sig, "signal dict missing triggered"
    assert sig.get("direction") in ("LONG", "SHORT", None)
    assert sig.get("source") == SOURCE
    print("sanity check passed; triggered =", sig["triggered"], "direction =", sig.get("direction"))

    # Verify the happy-path produces a LONG signal.
    assert sig["triggered"] is True, f"expected triggered=True, got {sig}"
    assert sig["direction"] == "LONG", f"expected LONG, got {sig['direction']}"
    assert sig["entry_price"] == one_h[-1]["close"]
    assert sig["sl"] < sig["entry_price"] < sig["tp"]

    # Sanity check no-signal case: remove bars.
    no_sig = trade_ats_ma_master(
        spot_price=100.0,
        asset="NQ",
        daily_bars=[],
        one_h_bars=[],
    )
    assert no_sig["triggered"] is False
    print("no-signal path passed")


# QA_REPORT
# Date: 2026-08-17
# Reviewer: QA agent (Kimi Code)
# Strategy: trade_ats_ma_master (Blueprint 5 / Master Pattern)
# Verdict: PASSED
#
# Checks performed:
#   1. Blueprint fidelity: Implementation matches the documented intent — daily
#      HTF 21 EMA trend + Phase 3 filter, 1H LTF 48 SMA mean-reversion entry
#      against HTF trend, dynamic SL/TP. LONG/SHORT market entry, sl/tp included
#      on triggered signals.
#   2. Lookahead bias: No future bars or future indicator values are used.
#      - EMA/MA/ATR/swing-extreme calculations only reference bars[-N:] and
#        current bar close.
#      - The current daily/1H close is the signal bar’s own close; no later
#        prices, no shifted indexes, no ahead-of-time stochastic/EMA values.
#   3. FUTURES dict contract: triggered, direction, confidence, entry_price,
#      signal_price, source, reason always present; sl and tp present whenever
#      triggered is True. Extra audit fields (htf_bias, ltf_ma, deviation_atr)
#      are harmless and consistent with sibling strategies.
#   4. py_compile: PASS.
#   5. Module sanity check (`python3 -m strategies.signals.trade_ats_ma_master`):
#      PASS — LONG happy-path and empty-bar no-signal path both pass.
#   6. Defensive programming: Returns no_signal (does not crash) for None/empty
#      bar lists, insufficient daily bars (<24), insufficient 1H bars for MA
#      (<48) or ATR (<15), zero/negative ATR, and missing spot_price/asset.
#   7. StateStore usage: Keyed by (asset.upper(), today, variant, max_reentries);
#      prune/tick_cooldowns/save pattern matches reference strategy. Verified
#      asset isolation — NQ and ES maintain independent entry_count/cooldown.
#      Disk persistence means external tests must clear the store directory for
#      deterministic results (the inline __main__ check already does this).
#
# Issues found: None.
# Fixes applied: None.
#
# Remaining concerns / notes:
#   - The strategy keys state by the real-time ET date (tu.get_et_now()). This
#     matches the reference futures strategies, but a backtest harness must
#     either run one calendar day at a time or reset the state store between
#     historical dates to avoid leaking cooldown/entry_count across days.
#   - The structural SL includes the current 1H bar in its lookback window; this
#     is a design choice that can produce a very tight stop when the signal bar
#     prints a long wick. It is not a bug, but worth monitoring in live results.
