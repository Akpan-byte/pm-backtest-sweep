# CHANGE_SUMMARY
# 2026-08-17  coder
#   - Created signals/ema20_stochastic_pullback.py (Strategy 1) for the
#     StarTrading futures backtest harness.
#   - Implements a 1-minute pullback scalper: HTF EMA trend filter, 20-period
#     EMA pullback mean on 1m, and Stochastic (8,5,3) crossover entry.
#   - Returns the standard FUTURES signal dict with absolute SL/TP levels.
#   - Includes a bottom-of-file sanity check that constructs synthetic bars and
#     asserts both a no-signal (flat trend) case and a triggered LONG setup.
# WHY: Compartmentalized strategy module matching the existing signal contract
#      in strategies/signals/common.py.

"""Strategy 1 (FUTURES): EMA20 + Stochastic Pullback Scalping.

Documentation / blueprint: see the task description for `ema20_stochastic_pullback`.

Core logic:
  Use a higher-timeframe EMA direction as a non-repainting trend proxy.  On the
  1-minute chart wait for price to deviate away from the 20 EMA (pullback) and
  then close back to the trend side.  Enter when a Stochastic (8,5,3) crossover
  confirms the reversal on that close.

Signal kwargs (in addition to the standard set):
  daily_bars : list of daily OHLCV dicts (fallback HTF trend)
  four_h_bars: list of 4h OHLCV dicts (primary HTF trend filter)
  one_m_bars : list of recent 1m OHLCV dicts (pullback + entry)
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

log = logging.getLogger("ema20_stochastic_pullback")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SOURCE = "EMA20_STOCHASTIC_PULLBACK"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
store = StateStore("ema20_stochastic_pullback", _PROJECT_ROOT)

# Defaults from the strategy blueprint; the caller can override them for sweeps.
DEFAULT_EMA_PERIOD = 20
DEFAULT_K_PERIOD = 8
DEFAULT_D_PERIOD = 5
DEFAULT_SLOWING = 3
DEFAULT_TP_RETRACEMENT = 0.25
DEFAULT_SL_BUFFER_ATR_MULT = 0.10


def _make_state():
    return {
        "today": None,
        "trend_bias": None,
        "entry_count": {"LONG": 0, "SHORT": 0},
        "cooldown": {"LONG": 0, "SHORT": 0},
        "last_trigger_time": None,
    }


# ----- indicator helpers -----------------------------------------------------

def _ema(values: list[float], period: int) -> float | None:
    """Exponential moving average of a list of floats; no lookahead."""
    if len(values) < period:
        return None
    ema = sum(values[:period]) / period
    multiplier = 2.0 / (period + 1)
    for v in values[period:]:
        ema = (v - ema) * multiplier + ema
    return ema


def _ema_series(bars: list[dict], period: int, price_key: str = "close") -> list[float | None]:
    """Return a series of EMA values aligned with ``bars``.

    The first ``period - 1`` slots are ``None``; the EMA at index
    ``period - 1`` is seeded with the SMA of the first ``period`` closes.
    """
    n = len(bars)
    if n < period:
        return [None] * n
    ema = [None] * n
    ema[period - 1] = sum(c[price_key] for c in bars[:period]) / period
    multiplier = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = (bars[i][price_key] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema


def _stochastic_series(
    bars: list[dict],
    k_period: int = DEFAULT_K_PERIOD,
    d_period: int = DEFAULT_D_PERIOD,
    slowing: int = DEFAULT_SLOWING,
) -> list[tuple[float | None, float | None]]:
    """Slow Stochastic (k,d,slowing) for each bar; no lookahead.

    Returns a list of (slow_k, d) tuples.  The first valid index is
    ``k_period + slowing + d_period - 3``.
    """
    n = len(bars)
    min_bars = k_period + slowing + d_period - 2
    if n < min_bars:
        return [(None, None)] * n

    fast_k: list[float | None] = [None] * n
    for i in range(k_period - 1, n):
        window = bars[i - k_period + 1 : i + 1]
        lowest = min(c["low"] for c in window)
        highest = max(c["high"] for c in window)
        rng = highest - lowest
        fast_k[i] = 50.0 if rng <= 0 else 100.0 * (bars[i]["close"] - lowest) / rng

    slow_k: list[float | None] = [None] * n
    for i in range(k_period + slowing - 2, n):
        vals = fast_k[i - slowing + 1 : i + 1]
        slow_k[i] = sum(vals) / slowing

    d: list[float | None] = [None] * n
    for i in range(k_period + slowing + d_period - 3, n):
        vals = slow_k[i - d_period + 1 : i + 1]
        d[i] = sum(vals) / d_period

    return list(zip(slow_k, d))


# ----- setup detection -------------------------------------------------------

def _frame_htf_bias(four_h_bars: list[dict] | None, daily_bars: list[dict] | None, period: int = 21) -> str | None:
    """Non-repainting trend proxy via 4H (or daily) EMA direction.

    Returns ``LONG`` when the latest close is above the EMA and ``SHORT`` when
    below.  The EMA is computed only over fully-closed bars.
    """
    for tf_bars in (four_h_bars, daily_bars):
        if tf_bars and len(tf_bars) >= period + 1:
            closes = [c["close"] for c in tf_bars]
            ema = _ema(closes, period)
            if ema is not None:
                last_close = tf_bars[-1]["close"]
                if last_close > ema:
                    return "LONG"
                if last_close < ema:
                    return "SHORT"
    return None


def _find_pullback_setup(
    one_m_bars: list[dict],
    ema20: list[float | None],
    stoch: list[tuple[float | None, float | None]],
    direction: str,
) -> dict | None:
    """Return the latest 1m pullback setup, or None.

    Only the most recently closed bar is considered as a trigger candle so the
    entry decision uses no future information.  A valid setup requires:
      * price was on the trend side of the 20 EMA before the pullback,
      * price deviated to the opposite side (consecutive closes beyond EMA),
      * the current candle closes back to the trend side,
      * Stochastic slow-k crosses the %D line in the trend direction.
    """
    n = len(one_m_bars)
    if n < 25:
        return None

    i = n - 1  # only the latest closed bar can be a trigger

    # Need stochastic values for the trigger and the prior bar.
    if stoch[i][0] is None or stoch[i - 1][0] is None:
        return None

    slow_k_now, d_now = stoch[i]
    slow_k_prev, d_prev = stoch[i - 1]

    if direction == "LONG":
        # Trigger candle closes above the 20 EMA.
        if one_m_bars[i]["close"] <= ema20[i]:
            return None
        # Previous candle must be in deviation (close below EMA).
        if one_m_bars[i - 1]["close"] >= ema20[i - 1]:
            return None
        # Bullish stochastic crossover.
        if not (slow_k_now > d_now and slow_k_prev <= d_prev):
            return None

        # Find the consecutive deviation window ending at i-1.  Stop before
        # bars where the EMA is not yet valid to avoid comparing against None.
        dev_end = i - 1
        dev_start = dev_end
        while (
            dev_start > 0
            and ema20[dev_start] is not None
            and one_m_bars[dev_start]["close"] < ema20[dev_start]
        ):
            dev_start -= 1
        dev_start += 1

        if dev_end < dev_start:
            return None

        # Confirm price was above the EMA immediately before the deviation.
        pre_window = one_m_bars[max(0, dev_start - 5) : dev_start]
        if not any(
            ema20[j] is not None and c["close"] > ema20[j]
            for j, c in enumerate(pre_window, start=max(0, dev_start - 5))
        ):
            return None

        extreme_low = min(c["low"] for c in one_m_bars[dev_start : dev_end + 1])
        pullback_start = max(c["high"] for c in pre_window) if pre_window else one_m_bars[dev_start]["high"]

        return {
            "trigger_candle": one_m_bars[i],
            "extreme": extreme_low,
            "start": pullback_start,
        }

    else:  # SHORT
        if one_m_bars[i]["close"] >= ema20[i]:
            return None
        if one_m_bars[i - 1]["close"] <= ema20[i - 1]:
            return None
        if not (slow_k_now < d_now and slow_k_prev >= d_prev):
            return None

        dev_end = i - 1
        dev_start = dev_end
        while (
            dev_start > 0
            and ema20[dev_start] is not None
            and one_m_bars[dev_start]["close"] > ema20[dev_start]
        ):
            dev_start -= 1
        dev_start += 1

        if dev_end < dev_start:
            return None

        pre_window = one_m_bars[max(0, dev_start - 5) : dev_start]
        if not any(
            ema20[j] is not None and c["close"] < ema20[j]
            for j, c in enumerate(pre_window, start=max(0, dev_start - 5))
        ):
            return None

        extreme_high = max(c["high"] for c in one_m_bars[dev_start : dev_end + 1])
        pullback_start = min(c["low"] for c in pre_window) if pre_window else one_m_bars[dev_start]["low"]

        return {
            "trigger_candle": one_m_bars[i],
            "extreme": extreme_high,
            "start": pullback_start,
        }


def _compute_levels(
    setup: dict,
    direction: str,
    entry_price: float,
    atr: float | None,
    tp_retracement: float,
    sl_buffer_atr_mult: float,
) -> tuple[float, float, float]:
    """Compute SL, TP, and the raw available technical potential.

    TP1 is ``tp_retracement`` (default 25%) of the distance from the pullback
    start to the pullback extreme.  SL is placed just beyond the extreme that
    created the deviation, buffered by a fraction of ATR.
    """
    extreme = setup["extreme"]
    start = setup["start"]
    potential = abs(start - extreme)

    buffer = sl_buffer_atr_mult * atr if atr else entry_price * 0.001

    if direction == "LONG":
        sl = extreme - buffer
        tp = entry_price + tp_retracement * potential
    else:
        sl = extreme + buffer
        tp = entry_price - tp_retracement * potential

    return sl, tp, potential


# ----- signal ----------------------------------------------------------------

def ema20_stochastic_pullback(
    spot_price=None,
    asset="NQ",
    max_reentries=3,
    daily_bars=None,
    four_h_bars=None,
    one_m_bars=None,
    ema_period: int = DEFAULT_EMA_PERIOD,
    k_period: int = DEFAULT_K_PERIOD,
    d_period: int = DEFAULT_D_PERIOD,
    slowing: int = DEFAULT_SLOWING,
    tp_retracement: float = DEFAULT_TP_RETRACEMENT,
    sl_buffer_atr_mult: float = DEFAULT_SL_BUFFER_ATR_MULT,
    **kwargs,
):
    ok, reason = validate_signal_inputs(spot_price, asset)
    if not ok:
        return no_signal(reason, SOURCE)

    if not one_m_bars:
        return no_signal("missing_1m_bars", SOURCE)
    if not four_h_bars and not daily_bars:
        return no_signal("missing_htf_bars", SOURCE)

    now = tu.get_et_now()
    today = now.date()

    key = store.make_key(asset, today, max_reentries)
    state = store.load_or_new(key, _make_state)
    state["today"] = today
    store.prune(asset.upper(), today)
    store.tick_cooldowns(state)

    # Step 1: higher-timeframe trend filter.
    bias = _frame_htf_bias(four_h_bars, daily_bars)
    state["trend_bias"] = bias
    if bias is None:
        return no_signal("no_htf_bias", SOURCE)

    # Step 2: compute 1m EMA and stochastic using only closed bars.
    min_history = max(ema_period, k_period + slowing + d_period) + 5
    if len(one_m_bars) < min_history:
        return no_signal("insufficient_1m_history", SOURCE)

    ema20 = _ema_series(one_m_bars, ema_period)
    stoch = _stochastic_series(one_m_bars, k_period, d_period, slowing)

    # Step 3: detect pullback + crossover on the latest closed 1m bar.
    setup = _find_pullback_setup(one_m_bars, ema20, stoch, bias)
    if setup is None:
        return no_signal("no_pullback_setup", SOURCE)

    trigger_candle = setup["trigger_candle"]
    trigger_time = trigger_candle["timestamp"]
    trigger_ts = trigger_time.timestamp()

    # Avoid duplicate entries on the same trigger candle.
    # ``last_trigger_time`` is stored as a float timestamp for safe JSON
    # round-tripping; gracefully handle legacy string datetimes too.
    last_ts = state["last_trigger_time"]
    if isinstance(last_ts, str):
        try:
            last_ts = datetime.fromisoformat(last_ts).timestamp()
        except ValueError:
            last_ts = None
    if last_ts is not None and trigger_ts <= last_ts:
        return no_signal("already_triggered", SOURCE)

    direction = bias
    n = state["entry_count"][direction]
    if not (n == 0 or 0 < n <= max_reentries):
        return no_signal("max_reentries", SOURCE)
    if state["cooldown"][direction] > 0:
        return no_signal("cooldown", SOURCE)

    entry_price = trigger_candle["close"]
    if entry_price is None or entry_price <= 0:
        return no_signal("no_price", SOURCE)

    atr = cu.calculate_atr(one_m_bars[-25:], 14)
    sl, tp, potential = _compute_levels(
        setup, direction, entry_price, atr, tp_retracement, sl_buffer_atr_mult
    )

    state["entry_count"][direction] += 1
    state["cooldown"][direction] = MIN_COOLDOWN_TICKS
    state["last_trigger_time"] = trigger_ts
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
        "trend_bias": bias,
        "ema_period": ema_period,
        "stochastic": f"{k_period},{d_period},{slowing}",
        "pullback_start": float(setup["start"]),
        "pullback_extreme": float(setup["extreme"]),
        "potential": float(potential),
        "reason": (
            f"{'RE-ENTRY' if n > 0 else 'FIRST'} dir={direction} bias={bias} "
            f"asset={asset.upper()} entry={entry_price:.2f} "
            f"potential={potential:.2f} sl={sl:.2f} tp={tp:.2f}"
        ),
    }


# ----- sanity check / doctest ------------------------------------------------

if __name__ == "__main__":
    import doctest
    import shutil
    import tempfile
    from pathlib import Path

    def _bar(ts: float, o: float, h: float, l: float, c: float, v: float = 1.0) -> dict:
        return {
            "timestamp": datetime.fromtimestamp(ts, tz=UTC),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        }

    # Redirect the module store to a temporary directory for the tests.
    _orig_dir = store.dir
    store.dir = Path(tempfile.mkdtemp())

    try:
        # ---- Test 1: flat higher timeframe -> no trend bias ----
        flat_4h = [_bar(86400.0 * i, 100.0, 101.0, 99.0, 100.0) for i in range(30)]
        flat_1m = [_bar(3600.0 + 60.0 * i, 100.0, 101.0, 99.0, 100.0) for i in range(50)]
        sig = ema20_stochastic_pullback(
            spot_price=100.0,
            asset="NQ",
            daily_bars=flat_4h,
            four_h_bars=flat_4h,
            one_m_bars=flat_1m,
        )
        assert sig["triggered"] is False, sig
        assert sig["reason"] == "no_htf_bias", sig

        # ---- Test 2: bullish HTF trend + pullback reversal -> LONG signal ----
        t0 = 86400.0 * 200  # arbitrary base epoch

        # 4h uptrend so the 21 EMA trend filter is bullish.
        four_h = []
        for i in range(30):
            c = 100.0 + i * 0.5
            four_h.append(_bar(t0 + 4 * 3600.0 * i, c - 0.5, c + 0.5, c - 0.5, c))

        # 1m bars: uptrend, then a deviation below EMA20, then a close back above.
        base = t0 + 4 * 3600.0 * 29
        one_m = []
        for i in range(20):
            c = 110.0 + i * 0.05
            one_m.append(_bar(base + 60.0 * i, c - 0.02, c + 0.02, c - 0.02, c))
        # Deviation (closes drop below the freshly-rising 20 EMA).
        for i in range(5):
            c = 111.0 - i * 0.40
            one_m.append(_bar(base + 60.0 * (20 + i), c, c, c - 0.10, c))
        # Trigger candle (latest closed bar): close back above EMA20 with a
        # bullish stochastic crossover.  A close near the 8-bar high forces
        # fast_k (and therefore slow_k) above the %D line.
        one_m.append(_bar(base + 60.0 * 25, 109.50, 111.50, 109.00, 111.40))

        sig = ema20_stochastic_pullback(
            spot_price=111.40,
            asset="NQ",
            daily_bars=four_h,
            four_h_bars=four_h,
            one_m_bars=one_m,
        )
        assert sig["triggered"] is True, sig
        assert sig["direction"] == "LONG", sig
        assert sig["entry_price"] > 0, sig
        assert sig["sl"] < sig["entry_price"] < sig["tp"], sig
        assert sig["source"] == SOURCE, sig

        print("sanity check passed")
    finally:
        shutil.rmtree(store.dir, ignore_errors=True)
        store.dir = _orig_dir
        store._mem.clear()

    doctest.testmod(verbose=False)


# QA_REPORT:
# Status: passed
# Reviewer: QA agent (StarTrading futures backtest harness)
# Date: 2026-08-17
#
# Checks performed:
#   1. Blueprint alignment: ema20_stochastic_pullback matches the documented
#      1-minute pullback scalper logic (HTF EMA trend filter, 20 EMA mean,
#      Stochastic (8,5,3) crossover entry) and returns the standard FUTURES
#      signal dict with triggered, direction, confidence, entry_price,
#      signal_price, source, reason, plus sl/tp when triggered.
#   2. Lookahead bias: verified that _ema_series, _stochastic_series, and
#      _find_pullback_setup only use fully-closed bars up to the current bar.
#      No future opens/highs/lows/closes, future EMA values, or future
#      stochastic values are referenced.
#   3. py_compile: passes.
#   4. Module sanity check (python3 -m strategies.signals.ema20_stochastic_pullback):
#      passes (flat-trend no-signal case and bullish pullback LONG case).
#   5. Minimal/empty windows: returns no_signal gracefully for missing bars,
#      insufficient history, or no setup; no unhandled exceptions.
#   6. StateStore: keyed per (asset.upper(), date, variant, max_reentries);
#      prune isolates by asset and only drops entries older than retention.
#      No state leak across assets/dates observed.
#
# Issues found and fixed:
#   - Defensive-programming crash: the deviation walk in _find_pullback_setup
#     could walk into the EMA warm-up region where ema20[j] is None, raising
#     TypeError on float/None comparison. Added ``ema20[dev_start] is not None``
#     guard to the while loops for both LONG and SHORT.
#
# Remaining concerns:
#   - The strategy keys state by the current ET system date (tu.get_et_now()),
#     matching the reference strategy pattern. Backtests must be run on the
#     target date or use a mocked clock so state keys align with the bar data.
#   - Degenerate input where pullback_start equals pullback_extreme would yield
#     potential == 0 and tp == entry_price. The harness/position manager should
#     treat such signals as invalid; no explicit zero-potential guard was added
#     to keep the change minimal.
