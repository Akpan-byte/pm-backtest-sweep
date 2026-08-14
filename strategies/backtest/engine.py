# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created strategies/backtest/engine.py: the core bar-by-bar futures
#     backtest driver for the four StarTrading signals.
#     * Loads 1m OHLCV (UTC-aware), filters to in-sample dates.
#     * Maintains rolling completed-bar windows (1m/5m/15m/1h/4h/1d) via
#       deques with no lookahead (higher-TF buckets only finalized once their
#       end boundary has passed).
#     * Monkeypatches strategies.core.time_utils.get_et_now/get_utc_now so the
#       signals' wall-clock session gates replay historical bar time.
#     * Calls the FUTURES editions of the signals (LONG/SHORT, market entry);
#       no Polymarket mapping or 0.85 cap.
#     * Simulates SL/TP exits per bar (SL assumed first if both touched);
#       hard 14:00 ET exit for Blueprint 1; single open position per strategy.
#     * Computes confirmed swing highs/lows for Blueprint 2 and feeds back a
#       trade_history of closed trades for its recovery loop.
#     * Isolates each signal's StateStore into a scratch dir per run.
# WHY: Backtest the 4 strategies on 10y NQ/ES/YM in-sample data with the
#      topstep-strats engine + 20k-sim metrics, then merge tagged trades for
#      instrument combos without re-running signals.
"""Futures backtest driver for the four StarTrading signals."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import time_utils as tu
from ..core.state_store import StateStore

log = logging.getLogger("strategies.backtest.engine")

EST = tu.EST
UTC = tu.UTC

# Rolling window sizes (bars) passed to each signal.  Larger than any internal
# slice the signals use (max internal lookback is 30), so detectors always see
# enough history without scanning the whole 10y dataset.
WINDOW_1M = 600
WINDOW_5M = 400
WINDOW_15M = 200
WINDOW_1H = 200
WINDOW_4H = 200
WINDOW_1D = 400

# Confirmation lag (in 5m bars) for a swing point before it is "confirmed".
SWING_LAG_5M = 3
MAX_SWINGS = 30

# Standard args supplied to every signal call.  The futures edition passes the
# market price and asset only; session gating is handled by time_utils.


def _period_ns(period: str) -> int:
    return {"5m": 5 * 60, "15m": 15 * 60, "1h": 3600, "4h": 4 * 3600}[period] * 10**9


def _bucket_end_ns(ts_ns: int, period_ns: int) -> int:
    return (ts_ns // period_ns) * period_ns + period_ns


def _day_end_ns(ts_ns: int) -> int:
    """UTC-midnight bucket end for a 1d bar (MOS uses UTC calendar days)."""
    day_ns = 86400 * 10**9
    return (ts_ns // day_ns) * day_ns + day_ns


class _TimePatch:
    """Mutable holder so the lambdas we inject into time_utils track bar time."""

    def __init__(self):
        self.utc: datetime | None = None
        self.et: datetime | None = None


def _swing_pivots(completed: deque, window: int) -> tuple[list, list]:
    """Confirmed trailing fractals over the 5m deque.

    A bar at index i (where i+window < len, i.e. already fully confirmed) is a
    swing high if its high strictly exceeds the highs of the ``window`` bars on
    either side; a swing low likewise.  No lookahead: only bars that have
    ``window`` completed bars after them are evaluated.
    """
    n = len(completed)
    if n < 2 * window + 1:
        return [], []
    arr = list(completed)
    highs: list[float] = []
    lows: list[float] = []
    # The most recently confirmable pivot is at n-1-window (needs `window`
    # future bars); older pivots were already caught on earlier ticks, so only
    # inspect this one slot for freshness.
    i = n - 1 - window
    mid = arr[i]
    left = arr[i - window:i]
    right = arr[i + 1:i + window + 1]
    if mid["high"] > max(c["high"] for c in left) and mid["high"] > max(c["high"] for c in right):
        highs.append(mid["high"])
    if mid["low"] < min(c["low"] for c in left) and mid["low"] < min(c["low"] for c in right):
        lows.append(mid["low"])
    return highs, lows


class StrategyHarness:
    """Drives one strategy over one symbol's 1m bars, producing tagged trades.

    The harness does not re-implement any strategy logic; it replays completed
    bar windows through each signal module exactly as the live runtime would,
    then simulates SL/TP exits in price space.
    """

    SIGNALS = {
        "fifteen_min_range_scalp": {
            "module": "fifteen_min_range_scalp",
            "func": "fifteen_min_range_scalp",
            "source": "15M_RANGE_SCALP",
            "max_reentries": 3,
            "needs": ("daily_bars", "four_h_bars", "fifteen_m_bars", "one_m_bars"),
        },
        "negative_rr_consolidation_sweeper": {
            "module": "negative_rr_consolidation_sweeper",
            "func": "negative_rr_consolidation_sweeper",
            "source": "NEG_RR_CONSOLIDATION",
            "max_reentries": 3,
            "needs": ("daily_bars",),
            "swings": True,
            "trade_history": True,
        },
        "mos_session_daily_draw": {
            "module": "mos_session_daily_draw",
            "func": "mos_session_daily_draw",
            "source": "MOS_SESSION_DRAW",
            "max_reentries": 1,
            "needs": ("daily_bars", "four_h_bars", "one_h_bars"),
            "pip_value": True,
        },
        "post_8am_bpr_magnet": {
            "module": "post_8am_bpr_magnet",
            "func": "post_8am_bpr_magnet",
            "source": "POST_8AM_BPR_MAGNET",
            "max_reentries": 3,
            "needs": ("one_m_bars", "five_m_bars", "fifteen_m_bars"),
            "pip_value": True,
        },
    }

    def __init__(
        self,
        strategy: str,
        symbol: str,
        point_value: float,
        pip_value: float = 1.0,
        max_reentries: int | None = None,
        scratch_root: Path | None = None,
        hard_exit_1400: bool = True,
    ):
        if strategy not in self.SIGNALS:
            raise KeyError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.cfg = self.SIGNALS[strategy]
        self.symbol = symbol.upper()
        self.point_value = point_value
        self.pip_value = pip_value
        self.max_reentries = max_reentries or self.cfg["max_reentries"]
        self.hard_exit_1400 = hard_exit_1400

        import importlib

        self.mod = importlib.import_module(f"strategies.signals.{self.cfg['module']}")
        self.fn = getattr(self.mod, self.cfg["func"])
        self.source = self.cfg["source"]

        # Isolate this run's signal state from the live state dir and from any
        # parallel worker (each worker gets its own scratch root).
        root = scratch_root or Path("/tmp/strategies_state")
        self.mod.store = StateStore(self.cfg["module"], root / self.symbol / strategy)

        self.trades: list[dict] = []
        self._open: dict | None = None
        self._trade_history: list[dict] = []
        self._swing_highs: list[float] = []
        self._swing_lows: list[float] = []
        self._five_m_for_swings = deque(maxlen=2 * SWING_LAG_5M + 1)

        self.patch = _TimePatch()
        self._orig_et = tu.get_et_now
        self._orig_utc = tu.get_utc_now

    # ----- time patch lifecycle -----
    def __enter__(self):
        tu.get_et_now = lambda: self.patch.et or datetime.now(EST)
        tu.get_utc_now = lambda: self.patch.utc or datetime.now(UTC)
        return self

    def __exit__(self, *exc):
        tu.get_et_now = self._orig_et
        tu.get_utc_now = self._orig_utc

    # ----- window maintenance -----
    def _windows(self):
        return {
            "one_m_bars": list(self._one_m),
            "five_m_bars": list(self._five_m),
            "fifteen_m_bars": list(self._fifteen_m),
            "one_h_bars": list(self._one_h),
            "four_h_bars": list(self._four_h),
            "daily_bars": list(self._daily),
        }

    def _init_windows(self):
        self._one_m = deque(maxlen=WINDOW_1M)
        self._five_m = deque(maxlen=WINDOW_5M)
        self._fifteen_m = deque(maxlen=WINDOW_15M)
        self._one_h = deque(maxlen=WINDOW_1H)
        self._four_h = deque(maxlen=WINDOW_4H)
        self._daily = deque(maxlen=WINDOW_1D)
        self._bucket = {
            "5m": {"start": None, "o": None, "h": None, "l": None, "c": None, "v": 0},
            "15m": {"start": None, "o": None, "h": None, "l": None, "c": None, "v": 0},
            "1h": {"start": None, "o": None, "h": None, "l": None, "c": None, "v": 0},
            "4h": {"start": None, "o": None, "h": None, "l": None, "c": None, "v": 0},
            "1d": {"start": None, "o": None, "h": None, "l": None, "c": None, "v": 0},
        }

    def _finalize_bucket(self, tf: str, end_ns: int):
        b = self._bucket[tf]
        if b["o"] is None:
            return
        d = {
            "timestamp": datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc),
            "open": b["o"], "high": b["h"], "low": b["l"],
            "close": b["c"], "volume": b["v"],
        }
        if tf == "5m":
            self._five_m.append(d)
            self._five_m_for_swings.append(d)
        elif tf == "15m":
            self._fifteen_m.append(d)
        elif tf == "1h":
            self._one_h.append(d)
        elif tf == "4h":
            self._four_h.append(d)
        else:
            self._daily.append(d)

    def _update_bucket(self, tf: str, start_ns: int, end_ns: int, bar: dict):
        b = self._bucket[tf]
        if b["start"] is None or start_ns != b["start"]:
            self._finalize_bucket(tf, start_ns)
            b["start"] = start_ns
            b["o"] = bar["open"]
            b["h"] = bar["high"]
            b["l"] = bar["low"]
            b["c"] = bar["close"]
            b["v"] = bar["volume"]
        else:
            b["h"] = max(b["h"], bar["high"])
            b["l"] = min(b["l"], bar["low"])
            b["c"] = bar["close"]
            b["v"] += bar["volume"]

    def _advance_windows(self, bar_ns: int, bar: dict):
        for tf, pn in (("5m", _period_ns("5m")), ("15m", _period_ns("15m")),
                       ("1h", _period_ns("1h")), ("4h", _period_ns("4h"))):
            end = _bucket_end_ns(bar_ns, pn)
            self._update_bucket(tf, end - pn, end, bar)
        self._update_bucket("1d", _day_end_ns(bar_ns) - 86400 * 10**9, _day_end_ns(bar_ns), bar)
        self._one_m.append(bar)

    # ----- trade simulation -----
    def _try_close(self, bar: dict, et: datetime, last: bool) -> dict | None:
        """Return an exit dict if the open position is stopped/targeted."""
        pos = self._open
        if pos is None:
            return None
        direction = pos["direction"]  # +1 long / -1 short
        low, high = bar["low"], bar["high"]
        sl, tp = pos["sl"], pos["tp"]
        # Conservative: assume the stop is hit first when both are touched.
        if direction == 1:
            if low <= sl:
                return {"exit_price": sl, "reason": "SL"}
            if high >= tp:
                return {"exit_price": tp, "reason": "TP"}
        else:
            if high >= sl:
                return {"exit_price": sl, "reason": "SL"}
            if low <= tp:
                return {"exit_price": tp, "reason": "TP"}
        if self.hard_exit_1400 and self.strategy == "fifteen_min_range_scalp" and et.time() >= datetime.strptime("14:00", "%H:%M").time():
            return {"exit_price": bar["close"], "reason": "HARD_EXIT_1400"}
        if last:
            return {"exit_price": bar["close"], "reason": "END_OF_DATA"}
        return None

    def _close_position(self, bar: dict, exit_price: float, reason: str):
        pos = self._open
        if pos is None:
            return
        pnl = (exit_price - pos["entry_price"]) * pos["direction"]
        self.trades.append({
            "symbol": self.symbol,
            "strategy": self.strategy,
            "source": self.source,
            "entry_time": pos["entry_time"],
            "direction": pos["direction"],
            "entry_price": round(pos["entry_price"], 6),
            "stop_loss": round(pos["sl"], 6),
            "take_profit": round(pos["tp"], 6),
            "exit_time": bar["timestamp"],
            "exit_price": round(exit_price, 6),
            "pnl": round(pnl, 6),
            "exit_reason": reason,
        })
        if self.cfg.get("trade_history"):
            self._trade_history.append({
                "id": len(self._trade_history),
                "profit_loss": pnl,
            })
            self._trade_history = self._trade_history[-20:]
        self._open = None

    # ----- main loop -----
    def run(self, df: pd.DataFrame) -> list[dict]:
        if df.empty:
            return self.trades
        ts = df.index.values.astype("datetime64[ns]").astype(np.int64)
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        v = df["volume"].to_numpy(float)
        n = len(ts)

        self._init_windows()
        with self:
            for i in range(n):
                ts_ns = int(ts[i])
                bar = {
                    "timestamp": datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc),
                    "open": o[i], "high": h[i], "low": l[i], "close": c[i], "volume": v[i],
                }
                self._advance_windows(ts_ns, bar)

                # Maintain Blueprint 2 swing pivots (confirmed on 5m).
                if self.cfg.get("swings") and len(self._five_m_for_swings) >= 2 * SWING_LAG_5M + 1:
                    hs, ls = _swing_pivots(self._five_m_for_swings, SWING_LAG_5M)
                    if hs:
                        self._swing_highs = (self._swing_highs + hs)[-MAX_SWINGS:]
                    if ls:
                        self._swing_lows = (self._swing_lows + ls)[-MAX_SWINGS:]

                # Pre-gates mirroring the signals' session checks so we do not
                # build per-call kwargs for bars the strategy can never act on.
                now_utc = bar["timestamp"]
                now_et = now_utc.astimezone(EST)
                self.patch.utc = now_utc
                self.patch.et = now_et

                call = False
                if self.strategy == "fifteen_min_range_scalp":
                    call = datetime.strptime("08:30", "%H:%M").time() <= now_et.time() < datetime.strptime("14:00", "%H:%M").time()
                elif self.strategy == "negative_rr_consolidation_sweeper":
                    call = True
                elif self.strategy == "mos_session_daily_draw":
                    call = now_utc.hour == 0 and now_utc.minute == 0
                elif self.strategy == "post_8am_bpr_magnet":
                    call = now_et.time() >= datetime.strptime("08:00", "%H:%M").time()

                # Close existing position (SL/TP/hard-exit) before evaluating
                # a new entry on the same bar.
                exit_ = self._try_close(bar, now_et, last=(i == n - 1))
                if exit_ is not None:
                    self._close_position(bar, exit_["exit_price"], exit_["reason"])

                if call and self._open is None:
                    self._eval_signal(bar, now_et, now_utc)
                else:
                    # Still allow a same-bar exit on the final iteration even
                    # if a signal was evaluated (open stays None then anyway).
                    pass
            # Force-close anything still open at end of data.
            if self._open is not None:
                self._close_position(self._one_m[-1] if self._one_m else bar, bar["close"], "END_OF_DATA")
        return self.trades

    def _eval_signal(self, bar: dict, now_et: datetime, now_utc: datetime):
        wins = self._windows()
        kwargs = dict(
            spot_price=bar["close"],
            asset=self.symbol,
            max_reentries=self.max_reentries,
        )
        for k in self.cfg["needs"]:
            kwargs[k] = wins[k]
        if self.cfg.get("swings"):
            kwargs["swing_highs"] = self._swing_highs
            kwargs["swing_lows"] = self._swing_lows
            kwargs["trade_history"] = self._trade_history
        if self.cfg.get("pip_value"):
            kwargs["pip_value"] = self.pip_value

        sig = self.fn(**kwargs)
        if not sig.get("triggered"):
            return
        direction = 1 if sig["direction"] == "LONG" else -1
        self._open = {
            "direction": direction,
            "entry_price": sig["entry_price"],
            "sl": sig["sl"],
            "tp": sig["tp"],
            "entry_time": bar["timestamp"],
        }
