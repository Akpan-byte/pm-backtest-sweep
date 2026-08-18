# CHANGE_SUMMARY
# 2026-08-17  integration-engineer + lead
#   - Registered seven new StarTrading strategy signals in
#     StrategyHarness.SIGNALS:
#       ema20_stochastic_pullback, sneaky_pivot, trident_pattern,
#       rhapsody_crt_msnr, trade_ats_ma_master, dumb_money_concepts,
#       brandontrades_supply_demand.
#   - Added per-strategy session pre-gate branches in _step() matching each
#     signal's documented time rules (24/7, NY open first 45m, London Killzone).
#   - Added a dedicated thirty_m_bars rolling window (250 bars) and increased
#     WINDOW_15M to 400 bars so trident_pattern has enough 30m history for its
#     200 EMA filter.
# WHY: Integrate the newly written signal modules into the backtest harness
#      while keeping the original four strategies untouched.
#
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
from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import time_utils as tu
from ..core.state_store import StateStore

log = logging.getLogger("strategies.backtest.engine")

EST = tu.EST
UTC = tu.UTC

# Session-gate boundaries, hoisted so the per-bar pre-gates never hit the slow
# datetime.strptime path (~0.4M calls per 10y portfolio sweep).
T_0300 = time(3, 0)
T_0630 = time(6, 30)
T_0800 = time(8, 0)
T_0830 = time(8, 30)
T_0930 = time(9, 30)
T_1015 = time(10, 15)
T_1000 = time(10, 0)
T_1400 = time(14, 0)
T_1530 = time(15, 30)
T_1545 = time(15, 45)
T_1550 = time(15, 50)
T_1555 = time(15, 55)

# Rolling window sizes (bars) passed to each signal.  Larger than any internal
# slice the signals use (max internal lookback is 30), so detectors always see
# enough history without scanning the whole 10y dataset.
WINDOW_1M = 600
WINDOW_5M = 400
WINDOW_15M = 400
WINDOW_30M = 250
WINDOW_1H = 200
WINDOW_4H = 200
WINDOW_1D = 400

# Confirmation lag (in 5m bars) for a swing point before it is "confirmed".
SWING_LAG_5M = 3
MAX_SWINGS = 30

# Standard args supplied to every signal call.  The futures edition passes the
# market price and asset only; session gating is handled by time_utils.


def _period_ns(period: str) -> int:
    return {"5m": 5 * 60, "15m": 15 * 60, "30m": 30 * 60, "1h": 3600, "4h": 4 * 3600}[period] * 10**9


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
        "ema20_stochastic_pullback": {
            "module": "ema20_stochastic_pullback",
            "func": "ema20_stochastic_pullback",
            "source": "EMA20_STOCHASTIC_PULLBACK",
            "max_reentries": 3,
            "needs": ("daily_bars", "four_h_bars", "one_m_bars"),
        },
        "sneaky_pivot": {
            "module": "sneaky_pivot",
            "func": "sneaky_pivot",
            "source": "SNEAKY_PIVOT",
            "max_reentries": 3,
            "needs": ("daily_bars", "fifteen_m_bars"),
        },
        "trident_pattern": {
            "module": "trident_pattern",
            "func": "trident_pattern",
            "source": "TRIDENT_PATTERN",
            "max_reentries": 1,
            "needs": ("thirty_m_bars", "fifteen_m_bars"),
            "pip_value": True,
        },
        "rhapsody_crt_msnr": {
            "module": "rhapsody_crt_msnr",
            "func": "rhapsody_crt_msnr",
            "source": "RHAPSODY_CRT_MSNR",
            "max_reentries": 3,
            "needs": ("daily_bars", "four_h_bars", "fifteen_m_bars"),
        },
        "trade_ats_ma_master": {
            "module": "trade_ats_ma_master",
            "func": "trade_ats_ma_master",
            "source": "ATS_MA_MASTER",
            "max_reentries": 3,
            "needs": ("daily_bars", "one_h_bars"),
        },
        "dumb_money_concepts": {
            "module": "dumb_money_concepts",
            "func": "dumb_money_concepts",
            "source": "DUMB_MONEY_CONCEPTS",
            "max_reentries": 3,
            "needs": ("daily_bars", "one_h_bars", "fifteen_m_bars"),
        },
        "brandontrades_supply_demand": {
            "module": "brandontrades_supply_demand",
            "func": "brandontrades_supply_demand",
            "source": "BRANDONTRADES_SUPPLY_DEMAND",
            "max_reentries": 3,
            "needs": ("five_m_bars", "fifteen_m_bars", "one_h_bars", "four_h_bars"),
        },
        "orb_vwap": {
            "module": "orb_vwap",
            "func": "orb_vwap",
            "source": "ORB_VWAP",
            "max_reentries": 0,
            "needs": ("one_m_bars",),
        },
        "vwap_sd_reversion": {
            "module": "vwap_sd_reversion",
            "func": "vwap_sd_reversion",
            "source": "VWAP_SD_REVERSION",
            "max_reentries": 0,
            "needs": ("one_m_bars",),
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
        dll: float | None = None,
        risk_pct: float | None = None,
        initial_capital: float = 100_000.0,
        max_drawdown: float | None = None,
        eod_drawdown: float | None = None,
        max_contracts: int | None = None,
        daily_profit_cap: float | None = None,
        trail_at_tp: bool = False,
        trail_distance: float | None = None,
        ledger: dict | None = None,
        session_entry_hour_utc: int = 0,
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
        # Daily loss limit (dollars).  None = disabled.  Enforced at bar level:
        # realized day PnL + open-position floating PnL may never go below -dll.
        # `ledger` (optional) is a shared portfolio ledger dict owned by a
        # PortfolioHarness: {'day', 'day_realized', 'day_halted', 'open_float_others', 'dll'}.
        # When set, all the day-PnL state is read/written through the ledger so a
        # portfolio-wide daily loss bucket is enforced across all instruments.
        self.dll = dll
        self.ledger = ledger
        self._ledger = ledger is not None
        if self._ledger:
            self.ledger["dll"] = self.dll
        self.risk_pct = risk_pct
        self.initial_capital = initial_capital
        # max_drawdown: intra-day trailing drawdown (peak-to-trough in equity).
        # Halt if equity drops this far from peak at any point. For firms like
        # Apex / Earn2Trade that enforce intra-day trailing DD.
        self.max_drawdown = max_drawdown
        # eod_drawdown: end-of-day trailing drawdown.  Halt if the close-of-day
        # equity is this far below the all-time equity peak.  For Topstep-style
        # rules where intra-day excursion is allowed as long as you recover by EOD.
        self.eod_drawdown = eod_drawdown
        self.max_contracts = max_contracts
        self.daily_profit_cap = daily_profit_cap
        self.session_entry_hour_utc = session_entry_hour_utc
        self.trail_at_tp = trail_at_tp
        # trail_distance: how far to trail stop after TP hit (in points).
        # None = use original SL distance as the trail distance.
        self.trail_distance = trail_distance
        self._peak_equity = initial_capital
        self._eod_peak_equity = initial_capital
        self._day_realized = 0.0
        self._day_key: datetime | None = None
        self._day_halted = False
        self._equity = initial_capital

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
            "thirty_m_bars": list(self._thirty_m),
            "one_h_bars": list(self._one_h),
            "four_h_bars": list(self._four_h),
            "daily_bars": list(self._daily),
        }

    def _init_windows(self):
        self._one_m = deque(maxlen=WINDOW_1M)
        self._five_m = deque(maxlen=WINDOW_5M)
        self._fifteen_m = deque(maxlen=WINDOW_15M)
        self._thirty_m = deque(maxlen=WINDOW_30M)
        self._one_h = deque(maxlen=WINDOW_1H)
        self._four_h = deque(maxlen=WINDOW_4H)
        self._daily = deque(maxlen=WINDOW_1D)
        self._bucket = {
            "5m": {"start": None, "o": None, "h": None, "l": None, "c": None, "v": 0},
            "15m": {"start": None, "o": None, "h": None, "l": None, "c": None, "v": 0},
            "30m": {"start": None, "o": None, "h": None, "l": None, "c": None, "v": 0},
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
        elif tf == "30m":
            self._thirty_m.append(d)
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
                       ("30m", _period_ns("30m")), ("1h", _period_ns("1h")),
                       ("4h", _period_ns("4h"))):
            end = _bucket_end_ns(bar_ns, pn)
            self._update_bucket(tf, end - pn, end, bar)
        self._update_bucket("1d", _day_end_ns(bar_ns) - 86400 * 10**9, _day_end_ns(bar_ns), bar)
        self._one_m.append(bar)

    # ----- trade simulation -----
    # Day-state accessors that route through the shared portfolio ledger when a
    # PortfolioHarness owns this harness.  `open_float_others` is the combined
    # floating PnL (dollars) of every OTHER open position in the portfolio, set
    # by the driver before each bar so each instrument's DLL trigger accounts
    # for concurrent positions on the other symbols.
    def _realized_day(self) -> float:
        return self.ledger["day_realized"] if self._ledger else self._day_realized

    def _set_realized_day(self, v: float):
        if self._ledger:
            self.ledger["day_realized"] = v
        else:
            self._day_realized = v

    def _is_halted(self) -> bool:
        return self.ledger["day_halted"] if self._ledger else self._day_halted

    def _set_halted(self, v: bool):
        if self._ledger:
            self.ledger["day_halted"] = v
        else:
            self._day_halted = v

    def _others_float(self) -> float:
        return self.ledger.get("open_float_others", 0.0) if self._ledger else 0.0

    def _dll_price(self, pos: dict) -> float | None:
        """Price at which the open position would push day PnL to exactly -dll."""
        if not self.dll:
            return None
        q = pos["qty"]
        pv = self.point_value
        # Current day PnL = realized (shared if portfolio) + floating of the
        # other open portfolio positions + this position's own floating at the
        # trigger price.  Only this position's term varies with price.
        base = self._realized_day() + self._others_float()
        if pos["direction"] == 1:
            return pos["entry_price"] + (-self.dll - base) / (q * pv)
        return pos["entry_price"] + (self.dll + base) / (q * pv)

    def _try_close(self, bar: dict, et: datetime, last: bool) -> dict | None:
        """Return an exit dict if the open position is stopped/targeted."""
        pos = self._open
        if pos is None:
            return None
        direction = pos["direction"]  # +1 long / -1 short
        low, high = bar["low"], bar["high"]
        sl, tp = pos["sl"], pos["tp"]
        # Daily loss limit: if the bar adverses through the DLL trigger level
        # before reaching the stop, flatten exactly at the limit.  When both
        # are touched in one bar the DLL level is reached first only if it sits
        # on the near side of the stop; otherwise the stop binds first.
        dllp = self._dll_price(pos)
        if direction == 1:
            if dllp is not None and dllp > sl and low <= dllp:
                return {"exit_price": dllp, "reason": "DLL"}
            if low <= sl:
                return {"exit_price": sl, "reason": "SL"}
            if high >= tp:
                if self.trail_at_tp and not pos.get("trailing"):
                    # TP hit: lock to breakeven, start trailing
                    trail_dist = self.trail_distance or abs(tp - pos["entry_price"])
                    pos["sl"] = pos["entry_price"]  # move SL to breakeven
                    pos["tp"] = 1e18 if direction == 1 else -1e18  # remove TP cap
                    pos["trailing"] = True
                    pos["trail_dist"] = trail_dist
                    pos["trail_peak"] = high
                elif pos.get("trailing"):
                    # Update trailing stop
                    pos["trail_peak"] = max(pos.get("trail_peak", high), high)
                    pos["sl"] = pos["trail_peak"] - pos["trail_dist"]
                else:
                    return {"exit_price": tp, "reason": "TP"}
            elif pos.get("trailing"):
                pos["trail_peak"] = max(pos.get("trail_peak", high), high)
                pos["sl"] = pos["trail_peak"] - pos["trail_dist"]
        else:
            if dllp is not None and dllp < sl and high >= dllp:
                return {"exit_price": dllp, "reason": "DLL"}
            if high >= sl:
                return {"exit_price": sl, "reason": "SL"}
            if low <= tp:
                if self.trail_at_tp and not pos.get("trailing"):
                    # TP hit: lock to breakeven, start trailing
                    trail_dist = self.trail_distance or abs(tp - pos["entry_price"])
                    pos["sl"] = pos["entry_price"]  # move SL to breakeven
                    pos["tp"] = -1e18 if direction == 1 else 1e18  # remove TP cap
                    pos["trailing"] = True
                    pos["trail_dist"] = trail_dist
                    pos["trail_peak"] = low
                elif pos.get("trailing"):
                    pos["trail_peak"] = min(pos.get("trail_peak", low), low)
                    pos["sl"] = pos["trail_peak"] + pos["trail_dist"]
                else:
                    return {"exit_price": tp, "reason": "TP"}
            elif pos.get("trailing"):
                pos["trail_peak"] = min(pos.get("trail_peak", low), low)
                pos["sl"] = pos["trail_peak"] + pos["trail_dist"]
        if self.hard_exit_1400 and self.strategy == "fifteen_min_range_scalp" and et.time() >= T_1400:
            return {"exit_price": bar["close"], "reason": "HARD_EXIT_1400"}
        if self.strategy == "orb_vwap" and et.time() >= T_1555:
            return {"exit_price": bar["close"], "reason": "FLATTEN_1555"}
        if self.strategy == "vwap_sd_reversion" and et.time() >= T_1550:
            return {"exit_price": bar["close"], "reason": "FLATTEN_1550"}
        if last:
            return {"exit_price": bar["close"], "reason": "END_OF_DATA"}
        return None

    def _close_position(self, bar: dict, exit_price: float, reason: str):
        pos = self._open
        if pos is None:
            return
        q = pos["qty"]
        pnl_points = (exit_price - pos["entry_price"]) * pos["direction"] * q
        pnl_dollars = pnl_points * self.point_value
        # Hard daily loss limit: never let cumulative day PnL cross -dll.  If a
        # closing trade would push the day below the limit, clamp it so the day
        # lands exactly on -dll (equivalent to the mid-trade DLL cut) and halt.
        day_before = self._realized_day()
        if self.dll and pnl_dollars < 0 and day_before + pnl_dollars < -self.dll:
            pnl_dollars = -self.dll - day_before
            pnl_points = pnl_dollars / self.point_value
            reason = "DLL"
        self._set_realized_day(day_before + pnl_dollars)
        self._equity += pnl_dollars
        # Intra-day trailing drawdown: halt if equity drops max_drawdown from peak.
        if self.max_drawdown is not None and self._equity > self._peak_equity:
            self._peak_equity = self._equity
        if self.max_drawdown is not None and self._peak_equity - self._equity >= self.max_drawdown:
            self._set_halted(True)
        if self.dll and self._realized_day() <= -self.dll:
            self._set_halted(True)
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
            "pnl": round(pnl_points, 6),
            "qty": q,
            "exit_reason": reason,
        })
        if self.cfg.get("trade_history"):
            self._trade_history.append({
                "id": len(self._trade_history),
                "profit_loss": pnl_points,
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
                bar = {
                    "timestamp": datetime.fromtimestamp(ts[i] / 1e9, tz=timezone.utc),
                    "open": o[i], "high": h[i], "low": l[i], "close": c[i], "volume": v[i],
                }
                self._step(ts[i], bar, i, last=(i == n - 1))
            # Force-close anything still open at end of data.
            if self._open is not None:
                self._close_position(self._one_m[-1] if self._one_m else bar, bar["close"], "END_OF_DATA")
        return self.trades

    def _step(self, ts_ns: int, bar: dict, i: int, last: bool):
        """Advance the harness by one bar.

        Extracted from the run() loop so a PortfolioHarness can drive several
        instruments in lockstep against one shared daily loss ledger.
        """
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
        # Re-assert the global time patch for THIS harness.  The patch is a
        # module-level singleton, so when a PortfolioHarness drives several
        # instruments each harness must re-install its own patch before every
        # `_step` or the signal time checks (get_et_now/get_utc_now) would read
        # whichever harness entered last.
        tu.get_et_now = lambda: self.patch.et or datetime.now(EST)
        tu.get_utc_now = lambda: self.patch.utc or datetime.now(UTC)

        call = False
        if self.strategy == "fifteen_min_range_scalp":
            call = T_0830 <= now_et.time() < T_1400
        elif self.strategy == "negative_rr_consolidation_sweeper":
            call = True
        elif self.strategy == "mos_session_daily_draw":
            call = now_utc.hour == self.session_entry_hour_utc and now_utc.minute == 0
        elif self.strategy == "post_8am_bpr_magnet":
            call = now_et.time() >= T_0800
        elif self.strategy == "ema20_stochastic_pullback":
            call = True
        elif self.strategy == "sneaky_pivot":
            call = T_0930 <= now_et.time() <= T_1015
        elif self.strategy == "trident_pattern":
            call = T_0300 <= now_et.time() <= T_0630
        elif self.strategy == "rhapsody_crt_msnr":
            call = True
        elif self.strategy == "trade_ats_ma_master":
            call = True
        elif self.strategy == "dumb_money_concepts":
            call = True
        elif self.strategy == "brandontrades_supply_demand":
            call = True
        elif self.strategy == "orb_vwap":
            call = T_0930 <= now_et.time() <= T_1545
        elif self.strategy == "vwap_sd_reversion":
            call = T_1000 <= now_et.time() <= T_1530

        # Daily loss limit day rollover (UTC trading day).  Reset the
        # realized-day counter and un-halt new entries at midnight UTC.
        # Standalone runs own their day key; portfolio runs share the ledger's.
        if self.dll is not None:
            day_key = bar["timestamp"].date()
            if self._ledger:
                if day_key != self.ledger.get("day"):
                    # EOD drawdown check: halt if equity is below peak - eod_drawdown.
                    if self.eod_drawdown is not None and self._peak_equity - self._equity >= self.eod_drawdown:
                        self._set_halted(True)
                    if self._equity > self._peak_equity:
                        self._peak_equity = self._equity
                    self.ledger["day"] = day_key
                    self.ledger["day_realized"] = 0.0
                    self.ledger["day_halted"] = False
            elif day_key != self._day_key:
                # EOD drawdown check
                if self.eod_drawdown is not None and self._peak_equity - self._equity >= self.eod_drawdown:
                    self._set_halted(True)
                if self._equity > self._peak_equity:
                    self._peak_equity = self._equity
                self._day_key = day_key
                self._day_realized = 0.0
                self._day_halted = False

        # EOD drawdown check when no DLL (standalone EOD DD mode).
        elif self.eod_drawdown is not None:
            day_key = bar["timestamp"].date()
            if day_key != self._day_key:
                if self._peak_equity - self._equity >= self.eod_drawdown:
                    self._set_halted(True)
                if self._equity > self._peak_equity:
                    self._peak_equity = self._equity
                self._day_key = day_key

        # Close existing position (SL/TP/DLL/hard-exit) before evaluating
        # a new entry on the same bar.
        exit_ = self._try_close(bar, now_et, last=last)
        if exit_ is not None:
            self._close_position(bar, exit_["exit_price"], exit_["reason"])

        if call and self._open is None and not self._is_halted():
            self._eval_signal(bar, now_et, now_utc)

    def _eval_signal(self, bar: dict, now_et: datetime, now_utc: datetime):
        wins = self._windows()
        kwargs = dict(
            spot_price=bar["close"],
            asset=self.symbol,
            max_reentries=self.max_reentries,
            point_value=self.point_value,
        )
        for k in self.cfg["needs"]:
            kwargs[k] = wins[k]
        if self.cfg.get("swings"):
            kwargs["swing_highs"] = self._swing_highs
            kwargs["swing_lows"] = self._swing_lows
            kwargs["trade_history"] = self._trade_history
        if self.cfg.get("pip_value"):
            kwargs["pip_value"] = self.pip_value
        if self.strategy == "mos_session_daily_draw":
            kwargs["session_entry_hour_utc"] = self.session_entry_hour_utc

        sig = self.fn(**kwargs)
        if not sig.get("triggered"):
            return
        # Daily profit cap: stop opening new trades once day realized PnL hits cap.
        if self.daily_profit_cap is not None and self._realized_day() >= self.daily_profit_cap:
            return
        direction = 1 if sig["direction"] == "LONG" else -1
        # Position sizing: fixed 1 contract (risk_pct None) or fixed-fractional
        # risk sizing where qty = risk_pct*equity / (stop_distance*point_value).
        qty = 1
        sl = float(sig["sl"])
        if self.risk_pct is not None:
            stop_dist = abs(float(sig["entry_price"]) - sl)
            if stop_dist > 0 and self.point_value > 0:
                qty = max(1, int((self.risk_pct * self._equity) / (stop_dist * self.point_value)))
        if self.max_contracts is not None:
            qty = min(qty, self.max_contracts)
        self._open = {
            "direction": direction,
            "entry_price": sig["entry_price"],
            "sl": sl,
            "tp": sig["tp"],
            "qty": qty,
            "entry_time": bar["timestamp"],
        }
