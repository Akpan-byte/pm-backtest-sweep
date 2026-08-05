"""Pure ORB strategy math for YM v5.

CHANGE_SUMMARY
2026-06-30  kilo
  - Extracted pure ORB strategy math from ym_orb_v4.py into TFEngine and
    StrategyProcessor.
  - Preserved exact formulas for opening range, triggers, entries, trailing
    stops, stop checks, EOD exits, PnL, and rearm eligibility.
  - Kept zero broker/SDK imports; execution layer (orders, websockets) lives
    outside this module.
WHY: v5 needs a testable, deterministic core that can be parity-checked
     against v4 while the legacy file remains untouched.

This module contains no broker/SDK I/O and no logging.  It is a direct
extraction of the mathematical rules from ``ym_orb_v4.py``:

* opening-range (OR) calculation
* entry-trigger computation and breakout checks
* stop / target / trailing-stop logic
* end-of-day exit logic
* rearm eligibility logic

All formulas are copied verbatim so that back-tests and parity tests
against v4 produce identical numbers.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np

from .config import TICK_VALUE, TIMEFRAMES
from .models import DayParams, StrategyConfig, TFParams, TFPosition, TradeRecord


class TFEngine:
    """One timeframe's pure ORB state machine.

    Mirrors the strategy logic of ``ym_orb_v4.py::TFEngine`` but strips out
    logging, persistence, async order placement, and WebSocket callbacks.
    """

    def __init__(self, tf_name: str, params: TFParams, config: StrategyConfig) -> None:
        self.tf_name: str = tf_name
        self.params: TFParams = params
        self.config: StrategyConfig = config

        self.max_entries: int = config.max_entries
        self.risk_per_trade: float = config.risk_per_trade
        self.sl_pts: float = config.sl_pts
        self.buffer_pts: float = config.buffer_pts
        self.baseline_index: float = config.baseline_index
        self.tick_value: float = getattr(config, "tick_value", TICK_VALUE)

        # Shared daily scaling parameters; StrategyProcessor mutates this object.
        self.day_params: DayParams = DayParams()

        # PnL accounting
        self.daily_pnl: float = 0.0
        self.cumulative_pnl: float = 0.0
        self.trade_history: list[TradeRecord] = []
        self.entries_taken: int = 0

        # OR state
        self.or_high: float = -np.inf
        self.or_low: float = np.inf
        self.buy_trigger: Optional[float] = None
        self.sell_trigger: Optional[float] = None

        # Position state
        self.position: Optional[TFPosition] = None
        self.current_date: Optional[date] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset_daily(self, date_value: date) -> None:
        """Reset all day-local state."""
        self.current_date = date_value
        self.daily_pnl = 0.0
        self.entries_taken = 0
        self.or_high = -np.inf
        self.or_low = np.inf
        self.buy_trigger = None
        self.sell_trigger = None
        self.position = None

    # ------------------------------------------------------------------
    # Opening range and triggers
    # ------------------------------------------------------------------
    def update_or(self, bar_high: float, bar_low: float, time_min: int) -> None:
        """Update the opening-range high/low while inside the OR window.

        The OR window is ``570 <= time_min < params.or_min`` (09:30-09:?? ET).
        """
        or_end = self.params.or_min
        if 570 <= time_min < or_end:
            if bar_high > self.or_high:
                self.or_high = bar_high
            if bar_low < self.or_low:
                self.or_low = bar_low

    def compute_triggers(self) -> tuple[Optional[float], Optional[float]]:
        """Compute buy/sell breakout triggers from the OR and daily buffer.

        Returns ``(buy_trigger, sell_trigger)`` or ``(None, None)`` if the
        OR has not been established yet.
        """
        S, sl_dist, buf = (
            self.day_params.S,
            self.day_params.sl_dist,
            self.day_params.buf,
        )
        del S, sl_dist  # kept for parity with v4 unpacking
        if not np.isfinite(self.or_high) or not np.isfinite(self.or_low):
            return None, None
        self.buy_trigger = self.or_high + buf
        self.sell_trigger = self.or_low - buf
        return self.buy_trigger, self.sell_trigger

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------
    def expected_entry_qty(self) -> int:
        """Contracts to trade based on risk parity.

        Formula from v4::TFEngine.expected_entry_qty.
        """
        S, sl_dist, buf = (
            self.day_params.S,
            self.day_params.sl_dist,
            self.day_params.buf,
        )
        del S, buf  # kept for parity with v4 unpacking
        risk_dollars = sl_dist * self.tick_value
        if risk_dollars <= 0:
            return 1
        return max(1, round(self.risk_per_trade / risk_dollars))

    def try_entry(
        self,
        bar_high: float,
        bar_low: float,
        timestamp: datetime,
        time_min: int,
    ) -> bool:
        """Check for a paper breakout entry on this bar.

        Returns ``True`` if a position was opened.  This is the pure-math
        equivalent of v4's paper-mode ``try_entry`` path (the live path,
        which places stop orders via an order helper, lives in the execution
        layer, not here).
        """
        if time_min < self.params.or_min or time_min >= 958:
            return False
        if self.entries_taken >= self.max_entries:
            return False
        if self.position is not None:
            return False
        if self.buy_trigger is None:
            self.compute_triggers()
        if self.buy_trigger is None:
            return False

        if bar_high >= self.buy_trigger:
            self._enter("Long", self.buy_trigger, timestamp)
            return True
        if bar_low <= self.sell_trigger:
            self._enter("Short", self.sell_trigger, timestamp)
            return True
        return False

    def _enter(self, direction: str, entry_price: float, timestamp: datetime) -> None:
        """Open a virtual position."""
        S, sl_dist, buf = (
            self.day_params.S,
            self.day_params.sl_dist,
            self.day_params.buf,
        )
        del S, buf  # kept for parity with v4 unpacking
        risk_dollars = sl_dist * self.tick_value
        qty = (
            max(1, round(self.risk_per_trade / risk_dollars))
            if risk_dollars > 0
            else 1
        )
        dir_sign = 1.0 if direction == "Long" else -1.0
        virtual_sl = entry_price - dir_sign * sl_dist
        self.position = TFPosition(
            direction=direction,
            entry_price=entry_price,
            entry_time=timestamp,
            qty=qty,
            virtual_sl=virtual_sl,
        )
        self.entries_taken += 1

    def on_entry_fill(
        self, side: str, fill_price: float, timestamp: datetime
    ) -> None:
        """Record a live entry fill (used by the execution layer).

        ``side`` is ``"buy"`` or ``"sell"`` to match v4's WebSocket payload.
        """
        direction = "Long" if side == "buy" else "Short"
        S, sl_dist, buf = (
            self.day_params.S,
            self.day_params.sl_dist,
            self.day_params.buf,
        )
        del S, buf  # kept for parity with v4 unpacking
        qty = 1  # platform entry was placed at size 1
        dir_sign = 1.0 if direction == "Long" else -1.0
        virtual_sl = fill_price - dir_sign * sl_dist
        self.position = TFPosition(
            direction=direction,
            entry_price=fill_price,
            entry_time=timestamp,
            qty=qty,
            virtual_sl=virtual_sl,
        )

    def _maybe_refund_entry_attempt(self) -> None:
        """Refund an entry slot if a placed entry pair died before any fill.

        Pure-math copy of v4::TFEngine._maybe_refund_entry_attempt.
        """
        if (
            self.position is None
            and self.entries_taken > 0
        ):
            self.entries_taken -= 1

    # ------------------------------------------------------------------
    # Stops, targets, and trailing
    # ------------------------------------------------------------------
    def trail(self, current_price: float) -> Optional[float]:
        """Update virtual_sl based on this TF's trig/sint/lock params.

        Returns the new stop level if it moved, otherwise ``None``.
        """
        if self.position is None or not np.isfinite(current_price):
            return None
        sd = self.day_params.sl_dist
        if sd <= 0 or not np.isfinite(self.position.entry_price):
            return None
        trig = self.params.trig
        sint = self.params.sint
        lock = self.params.lock
        direction_sign = 1.0 if self.position.direction == "Long" else -1.0
        profit_r = direction_sign * (current_price - self.position.entry_price) / sd
        if profit_r > self.position.max_r:
            self.position.max_r = profit_r
        if self.position.max_r >= trig:
            steps = np.floor((self.position.max_r - trig) / sint) + 1
            locked_r = steps * sint * lock
            locked_r = min(locked_r, max(0.0, self.position.max_r - 1e-9))
            new_sl = self.position.entry_price + direction_sign * locked_r * sd
            if self.position.direction == "Long":
                new_sl = max(self.position.virtual_sl, new_sl)
            else:
                new_sl = min(self.position.virtual_sl, new_sl)
            if abs(new_sl - self.position.virtual_sl) >= 1.0:
                self.position.virtual_sl = new_sl
                return new_sl
        return None

    def check_stop(self, bar_high: float, bar_low: float) -> Optional[float]:
        """Return exit price if the virtual stop is hit, else ``None``.

        Uses bar HIGH for short stops and bar LOW for long stops — matching
        v4's sweep logic.
        """
        if self.position is None:
            return None
        if self.position.direction == "Long" and bar_low <= self.position.virtual_sl:
            return self.position.virtual_sl
        if self.position.direction == "Short" and bar_high >= self.position.virtual_sl:
            return self.position.virtual_sl
        return None

    def check_eod(self, bar_close: float, time_min: int) -> Optional[float]:
        """Return exit price at end of day (time_min >= 958), else ``None``."""
        if self.position is None:
            return None
        if time_min >= 958:
            return bar_close
        return None

    def close(
        self, exit_price: float, exit_reason: str, timestamp: datetime
    ) -> float:
        """Close this TF's position and record PnL.

        Returns the net PnL of the trade.
        """
        if self.position is None:
            return 0.0
        pos = self.position
        self.position = None
        dir_sign = 1.0 if pos.direction == "Long" else -1.0
        gross = pos.qty * dir_sign * (exit_price - pos.entry_price) * self.tick_value
        fee_rate = 0.000016
        friction = pos.qty * (pos.entry_price + exit_price) * fee_rate
        net = gross - friction
        self.daily_pnl += net
        self.cumulative_pnl += net
        rec = TradeRecord(
            tf=self.tf_name,
            direction=pos.direction,
            entry_price=round(pos.entry_price, 2),
            exit_price=round(exit_price, 2),
            qty=pos.qty,
            gross=round(gross, 2),
            net=round(net, 2),
            exit_reason=exit_reason,
            entry_time=pos.entry_time.strftime("%H:%M:%S") if pos.entry_time else "",
            exit_time=timestamp.strftime("%H:%M:%S"),
            duration_mins=(
                int((timestamp - pos.entry_time).total_seconds() / 60)
                if pos.entry_time
                else 0
            ),
        )
        self.trade_history.append(rec)
        return net

    # ------------------------------------------------------------------
    # Rearm
    # ------------------------------------------------------------------
    def can_rearm(self) -> bool:
        """Return ``True`` if this TF is eligible to place fresh entry stops."""
        return (
            self.entries_taken < self.max_entries
            and self.buy_trigger is not None
            and self.sell_trigger is not None
            and self.position is None
        )

    def rearm(self, timestamp: datetime, time_min: int) -> bool:
        """Mark this TF as re-armed for fresh entry stops.

        The actual order placement is performed by the execution layer.  This
        pure method only updates the ``entries_taken`` counter so that
        back-test behaviour matches v4's live rearm logic.
        """
        if not self.can_rearm() or time_min >= 958:
            return False
        self.entries_taken += 1
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def virtual_sl(self) -> Optional[float]:
        return self.position.virtual_sl if self.position else None

    @property
    def is_open(self) -> bool:
        return self.position is not None

    @property
    def direction(self) -> Optional[str]:
        return self.position.direction if self.position else None

    @property
    def qty(self) -> int:
        return self.position.qty if self.position else 0


class StrategyProcessor:
    """Pure bar-by-bar processor that drives one ``TFEngine`` per timeframe.

    This is the v5 equivalent of the strategy portion of
    ``ym_orb_v4.py::OrbOrchestrator.process_bar``.  It contains no platform,
    websocket, or order-manager code.
    """

    def __init__(
        self,
        config: StrategyConfig,
        tfs: Optional[dict[str, TFParams]] = None,
    ) -> None:
        self.config: StrategyConfig = config
        self.day_params: DayParams = DayParams()
        self.engines: dict[str, TFEngine] = {}
        for tf_name, raw in (tfs or TIMEFRAMES).items():
            params = TFParams(**raw)
            engine = TFEngine(tf_name, params, config)
            engine.day_params = self.day_params
            self.engines[tf_name] = engine
        self.current_date: Optional[date] = None
        self.last_bar_ts: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset_daily(self, date_value: date) -> None:
        """Reset the processor and every engine for a new session."""
        self.current_date = date_value
        self.day_params = DayParams()
        for eng in self.engines.values():
            eng.day_params = self.day_params
            eng.reset_daily(date_value)

    def set_day_params(self, open_price: float) -> None:
        """Set daily scaling from the session open price."""
        if open_price > 0:
            raw = open_price / float(self.config.baseline_index)
            if raw < 0.7 or raw > 3.0:
                raw = max(0.7, min(3.0, raw))
            self.day_params.S = raw
            self.day_params.sl_dist = self.config.sl_pts * self.day_params.S
            self.day_params.buf = self.config.buffer_pts * self.day_params.S

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def compute_tightest(self) -> tuple[Optional[float], Optional[str]]:
        """Return the tightest virtual stop among all open positions."""
        longs = [
            eng.virtual_sl
            for eng in self.engines.values()
            if eng.is_open and eng.direction == "Long"
        ]
        shorts = [
            eng.virtual_sl
            for eng in self.engines.values()
            if eng.is_open and eng.direction == "Short"
        ]
        if longs and not shorts:
            return max(longs), "Long"
        if shorts and not longs:
            return min(shorts), "Short"
        return None, None

    def net_expected(self) -> tuple[int, Optional[str]]:
        """Return the net expected position size and direction."""
        long_q = sum(
            eng.qty
            for eng in self.engines.values()
            if eng.is_open and eng.direction == "Long"
        )
        short_q = sum(
            eng.qty
            for eng in self.engines.values()
            if eng.is_open and eng.direction == "Short"
        )
        net = long_q - short_q
        if net > 0:
            return net, "Long"
        if net < 0:
            return -net, "Short"
        return 0, None

    # ------------------------------------------------------------------
    # Main bar handler
    # ------------------------------------------------------------------
    def process_bar(
        self,
        ts: datetime,
        o: float,
        h: float,
        l: float,
        c: float,
        time_min: int,
    ) -> dict:
        """Process one bar through the full ORB pipeline.

        Returns a summary dict with keys ``entered``, ``tightest_sl``,
        ``net_expected``, and ``closed_trades`` for the bar.
        """
        if not all(np.isfinite(x) for x in (o, h, l, c)):
            return {
                "entered": False,
                "tightest_sl": (None, None),
                "net_expected": (0, None),
                "closed_trades": [],
            }

        if self.current_date is None:
            self.reset_daily(ts.date())
        if ts.date() != self.current_date:
            self.reset_daily(ts.date())
        if time_min == 570:
            self.set_day_params(o)

        # Update OR for each engine
        for eng in self.engines.values():
            eng.update_or(h, l, time_min)

        # Trail + check stops/EOD
        closed_trades: list[TradeRecord] = []
        for eng in self.engines.values():
            if eng.is_open:
                eng.trail(c)
                sl = eng.check_stop(h, l)
                eod_price = eng.check_eod(c, time_min)
                if sl is not None:
                    exit_price = sl
                    exit_reason = "SL"
                elif eod_price is not None:
                    exit_price = eod_price
                    exit_reason = "EOD"
                else:
                    exit_price = None
                    exit_reason = None
                if exit_price is not None:
                    eng.close(exit_price, exit_reason, ts)
                    closed_trades.extend(eng.trade_history[-1:])

        tightest_sl = self.compute_tightest()

        # Try entries — cap at max_contracts
        entered = False
        open_contracts = sum(eng.qty for eng in self.engines.values() if eng.is_open)
        available = self.config.max_contracts - open_contracts
        for eng in self.engines.values():
            if available <= 0:
                break
            entry_qty = eng.expected_entry_qty()
            if entry_qty > available:
                continue
            did_enter = eng.try_entry(h, l, ts, time_min)
            if did_enter:
                entered = True
                available -= entry_qty

        return {
            "entered": entered,
            "tightest_sl": tightest_sl,
            "net_expected": self.net_expected(),
            "closed_trades": closed_trades,
        }
