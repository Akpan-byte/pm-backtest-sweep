#!/usr/bin/env python3
"""
JJ Simon NQ Fair-Price Strategy — Modular Engine

A clean, self-contained implementation of the Fair-Price state machine.
This module is intentionally dependency-light (numpy + pandas) so it can be
imported by backtest runners, live-bot adapters, and research notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Session timing in minute-of-day (US Eastern)
# -----------------------------------------------------------------------------
SCHEDULED_NEWS_START = 510   # 08:30
SCHEDULED_NEWS_END = 540     # 09:00
OPENING_CONT_START = 570     # 09:30
OPENING_CONT_END = 580       # 09:40
MEAN_REVERSION_START = 580   # 09:40
HARD_CUTOFF = 660            # 11:00
PM_START = 840               # 14:00
PM_REVERSION_START = 850     # 14:10
PM_END = 900                 # 15:00

ANCHOR_829 = (8, 29)
ANCHOR_929 = (9, 29)
ANCHOR_1359 = (13, 59)


@dataclass
class FairPriceConfig:
    """Runtime configuration for one strategy instance."""
    profile: str
    starting_balance: float
    sl_pts: float
    tp_pts: float
    risk_pct: float
    bos_lookback: int
    mean_reversion_distance: float
    news_spike_threshold: float
    dynamic_candle_trigger: float
    dynamic_sl_pts: float
    dynamic_tp_pts: float
    dynamic_size_reduction: float
    enable_pm_session: bool
    point_value: float
    tick_size: float
    max_morning_trades: int = 3
    max_consecutive_losses: int = 2


@dataclass
class Trade:
    """One completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    qty: float
    gross: float
    fee: float
    net: float
    exit_reason: str
    balance_after: float


class FairPriceAnchor:
    """Immutable fair-price reference zone built from a single candle body."""

    def __init__(self, open_price: float, close_price: float):
        self.high = max(open_price, close_price)
        self.low = min(open_price, close_price)
        self.mid = (self.high + self.low) / 2.0

    def distance(self, price: float) -> float:
        return price - self.mid

    def __repr__(self) -> str:
        return f"FairPriceAnchor({self.low:.2f} - {self.high:.2f}, mid={self.mid:.2f})"


class JJSimonEngine:
    """
    Bar-by-bar state machine for the JJ Simon Fair-Price strategy.

    Usage:
        config = FairPriceConfig(...)
        engine = JJSimonEngine(config)
        for bar in bars:
            engine.on_bar(bar)
        trades = engine.closed_trades
    """

    def __init__(self, config: FairPriceConfig):
        self.cfg = config
        self.closed_trades: list[Trade] = []

        # Daily reset state
        self._balance = config.starting_balance
        self._news_anchor: Optional[FairPriceAnchor] = None
        self._ny_anchor: Optional[FairPriceAnchor] = None
        self._pm_anchor: Optional[FairPriceAnchor] = None
        self._reset_anchor: Optional[FairPriceAnchor] = None
        self._news_event_pending = False
        self._consolidation: list[tuple[float, float, float, float]] = []

        # Position state
        self._in_position = False
        self._direction = 0  # 1 long, -1 short
        self._entry_price = 0.0
        self._entry_time: Optional[pd.Timestamp] = None
        self._sl = 0.0
        self._tp = 0.0
        self._qty = 0.0

        # Session counters
        self._morning_trades = 0
        self._consecutive_losses = 0

        # Trigger buffers
        self._bos_highs: np.ndarray = np.zeros(config.bos_lookback, dtype=np.float64)
        self._bos_lows: np.ndarray = np.zeros(config.bos_lookback, dtype=np.float64)
        self._bos_count = 0
        self._bos_idx = 0
        self._prev_body = 0.0

        # Current date for daily reset
        self._current_date: Optional[object] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def on_bar(self, bar: pd.Series) -> Optional[Trade]:
        """
        Process one 1-minute bar (convenience wrapper for interactive use).

        Bar fields required:
            timestamp, open, high, low, close, minute_of_day, date,
            is_829, is_929, is_1359
        """
        return self._on_bar_scalar(
            timestamp=bar["timestamp"],
            open_price=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
            minute_of_day=int(bar["minute_of_day"]),
            date=bar["date"],
            is_829=bool(bar["is_829"]),
            is_929=bool(bar["is_929"]),
            is_1359=bool(bar["is_1359"]),
        )

    def _on_bar_scalar(
        self,
        timestamp: pd.Timestamp,
        open_price: float,
        high: float,
        low: float,
        close: float,
        minute_of_day: int,
        date,
        is_829: bool,
        is_929: bool,
        is_1359: bool,
    ) -> Optional[Trade]:
        """Fast scalar path used by both on_bar and run_backtest."""
        # Daily reset
        if date != self._current_date:
            self._current_date = date
            self._news_anchor = None
            self._ny_anchor = None
            self._pm_anchor = None
            self._reset_anchor = None
            self._news_event_pending = False
            self._consolidation = []
            self._morning_trades = 0
            self._consecutive_losses = 0
            self._close_position()
            self._bos_count = 0
            self._bos_idx = 0
            self._prev_body = 0.0

        # Anchors
        if is_829:
            self._news_anchor = FairPriceAnchor(open_price, close)
        if is_929:
            self._ny_anchor = FairPriceAnchor(open_price, close)
        if is_1359:
            self._pm_anchor = FairPriceAnchor(open_price, close)

        # News spike detection
        if not self._in_position:
            phase = self._phase(minute_of_day)
            if phase not in ("news_phase", "opening_continuation", "pre_market"):
                candle_range = high - low
                if candle_range > self.cfg.news_spike_threshold and not self._news_event_pending:
                    self._news_event_pending = True
                    self._consolidation = []

        # Consolidation anchor
        if self._news_event_pending:
            self._consolidation.append((open_price, high, low, close))
            if len(self._consolidation) >= 3:
                avg_high = float(np.mean([b[1] for b in self._consolidation[:3]]))
                avg_low = float(np.mean([b[2] for b in self._consolidation[:3]]))
                self._reset_anchor = FairPriceAnchor(avg_low, avg_high)
                self._news_event_pending = False
                self._consolidation = []

        # Exit check
        trade = self._check_exit_scalar(timestamp, high, low, close, minute_of_day)
        if trade is not None:
            self.closed_trades.append(trade)
            return trade

        # Entry check
        self._check_entry_scalar(timestamp, open_price, high, low, close, minute_of_day)

        # Update BOS buffer
        if self._bos_count < self.cfg.bos_lookback:
            self._bos_highs[self._bos_count] = high
            self._bos_lows[self._bos_count] = low
            self._bos_count += 1
        else:
            self._bos_highs[self._bos_idx] = high
            self._bos_lows[self._bos_idx] = low
            self._bos_idx = (self._bos_idx + 1) % self.cfg.bos_lookback

        self._prev_body = abs(close - open_price)
        return None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _maybe_reset_day(self, bar: pd.Series) -> None:
        d = bar["date"]
        if d != self._current_date:
            self._current_date = d
            self._news_anchor = None
            self._ny_anchor = None
            self._pm_anchor = None
            self._reset_anchor = None
            self._news_event_pending = False
            self._consolidation = []
            self._morning_trades = 0
            self._consecutive_losses = 0
            self._close_position()
            self._bos_count = 0
            self._bos_idx = 0
            self._prev_body = 0.0

    def _detect_anchors(self, bar: pd.Series) -> None:
        o, c = bar["open"], bar["close"]
        if bar["is_829"]:
            self._news_anchor = FairPriceAnchor(o, c)
        if bar["is_929"]:
            self._ny_anchor = FairPriceAnchor(o, c)
        if bar["is_1359"]:
            self._pm_anchor = FairPriceAnchor(o, c)

    def _phase(self, mod: int) -> str:
        cfg = self.cfg
        if mod < SCHEDULED_NEWS_START:
            return "pre_market"
        if SCHEDULED_NEWS_START <= mod < SCHEDULED_NEWS_END:
            return "news_phase"
        if OPENING_CONT_START <= mod < OPENING_CONT_END:
            return "opening_continuation"
        if MEAN_REVERSION_START <= mod < HARD_CUTOFF:
            return "mean_reversion"
        if cfg.enable_pm_session and PM_START <= mod < PM_END:
            return "afternoon"
        if mod >= HARD_CUTOFF:
            return "hard_cutoff"
        return "post_session"

    def _active_anchor(self, phase: str) -> Optional[FairPriceAnchor]:
        if self._reset_anchor is not None:
            return self._reset_anchor
        if phase == "news_phase":
            return self._news_anchor
        if phase in ("opening_continuation", "mean_reversion"):
            return self._ny_anchor
        if phase == "afternoon":
            return self._pm_anchor
        return None

    def _detect_news_spike(self, bar: pd.Series) -> None:
        if self._in_position:
            return
        phase = self._phase(bar["minute_of_day"])
        if phase in ("news_phase", "opening_continuation", "pre_market"):
            return
        candle_range = bar["high"] - bar["low"]
        if candle_range > self.cfg.news_spike_threshold and not self._news_event_pending:
            self._news_event_pending = True
            self._consolidation = []

    def _build_consolidation_anchor(self, bar: pd.Series) -> None:
        if not self._news_event_pending:
            return
        self._consolidation.append((bar["open"], bar["high"], bar["low"], bar["close"]))
        if len(self._consolidation) >= 3:
            avg_high = float(np.mean([b[1] for b in self._consolidation[:3]]))
            avg_low = float(np.mean([b[2] for b in self._consolidation[:3]]))
            self._reset_anchor = FairPriceAnchor(avg_low, avg_high)
            self._news_event_pending = False
            self._consolidation = []

    def _check_exit(self, bar: pd.Series) -> Optional[Trade]:
        if not self._in_position:
            return None

        exit_price: Optional[float] = None
        exit_reason: Optional[str] = None
        h, l, c = bar["high"], bar["low"], bar["close"]

        if self._direction == 1:
            if l <= self._sl:
                exit_price = self._sl
                exit_reason = "stop_loss"
            elif self._tp > 0 and h >= self._tp:
                exit_price = self._tp
                exit_reason = "take_profit"
        else:
            if h >= self._sl:
                exit_price = self._sl
                exit_reason = "stop_loss"
            elif self._tp > 0 and l <= self._tp:
                exit_price = self._tp
                exit_reason = "take_profit"

        # Hard cutoff: flatten at 11:00 AM EST (or PM end if in afternoon)
        if exit_price is None and bar["minute_of_day"] >= HARD_CUTOFF:
            exit_price = c
            exit_reason = "hard_cutoff"

        # Break-even move at 09:30 open
        if exit_price is None and bar["minute_of_day"] == OPENING_CONT_START and self._entry_price > 0:
            tick = self.cfg.tick_size
            if self._direction == 1:
                new_sl = self._entry_price + tick
                if new_sl > self._sl:
                    self._sl = new_sl
            else:
                new_sl = self._entry_price - tick
                if new_sl < self._sl:
                    self._sl = new_sl

        if exit_price is None:
            return None

        gross_pts = (exit_price - self._entry_price) if self._direction == 1 else (self._entry_price - exit_price)
        gross = gross_pts * self.cfg.point_value * self._qty
        fee = abs(gross) * 0.0  # fees applied externally if needed
        net = gross - fee
        self._balance += net

        trade = Trade(
            entry_time=self._entry_time,
            exit_time=bar["timestamp"],
            direction="long" if self._direction == 1 else "short",
            entry_price=round(float(self._entry_price), 4),
            exit_price=round(float(exit_price), 4),
            qty=round(float(self._qty), 4),
            gross=round(float(gross), 4),
            fee=round(float(fee), 4),
            net=round(float(net), 4),
            exit_reason=exit_reason,
            balance_after=round(float(self._balance), 4),
        )

        if net < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._close_position()
        return trade

    def _check_exit_scalar(
        self,
        timestamp: pd.Timestamp,
        high: float,
        low: float,
        close: float,
        minute_of_day: int,
    ) -> Optional[Trade]:
        if not self._in_position:
            return None

        exit_price: Optional[float] = None
        exit_reason: Optional[str] = None

        if self._direction == 1:
            if low <= self._sl:
                exit_price = self._sl
                exit_reason = "stop_loss"
            elif self._tp > 0 and high >= self._tp:
                exit_price = self._tp
                exit_reason = "take_profit"
        else:
            if high >= self._sl:
                exit_price = self._sl
                exit_reason = "stop_loss"
            elif self._tp > 0 and low <= self._tp:
                exit_price = self._tp
                exit_reason = "take_profit"

        if exit_price is None and minute_of_day >= HARD_CUTOFF:
            exit_price = close
            exit_reason = "hard_cutoff"

        if exit_price is None and minute_of_day == OPENING_CONT_START and self._entry_price > 0:
            tick = self.cfg.tick_size
            if self._direction == 1:
                new_sl = self._entry_price + tick
                if new_sl > self._sl:
                    self._sl = new_sl
            else:
                new_sl = self._entry_price - tick
                if new_sl < self._sl:
                    self._sl = new_sl

        if exit_price is None:
            return None

        gross_pts = (exit_price - self._entry_price) if self._direction == 1 else (self._entry_price - exit_price)
        gross = gross_pts * self.cfg.point_value * self._qty
        fee = abs(gross) * 0.0
        net = gross - fee
        self._balance += net

        trade = Trade(
            entry_time=self._entry_time,
            exit_time=timestamp,
            direction="long" if self._direction == 1 else "short",
            entry_price=round(float(self._entry_price), 4),
            exit_price=round(float(exit_price), 4),
            qty=round(float(self._qty), 4),
            gross=round(float(gross), 4),
            fee=round(float(fee), 4),
            net=round(float(net), 4),
            exit_reason=exit_reason,
            balance_after=round(float(self._balance), 4),
        )

        if net < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._close_position()
        return trade

    def _check_entry_scalar(
        self,
        timestamp: pd.Timestamp,
        open_price: float,
        high: float,
        low: float,
        close: float,
        minute_of_day: int,
    ) -> None:
        if self._in_position:
            return

        phase = self._phase(minute_of_day)
        if phase not in ("news_phase", "opening_continuation", "mean_reversion", "afternoon"):
            return
        if self._morning_trades >= self.cfg.max_morning_trades:
            return
        if self._consecutive_losses >= self.cfg.max_consecutive_losses:
            return

        anchor = self._active_anchor(phase)
        if anchor is None:
            return

        body = abs(close - open_price)
        is_displacement = False
        disp_bullish = False
        if self._prev_body > 0 and body > self._prev_body:
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low
            if body > 0 and upper_wick < body * 0.3 and lower_wick < body * 0.3:
                is_displacement = True
                disp_bullish = close > open_price

        is_bos = False
        bos_bullish = False
        if self._bos_count >= self.cfg.bos_lookback:
            highest = float(np.max(self._bos_highs[: self._bos_count]))
            lowest = float(np.min(self._bos_lows[: self._bos_count]))
            if close > highest:
                is_bos = True
                bos_bullish = True
            elif close < lowest:
                is_bos = True
                bos_bullish = False

        if not (is_displacement or is_bos):
            return

        distance = anchor.distance(close)
        direction = 0
        signal = None

        if phase == "news_phase":
            direction, signal = self._mean_reversion_signal(distance, is_bos, bos_bullish, is_displacement, disp_bullish)
        elif phase == "opening_continuation":
            direction, signal = self._continuation_signal(is_bos, bos_bullish, is_displacement, disp_bullish)
        elif phase == "mean_reversion":
            direction, signal = self._mean_reversion_signal(distance, is_bos, bos_bullish, is_displacement, disp_bullish)
        elif phase == "afternoon":
            if minute_of_day < PM_REVERSION_START:
                direction, signal = self._continuation_signal(is_bos, bos_bullish, is_displacement, disp_bullish)
            else:
                direction, signal = self._mean_reversion_signal(distance, is_bos, bos_bullish, is_displacement, disp_bullish)

        if direction == 0 or signal is None:
            return

        self._enter_scalar(timestamp, close, open_price, direction, signal)

    def _enter_scalar(self, timestamp: pd.Timestamp, close: float, open_price: float, direction: int, signal: str) -> None:
        body = abs(close - open_price)

        sl_pts = self.cfg.sl_pts
        tp_pts = self.cfg.tp_pts
        if body > self.cfg.dynamic_candle_trigger:
            sl_pts = max(sl_pts, self.cfg.dynamic_sl_pts)
            tp_pts = max(tp_pts, self.cfg.dynamic_tp_pts)

        risk_dollars = self._balance * self.cfg.risk_pct
        risk_per_contract = sl_pts * self.cfg.point_value
        qty = max(1, round(risk_dollars / risk_per_contract)) if risk_per_contract > 0 else 1

        if body > self.cfg.dynamic_candle_trigger:
            qty = max(1, round(qty * self.cfg.dynamic_size_reduction))

        if direction == 1:
            sl = close - sl_pts
            tp = close + tp_pts
        else:
            sl = close + sl_pts
            tp = close - tp_pts

        self._in_position = True
        self._direction = direction
        self._entry_price = close
        self._entry_time = timestamp
        self._sl = sl
        self._tp = tp
        self._qty = qty
        self._morning_trades += 1

    def _mean_reversion_signal(
        self,
        distance: float,
        is_bos: bool,
        bos_bullish: bool,
        is_displacement: bool,
        disp_bullish: bool,
    ) -> tuple[int, Optional[str]]:
        if abs(distance) < self.cfg.mean_reversion_distance:
            return 0, None
        if distance > 0 and is_bos and not bos_bullish:
            return -1, "mean_reversion"
        if distance < 0 and is_bos and bos_bullish:
            return 1, "mean_reversion"
        if distance > 0 and is_displacement and not disp_bullish:
            return -1, "mean_reversion"
        if distance < 0 and is_displacement and disp_bullish:
            return 1, "mean_reversion"
        return 0, None

    def _continuation_signal(
        self,
        is_bos: bool,
        bos_bullish: bool,
        is_displacement: bool,
        disp_bullish: bool,
    ) -> tuple[int, Optional[str]]:
        if is_displacement:
            return (1 if disp_bullish else -1), "opening_continuation"
        if is_bos:
            return (1 if bos_bullish else -1), "opening_continuation"
        return 0, None

    def _enter(self, bar: pd.Series, direction: int, signal: str) -> None:
        c = bar["close"]
        body = abs(bar["close"] - bar["open"])

        sl_pts = self.cfg.sl_pts
        tp_pts = self.cfg.tp_pts
        if body > self.cfg.dynamic_candle_trigger:
            sl_pts = max(sl_pts, self.cfg.dynamic_sl_pts)
            tp_pts = max(tp_pts, self.cfg.dynamic_tp_pts)

        risk_dollars = self._balance * self.cfg.risk_pct
        risk_per_contract = sl_pts * self.cfg.point_value
        qty = max(1, round(risk_dollars / risk_per_contract)) if risk_per_contract > 0 else 1

        # Dynamic size reduction on large trigger candles
        if body > self.cfg.dynamic_candle_trigger:
            qty = max(1, round(qty * self.cfg.dynamic_size_reduction))

        if direction == 1:
            sl = c - sl_pts
            tp = c + tp_pts
        else:
            sl = c + sl_pts
            tp = c - tp_pts

        self._in_position = True
        self._direction = direction
        self._entry_price = c
        self._entry_time = bar["timestamp"]
        self._sl = sl
        self._tp = tp
        self._qty = qty
        self._morning_trades += 1

    def _is_displacement(self, bar: pd.Series) -> tuple[bool, bool]:
        body = abs(bar["close"] - bar["open"])
        if self._prev_body <= 0 or body <= self._prev_body:
            return False, False
        upper_wick = bar["high"] - max(bar["open"], bar["close"])
        lower_wick = min(bar["open"], bar["close"]) - bar["low"]
        if body > 0 and upper_wick < body * 0.3 and lower_wick < body * 0.3:
            return True, bar["close"] > bar["open"]
        return False, False

    def _is_bos(self, bar: pd.Series) -> tuple[bool, bool]:
        if self._bos_count < self.cfg.bos_lookback:
            return False, False
        highest = float(np.max(self._bos_highs[: self._bos_count]))
        lowest = float(np.min(self._bos_lows[: self._bos_count]))
        if bar["close"] > highest:
            return True, True
        if bar["close"] < lowest:
            return True, False
        return False, False

    def _update_bos_buffer(self, bar: pd.Series) -> None:
        if self._bos_count < self.cfg.bos_lookback:
            self._bos_highs[self._bos_count] = bar["high"]
            self._bos_lows[self._bos_count] = bar["low"]
            self._bos_count += 1
        else:
            self._bos_highs[self._bos_idx] = bar["high"]
            self._bos_lows[self._bos_idx] = bar["low"]
            self._bos_idx = (self._bos_idx + 1) % self.cfg.bos_lookback

    def _close_position(self) -> None:
        self._in_position = False
        self._direction = 0
        self._entry_price = 0.0
        self._entry_time = None
        self._sl = 0.0
        self._tp = 0.0
        self._qty = 0.0


# -----------------------------------------------------------------------------
# Convenience runners
# -----------------------------------------------------------------------------
def load_data(symbol: str, data_dir: str = "/config/fvg_execution_engine/backtests/data") -> pd.DataFrame:
    """Load 1-minute futures CSV and add derived columns used by the engine."""
    from pathlib import Path

    path = Path(data_dir) / symbol / "M1.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing data: {path}")

    df = pd.read_csv(path)
    if "ts" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["is_829"] = (df["hour"] == ANCHOR_829[0]) & (df["minute"] == ANCHOR_829[1])
    df["is_929"] = (df["hour"] == ANCHOR_929[0]) & (df["minute"] == ANCHOR_929[1])
    df["is_1359"] = (df["hour"] == ANCHOR_1359[0]) & (df["minute"] == ANCHOR_1359[1])
    return df


def run_backtest(df: pd.DataFrame, config: FairPriceConfig) -> tuple[list[Trade], float]:
    """Run the full bar-by-bar backtest and return (trades, final_balance)."""
    engine = JJSimonEngine(config)

    # Fast numpy path for batch backtests
    timestamps = df["timestamp"].values
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    minutes = df["minute_of_day"].values.astype(np.int32)
    dates = df["date"].values
    is_829 = df["is_829"].values
    is_929 = df["is_929"].values
    is_1359 = df["is_1359"].values

    for i in range(len(df)):
        engine._on_bar_scalar(
            timestamp=pd.Timestamp(timestamps[i]),
            open_price=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            minute_of_day=int(minutes[i]),
            date=dates[i],
            is_829=bool(is_829[i]),
            is_929=bool(is_929[i]),
            is_1359=bool(is_1359[i]),
        )

    return engine.closed_trades, engine._balance


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert a list of Trade objects to a DataFrame."""
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        rows.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "qty": t.qty,
            "gross": t.gross,
            "fee": t.fee,
            "net": t.net,
            "exit_reason": t.exit_reason,
            "balance_after": t.balance_after,
        })
    return pd.DataFrame(rows)
