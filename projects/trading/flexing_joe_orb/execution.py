"""Execution engine for Flexing Joe ORB signals."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .models import Signal, StrategyConfig, Trade


class FuturesExecutionEngine:
    """Execute a list of :class:`Signal` objects against 1-minute OHLCV data.

    The engine applies adverse slippage on entry, commission per side, and an
    end-of-day close at the configured session end time.  Risk rules such as
    daily loss limits and trailing drawdown are modelled separately by the
    prop-firm simulator so the pure strategy backtest can run the full period.
    """

    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def _adverse_entry_price(self, price: float, direction: int) -> float:
        """Apply adverse slippage to the entry fill."""
        if direction == 1:
            return price + self.cfg.slippage_points
        return price - self.cfg.slippage_points

    def execute_signals(
        self,
        df_1m: pd.DataFrame,
        signals: List[Signal],
    ) -> Tuple[List[Trade], Dict[str, float]]:
        """Run all signals through 1-minute bars and return trades + summary."""
        df = df_1m.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
        df = df.sort_index()

        # Pre-compute numpy arrays once to avoid expensive pandas timestamp
        # operations inside the per-signal loop.
        dtidx_et = df.index
        # Use naive UTC nanoseconds for fast timestamp comparisons.
        dtidx_ns = dtidx_et.tz_convert("UTC").tz_localize(None).astype("datetime64[ns]")
        timestamps = dtidx_ns.to_numpy()
        timestamps_ns = dtidx_ns.astype(np.int64)
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)

        minutes = dtidx_et.hour.to_numpy() * 60 + dtidx_et.minute.to_numpy()
        years = dtidx_et.year.to_numpy()
        months = dtidx_et.month.to_numpy()
        days = dtidx_et.day.to_numpy()
        date_ints = years * 10000 + months * 100 + days

        end_h, end_m = (int(x) for x in self.cfg.session_end_time.split(":"))
        end_minutes = end_h * 60 + end_m

        balance = self.cfg.initial_account_size
        peak_balance = balance
        max_dd_dollars = 0.0

        trades: List[Trade] = []
        daily_pnl: Dict[str, float] = {}
        daily_halted: set = set()

        for sig in signals:
            entry_ts = pd.Timestamp(sig.timestamp)
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_ts = entry_ts.tz_convert("America/New_York")
            date_str = entry_ts.strftime("%Y-%m-%d")
            entry_date_int = (
                entry_ts.year * 10000 + entry_ts.month * 100 + entry_ts.day
            )

            # Daily loss limit halt (cumulative per day).
            if date_str in daily_halted:
                continue
            if daily_pnl.get(date_str, 0.0) <= -self.cfg.daily_loss_limit:
                daily_halted.add(date_str)
                continue

            entry_ns = int(entry_ts.value)
            entry_idx = int(np.searchsorted(timestamps_ns, entry_ns, side="left"))
            if entry_idx >= len(timestamps):
                continue
            # Ensure we are on or after the signal timestamp.
            if timestamps_ns[entry_idx] < entry_ns:
                entry_idx += 1
                if entry_idx >= len(timestamps):
                    continue

            if entry_idx + 1 >= len(timestamps):
                continue

            contracts = sig.contracts or self.cfg.contracts_per_trade
            entry_price = self._adverse_entry_price(sig.entry_price, sig.direction)

            # Fixed slippage and commission costs for the round turn.
            slip_cost = self.cfg.slippage_points * 2.0 * self.cfg.point_value * contracts
            comm_cost = self.cfg.commission_per_contract * 2.0 * contracts

            exit_reason = "EOD_CLOSE"
            exit_idx: int | None = None

            # Scan forward bars strictly after entry.
            for j in range(entry_idx + 1, len(timestamps)):
                if date_ints[j] != entry_date_int:
                    # Next calendar day; close at the previous bar.
                    exit_idx = j - 1
                    break

                if minutes[j] >= end_minutes:
                    # Session-end bar reached.  Close at the last tradeable bar
                    # before it, or at this bar if none exist after entry.
                    if j == entry_idx + 1:
                        exit_idx = j
                    else:
                        exit_idx = j - 1
                    break

                if sig.direction == 1:  # LONG
                    if lows[j] <= sig.stop_price:
                        exit_idx = j
                        exit_reason = "STOP"
                        break
                    if highs[j] >= sig.target_price:
                        exit_idx = j
                        exit_reason = "TARGET"
                        break
                else:  # SHORT
                    if highs[j] >= sig.stop_price:
                        exit_idx = j
                        exit_reason = "STOP"
                        break
                    if lows[j] <= sig.target_price:
                        exit_idx = j
                        exit_reason = "TARGET"
                        break

            if exit_idx is None:
                exit_idx = len(timestamps) - 1

            if exit_reason == "STOP":
                exit_price = sig.stop_price
            elif exit_reason == "TARGET":
                exit_price = sig.target_price
            else:
                exit_price = float(closes[exit_idx])

            exit_ts = pd.Timestamp(timestamps[exit_idx]).tz_localize("UTC").tz_convert("America/New_York")
            entry_ts_out = pd.Timestamp(timestamps[entry_idx]).tz_localize("UTC").tz_convert("America/New_York")

            pnl_pts = (exit_price - entry_price) * sig.direction
            gross_pnl = pnl_pts * self.cfg.point_value * contracts
            net_pnl = gross_pnl - comm_cost - slip_cost

            # Enforce daily loss limit cumulatively.
            projected_daily = daily_pnl.get(date_str, 0.0) + net_pnl
            if projected_daily < -self.cfg.daily_loss_limit:
                allowable = -self.cfg.daily_loss_limit - daily_pnl.get(date_str, 0.0)
                net_pnl = allowable
                gross_pnl = net_pnl + comm_cost + slip_cost
                exit_reason = "DAILY_LIMIT_HALT"
                daily_halted.add(date_str)

            daily_pnl[date_str] = daily_pnl.get(date_str, 0.0) + net_pnl
            balance += net_pnl
            if balance > peak_balance:
                peak_balance = balance

            dd = peak_balance - balance
            if dd > max_dd_dollars:
                max_dd_dollars = dd

            trades.append(
                Trade(
                    entry_time=entry_ts_out,
                    exit_time=exit_ts,
                    direction=sig.direction,
                    entry_price=round(entry_price, 4),
                    exit_price=round(exit_price, 4),
                    contracts=contracts,
                    gross_pnl=round(gross_pnl, 4),
                    commission=round(comm_cost, 4),
                    slippage=round(slip_cost, 4),
                    net_pnl=round(net_pnl, 4),
                    exit_reason=exit_reason,
                )
            )

        summary = {
            "symbol": self.cfg.symbol,
            "point_value": self.cfg.point_value,
            "initial_balance": self.cfg.initial_account_size,
            "final_balance": round(balance, 2),
            "net_pnl": round(balance - self.cfg.initial_account_size, 2),
            "total_trades": len(trades),
            "win_rate": (
                round(sum(1 for t in trades if t.net_pnl > 0) / len(trades) * 100.0, 2)
                if trades
                else 0.0
            ),
            "max_dd_dollars": round(max_dd_dollars, 2),
            "max_dd_pct": round(
                (max_dd_dollars / peak_balance * 100.0) if peak_balance > 0 else 0.0, 2
            ),
            "daily_loss_halted_days": len(daily_halted),
        }
        return trades, summary
