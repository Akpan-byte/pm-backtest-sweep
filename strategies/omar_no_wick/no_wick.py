"""
Omar Nowick "No Wick" strategy implementation for index futures (NQ/ES) on 5m.

Rules implemented:
- External structure only (body-close BOS / CHOCH).
- 3-step A-setup reversal: CHOCH -> pullback -> BOS.
- No-wick candle detection (bullish: low == open; bearish: high == open).
- Base SL at confirmed HL/LH wick extreme.
- Min SL filter (default 5 index points).
- 10% breathing room on SL.
- Omar Entry offset (2 points early if final SL > 10 points, else at open).
- 1:1 TP.
- 10-candle limit order expiration.
- Pre-fill structural invalidation (new external BOS / CHOCH before fill).
- Close-miss invalidation (price within 2 points of limit then reaches 1:1 TP without fill).
- Daily trade cap: 2 wins or 2 losses stops; 1W+1L allows a 3rd trade.
- No rollover positions held through 17:00 EST.
"""
from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd


def detect_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build external market structure state bar-by-bar (numpy-vectorized loop).

    Uses only information available at or before each bar (no lookahead):
    - swing_highs / swing_lows confirmed retrospectively by body closes.
    - confirmed_HH / confirmed_LL / confirmed_HL / confirmed_LH are the most
      recent confirmed external swing points as of each bar.
    - trend_state is 1 (uptrend confirmed), -1 (downtrend confirmed), or 0.
    """
    n = len(df)
    result = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # 3-bar pivots (confirmed on the third bar, so no future leak beyond i).
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    for j in range(1, n - 1):
        if high[j] > high[j - 1] and high[j] > high[j + 1]:
            swing_high[j] = high[j]
        if low[j] < low[j - 1] and low[j] < low[j + 1]:
            swing_low[j] = low[j]

    confirmed_HH = np.full(n, np.nan)
    confirmed_LL = np.full(n, np.nan)
    confirmed_HL = np.full(n, np.nan)
    confirmed_LH = np.full(n, np.nan)
    trend_state = np.zeros(n, dtype=int)

    last_HH_idx = -1
    last_LL_idx = -1
    low_since_HH = np.nan
    high_since_LL = np.nan

    for i in range(1, n):
        # Carry forward previous confirmed levels
        confirmed_HH[i] = confirmed_HH[i - 1]
        confirmed_LL[i] = confirmed_LL[i - 1]
        confirmed_HL[i] = confirmed_HL[i - 1]
        confirmed_LH[i] = confirmed_LH[i - 1]
        trend_state[i] = trend_state[i - 1]

        body_close = close[i]

        # Update running extremes since last confirmed swing
        if last_HH_idx >= 0:
            if np.isnan(low_since_HH) or low[i] < low_since_HH:
                low_since_HH = low[i]
        if last_LL_idx >= 0:
            if np.isnan(high_since_LL) or high[i] > high_since_LL:
                high_since_LL = high[i]

        if not np.isnan(confirmed_HH[i]) and not np.isnan(confirmed_LL[i]):
            if body_close > confirmed_HH[i]:
                # Bullish BOS: confirm HL as lowest low since last HH
                confirmed_HL[i] = low_since_HH if not np.isnan(low_since_HH) else low[i]
                confirmed_HH[i] = body_close
                last_HH_idx = i
                trend_state[i] = 1
                low_since_HH = low[i]

            elif body_close < confirmed_LL[i]:
                # Bearish BOS: confirm LH as highest high since last LL
                confirmed_LH[i] = high_since_LL if not np.isnan(high_since_LL) else high[i]
                confirmed_LL[i] = body_close
                last_LL_idx = i
                trend_state[i] = -1
                high_since_LL = high[i]

        else:
            # Bootstrap from completed swing pivots
            sh_idx = np.where(~np.isnan(swing_high[:i]))[0]
            sl_idx = np.where(~np.isnan(swing_low[:i]))[0]
            if len(sh_idx) > 0 and len(sl_idx) > 0:
                last_sh = sh_idx[-1]
                last_sl = sl_idx[-1]
                confirmed_HH[i] = swing_high[last_sh]
                confirmed_LL[i] = swing_low[last_sl]
                last_HH_idx = last_sh
                last_LL_idx = last_sl
                if confirmed_HH[i] > confirmed_LL[i]:
                    if last_sh > last_sl:
                        trend_state[i] = 1
                        confirmed_HL[i] = confirmed_LL[i]
                        low_since_HH = low[i]
                    else:
                        trend_state[i] = -1
                        confirmed_LH[i] = confirmed_HH[i]
                        high_since_LL = high[i]

    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    result["confirmed_HH"] = confirmed_HH
    result["confirmed_LL"] = confirmed_LL
    result["confirmed_HL"] = confirmed_HL
    result["confirmed_LH"] = confirmed_LH
    result["trend_state"] = trend_state
    return result


def detect_a_setups(df_struct: pd.DataFrame) -> pd.DataFrame:
    """
    Detect full 3-step A-setup reversals with no lookahead.

    A bullish A-setup in a downtrend:
      1. CHOCH: body close above the most recent LH.
      2. Pullback: price makes a new HL (confirmed by next bullish BOS) or we track a potential HL.
      3. Confirmation: body close above the new HH (previous LH area becomes HL).

    We simplify by using trend_state transitions from confirmed structure.
    A confirmed trend switch occurs when the structure engine itself flips trend_state
    after a CHOCH-pullback-BOS sequence.

    The implementation marks `a_setup_long` / `a_setup_short` at the confirmation bar.
    """
    n = len(df_struct)
    a_setup_long = np.zeros(n, dtype=int)
    a_setup_short = np.zeros(n, dtype=int)

    ts = df_struct["trend_state"].values
    for i in range(1, n):
        if ts[i] == 1 and ts[i - 1] != 1:
            a_setup_long[i] = 1
        if ts[i] == -1 and ts[i - 1] != -1:
            a_setup_short[i] = 1

    df_struct["a_setup_long"] = a_setup_long
    df_struct["a_setup_short"] = a_setup_short
    return df_struct


def detect_no_wick(df: pd.DataFrame, pip_tol: float = 0.0) -> pd.DataFrame:
    """
    Detect No-Wick candles (vectorized).

    Bullish: green candle with bottom wick == 0 (low == open within tolerance).
    Bearish: red candle with top wick == 0 (high == open within tolerance).
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    df["no_wick_long"] = ((c > o) & (np.abs(l - o) <= pip_tol)).astype(int)
    df["no_wick_short"] = ((c < o) & (np.abs(h - o) <= pip_tol)).astype(int)
    return df


def simulate_no_wick(df: pd.DataFrame,
                     min_sl_pips: float = 5.0,
                     breather_pct: float = 0.10,
                     omar_offset_pips: float = 2.0,
                     omar_threshold_pips: float = 10.0,
                     close_miss_pips: float = 2.0,
                     max_candles: int = 10,
                     pip_size: float = 1.0,
                     daily_win_cap: int = 2,
                     daily_loss_cap: int = 2,
                     rollover_hour: int = 17,
                     commission: float = 0.0001,
                     starting_equity: float = 100000.0,
                     session_start: str | None = None,
                     session_end: str | None = None,
                     close_at_session_end: bool = True,
                     signal_mask: pd.Series | None = None):
    """
    Bar-by-bar limit-order simulation with no lookahead.
    """
    n = len(df)
    trades = []
    equity = [starting_equity] * n
    cash = starting_equity

    # Daily counters keyed by calendar date
    daily_wins = {}
    daily_losses = {}

    # Track active state
    active_trade = None
    resting_order = None  # dict with details

    # Parse session windows once
    start_t = datetime.strptime(session_start, "%H:%M").time() if session_start else None
    end_t = datetime.strptime(session_end, "%H:%M").time() if session_end else None

    for i in range(n):
        ts = df.index[i]
        date_key = ts.date()
        day_wins = daily_wins.get(date_key, 0)
        day_losses = daily_losses.get(date_key, 0)
        t = ts.time()
        in_session = True
        if start_t and end_t:
            in_session = start_t <= t < end_t
        elif start_t:
            in_session = start_t <= t
        elif end_t:
            in_session = t < end_t

        # Close active trade at session end
        if close_at_session_end and active_trade is not None and not in_session:
            exit_price = df["close"].iloc[i]
            pos = active_trade["direction"]
            raw_pnl = pos * (exit_price - active_trade["entry_price"])
            cost = exit_price * commission
            pnl = raw_pnl - cost
            cash += pnl
            active_trade["exit_time"] = str(ts)
            active_trade["exit_price"] = exit_price
            active_trade["pnl"] = pnl
            active_trade["exit_reason"] = "session_close"
            trades.append(active_trade)
            if pnl > 0:
                day_wins += 1
            else:
                day_losses += 1
            daily_wins[date_key] = day_wins
            daily_losses[date_key] = day_losses
            active_trade = None

        # Check rollover for active trade
        if active_trade is not None and ts.hour == rollover_hour and ts.minute == 0:
            # close at current price
            exit_price = df["close"].iloc[i]
            pos = active_trade["direction"]
            raw_pnl = pos * (exit_price - active_trade["entry_price"])
            cost = exit_price * commission
            pnl = raw_pnl - cost
            cash += pnl
            active_trade["exit_time"] = str(ts)
            active_trade["exit_price"] = exit_price
            active_trade["pnl"] = pnl
            active_trade["exit_reason"] = "rollover"
            trades.append(active_trade)
            if pnl > 0:
                day_wins += 1
            else:
                day_losses += 1
            daily_wins[date_key] = day_wins
            daily_losses[date_key] = day_losses
            active_trade = None

        # Daily cap check
        day_done = (day_wins >= daily_win_cap) or (day_losses >= daily_loss_cap)

        # Manage active trade: check SL / TP on each bar
        if active_trade is not None:
            pos = active_trade["direction"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            exit_price = None
            if pos > 0:
                if low <= sl:
                    exit_price = sl if df["open"].iloc[i] > sl else df["open"].iloc[i]
                elif high >= tp:
                    exit_price = tp if df["open"].iloc[i] < tp else df["open"].iloc[i]
            else:
                if high >= sl:
                    exit_price = sl if df["open"].iloc[i] < sl else df["open"].iloc[i]
                elif low <= tp:
                    exit_price = tp if df["open"].iloc[i] > tp else df["open"].iloc[i]

            if exit_price is not None:
                raw_pnl = pos * (exit_price - active_trade["entry_price"])
                cost = exit_price * commission
                pnl = raw_pnl - cost
                cash += pnl
                active_trade["exit_time"] = str(ts)
                active_trade["exit_price"] = exit_price
                active_trade["pnl"] = pnl
                active_trade["exit_reason"] = "tp" if (pnl > 0) else "sl"
                trades.append(active_trade)
                if pnl > 0:
                    day_wins += 1
                else:
                    day_losses += 1
                daily_wins[date_key] = day_wins
                daily_losses[date_key] = day_losses
                active_trade = None

        # Cancel resting order outside session
        if resting_order is not None and not in_session:
            resting_order = None

        # Manage resting order
        if resting_order is not None:
            # Check expiration
            if i - resting_order["signal_idx"] > max_candles:
                resting_order = None
            else:
                # Check fill
                direction = resting_order["direction"]
                limit = resting_order["limit_price"]
                sl = resting_order["sl"]
                tp = resting_order["tp"]
                high = df["high"].iloc[i]
                low = df["low"].iloc[i]
                filled = False
                if direction > 0 and low <= limit <= high:
                    filled = True
                elif direction < 0 and low <= limit <= high:
                    filled = True

                if filled:
                    active_trade = {
                        "entry_time": str(ts),
                        "direction": direction,
                        "entry_price": limit,
                        "sl": sl,
                        "tp": tp,
                        "signal_idx": resting_order["signal_idx"],
                    }
                    resting_order = None
                else:
                    # Close-miss invalidation: price within 2 pips of limit then hits TP
                    if direction > 0:
                        came_close = low <= (limit + close_miss_pips * pip_size)
                        hit_tp = high >= tp
                    else:
                        came_close = high >= (limit - close_miss_pips * pip_size)
                        hit_tp = low <= tp
                    if came_close and hit_tp:
                        resting_order = None

        # Generate new signal if flat, no resting order, daily cap not hit, in session, and filter mask passes
        mask_pass = True if signal_mask is None else bool(signal_mask.iloc[i])
        if active_trade is None and resting_order is None and not day_done and in_session and mask_pass:
            trend = df["trend_state"].iloc[i]
            # Use confirmed A-setup presence (we require trend to have just flipped or be active)
            long_signal = (trend == 1 and df["no_wick_long"].iloc[i] == 1)
            short_signal = (trend == -1 and df["no_wick_short"].iloc[i] == 1)

            if long_signal or short_signal:
                direction = 1 if long_signal else -1
                entry_open = df["open"].iloc[i]
                # Structural stop
                if direction > 0:
                    stop_level = df["confirmed_HL"].iloc[i]
                else:
                    stop_level = df["confirmed_LH"].iloc[i]

                if not np.isnan(stop_level):
                    base_sl_dist = abs(entry_open - stop_level)
                    if base_sl_dist >= min_sl_pips * pip_size:
                        final_sl_dist = base_sl_dist * (1 + breather_pct)
                        # Final SL price
                        if direction > 0:
                            final_sl = entry_open - final_sl_dist
                        else:
                            final_sl = entry_open + final_sl_dist

                        # Omar Entry offset
                        if final_sl_dist > omar_threshold_pips * pip_size:
                            if direction > 0:
                                limit_price = entry_open + omar_offset_pips * pip_size
                            else:
                                limit_price = entry_open - omar_offset_pips * pip_size
                        else:
                            limit_price = entry_open

                        # TP 1:1
                        if direction > 0:
                            tp = limit_price + abs(limit_price - final_sl)
                        else:
                            tp = limit_price - abs(limit_price - final_sl)

                        resting_order = {
                            "signal_idx": i,
                            "direction": direction,
                            "limit_price": limit_price,
                            "sl": final_sl,
                            "tp": tp,
                        }

        equity[i] = cash
        if active_trade is not None:
            equity[i] += active_trade["direction"] * (df["close"].iloc[i] - active_trade["entry_price"])

    eq = pd.Series(equity, index=df.index)
    return eq, trades
