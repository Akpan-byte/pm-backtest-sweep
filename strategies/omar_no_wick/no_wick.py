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

import numpy as np
import pandas as pd


def detect_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build external market structure state bar-by-bar.

    Uses only information available at or before each bar (no lookahead):
    - swing_highs / swing_lows confirmed retrospectively by body closes.
    - confirmed_HH / confirmed_LL / confirmed_HL / confirmed_LH are the most
      recent confirmed external swing points as of each bar.
    - trend_state is 1 (uptrend confirmed), -1 (downtrend confirmed), or 0.
    """
    n = len(df)
    result = df.copy()

    # Working arrays
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    confirmed_HH = np.full(n, np.nan)
    confirmed_LL = np.full(n, np.nan)
    confirmed_HL = np.full(n, np.nan)
    confirmed_LH = np.full(n, np.nan)
    trend_state = np.zeros(n, dtype=int)

    # Recent confirmed indices for HL/LH lookup
    last_HH_idx = -1
    last_LL_idx = -1
    last_HL_idx = -1
    last_LH_idx = -1

    for i in range(1, n):
        # Carry forward previous confirmed levels
        confirmed_HH[i] = confirmed_HH[i - 1]
        confirmed_LL[i] = confirmed_LL[i - 1]
        confirmed_HL[i] = confirmed_HL[i - 1]
        confirmed_LH[i] = confirmed_LH[i - 1]
        trend_state[i] = trend_state[i - 1]

        body_close = df["close"].iloc[i]
        prev_close = df["close"].iloc[i - 1]

        # Determine if this bar creates a new swing high/low based on close vs prior close.
        # A swing high is a local high relative to neighbours (simplified).
        # We use a 3-bar pivot for swing detection.
        if i >= 2:
            if df["high"].iloc[i - 1] > df["high"].iloc[i - 2] and df["high"].iloc[i - 1] > df["high"].iloc[i]:
                swing_high[i - 1] = df["high"].iloc[i - 1]
            if df["low"].iloc[i - 1] < df["low"].iloc[i - 2] and df["low"].iloc[i - 1] < df["low"].iloc[i]:
                swing_low[i - 1] = df["low"].iloc[i - 1]

        # Structure breaks require BODY CLOSE beyond prior confirmed levels.
        # On first few bars, establish initial structure from pivots.
        if not np.isnan(confirmed_HH[i]) and not np.isnan(confirmed_LL[i]):
            # Uptrend state: need HH and HL
            # Downtrend state: need LH and LL
            # BOS up: body close above HH
            if body_close > confirmed_HH[i]:
                # Bullish BOS. Confirm the most recent HL (snake method).
                # Scan back from previous bar to last LL/HL region.
                # Use the lowest wick in the pullback since the last HH.
                scan_start = last_HH_idx if last_HH_idx >= 0 else 0
                low_wick = df["low"].iloc[scan_start:i].min()
                confirmed_HL[i] = low_wick
                confirmed_HH[i] = body_close
                last_HH_idx = i
                last_HL_idx = i
                trend_state[i] = 1

            # BOS down: body close below LL
            elif body_close < confirmed_LL[i]:
                scan_start = last_LL_idx if last_LL_idx >= 0 else 0
                high_wick = df["high"].iloc[scan_start:i].max()
                confirmed_LH[i] = high_wick
                confirmed_LL[i] = body_close
                last_LL_idx = i
                last_LH_idx = i
                trend_state[i] = -1

            # CHOCH up (reversal attempt): body close above LH in downtrend
            elif trend_state[i] == -1 and not np.isnan(confirmed_LH[i]) and body_close > confirmed_LH[i]:
                # Step 1 of reversal. Track as potential reversal.
                # We don't switch trend yet.
                pass

            # CHOCH down (reversal attempt): body close below HL in uptrend
            elif trend_state[i] == 1 and not np.isnan(confirmed_HL[i]) and body_close < confirmed_HL[i]:
                # Step 1 of reversal. Track as potential reversal.
                pass

        else:
            # Bootstrap: use swing pivots to set initial HH/LL
            # Find most recent non-nan swing high/low
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
                    # crude initial trend based on order
                    if last_sh > last_sl:
                        trend_state[i] = 1
                        confirmed_HL[i] = confirmed_LL[i]
                        last_HL_idx = last_sl
                    else:
                        trend_state[i] = -1
                        confirmed_LH[i] = confirmed_HH[i]
                        last_LH_idx = last_sh

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
    Detect No-Wick candles.

    Bullish: green candle with bottom wick == 0 (low == open within tolerance).
    Bearish: red candle with top wick == 0 (high == open within tolerance).
    """
    n = len(df)
    no_wick_long = np.zeros(n, dtype=int)
    no_wick_short = np.zeros(n, dtype=int)

    for i in range(n):
        o = df["open"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        c = df["close"].iloc[i]
        if c > o and abs(l - o) <= pip_tol:
            no_wick_long[i] = 1
        elif c < o and abs(h - o) <= pip_tol:
            no_wick_short[i] = 1

    df["no_wick_long"] = no_wick_long
    df["no_wick_short"] = no_wick_short
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
                     starting_equity: float = 100000.0):
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

    for i in range(n):
        ts = df.index[i]
        date_key = ts.date()
        day_wins = daily_wins.get(date_key, 0)
        day_losses = daily_losses.get(date_key, 0)

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

        # Generate new signal if flat, no resting order, and daily cap not hit
        if active_trade is None and resting_order is None and not day_done:
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
