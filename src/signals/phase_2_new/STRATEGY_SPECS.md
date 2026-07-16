# Wave 1 Strategy Specs

All signals are for Polymarket BTC 5m UP/DOWN markets. Use the interface in `INTERFACE.md`.

## prob_convexity_fade
**Idea:** The binary price is a probability, so the same spot move should have a smaller probability impact when the market is already near 0 or 1. Fade moves that look too large relative to the spot change.
**Logic:**
- Compute `prob = yp / (yp + np_val)` if both > 0 else 0.5.
- Compute recent spot return over last N=10 ticks: `ret = (spot_price - spot_history[-10]) / spot_history[-10]`.
- Compute implied delta: `delta = (prob - 0.5) / (spot_price - strike + 1e-9) * strike` (signed distance from strike).
- Compute realized delta over the last M=30 ticks using linear regression of prob vs spot.
- If the recent prob move is > 1.5x the realized delta and prob > 0.65, trigger NO (fade). If prob move is > 1.5x and prob < 0.35, trigger YES.
- Confidence scales with overreaction ratio.

## prob_convexity_trend
**Idea:** When spot keeps trending and the binary probability is lagging (underpriced convexity), buy the direction.
**Logic:**
- Require spot has moved > 0.1% from strike in the last 10 ticks.
- If spot is above strike and prob < 0.6, trigger YES (probability should catch up).
- If spot is below strike and prob > 0.4, trigger NO.
- Confidence scales with how far spot is from strike and how low/high prob is.

## expiry_convergence_90_300
**Idea:** Between 90s and 300s before expiry, if spot is clearly above/below strike, the binary should converge to 1/0. Buy the likely winner if it is still cheap.
**Logic:**
- Only run if `90 <= rem_sec <= 300`.
- Compute `dist = (spot_price - strike) / strike`.
- If `dist > 0.0005` and `yp <= 0.85`, trigger YES.
- If `dist < -0.0005` and `np_val <= 0.85`, trigger NO.
- Confidence scales with absolute distance.

## expiry_pin_guard
**Idea:** Near expiry, avoid trades when spot is hovering near the strike (pin risk / chop).
**Logic:**
- Only run if `rem_sec <= 90`.
- Compute `dist = abs(spot_price - strike) / strike`.
- If `dist < 0.0003`, return triggered=False (guard blocks entry).
- Otherwise, allow underlying logic; here simply return triggered=False always — this is a negative filter to be combined with other signals.
- Note: as a standalone signal it never triggers; it will be used as a gate in a composite.

## spot_shock_overreaction
**Idea:** After a sharp BTC spot move in a short time, Polymarket retail chases and overprices the move. Fade once the spot shock exhausts.
**Logic:**
- Compute 5-tick return: `r5 = (spot_price - spot_history[-5]) / spot_history[-5]` if len >= 5.
- Compute 20-tick volatility (std of returns).
- If `abs(r5) > 2.5 * vol` and the last 2 ticks show deceleration (acceleration opposite sign), fade.
- If r5 > 0 and decelerating, trigger NO at `np_val`.
- If r5 < 0 and decelerating, trigger YES at `yp`.
- Confidence based on shock magnitude.

## topbook_imbalance_rate
**Idea:** Use the rate of change of top-of-book sizes (yes bid size vs no ask size) as a microstructure signal.
**Logic:**
- Track previous `ub` (YES bid size) and `da` (NO ask size) per market in `_STATE`.
- Compute change: `d_ub = ub - prev_ub`, `d_da = da - prev_da`.
- If `d_ub` is strongly positive over last 3 ticks and `yp <= 0.80`, trigger YES.
- If `d_da` is strongly positive over last 3 ticks and `np_val <= 0.80`, trigger NO.
- Confidence scales with magnitude of size change relative to average size.

## time_of_day_bias
**Idea:** Different hours of the day have different directional bias in BTC 5m markets.
**Logic:**
- Parse `start_date_iso` to get ET hour.
- Pre-compute (or hard-code initial guesses) profitable hours: 09:30–11:30 ET and 14:00–16:00 ET are breakout-friendly; 11:30–14:00 and overnight are mean-reversion-friendly.
- If hour is in breakout window and spot has made a new 20-tick high, trigger YES; new 20-tick low → NO.
- If hour is in mean-reversion window and spot has made a new 20-tick high, trigger NO; new low → YES.
- Confidence 0.5 fixed initially; later optimized.

## adaptive_lookback_vol
**Idea:** Adjust the effective lookback for breakout/mean-reversion based on realized volatility.
**Logic:**
- Compute realized vol over last 20 ticks: `vol = stdev of returns`.
- If vol is in top 25% of recent history, use short lookback ( breakout-friendly) and trigger on 5-tick high/low breakout.
- If vol is in bottom 25%, use long lookback and fade extremes (z-score > 1.5).
- Direction: high vol → follow 5-tick momentum; low vol → fade z-score.

## vol_compression_breakout
**Idea:** When spot volatility compresses then expands, price tends to break out.
**Logic:**
- Compute ATR over last 10 ticks and last 30 ticks.
- If 10-tick ATR < 30-tick ATR * 0.6 for at least 5 ticks, then a sudden expansion (current 3-tick range > 1.5x 10-tick ATR) triggers.
- Direction is the direction of the expansion (close near high → YES, near low → NO).

## kalman_signal_filter
**Idea:** Use a simple Kalman filter to extract the slow trend from noisy spot price. Only take directional trades when spot is clearly above/below the filtered trend.
**Logic:**
- Maintain a simple exponential-Kalman estimate per market: `state = alpha * spot_price + (1-alpha) * prev_state`, with `alpha = 0.15`.
- If spot_price > state * 1.0002 and `yp <= 0.80`, trigger YES.
- If spot_price < state * 0.9998 and `np_val <= 0.80`, trigger NO.
- Confidence based on deviation size.

## noise_signal_decomp
**Idea:** Decompose spot into fast noise and slow signal using two EMAs. Only trade when fast noise has moved back toward the slow signal (i.e., a pullback in trend).
**Logic:**
- Fast EMA (alpha=0.4) and slow EMA (alpha=0.1).
- If slow EMA is rising and fast EMA just crossed back above slow EMA, trigger YES.
- If slow EMA is falling and fast EMA just crossed back below slow EMA, trigger NO.

## kronos_5m_filter
**Idea:** Use pre-computed Kronos-mini 5m close forecasts to confirm or block the trade direction.
**Logic:**
- At module import, try to load `/tmp/kronos_forecasts_5m.parquet` indexed by `market_id` and forecast timestamp.
- If current market_id has a forecast for the current snapshot time, compare predicted close to current spot.
- If predicted close > spot * 1.0001 and `yp <= 0.80`, trigger YES.
- If predicted close < spot * 0.9999 and `np_val <= 0.80`, trigger NO.
- Confidence based on predicted move magnitude.
- If forecast file missing, return triggered=False.

## kronos_agreement_sizing
**Idea:** Same as kronos_5m_filter but returns the forecast direction/confidence to be used by a meta sizer, not as a standalone entry.
**Logic:**
- Load `/tmp/kronos_forecasts_5m.parquet`.
- Return triggered=False always, but include `direction` and `confidence` for a composite signal to read.
- This is a sizing helper; standalone it does not trade.
