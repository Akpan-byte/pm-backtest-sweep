# CHANGE_SUMMARY
# 2026-08-12  opencode
#   - Aligned port to the Drive 10-year reference (topstep_150k_paper_config.json):
#     CONTRACTS=2, MAX_ENTRIES=3, volatility filter MIN_ORB_MULT=0.7
#     (rolling median of prior 20 session-days, skip when ORB range < 0.7x median).
#   - Added counter breakeven lock (stop->entry at 1x ORB target, ride to session
#     end) matching BOTH reference simulate_counter_trades and live kernel adapter.
#   - Switched counter arming to reference semantics (first main trade net_pnl <= 0).
#   - Matched reference cost economics: main = contracts-scaled gross + costs;
#     counter = 1-contract gross with flat slip (SLIPPAGE_PTS*PV/TICK) + commission.
#   - Added `prop` mode: runs 7 session combos x {reg, -900} at c2/me3/filter0.7,
#     then runs v4 prop-farm sim for 150K standard (daily cap 3000) + consistency
#     (daily cap 2500) per topstep_150k_paper_config.
# 2026-08-12  opencode
#   - Parameterized MAX_ENTRIES via sys.argv[4] (prop mode) and suffixed output
#     filenames per cap (_me{N}) so me3/me8/me12 sweeps coexist.
# 2026-08-12  opencode
#   - Added parallel-worker slicing for GitHub Actions: optional sys.argv[5]
#     (comma-separated combos) and sys.argv[6] ("reg"/"lim"/"both").
#     Each of 20 GHA workers runs a disjoint subset of the 42 scenarios.
# 2026-08-12  opencode
#   - Made the prop-farm import portable for GHA: try flexing_joe_orb/scripts
#     (checked in), then v5_orb_nq_backtest, instead of the hardcoded /config
#     path. Copied prop_farm_simulator_v4.py into scripts/.
# WHY: User asked to parallelize the 10-year session matrix (7 combos x
#      {reg,lim} x {me3,me8,me12}) across 20 GHA workers instead of one
#      slow sequential local run.
"""Multi-session Flexing Joe ORB backtest on real Topstep API NQ 1-min data.

Faithful synchronous port of the live kernel adapter
(execution-kernel/src/execution_kernel/strategy/flexing_joe_orb.py):
  - per-session ORB capture, bias, 10m-breakout + 2m-EMA20 pullback entries
  - reentries capped at MAX_ENTRIES per session-day (default unlimited)
  - per-session counter-strategy (arms after the session's first losing main trade)
  - counter breakeven lock: at 1x ORB target move stop -> entry, ride to session end
    (matches both reference 10-year simulate_counter_trades and live kernel adapter)
  - optional volatility filter (skip session-days whose ORB range is < MIN_ORB_MULT
    x rolling median of prior VOL_FILTER_LOOKBACK session-days)
  - optional -$900/session-day entry halt

Session windows (standard CME macros, user-confirmed):
  - Asian : ORB 19:00-19:30 ET, entries 19:40 -> 03:00 ET
  - London: ORB 03:00-03:30 ET, entries 03:40 -> 09:30 ET
  - NY    : ORB 09:30-10:00 ET, entries 10:10 -> 16:00 ET

Each session is an independent strategy instance over the session-day window
(18:00 ET -> next 18:00 ET). The daily loss limit, when enabled, is a shared
per-session-day entry halt (open positions still exit at stop/target/flatten).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from flexing_joe_orb.data import load_ohlcv_csv

SESSION_DAY_START_MIN = 18 * 60  # 18:00 ET rollover
POINT_VALUE_NQ = 20.0
TICK_SIZE_NQ = 0.25
COMMISSION = 2.50          # per contract per side
SLIPPAGE_PTS = 0.25        # adverse entry fill + round-turn cost
TARGET_MULTIPLE = 2.0
EMA_PERIOD = 20
MAX_ENTRIES = 999          # reentries variant (unlimited; reference uses 3)
CONTRACTS = 1              # contracts per trade (reference 10-year uses 2)
# Volatility filter: skip a session-day when its ORB range is below this
# multiple of the rolling median of the prior VOL_FILTER_LOOKBACK ranges.
MIN_ORB_MULT: Optional[float] = None
VOL_FILTER_LOOKBACK = 20


@dataclass
class SessionSpec:
    name: str
    orb_start_mss: int   # minutes since 18:00 ET
    orb_end_mss: int
    session_end_mss: int  # flatten time (minutes since 18:00 ET)


SESSIONS = {
    "asian": SessionSpec("asian", 60, 90, 540),       # 19:00-19:30 -> 03:00
    "london": SessionSpec("london", 540, 570, 930),   # 03:00-03:30 -> 09:30
    "ny": SessionSpec("ny", 930, 960, 1200),          # 09:30-10:00 -> 16:00
}

SESSION_COMBOS = {
    "ny_only": ["ny"],
    "london_only": ["london"],
    "asian_only": ["asian"],
    "all_three": ["asian", "london", "ny"],
    "ny_asian": ["asian", "ny"],
    "ny_london": ["london", "ny"],
    "london_asian": ["asian", "london"],
}

MSS_PERIOD = 24 * 60


def _mss(time_min: int) -> int:
    """Minutes since 18:00 ET (session-day start)."""
    return (time_min - SESSION_DAY_START_MIN) % MSS_PERIOD


@dataclass
class _TradeRec:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    session: str
    kind: str              # "main" or "counter"
    direction: int
    entry_price: float
    exit_price: float
    contracts: int
    gross_pnl: float
    net_pnl: float
    exit_reason: str


class _SessionState:
    """One independent FlexingJoeOrbAdapter instance for one session-day."""

    def __init__(self, spec: SessionSpec):
        self.spec = spec
        self.orb_high: Optional[float] = None
        self.orb_low: Optional[float] = None
        self.bias_allow_long = True
        self.bias_allow_short = True
        self.entries_today = 0
        self.long_taken = False
        self.short_taken = False
        # buffers
        self.m2_bars: List[dict] = []
        self.m10_bars: List[dict] = []
        self._m2_partial: Optional[dict] = None
        self._m10_partial: Optional[dict] = None
        # pending signal bookkeeping
        self.pending_breakouts: List[dict] = []
        self.pending_entry: Optional[dict] = None
        self.emitted_signals: set = set()
        # position
        self.in_position = False
        self.pos_dir = 0
        self.pos_entry = 0.0
        self.pos_stop = 0.0
        self.pos_target = 0.0
        self.pos_kind = ""
        # counter
        self.counter_armed = False
        self.counter_dir = 0
        self.counter_entry = 0.0
        self.first_main_done = False
        self.first_main_pnl = 0.0
        self.counter_be_locked = False
        # volatility filter: filtered-out session-days take no entries
        self.filtered_out = False
        self._vol_checked = False
        # day pnl
        self.realized_today = 0.0
        self.trades: List[_TradeRec] = []
        self.flattened = False
        self.session_over = False

    def reset_day(self) -> None:
        self.orb_high = None
        self.orb_low = None
        self.bias_allow_long = True
        self.bias_allow_short = True
        self.entries_today = 0
        self.long_taken = False
        self.short_taken = False
        self.m2_bars = []
        self.m10_bars = []
        self._m2_partial = None
        self._m10_partial = None
        self.pending_breakouts = []
        self.pending_entry = None
        self.emitted_signals = set()
        self.in_position = False
        self.pos_dir = 0
        self.pos_entry = 0.0
        self.pos_stop = 0.0
        self.pos_target = 0.0
        self.pos_kind = ""
        self.counter_armed = False
        self.counter_dir = 0
        self.counter_entry = 0.0
        self.first_main_done = False
        self.first_main_pnl = 0.0
        self.counter_be_locked = False
        self.filtered_out = False
        self._vol_checked = False
        self.realized_today = 0.0
        self.flattened = False
        self.session_over = False


def _port_session(
    st: _SessionState,
    bar: dict,
    prior_high: float,
    prior_low: float,
    prior_close16: float,
    today_open: float,
    london_orb: Optional[tuple[float, float]],
    halt: bool,
    orb_hist: Optional[List[float]] = None,
) -> List[_TradeRec]:
    """Process one 1m bar for one session. Returns trades that closed on it.

    Mirrors FlexingJoeOrbAdapter.on_bar: buffers -> bias -> 10m finalize
    (breakout) -> 2m finalize (pullback) -> maybe emit entry -> manage
    position (stop/target/session-end flatten) -> counter entry.
    """
    ts = bar["ts"]
    time_min = bar["time_min"]
    mss = _mss(time_min)
    open_ = bar["open"]
    high = bar["high"]
    low = bar["low"]
    close = bar["close"]
    spec = st.spec
    trades: List[_TradeRec] = []

    # --- ORB capture ---
    if spec.orb_start_mss <= mss < spec.orb_end_mss:
        if st.orb_high is None or high > st.orb_high:
            st.orb_high = high
        if st.orb_low is None or low < st.orb_low:
            st.orb_low = low

    # --- volatility filter (evaluated once ORB window closes) ---
    # Skip this session-day entirely when today's ORB range is below
    # MIN_ORB_MULT x rolling median of the prior VOL_FILTER_LOOKBACK ranges.
    # Mirrors signals.py generate_all_signals (min_periods = lookback//2).
    if (
        MIN_ORB_MULT is not None
        and not st._vol_checked
        and mss >= spec.orb_end_mss
        and st.orb_high is not None
        and st.orb_low is not None
        and orb_hist is not None
    ):
        st._vol_checked = True
        today_orb_range = st.orb_high - st.orb_low
        if orb_hist:
            lookback = VOL_FILTER_LOOKBACK
            window = orb_hist[-lookback:] if len(orb_hist) > lookback else orb_hist
            minp = max(1, lookback // 2)
            median_orb = float(np.median(window) if len(window) >= minp else np.nan)
        else:
            median_orb = np.nan
        if (
            pd.notna(today_orb_range)
            and pd.notna(median_orb)
            and median_orb > 0
            and today_orb_range < MIN_ORB_MULT * median_orb
        ):
            st.filtered_out = True
        orb_hist.append(today_orb_range)

    # --- buffers ---
    m2_start = (time_min // 2) * 2
    if st._m2_partial is None or st._m2_partial["time_min"] != m2_start:
        st._m2_partial = {"ts": ts, "time_min": m2_start, "open": open_,
                          "high": high, "low": low, "close": close}
    p2 = st._m2_partial
    p2["high"] = max(p2["high"], high)
    p2["low"] = min(p2["low"], low)
    p2["close"] = close
    if time_min == m2_start + 1:
        st.m2_bars.append(dict(p2))

    m10_start = (time_min // 10) * 10
    if st._m10_partial is None or st._m10_partial["time_min"] != m10_start:
        st._m10_partial = {"ts": ts, "time_min": m10_start, "open": open_,
                           "high": high, "low": low, "close": close}
    p10 = st._m10_partial
    p10["high"] = max(p10["high"], high)
    p10["low"] = min(p10["low"], low)
    p10["close"] = close
    if time_min == m10_start + 9:
        st.m10_bars.append(dict(p10))

    # --- bias (computed fresh each bar from known-at-the-time inputs) ---
    gap_pct = (today_open - prior_close16) / prior_close16 * 100.0 if prior_close16 else 0.0
    above_pdh = today_open > prior_high
    below_pdl = today_open < prior_low
    if london_orb is not None:
        above_london = today_open > london_orb[0]
        below_london = today_open < london_orb[1]
    else:
        above_london = below_london = False
    bias_score = 0
    bias_score += 1 if gap_pct > 0 else (-1 if gap_pct < 0 else 0)
    bias_score += 1 if above_pdh else (-1 if below_pdl else 0)
    bias_score += 1 if above_london else (-1 if below_london else 0)
    allow_long = not (bias_score <= -2)
    allow_short = not (bias_score >= 2)
    st.bias_allow_long = allow_long
    st.bias_allow_short = allow_short

    can_enter = (not halt) and (not st.flattened) and (not st.session_over) and (not st.filtered_out)
    if _mss(time_min) >= spec.session_end_mss:
        st.session_over = True
    if can_enter and not st.in_position and st.entries_today < MAX_ENTRIES:
        # --- 10m finalize: ORB breakout (only after ORB window) ---
        if st.orb_high is not None and st.orb_low is not None and st.m10_bars:
            latest = st.m10_bars[-1]
            if latest["time_min"] not in st.emitted_signals:
                l_tm = latest["time_min"]
                if _mss(l_tm) >= spec.orb_end_mss:
                    direction = None
                    if latest["close"] > st.orb_high and st.bias_allow_long:
                        direction = 1
                    elif latest["close"] < st.orb_low and st.bias_allow_short:
                        direction = -1
                    if direction is not None:
                        st.pending_breakouts.append({
                            "direction": direction,
                            "breakout_start": l_tm,
                            "start_minutes": l_tm + 10,
                            "resolved": False,
                        })
                        st.emitted_signals.add(latest["time_min"])
        # --- 2m finalize: EMA20 pullback -> pending entry ---
        if st.pending_breakouts and st.m2_bars:
            closes = [b["close"] for b in st.m2_bars]
            if len(closes) >= EMA_PERIOD:
                ema_arr = pd.Series(closes).ewm(span=EMA_PERIOD, adjust=False).mean().to_numpy()
                ema_value = float(ema_arr[-1])
                if not np.isnan(ema_value):
                    last2 = st.m2_bars[-1]
                    l2_tm = last2["time_min"]
                    if st.pending_entry is None and not st.in_position:
                        for pb in st.pending_breakouts:
                            if pb["resolved"]:
                                continue
                            if l2_tm < pb["start_minutes"]:
                                continue
                            if pb["direction"] == 1:
                                hit = last2["low"] <= ema_value and last2["close"] > ema_value
                            else:
                                hit = last2["high"] >= ema_value and last2["close"] < ema_value
                            if hit:
                                pb["resolved"] = True
                                st.pending_entry = {
                                    "direction": pb["direction"],
                                    "entry_time_min": l2_tm + 1,
                                }
                                break
                    st.pending_breakouts = [
                        pb for pb in st.pending_breakouts
                        if not pb["resolved"] and _mss(pb["start_minutes"]) <= spec.session_end_mss
                    ]

        # --- emit main entry at next bar open ---
        if st.pending_entry is not None and time_min >= st.pending_entry["entry_time_min"]:
            direction = st.pending_entry["direction"]
            if not st.in_position:
                st.pending_entry = None
                orb_range = st.orb_high - st.orb_low
                if orb_range > 0 and _mss(time_min) <= spec.session_end_mss:
                    slip = SLIPPAGE_PTS if direction == 1 else -SLIPPAGE_PTS
                    entry_price = open_ + slip
                    if direction == 1:
                        stop_price = st.orb_low
                        target_price = open_ + TARGET_MULTIPLE * orb_range
                    else:
                        stop_price = st.orb_high
                        target_price = open_ - TARGET_MULTIPLE * orb_range
                    st.in_position = True
                    st.pos_dir = direction
                    st.pos_entry = entry_price
                    st.pos_stop = stop_price
                    st.pos_target = target_price
                    st.pos_kind = "main"
                    st.entries_today += 1
                    if direction == 1:
                        st.long_taken = True
                    else:
                        st.short_taken = True
            else:
                st.pending_entry = None

    # --- manage open position: stop / target / session-end flatten ---
    if st.in_position:
        reason = None
        exit_price = None
        if _mss(time_min) >= spec.session_end_mss:
            reason = "SESSION_END"
            exit_price = close
        elif st.pos_dir == 1:
            if low <= st.pos_stop:
                reason, exit_price = "STOP", st.pos_stop
            elif high >= st.pos_target:
                if st.pos_kind == "counter" and not st.counter_be_locked:
                    st.pos_stop = st.pos_entry
                    st.counter_be_locked = True
                else:
                    reason, exit_price = "TARGET", st.pos_target
        else:
            if high >= st.pos_stop:
                reason, exit_price = "STOP", st.pos_stop
            elif low <= st.pos_target:
                if st.pos_kind == "counter" and not st.counter_be_locked:
                    st.pos_stop = st.pos_entry
                    st.counter_be_locked = True
                else:
                    reason, exit_price = "TARGET", st.pos_target
        if reason is not None:
            st.in_position = False
            st.flattened = True if reason == "SESSION_END" else st.flattened
            pnl_pts = (exit_price - st.pos_entry) * st.pos_dir
            # Reference economics (wrapped_NQ_c2_me3.json):
            #   main  : gross = pts * PV * CONTRACTS ; costs scale with contracts
            #   counter: gross = pts * PV (1 contract) ; flat slip + commission
            if st.pos_kind == "counter":
                gross = pnl_pts * POINT_VALUE_NQ
                slip_cost = SLIPPAGE_PTS * POINT_VALUE_NQ / TICK_SIZE_NQ
                comm_cost = COMMISSION * 2.0
            else:
                gross = pnl_pts * POINT_VALUE_NQ * CONTRACTS
                slip_cost = SLIPPAGE_PTS * 2.0 * POINT_VALUE_NQ * CONTRACTS
                comm_cost = COMMISSION * 2.0 * CONTRACTS
            net = gross - comm_cost - slip_cost
            trades.append(_TradeRec(
                entry_time=ts, exit_time=ts, session=spec.name, kind=st.pos_kind,
                direction=st.pos_dir, entry_price=round(st.pos_entry, 4),
                exit_price=round(exit_price, 4), contracts=CONTRACTS,
                gross_pnl=round(gross, 2), net_pnl=round(net, 2), exit_reason=reason,
            ))
            st.realized_today += net
            # counter arming on first main trade loss
            if st.pos_kind == "main" and not st.first_main_done:
                st.first_main_done = True
                st.first_main_pnl = net
                if net <= 0:
                    st.counter_armed = True
                    st.counter_dir = -st.pos_dir
                    st.counter_entry = st.pos_entry + (
                        gross / (POINT_VALUE_NQ * st.pos_dir)
                    )
            st.pos_dir = 0
            st.pos_entry = 0.0
            st.pos_stop = 0.0
            st.pos_target = 0.0
            st.pos_kind = ""
            st.counter_be_locked = False

    # --- counter entry if armed and flat ---
    if can_enter and st.counter_armed and not st.in_position and st.first_main_done:
        if st.first_main_pnl < 0 and st.orb_high is not None and st.orb_low is not None:
            orb_range = st.orb_high - st.orb_low
            direction = st.counter_dir
            entry_price = st.counter_entry
            if direction == 1:
                stop_price = st.orb_low
                target_price = entry_price + orb_range
            else:
                stop_price = st.orb_high
                target_price = entry_price - orb_range
            st.counter_armed = False
            st.in_position = True
            st.pos_dir = direction
            st.pos_entry = entry_price
            st.pos_stop = stop_price
            st.pos_target = target_price
            st.pos_kind = "counter"

    return trades


def _run_combo(
    df_et: pd.DataFrame,
    combo_sessions: List[str],
    use_daily_loss: bool,
    daily_loss_limit: float,
) -> Dict[str, Any]:
    """Run a session combo over the whole dataset. Returns trades + daily pnl."""
    et_idx = df_et.index
    years = et_idx.year.to_numpy()
    months = et_idx.month.to_numpy()
    days = et_idx.day.to_numpy()
    hours = et_idx.hour.to_numpy()
    minutes = et_idx.minute.to_numpy()
    time_mins = hours * 60 + minutes
    session_dates = []
    for y, m, d, h in zip(years, months, days, hours):
        dt = pd.Timestamp(year=y, month=m, day=d)
        if h >= 18:
            dt = dt + pd.Timedelta(days=1)
        session_dates.append(int(dt.strftime("%Y%m%d")))
    session_dates = np.array(session_dates)
    opens = df_et["open"].to_numpy(dtype=float)
    highs = df_et["high"].to_numpy(dtype=float)
    lows = df_et["low"].to_numpy(dtype=float)
    closes = df_et["close"].to_numpy(dtype=float)

    # day boundaries (full sessions only: require a 16:00 ET bar in the window)
    unique_days = []
    for sd in np.unique(session_dates):
        mask = session_dates == sd
        tms = time_mins[mask]
        if (16 * 60) in tms:  # has RTH end -> treat as complete
            unique_days.append(sd)
    unique_days.sort()

    all_trades: List[_TradeRec] = []
    daily_pnl: Dict[str, float] = {}
    daily_halted: Dict[str, bool] = {}
    orb_hist: Dict[str, List[float]] = {s: [] for s in combo_sessions}

    for di, sd in enumerate(unique_days):
        mask = session_dates == sd
        idxs = np.where(mask)[0]
        states = {s: _SessionState(SESSIONS[s]) for s in combo_sessions}
        day_realized = 0.0
        day_halted = False

        # today's open = first bar of session-day (18:00 ET)
        first_bar = idxs[0]
        today_open_of = float(opens[first_bar])

        # prior session-day stats for bias (immediately-prior day only)
        if di == 0:
            prior_high = today_open_of
            prior_low = today_open_of
            prior_close16 = today_open_of
        else:
            prior_sd = unique_days[di - 1]
            pm = session_dates == prior_sd
            pidx = np.where(pm)[0]
            prior_high = float(highs[pm].max())
            prior_low = float(lows[pm].min())
            hit = np.where(time_mins[pidx] == 16 * 60)[0]
            if hit.size:
                prior_close16 = float(closes[pidx[hit[-1]]])
            else:
                prior_close16 = float(closes[pidx[-1]])

        # London ORB for bias: available only once 03:30 passed. For Asian
        # (trades before 03:00) London ORB is in the future -> excluded.
        london_orb: Optional[tuple[float, float]] = None

        for i in idxs:
            bar = {
                "ts": et_idx[i], "time_min": int(time_mins[i]),
                "open": float(opens[i]), "high": float(highs[i]),
                "low": float(lows[i]), "close": float(closes[i]),
            }
            tm = int(time_mins[i])
            # update London ORB (03:00-03:30 ET) as bars arrive
            if 3 * 60 <= tm <= 3 * 60 + 30:
                hh = float(highs[i]); ll = float(lows[i])
                if london_orb is None:
                    london_orb = (hh, ll)
                else:
                    london_orb = (max(london_orb[0], hh), min(london_orb[1], ll))
            for sname in combo_sessions:
                st = states[sname]
                # London-ORB bias input only for sessions trading after 03:30
                use_london = None
                if sname in ("london", "ny") and london_orb is not None:
                    use_london = london_orb
                closed = _port_session(
                    st, bar, prior_high, prior_low, prior_close16,
                    today_open_of, use_london, day_halted,
                    orb_hist=orb_hist[sname],
                )
                for t in closed:
                    day_realized += t.net_pnl
                    daily_pnl[sd] = daily_pnl.get(sd, 0.0) + t.net_pnl
                    all_trades.append(t)
                    if use_daily_loss and day_realized <= -daily_loss_limit:
                        day_halted = True
        # end-of-day: flatten any leftover positions at last close
        for sname in combo_sessions:
            st = states[sname]
            if st.in_position:
                last_close = float(closes[idxs[-1]])
                pnl_pts = (last_close - st.pos_entry) * st.pos_dir
                if st.pos_kind == "counter":
                    gross = pnl_pts * POINT_VALUE_NQ
                    slip_cost = SLIPPAGE_PTS * POINT_VALUE_NQ / TICK_SIZE_NQ
                    comm_cost = COMMISSION * 2.0
                else:
                    gross = pnl_pts * POINT_VALUE_NQ * CONTRACTS
                    slip_cost = SLIPPAGE_PTS * 2.0 * POINT_VALUE_NQ * CONTRACTS
                    comm_cost = COMMISSION * 2.0 * CONTRACTS
                net = gross - comm_cost - slip_cost
                all_trades.append(_TradeRec(
                    entry_time=et_idx[idxs[-1]], exit_time=et_idx[idxs[-1]],
                    session=sname, kind=st.pos_kind, direction=st.pos_dir,
                    entry_price=round(st.pos_entry, 4), exit_price=round(last_close, 4),
                    contracts=CONTRACTS, gross_pnl=round(gross, 2), net_pnl=round(net, 2),
                    exit_reason="SESSION_END",
                ))
                daily_pnl[sd] = daily_pnl.get(sd, 0.0) + net

    daily_pnl = {str(k): round(v, 2) for k, v in sorted(daily_pnl.items())}
    return {"trades": all_trades, "daily_pnl": daily_pnl}


def _metrics(trades: List[_TradeRec], daily_pnl: Dict[str, float]) -> Dict[str, Any]:
    if not trades:
        return {"total_trades": 0, "net_pnl": 0.0}
    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    net = float(pnls.sum())
    win_rate = float((pnls > 0).mean() * 100.0)
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")
    eq = np.concatenate([[0.0], np.cumsum(pnls)])
    mdd = float(np.max(np.maximum.accumulate(eq) - eq))
    n_days = len(daily_pnl) or 1
    return {
        "total_trades": int(len(trades)),
        "main_trades": int(sum(1 for t in trades if t.kind == "main")),
        "counter_trades": int(sum(1 for t in trades if t.kind == "counter")),
        "net_pnl": round(net, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "avg_trade": round(float(pnls.mean()), 2),
        "avg_win": round(float(wins.mean()), 2) if wins.size else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if losses.size else 0.0,
        "max_dd_dollars": round(mdd, 2),
        "trades_per_day": round(len(trades) / n_days, 2),
        "active_days": n_days,
        "winning_days": int(sum(1 for v in daily_pnl.values() if v > 0)),
        "losing_days": int(sum(1 for v in daily_pnl.values() if v < 0)),
        "sum_win_days": round(sum(v for v in daily_pnl.values() if v > 0), 2),
        "sum_loss_days": round(sum(v for v in daily_pnl.values() if v < 0), 2),
        "daily_pnl_clusters": _clusters(daily_pnl),
    }


def _clusters(daily_pnl: Dict[str, float]) -> Dict[str, Any]:
    vals = sorted(daily_pnl.values())
    def buckets(vals_, edges):
        counts = {}
        for e in edges:
            counts[e] = 0
        for v in vals_:
            for e in edges:
                if v <= e:
                    counts[e] += 1
                    break
        return counts
    return {
        "loss_days": buckets(vals, [-250, -500, -900, -1500, -2500, float("inf")]),
        "win_days": buckets([-v for v in vals if v > 0],
                           [-250, -500, -1000, -2500, -5000, float("inf")]),
    }


def _time_to_target(
    daily_pnl: Dict[str, float],
    targets: List[float],
) -> Dict[str, Any]:
    """For each start day, days until cumulative PnL >= target (walk forward)."""
    dates = sorted(daily_pnl.keys())
    pnls = [daily_pnl[d] for d in dates]
    out: Dict[str, Any] = {"by_target": {}, "summary": {}}
    for tg in targets:
        rows = []
        reached = []
        for s in range(len(dates)):
            cum = 0.0
            days_to = None
            for k in range(s, len(dates)):
                cum += pnls[k]
                if cum >= tg:
                    days_to = k - s + 1
                    break
            rows.append({"start": dates[s], "days_to_target": days_to, "cum_end": round(cum, 2)})
            if days_to is not None:
                reached.append(days_to)
        out["by_target"][f"{tg:g}"] = rows
        out["summary"][f"{tg:g}"] = {
            "starts": len(rows),
            "reached": len(reached),
            "never_reached": len(rows) - len(reached),
            "pct_reached": round(100.0 * len(reached) / len(rows), 1) if rows else 0.0,
            "median_days": float(np.median(reached)) if reached else None,
            "mean_days": round(float(np.mean(reached)), 2) if reached else None,
        }
    return out


def _prop_sim(daily: Dict[str, float], account_size: str, path: str,
              daily_cap: Optional[float]) -> Dict[str, Any]:
    """Replay daily PnL through the v4 Topstep prop-farm simulator.

    Mirrors run_reentry_cap_sweep.simulate(): clip daily PnL at the cap, then
    feed through simulate_prop_farm (XFA scaling, combine lifecycle).
    """
    root = Path(__file__).resolve().parents[2]  # .../projects/trading
    for cand in (Path(__file__).parent / "scripts",
                 root / "v5_orb_nq_backtest",
                 root.parent):
        if (cand / "scripts" / "prop_farm_simulator_v4.py").exists() or (
                cand / "prop_farm_simulator_v4.py").exists():
            sys.path.insert(0, str(cand))
            break
    from scripts.prop_farm_simulator_v4 import CostAssumptions, make_rules, simulate_prop_farm

    df = pd.DataFrame(
        [{"date": pd.to_datetime(d).date(), "net_pnl": v} for d, v in daily.items()]
    ).set_index("date").sort_index()
    if daily_cap is not None and daily_cap > 0:
        df["net_pnl"] = df["net_pnl"].clip(upper=daily_cap)

    rules = make_rules(account_size)
    res = simulate_prop_farm(
        df, f"session_matrix_{path}", rules, CostAssumptions(), path,
        reset_after_payout=False, desired_contracts=1,
    )
    s = res.summary()
    n_payouts = sum(len(a.payouts) for a in res.attempts)
    s["n_payouts"] = n_payouts
    s["payouts_per_month"] = round(n_payouts / (len(df) / 365.25 * 12), 2)
    s["daily_cap"] = daily_cap
    return s


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("legacy", "prop") else "legacy"
    if mode == "legacy":
        data_path = sys.argv[2] if len(sys.argv) > 2 else (
            "/config/projects/trading/flexing_joe_orb/market_data/NQ_topstep_api_1min.csv")
        out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else
                       "/config/projects/trading/flexing_joe_orb/reports")
    else:  # prop: reference-aligned (c2 me3 filter0.7) + 150K std/consistency
        data_path = sys.argv[2] if len(sys.argv) > 2 else (
            "/config/projects/trading/flexing_joe_orb/market_data/NQ_topstep_api_1min.csv")
        out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else
                       "/config/projects/trading/flexing_joe_orb/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv_csv(data_path)
    df_et = df.tz_convert("America/New_York")

    if mode == "prop":
        # Reference-aligned parameters from topstep_150k_paper_config.json.
        # Max reentries per session-day configurable via sys.argv[4] (default 3).
        global CONTRACTS, MAX_ENTRIES, MIN_ORB_MULT
        CONTRACTS = 2
        MAX_ENTRIES = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        MIN_ORB_MULT = 0.7
        daily_caps = {"standard": 3000.0, "consistency": 2500.0}

    # Optional slicing for parallel workers (GitHub Actions):
    #   sys.argv[5] = comma-separated combos to run (default: all)
    #   sys.argv[6] = "reg" | "lim" | "both" (default: both)
    combo_filter = [c.strip() for c in sys.argv[5].split(",")] if len(sys.argv) > 5 and sys.argv[5] else list(SESSION_COMBOS)
    variant_filter = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else "both"
    bad = [c for c in combo_filter if c not in SESSION_COMBOS]
    if bad:
        raise SystemExit(f"Unknown combo(s): {bad}")
    if variant_filter not in ("reg", "lim", "both"):
        raise SystemExit(f"Unknown variant filter: {variant_filter}")

    results: Dict[str, Any] = {}
    for combo, sessions in SESSION_COMBOS.items():
        if combo not in combo_filter:
            continue
        for use_lim in ((False, True) if variant_filter == "both" else
                        (False,) if variant_filter == "reg" else
                        (True,)):
            key = f"{combo}_lim" if use_lim else f"{combo}_reg"
            res = _run_combo(df_et, sessions, use_lim, 900.0)
            daily = res["daily_pnl"]
            trades = res["trades"]
            met = _metrics(trades, daily)
            t2t = _time_to_target(daily, [2000.0, 4000.0])
            entry = {
                "combo": combo,
                "sessions": sessions,
                "daily_loss_limit": 900.0 if use_lim else None,
                "metrics": met,
                "time_to_target": t2t,
                "n_trades": len(trades),
                "params": {"contracts": CONTRACTS, "max_entries": MAX_ENTRIES,
                           "min_orb_mult": MIN_ORB_MULT},
            }
            if mode == "prop":
                entry["prop"] = {}
                for path_name, cap in daily_caps.items():
                    entry["prop"][path_name] = _prop_sim(daily, "150K", path_name, cap)
            results[key] = entry
            trade_json = [
                {"entry_time": t.entry_time.isoformat(), "exit_time": t.exit_time.isoformat(),
                 "session": t.session, "kind": t.kind, "direction": t.direction,
                 "entry_price": t.entry_price, "exit_price": t.exit_price,
                 "gross_pnl": t.gross_pnl, "net_pnl": t.net_pnl, "exit_reason": t.exit_reason}
                for t in trades
            ]
            me_sfx = f"_me{MAX_ENTRIES}" if mode == "prop" else ""
            (out_dir / f"session_matrix{me_sfx}_{key}_trades.json").write_text(
                json.dumps(trade_json, indent=2))
            (out_dir / f"session_matrix{me_sfx}_{key}_daily.json").write_text(
                json.dumps(daily, indent=2))
            if mode == "prop":
                tag = f"{key} c{CONTRACTS} me{MAX_ENTRIES} f{MIN_ORB_MULT}"
            else:
                tag = key
            print(f"{tag:22s} trades={len(trades):4d} net=${met['net_pnl']:>+12,.2f} "
                  f"WR={met['win_rate']:5.1f}% DD=${met['max_dd_dollars']:>+12,.0f}")

    summary = {k: {"combo": v["combo"], "sessions": v["sessions"],
                   "daily_loss_limit": v["daily_loss_limit"], "metrics": v["metrics"],
                   "params": v["params"]}
               for k, v in results.items()}
    if mode == "prop":
        for k, v in results.items():
            summary[k]["prop"] = v["prop"]
    suffix = "_prop" if mode == "prop" else ""
    if mode == "prop":
        suffix = f"_prop_me{MAX_ENTRIES}"
    (out_dir / f"session_matrix_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nSaved summary + trades + daily JSONs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
