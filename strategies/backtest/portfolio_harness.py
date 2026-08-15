# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created strategies/backtest/portfolio_harness.py: multi-instrument driver
#     that runs several (strategy, symbol) StrategyHarness instances in lockstep
#     against one shared daily loss limit ledger.  The ledger enforces a
#     PORTFOLIO-level hard DLL: realized day PnL + the floating PnL of EVERY
#     open position in the portfolio may never cross -dll, and when it does all
#     harnesses halt new entries for the rest of the (UTC) day.
#   - Each harness's DLL trigger accounts for concurrent open positions on the
#     other symbols via ledger["open_float_others"], so the cut price is the
#     price where the WHOLE portfolio lands exactly on -dll, not just the one
#     instrument.
# WHY: The user requested portfolio-level (not per-instrument) DLL for the
#      corrected reruns; per-symbol engines alone understate the tail.

"""Portfolio-level daily loss limit backtest driver.

Runs one StrategyHarness per (strategy, symbol) over its own 1-min bars while
sharing a single daily loss ledger.  Bars are stepped in chronological order
across the union of each instrument's minute grid; the ledger's day PnL is the
sum of realized dollars across all constituent harnesses, and every DLL trigger
price is computed against the portfolio-wide shortfall so a losing day on one
instrument can flatten a position on another.

Only the winners book is wired here: mos_session_daily_draw + fifteen_min_
range_scalp over {NQ, ES, YM}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # /config
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))

from strategies.backtest.engine import StrategyHarness  # noqa: E402

POINT_VALUES = {"NQ": 20.0, "ES": 50.0, "YM": 5.0}

WINNERS_COMBOS = [
    ("mos_session_daily_draw", "NQ"),
    ("mos_session_daily_draw", "ES"),
    ("mos_session_daily_draw", "YM"),
    ("fifteen_min_range_scalp", "NQ"),
    ("fifteen_min_range_scalp", "ES"),
    ("fifteen_min_range_scalp", "YM"),
]


def load_1m_fast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else "ts"
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


class PortfolioHarness:
    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        combos: list[tuple[str, str]] | None = None,
        dll: float | None = None,
        risk_pct: float | None = None,
        initial_capital: float = 100_000.0,
        max_drawdown: float | None = None,
        eod_drawdown: float | None = None,
        scratch_root: Path | None = None,
    ):
        combos = combos or WINNERS_COMBOS
        self.dll = dll
        self.ledger = {
            "day": None,
            "day_realized": 0.0,
            "day_halted": False,
            "dll": dll,
            "open_float_others": 0.0,
        }
        self.harnesses: list[StrategyHarness] = []
        self._data = {}
        for strat, sym in combos:
            if sym not in data:
                raise KeyError(f"missing data for {sym}")
            df = data[sym]
            self._data[sym] = df
            h = StrategyHarness(
                strategy=strat,
                symbol=sym,
                point_value=POINT_VALUES[sym],
                pip_value=1.0,
                scratch_root=scratch_root,
                dll=dll,
                risk_pct=risk_pct,
                initial_capital=initial_capital,
                max_drawdown=max_drawdown,
                eod_drawdown=eod_drawdown,
                ledger=self.ledger,
            )
            self.harnesses.append(h)
        self.point_values = POINT_VALUES

    def _others_float(self, skip_h: StrategyHarness, latest_close: dict[str, float | None]) -> float:
        tot = 0.0
        for h in self.harnesses:
            if h is skip_h:
                continue
            pos = h._open
            if pos is None:
                continue
            c = latest_close.get(h.symbol)
            if c is None:
                continue
            tot += (c - pos["entry_price"]) * pos["direction"] * pos["qty"] * h.point_value
        return tot

    def run(self) -> list[dict]:
        """Step every harness through its own bar grid, sharing the ledger.

        Pastes the portfolio open-floating shortfall into the shared ledger
        before each harness's bar so that harness's DLL trigger price accounts
        for the other symbols' concurrent positions.
        """
        # Pre-extract per-symbol numpy arrays + a merged chronological ts sweep.
        arrays = {}
        n_ts = {}
        for sym, df in self._data.items():
            ts = df.index.values.astype("datetime64[ns]").astype(int)
            arrays[sym] = {
                "ts": ts,
                "o": df["open"].to_numpy(float),
                "h": df["high"].to_numpy(float),
                "l": df["low"].to_numpy(float),
                "c": df["close"].to_numpy(float),
                "v": df["volume"].to_numpy(float),
            }
            n_ts[sym] = len(ts)

        # Pointers advance per symbol as the sweep reaches each timestamp.
        ptr = {sym: 0 for sym in self._data}
        latest_close: dict[str, float | None] = {sym: None for sym in self._data}

        # Union of all symbol timestamps, in order.
        all_ts = sorted({int(x) for s in arrays.values() for x in s["ts"]})

        from datetime import datetime, timezone

        for h in self.harnesses:
            h._init_windows()
        entered = []
        for h in self.harnesses:
            entered.append(h.__enter__())
        try:
            from collections import defaultdict

            by_ts: dict[int, list] = defaultdict(list)
            for sym in self._data:
                a = arrays[sym]
                for i in range(n_ts[sym]):
                    by_ts[int(a["ts"][i])].append(sym)
            for ts_ns in all_ts:
                for sym in by_ts[ts_ns]:
                    a = arrays[sym]
                    i = ptr[sym]
                    # Advance all harnesses that trade this symbol to this bar.
                    for h in self.harnesses:
                        if h.symbol != sym:
                            continue
                        bar = {
                            "timestamp": datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc),
                            "open": a["o"][i], "high": a["h"][i], "low": a["l"][i],
                            "close": a["c"][i], "volume": a["v"][i],
                        }
                        # Update the shared portfolio shortfall from the OTHER
                        # open positions at their latest closes before this
                        # harness decides its own close/entry.
                        self.ledger["open_float_others"] = self._others_float(h, latest_close)
                        h._step(int(a["ts"][i]), bar, i, last=(i == n_ts[sym] - 1))
                    ptr[sym] = i + 1
                    if i < n_ts[sym]:
                        latest_close[sym] = a["c"][i]
        finally:
            for h in self.harnesses:
                h.__exit__(None, None, None)

        # Force-close anything still open at end of the union window.
        for h in self.harnesses:
            if h._open is not None:
                ts_last = self._data[h.symbol].index[-1]
                bar = {
                    "timestamp": ts_last,
                    "open": arrays[h.symbol]["o"][-1],
                    "high": arrays[h.symbol]["h"][-1],
                    "low": arrays[h.symbol]["l"][-1],
                    "close": arrays[h.symbol]["c"][-1],
                    "volume": arrays[h.symbol]["v"][-1],
                }
                h._close_position(bar, arrays[h.symbol]["c"][-1], "END_OF_DATA")

        trades = []
        for h in self.harnesses:
            trades.extend(h.trades)
        trades.sort(key=lambda t: t["entry_time"])
        return trades