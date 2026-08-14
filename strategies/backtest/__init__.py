# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Created strategies/backtest/ package: a futures backtest harness for the
#     four StarTrading signals. Drives each signal bar-by-bar on 1m OHLCV,
#     monkeypatches time_utils so the signals' ET/UTC wall-clock gates replay
#     historical time, maps YES/NO -> LONG/SHORT, simulates SL/TP exits, tags
#     every trade with its symbol, and feeds the topstep-strats engine + metrics.
# WHY: Backtest the 4 intraday strategies on 10y NQ/ES/YM data (in-sample
#      2016-06-01..2023-12-31), then merge symbol-tagged trades for instrument
#      combos without re-running signals.
