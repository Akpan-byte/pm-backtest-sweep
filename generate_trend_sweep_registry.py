#!/usr/bin/env python3
# CHANGE_SUMMARY
# 2026-07-17  assistant
#   - Generates trend_sweep_registry.json and per-variant batch files.
#   - Drops slow / heavy estimators that bottleneck 2k-sample backtests.
# WHY: The original 660-variant registry included O(n^2)/sklearn/scipy estimators
#      that made GitHub Actions and laptop batches run 10x+ slower than needed.

import json
import itertools
import os
from typing import Any, Dict, List

BASE_PARAMS = {
    "module": "phase_2_new.trend_family_sweep",
    "fn": "trend_family_signal",
    "params": [
        "spot_price", "strike", "rem_sec", "elapsed_sec", "duration_sec",
        "yp", "np_val", "yes_ask", "no_ask", "spot_history",
        "yp_history", "np_history", "tf_hint", "market_id",
        "start_date_iso", "config",
    ],
    "tf_hint": "5m",
    "entry_min": 0.05,
    "time_guard": 5.0,
    "confidence_scale": 0.002,
}

# Estimators that are fast enough for a per-tick 2k-sample backtest.
# Removed: theilsen, ransac, savgol, hp_filter, huber_regression, recursive_least_squares.
TREND_GRID: List[Dict[str, Any]] = [
    # type, lookbacks, deviation_pct, entry_max, extra_params
    ("ema", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alpha": [0.1, 0.2, 0.3]}),
    ("dema", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alpha": [0.1, 0.2]}),
    ("tema", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alpha": [0.1, 0.2]}),
    ("zlema", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alpha": [0.1, 0.2]}),
    ("sma", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("wma", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("smma", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alpha": [0.1, 0.2, 0.3]}),
    ("hma", [10, 20], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("alma", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alma_offset": [0.75, 0.85], "alma_sigma": [6.0]}),
    ("mcginley", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"mcginley_k": [0.6]}),
    ("kama", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"kama_er_len": [10], "kama_fast": [0.6667], "kama_slow": [0.0645]}),
    ("frama", [50, 100], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("vidya", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alpha": [0.2], "vidya_cmo_len": [9]}),
    ("linreg", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("polyreg2", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("polyreg3", [50, 100], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("bayesian_regression", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"bayes_alpha": [1.0]}),
    ("ridge_regression", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"ridge_alpha": [1.0]}),
    ("perceptron_trend", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"perceptron_lr": [0.01]}),
    ("gaussian", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"gaussian_sigma": [2.0]}),
    ("median", [10, 20], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("butterworth", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alpha": [0.2]}),
    ("donchian_mid", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("bollinger_mid", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("keltner_mid", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("minmax_mid", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("range_mid", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("tenkan_sen", [10, 20], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("kijun_sen", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("parabolic_sar", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"psar_af": [0.02], "psar_max_af": [0.2]}),
    ("supertrend", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"supertrend_mult": [3.0]}),
    ("cumulative_avg", [50, 100], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("running_mean", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("sign_filter", [10, 20], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("mode_filter", [10, 20], [0.02, 0.05, 0.1], [0.80, 0.85], {"mode_tick": [1.0]}),
    ("vwap_ticks", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
    ("pwma", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"power": [2.0]}),
    ("exp_wma", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"decay": [0.9]}),
    ("kernel_regression", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"kernel_bw": [2.0]}),
    ("loess", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"kernel_bw": [2.0]}),
    ("holt", [20, 50], [0.02, 0.05, 0.1], [0.80, 0.85], {"alpha": [0.1, 0.2], "holt_beta": [0.05]}),
    ("atr_mid", [10, 20], [0.02, 0.05, 0.1], [0.80, 0.85], {}),
]


def _name(type_: str, lookback: int, deviation_pct: float, entry_max: float, extra: Dict[str, Any]) -> str:
    parts = [f"tf_{type_}", f"lb{lookback}", f"dev{int(deviation_pct*100):03d}", f"emax{int(entry_max*100):02d}"]
    for k, v in sorted(extra.items()):
        # normalize floats: 0.6667 -> 0667, 0.0645 -> 0065
        if isinstance(v, float):
            if v < 1:
                val = str(v).replace(".", "")[:4].rjust(4, "0")
            else:
                val = str(int(v))
        else:
            val = str(v)
        parts.append(f"{k[:3]}{val}")
    return "_".join(parts)


def build_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    for type_, lookbacks, devs, emaxs, extra_grid in TREND_GRID:
        keys = list(extra_grid.keys())
        vals = [extra_grid[k] for k in keys]
        for lookback, dev, emax, extra_vals in itertools.product(lookbacks, devs, emaxs, itertools.product(*vals)):
            extra = dict(zip(keys, extra_vals))
            name = _name(type_, lookback, dev, emax, extra)
            entry = dict(BASE_PARAMS)
            entry.update({
                "trend_type": type_,
                "lookback": lookback,
                "deviation_pct": dev,
                "entry_max": emax,
            })
            entry.update(extra)
            registry[name] = entry
    return registry


def write_batches(registry: Dict[str, Any], n_batches: int = 20, out_dir: str = "trend_sweep_batches") -> None:
    os.makedirs(out_dir, exist_ok=True)
    names = list(registry.keys())
    batch_size = (len(names) + n_batches - 1) // n_batches
    for i in range(n_batches):
        chunk = names[i * batch_size:(i + 1) * batch_size]
        with open(os.path.join(out_dir, f"batch_{i:02d}.txt"), "w") as f:
            f.write(",".join(chunk))
    print(f"Wrote {len(names)} variants into {n_batches} batches (~{batch_size} each).")


if __name__ == "__main__":
    reg = build_registry()
    with open("trend_sweep_registry.json", "w") as f:
        json.dump(reg, f, indent=2)
    print(f"Wrote trend_sweep_registry.json with {len(reg)} variants.")
    write_batches(reg, n_batches=20, out_dir="trend_sweep_batches")
