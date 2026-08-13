#!/usr/bin/env python3
"""Real market regime analysis using 1-minute OHLCV data.

Computes observable daily regimes from price action and merges with the
strategy's daily PnL to see how performance varies by market condition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, "/config/projects/trading")
from flexing_joe_orb.data import load_ohlcv_csv


def daily_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute daily price-action features from 1-minute bars."""
    df = df_1m.copy()
    if df.index.name != "timestamp":
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    df["date"] = df.index.date

    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    # Prior day close
    daily["prior_close"] = daily["close"].shift(1)
    daily["gap_pct"] = (daily["open"] - daily["prior_close"]) / daily["prior_close"] * 100.0

    # Daily range / true range
    daily["range"] = daily["high"] - daily["low"]
    daily["atr_20"] = daily["range"].rolling(20).mean()
    daily["rel_range"] = daily["range"] / daily["atr_20"]

    # Trend relative to 20-day EMA of close
    daily["ema_20"] = daily["close"].ewm(span=20, adjust=False).mean()
    daily["above_ema20"] = daily["close"] > daily["ema_20"]

    # 5-day momentum
    daily["ret_5d"] = (daily["close"] - daily["close"].shift(5)) / daily["close"].shift(5) * 100.0

    # Day of week / month
    daily["dow"] = pd.to_datetime(daily.index).day_name()
    daily["month"] = pd.to_datetime(daily.index).month

    # ORB range (09:30-09:45 ET)
    df["time_min"] = df.index.hour * 60 + df.index.minute
    orb = df[(df["time_min"] >= 570) & (df["time_min"] < 630)].groupby("date").agg(
        orb_high=("high", "max"),
        orb_low=("low", "min"),
    )
    orb["orb_range"] = orb["orb_high"] - orb["orb_low"]
    daily = daily.join(orb)

    return daily


def _qcut_safe(s: pd.Series, q: int, labels: List[str]) -> pd.Series:
    """Quantile-bin a series, gracefully handling duplicate edges."""
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except ValueError:
        # Duplicate-heavy edges can produce fewer bins than labels; trim labels.
        edges = s.quantile(np.linspace(0, 1, q + 1)).drop_duplicates().values
        n_bins = len(edges) - 1
        if n_bins <= 0:
            return pd.Series(np.nan, index=s.index)
        use_labels = labels[:n_bins]
        return pd.cut(s, bins=edges, labels=use_labels, include_lowest=True)


def classify_regimes(daily: pd.DataFrame) -> pd.DataFrame:
    """Add regime label columns based on quartiles/categories."""
    d = daily.copy()
    d["gap_regime"] = _qcut_safe(d["gap_pct"].abs(), q=3, labels=["small_gap", "med_gap", "large_gap"])
    d["vol_regime"] = _qcut_safe(d["rel_range"].fillna(d["rel_range"].median()), q=3, labels=["low_vol", "med_vol", "high_vol"])
    d["trend_regime"] = np.where(
        d["ret_5d"] > 1.0, "uptrend",
        np.where(d["ret_5d"] < -1.0, "downtrend", "sideways")
    )
    d["ema_regime"] = np.where(d["above_ema20"], "above_ema20", "below_ema20")
    return d


def analyze_by_regime(daily: pd.DataFrame, daily_pnl: pd.Series) -> Dict[str, Any]:
    daily["pnl"] = daily_pnl.reindex(pd.to_datetime(daily.index)).fillna(0.0)
    daily["win_day"] = daily["pnl"] > 0

    results = {}
    for regime_col in ["gap_regime", "vol_regime", "trend_regime", "ema_regime", "dow", "month"]:
        if regime_col not in daily.columns:
            continue
        groups = daily.groupby(regime_col)
        regimes = {}
        for name, grp in groups:
            if len(grp) < 5:
                continue
            regimes[str(name)] = {
                "days": int(len(grp)),
                "traded_days": int((grp["pnl"] != 0).sum()),
                "win_days": int(grp["win_day"].sum()),
                "win_rate_pct": round(float(grp["win_day"].mean()) * 100, 2),
                "avg_daily_pnl": round(float(grp["pnl"].mean()), 2),
                "total_pnl": round(float(grp["pnl"].sum()), 2),
                "avg_orb_range": round(float(grp["orb_range"].mean()), 2) if "orb_range" in grp else None,
            }
        results[regime_col] = regimes
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Backtest JSON")
    parser.add_argument("--data-path", required=True, help="1-min OHLCV CSV")
    parser.add_argument("--output", required=True, help="Output JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and len(raw) == 1 and isinstance(list(raw.values())[0], dict):
        result = list(raw.values())[0]
    else:
        result = raw

    daily = result.get("daily_pnl", {})
    daily_pnl = pd.Series(daily, dtype=float)
    daily_pnl.index = pd.to_datetime(daily_pnl.index)

    df_1m = load_ohlcv_csv(args.data_path)
    feat = daily_features(df_1m)
    feat = classify_regimes(feat)
    analysis = analyze_by_regime(feat, daily_pnl)

    out = {
        "source_file": args.input,
        "data_path": args.data_path,
        "regime_analysis": analysis,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved real regime analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
