# New Signal Interface

## Output files
Create a directory `/config/new_signals/<strategy_name>/` containing:
- `signal.py` — the signal function and any helpers/state.
- `registry.json` — a single JSON object (not array) with the registry entry.

## Signal function
Name: `<strategy_name>_signal`

Accept these keyword arguments and ignore the rest with `**kwargs`:

```python
{
    "spot_price": float,       # current BTC spot price from snapshot
    "strike": float,           # market open oracle price / strike
    "rem_sec": float,          # seconds remaining until expiry
    "elapsed_sec": float,      # seconds since market opened
    "duration_sec": float,     # total market duration (300 for 5m)
    "yp": float,               # YES best bid price
    "np_val": float,           # NO best bid price
    "yes_ask": float,          # YES best ask
    "no_ask": float,           # NO best ask
    "spot_history": list,      # rolling list of recent spot prices
    "yp_history": list,        # rolling list of recent YES prices
    "np_history": list,        # rolling list of recent NO prices
    "tf_hint": str,            # "5m", "15m", etc.
    "market_id": str,          # unique market id
    "start_date_iso": str,     # ISO timestamp of market open
}
```

## Return dict
```python
{
    "triggered": bool,
    "direction": "YES" | "NO" | None,
    "confidence": float,       # 0.0 - 1.0
    "signal_price": float,     # spot_price at signal time
    "entry_price": float,      # for YES use yp, for NO use np_val
    "source": str,             # uppercase strategy name
    "reason": str,             # human-readable reason
}
```

## Rules
1. Do **not** trigger within the first 5 seconds or last 5 seconds of the market.
2. Only enter if `entry_price <= 0.85` and `entry_price >= 0.05`.
3. Use module-level `_STATE = {}` keyed by `market_id` for any state you need to keep across snapshots.
4. Do **not** make network calls inside the signal function. Backtests run offline.
5. For signals that need external precomputed data (Kronos, SPY), load a local parquet/CSV file at module import and return neutral if missing.
6. Add a CHANGE_SUMMARY block at the top of `signal.py`.
7. Keep it simple and fast — the function runs once per snapshot per market.
