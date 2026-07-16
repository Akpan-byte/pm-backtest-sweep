# CHANGE_SUMMARY
# 2026-07-16  kilo
#   - Phase 1 VWAP retuning: replaced the blind 70-config grid with a focused
#     48-config candidate set (12 bases × 4 exit variants).
#   - Added support for direction_flip and exit-rule fields in _add().
#   - Kept a commented recipe for expanding back to the full parameter grid.
# WHY: The original sweep held every trade to expiry; disciplined exits and a
#      smaller, retuned grid are the fastest path to a deployable VWAP family.

VWAP_STRATEGIES = {}

_BASE_PARAMS = [
    "spot_price", "yp", "np_val", "yes_ask", "no_ask", "rem_sec",
    "elapsed_sec", "duration_sec", "z_score", "spread",
    "orderbook_up", "orderbook_down", "yp_history", "np_history",
    "book_imbalance_val", "config",
]


def _add(
    name,
    mode,
    lookback=60,
    threshold=1.0,
    entry_max=0.80,
    direction_flip=False,
    take_profit_pct=None,
    stop_loss_pct=None,
    trailing_stop_pct=None,
    max_hold_sec=None,
    min_rem_sec=None,
    vwap_reversion=False,
    **extra,
):
    cfg = {
        "module": "phase_2.vwap_factory",
        "fn": "vwap_factory_signal",
        "params": _BASE_PARAMS,
        "mode": mode,
        "lookback": lookback,
        "threshold": threshold,
        "entry_max": entry_max,
        "name": name,
    }
    if direction_flip:
        cfg["direction_flip"] = True
    if take_profit_pct is not None:
        cfg["take_profit_pct"] = float(take_profit_pct)
    if stop_loss_pct is not None:
        cfg["stop_loss_pct"] = float(stop_loss_pct)
    if trailing_stop_pct is not None:
        cfg["trailing_stop_pct"] = float(trailing_stop_pct)
    if max_hold_sec is not None:
        cfg["max_hold_sec"] = float(max_hold_sec)
    if min_rem_sec is not None:
        cfg["min_rem_sec"] = float(min_rem_sec)
    if vwap_reversion:
        cfg["vwap_reversion"] = True
    cfg.update(extra)
    VWAP_STRATEGIES[name] = cfg


# ---------------------------------------------------------------------------
# Reduced 48-config candidate set (Phase 1)
# 12 base configs × 4 exit variants: _hold, _tp3_sl2, _trail2_sl2, _rev90
# ---------------------------------------------------------------------------
_BASE_CONFIGS = [
    # family, mode, lookback, threshold, extra
    ("btc_fade_lb60_th100", "btc_vwap_fade", 60, 0.10, {}),
    ("btc_trend_lb60_cf2", "btc_vwap_trend", 60, 0.10, {"confirmation": 2}),
    ("pm_fade_lb60_th15", "pm_vwap_fade", 60, 1.5, {"use_pm": True}),
    ("pm_trend_lb60_cf2", "pm_vwap_trend", 60, 1.0, {"use_pm": True, "confirmation": 2}),
    ("pm_div_lb120_th100", "pm_vwap_divergence", 120, 1.0, {}),
    ("book_fade_lb60_th100", "book_vwap_fade", 60, 1.0, {"use_book": True}),
    ("book_imb_lb60_th4", "book_imbalance_vwap", 60, 4.0, {"use_book": True}),
    ("book_imb_lb60_th4_inv", "book_imbalance_vwap", 60, 4.0, {"use_book": True, "direction_flip": True}),
    ("ofi_lb60_th4", "vwap_ofi", 60, 4.0, {"use_book": True}),
    ("ofi_lb60_th4_inv", "vwap_ofi", 60, 4.0, {"use_book": True, "direction_flip": True}),
    ("ladder_lb60_th100", "vwap_ladder", 60, 1.0, {}),
    ("regime_lb120_th150", "vwap_regime_flip", 120, 1.5, {"flip_threshold": 0.0001}),
]

_VARIANTS = {
    "_hold": {},
    "_tp3_sl2": {"take_profit_pct": 3.0, "stop_loss_pct": 2.0},
    "_trail2_sl2": {"trailing_stop_pct": 2.0, "stop_loss_pct": 2.0},
    "_rev90": {"vwap_reversion": True, "max_hold_sec": 90},
}

for _base_name, _mode, _lb, _th, _extra in _BASE_CONFIGS:
    for _suffix, _rules in _VARIANTS.items():
        _cfg_extra = {**_extra, **_rules, "entry_max": 0.80, "cooldown_ticks": 5}
        _add(
            f"phase_2.vwap_{_base_name}{_suffix}",
            _mode,
            lookback=_lb,
            threshold=_th,
            **_cfg_extra,
        )



# Inverse counterparts for price-based modes that were negative on the sample
_INV_BASE_CONFIGS = [
    ("btc_trend_lb60_cf2_inv", "btc_vwap_trend", 60, 0.10, {"confirmation": 2, "direction_flip": True}),
    ("pm_trend_lb60_cf2_inv", "pm_vwap_trend", 60, 1.0, {"use_pm": True, "confirmation": 2, "direction_flip": True}),
    ("pm_div_lb120_th100_inv", "pm_vwap_divergence", 120, 1.0, {"direction_flip": True}),
    ("regime_lb120_th150_inv", "vwap_regime_flip", 120, 1.5, {"flip_threshold": 0.0001, "direction_flip": True}),
]

for _base_name, _mode, _lb, _th, _extra in _INV_BASE_CONFIGS:
    for _suffix, _rules in _VARIANTS.items():
        _cfg_extra = {**_extra, **_rules, "entry_max": 0.80, "cooldown_ticks": 5}
        _add(
            f"phase_2.vwap_{_base_name}{_suffix}",
            _mode,
            lookback=_lb,
            threshold=_th,
            **_cfg_extra,
        )

assert len(VWAP_STRATEGIES) == 64, f"expected 64 VWAP strategies, got {len(VWAP_STRATEGIES)}"

# ---------------------------------------------------------------------------
# Full-grid expansion recipe (uncomment and tune after Phase 1)
# ---------------------------------------------------------------------------
# lookbacks = [30, 60, 90, 120]
# btc_thresholds = [0.05, 0.10, 0.15, 0.20]
# pm_thresholds = [0.5, 1.0, 1.5, 2.0]
# div_thresholds = [0.3, 0.5, 0.75, 1.0]
# book_thresholds = [0.5, 1.0, 1.5, 2.0]
# imbalance_thresholds = [2, 3, 4, 5]
# entry_maxes = [0.75, 0.80, 0.85]
# cooldowns = [2, 3, 5, 10]
# confirmations = [2, 3, 4]
# band_ns = [0.3, 0.5, 0.7]
# orb_windows = [30, 60, 120]
#
# exit_variants = {
#     "_hold": {},
#     "_tp3_sl2": {"take_profit_pct": 3.0, "stop_loss_pct": 2.0},
#     "_trail2_sl2": {"trailing_stop_pct": 2.0, "stop_loss_pct": 2.0},
#     "_rev90": {"vwap_reversion": True, "max_hold_sec": 90},
# }
