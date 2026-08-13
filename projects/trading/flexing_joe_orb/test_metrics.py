"""Quick smoke tests for metrics.py."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from flexing_joe_orb.metrics import (
    compute_deflated_sharpe_ratio,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    compute_sortino,
    compute_true_benjamini_hochberg_fdr,
    compute_true_whites_reality_check,
    compute_win_rate,
    summarize_metrics,
)
from flexing_joe_orb.models import Trade


def make_trade(net_pnl: float, direction: int = 1) -> Trade:
    ts = datetime(2024, 1, 1, 10, 0, 0)
    return Trade(
        entry_time=ts,
        exit_time=ts,
        direction=direction,
        entry_price=100.0,
        exit_price=100.0 + net_pnl / 20.0 * direction,
        contracts=1,
        gross_pnl=net_pnl,
        commission=0.0,
        slippage=0.0,
        net_pnl=net_pnl,
        exit_reason="TP",
    )


def test_basic_metrics():
    pnls = np.array([100.0, -50.0, 75.0, -25.0, 200.0])
    assert compute_win_rate(pnls) == 60.0
    assert compute_profit_factor(pnls) == 375.0 / 75.0
    assert compute_sharpe(pnls, trades_per_day=1.0) > 0.0
    assert compute_sortino(pnls, trades_per_day=1.0) > 0.0


def test_max_drawdown():
    equity = np.array([100.0, 110.0, 105.0, 115.0, 100.0])
    dd, peak, trough = compute_max_drawdown(equity)
    assert dd == 15.0 / 115.0 * 100.0
    assert peak == 3
    assert trough == 4


def test_dsr():
    dsr = compute_deflated_sharpe_ratio(
        sharpe=1.5,
        total_trades=100,
        skew=-0.5,
        kurt=3.0,
        num_trials=100,
    )
    assert 0.0 <= dsr <= 1.0


def test_wrc_fdr():
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.001, 0.02, size=(3, 50))
    pvals = compute_true_whites_reality_check(matrix, num_bootstraps=500)
    assert len(pvals) == 3
    qvals = compute_true_benjamini_hochberg_fdr(pvals)
    assert len(qvals) == 3
    assert np.all((qvals >= 0.0) & (qvals <= 1.0))


def test_summarize_metrics():
    trades = [
        make_trade(100.0),
        make_trade(-50.0),
        make_trade(75.0),
        make_trade(200.0),
    ]
    daily_pnl = {"2024-01-01": 325.0}
    summary = summarize_metrics(trades, daily_pnl)
    assert summary["total_trades"] == 4
    assert summary["win_rate"] == 75.0
    assert summary["net_pnl"] == 325.0
    assert summary["trades_per_day"] == 4.0
    assert "wrc_pvalue" in summary
    assert "fdr_qvalue" in summary
    print("Summary:", summary)


def test_empty_trades():
    summary = summarize_metrics([], {})
    assert summary["total_trades"] == 0
    assert summary["wrc_pvalue"] == 0.5


if __name__ == "__main__":
    test_basic_metrics()
    test_max_drawdown()
    test_dsr()
    test_wrc_fdr()
    test_summarize_metrics()
    test_empty_trades()
    print("All metrics tests passed.")
