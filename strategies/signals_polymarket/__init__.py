# CHANGE_SUMMARY
# 2026-08-14  coder
#   - ARCHIVED package: the original Polymarket binary-contract edition of the
#     four StarTrading signals (YES/NO direction, yes_ask/no_ask/yp/np_val
#     entry, 0.85 price cap, rem_sec time-gate).
#   - Kept verbatim for reference / potential Polymarket testing; the active
#     futures edition lives in strategies/signals/.
# WHY: Split the binary-contract adaptation out of the futures-native suite;
#      see docs/FUTURES_VS_POLYMARKET.md.
#
# 2026-08-14  kilo
#   - Created signals/__init__.py exporting the four Blueprint signal functions
#     from the compartmentalized package.
# WHY: Compartmentalization; see docs/BLUEPRINTS.md.

"""The four StarTrading Blueprint signal functions (ARCHIVED Polymarket edition).

Direction vocabulary is YES/NO; entry uses binary yes_ask/no_ask with a 0.85
price cap and a rem_sec time-gate.  This is the original Polymarket-flavored
contract, kept only as a reference.  Use strategies/signals/ (LONG/SHORT,
market entry) for futures backtesting.
"""

from .fifteen_min_range_scalp import fifteen_min_range_scalp
from .negative_rr_consolidation_sweeper import negative_rr_consolidation_sweeper
from .mos_session_daily_draw import mos_session_daily_draw
from .post_8am_bpr_magnet import post_8am_bpr_magnet

__all__ = [
    "fifteen_min_range_scalp",
    "negative_rr_consolidation_sweeper",
    "mos_session_daily_draw",
    "post_8am_bpr_magnet",
]
