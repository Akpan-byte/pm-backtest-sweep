# CHANGE_SUMMARY
# 2026-08-14  coder
#   - Updated package docstring: this package now holds the FUTURES /
#     normal-trading signal functions (LONG/SHORT, market-price entry).
#   - The Polymarket-flavored originals live in strategies/signals_polymarket/.
# WHY: Split the binary-contract adaptation out of the futures-native suite.
#
# 2026-08-14  kilo
#   - Created signals/__init__.py exporting the four Blueprint signal functions
#     from the compartmentalized package.
# WHY: Compartmentalization; see docs/BLUEPRINTS.md.

"""The four StarTrading Blueprint signal functions (FUTURES edition).

Direction vocabulary is LONG/SHORT; entry is at the market price.  The
Polymarket binary-contract adaptation is archived in strategies/signals_polymarket.
"""

from .fifteen_min_range_scalp import fifteen_min_range_scalp
from .negative_rr_consolidation_sweeper import negative_rr_consolidation_sweeper
from .mos_session_daily_draw import mos_session_daily_draw
from .post_8am_bpr_magnet import post_8am_bpr_magnet
from .orb_vwap import orb_vwap
from .vwap_sd_reversion import vwap_sd_reversion

__all__ = [
    "fifteen_min_range_scalp",
    "negative_rr_consolidation_sweeper",
    "mos_session_daily_draw",
    "post_8am_bpr_magnet",
    "orb_vwap",
    "vwap_sd_reversion",
]
