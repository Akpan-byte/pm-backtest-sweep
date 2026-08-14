# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Rewrote strategies/__init__.py to expose the compartmentalized package:
#     core/* building blocks and signals/* Blueprint functions. Documentation
#     lives in docs/ (see docs/BLUEPRINTS.md).
# WHY: Compartmentalization of the four StarTrading intraday strategies.

"""StarTrading-style intraday strategy suite (compartmentalized).

Layout:
  strategies/core/     time_utils, candle_utils, detectors, state_store
  strategies/signals/  the four Blueprint signal functions
  strategies/docs/     master blueprints + per-strategy changelogs
"""

from . import core
from . import signals

__all__ = ["core", "signals"]
