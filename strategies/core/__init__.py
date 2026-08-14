# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Created core/__init__.py exporting the granular time/candle/detector/state
#     modules so strategies import from one namespace.
# WHY: Compartmentalization; see docs/BLUEPRINTS.md.

"""Core building blocks for the StarTrading intraday strategies."""

from . import time_utils, candle_utils, detectors, state_store

__all__ = ["time_utils", "candle_utils", "detectors", "state_store"]
