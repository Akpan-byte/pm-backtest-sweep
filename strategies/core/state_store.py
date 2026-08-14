# CHANGE_SUMMARY
# 2026-08-14  kilo
#   - Created core/state_store.py, a generic per-(asset,date,variant,max_reentries)
#     state store. Encapsulates key generation, atomic disk persistence
#     (default=str for date/datetime), load, and 2-day prune so the four signal
#     modules no longer duplicate this boilerplate.
# WHY: Granular compartmentalization; see docs/BLUEPRINTS.md.

"""Generic on-disk state store for the StarTrading strategies.

Each strategy owns one StateStore keyed by its variant name.  State is a plain
dict; date/datetime fields are serialized with default=str (they are audit-only
and not used in recomputation, so string form on reload is safe).
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("strategies.core.state_store")

EST = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

DEFAULT_RETENTION_DAYS = 2


class StateStore:
    def __init__(self, variant: str, project_root: Path, retention_days: int = DEFAULT_RETENTION_DAYS):
        self.variant = variant
        self.dir = project_root / "state" / variant
        self.retention_days = retention_days
        self._mem: dict = {}

    # ----- key handling -----
    def make_key(self, asset: str, date, max_reentries) -> tuple:
        return (asset.upper(), date, self.variant, max_reentries)

    def _path(self, key) -> Path:
        asset, today, variant, mx = key
        return self.dir / f"{asset}_{today.isoformat()}_{variant}_{mx}.json"

    # ----- lifecycle -----
    def load_or_new(self, key, make_state_fn) -> dict:
        if key in self._mem:
            return self._mem[key]
        loaded = self._load_file(key)
        if loaded is not None:
            self._mem[key] = loaded
            return loaded
        st = make_state_fn()
        self._mem[key] = st
        return st

    def save(self, key, state: dict) -> None:
        try:
            if not state or state.get("today") is None:
                return
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self._path(key)
            payload = {
                "asset": key[0],
                "date": key[1].isoformat(),
                "variant": key[2],
                "max_reentries": key[3],
                "state": state,
            }
            fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix=".tmp_", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, default=str)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        except Exception as exc:  # persistence must never break trading
            log.warning("%s: save failed %s: %s", self.variant, key, exc)

    def _load_file(self, key) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            state = payload.get("state", {})
            state["today"] = key[1]
            return state
        except Exception as exc:
            log.warning("%s: load failed %s: %s", self.variant, key, exc)
            return None

    def prune(self, asset: str, today) -> None:
        cutoff = today - timedelta(days=self.retention_days)
        stale = [k for k in self._mem if k[0] == asset and k[1] < cutoff]
        for k in stale:
            del self._mem[k]

    def tick_cooldowns(self, state: dict) -> None:
        """Decrement per-direction cooldown counters (called each tick).

        Direction-agnostic: iterates whatever keys are present so both the
        futures (LONG/SHORT) and archived Polymarket (YES/NO) editions work.
        """
        for d in list(state.get("cooldown", {})):
            if state["cooldown"][d] > 0:
                state["cooldown"][d] -= 1
