"""Durable draft state.

The board previously kept every pick in a dictionary in memory, which meant a
closed terminal, a slept laptop or a stray Ctrl+C threw away the whole draft with
no way back. That is a poor trade in any case, and a bad one here: this league
allows three hours per pick, so a draft runs over days rather than an evening.

State is written after every mutation and reloaded on startup. Writes go to a
temporary file and are then renamed, so a crash midway through a save leaves the
previous good file intact rather than a truncated one.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_PROCESSED, LEAGUE

STATE_DIR = DATA_PROCESSED / "draft_state"


def state_path(slot: int, season: int) -> Path:
    return STATE_DIR / f"draft_{season}_slot{slot}.json"


def save(slot: int, season: int, drafted: dict[str, str]) -> Path:
    """Write draft state atomically."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path(slot, season)
    payload = {
        "season": season,
        "slot": slot,
        "teams": LEAGUE["teams"],
        "rounds": LEAGUE["rounds"],
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "picks": len(drafted),
        "drafted": drafted,
    }
    # Write-then-rename: a crash mid-write cannot corrupt the existing state.
    fd, tmp = tempfile.mkstemp(dir=str(STATE_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load(slot: int, season: int) -> tuple[dict[str, str], dict]:
    """Return (drafted, metadata). Empty dict when there is nothing to resume."""
    path = state_path(slot, season)
    if not path.exists():
        return {}, {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}, {}

    # Refuse state saved under a different league shape rather than silently
    # resuming a draft whose pick numbers no longer mean the same thing.
    if payload.get("teams") != LEAGUE["teams"] or payload.get("slot") != slot:
        return {}, {"mismatch": True, "saved_at": payload.get("saved_at")}

    drafted = payload.get("drafted") or {}
    return dict(drafted), {
        "saved_at": payload.get("saved_at"),
        "picks": len(drafted),
        "path": str(path),
    }


def clear(slot: int, season: int) -> None:
    state_path(slot, season).unlink(missing_ok=True)
