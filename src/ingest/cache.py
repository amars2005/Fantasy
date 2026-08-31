"""Disk cache for anything fetched over the network.

Completed seasons never change, so they are cached forever. Anything tied to the
current season (ADP, rankings, rosters) carries a max-age so a stale draft board
is impossible.

Staleness is a softer failure than absence. If a source is unreachable and we
hold an expired copy, serving the expired copy with a warning beats raising --
the alternative on draft morning is no board at all.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable

import polars as pl

from src.config import DATA_RAW


def cached(
    name: str,
    loader: Callable[[], pl.DataFrame],
    max_age_hours: float | None = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Return a cached frame, fetching via `loader` on miss or expiry.

    `max_age_hours=None` means cache forever (use for completed seasons).
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW / f"{name}.parquet"

    if path.exists() and not refresh:
        fresh = max_age_hours is None or (
            (time.time() - path.stat().st_mtime) < max_age_hours * 3600
        )
        if fresh:
            return pl.read_parquet(path)

    try:
        df = loader()
    except Exception as exc:
        # Expired but present beats absent. The caller asked for a refresh and
        # could not have one; hand back what we have and say so, loudly enough
        # to notice but not so loudly that a draft stops.
        if path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600
            warnings.warn(
                f"{name}: refresh failed ({exc.__class__.__name__}); "
                f"serving cached copy {age_h:.1f}h old",
                RuntimeWarning,
                stacklevel=2,
            )
            return pl.read_parquet(path)
        raise
    df.write_parquet(path)
    return df


def cache_path(name: str) -> Path:
    return DATA_RAW / f"{name}.parquet"
