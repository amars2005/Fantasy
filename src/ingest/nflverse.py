"""Cached nflverse loaders.

Thin wrappers over nflreadpy so the rest of the codebase never worries about
network calls or refresh policy.
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl

from src.config import HISTORY_SEASONS, SEASON
from src.ingest.cache import cached

# Current-season data moves; refresh daily during draft season.
CURRENT_MAX_AGE = 12.0


def player_stats(seasons: list[int] | None = None, level: str = "week") -> pl.DataFrame:
    seasons = seasons or HISTORY_SEASONS
    key = f"player_stats_{level}_{min(seasons)}_{max(seasons)}"
    return cached(key, lambda: nfl.load_player_stats(seasons=seasons, summary_level=level))


def schedules(seasons: list[int] | None = None) -> pl.DataFrame:
    """Game schedules including Vegas spread_line / total_line."""
    seasons = seasons or (HISTORY_SEASONS + [SEASON])
    key = f"schedules_{min(seasons)}_{max(seasons)}"
    return cached(
        key,
        lambda: nfl.load_schedules(seasons=seasons),
        max_age_hours=CURRENT_MAX_AGE,
    )


def ff_rankings() -> pl.DataFrame:
    """FantasyPros expert consensus rankings (updated daily in season)."""
    return cached("ff_rankings", nfl.load_ff_rankings, max_age_hours=CURRENT_MAX_AGE)


def ff_playerids() -> pl.DataFrame:
    """Cross-platform player ID crosswalk."""
    return cached("ff_playerids", nfl.load_ff_playerids, max_age_hours=24 * 7)


def rosters(season: int = SEASON) -> pl.DataFrame:
    return cached(
        f"rosters_{season}",
        lambda: nfl.load_rosters(seasons=[season]),
        max_age_hours=CURRENT_MAX_AGE,
    )


def depth_charts(season: int = SEASON) -> pl.DataFrame:
    return cached(
        f"depth_charts_{season}",
        lambda: nfl.load_depth_charts(seasons=[season]),
        max_age_hours=CURRENT_MAX_AGE,
    )


def snap_counts(seasons: list[int] | None = None) -> pl.DataFrame:
    seasons = seasons or HISTORY_SEASONS
    key = f"snap_counts_{min(seasons)}_{max(seasons)}"
    return cached(key, lambda: nfl.load_snap_counts(seasons=seasons))


def players() -> pl.DataFrame:
    return cached("players", nfl.load_players, max_age_hours=24 * 7)
