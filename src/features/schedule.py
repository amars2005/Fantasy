"""Schedule strength, with the playoff weeks weighted separately.

Season-long strength of schedule is close to useless for fantasy: over seventeen
games it mostly averages out, and the market has already priced it. The weeks that
are *not* interchangeable are the ones the title is decided in.

This league seeds after fourteen weeks and plays a three-week bracket in NFL weeks
15-17. A player facing soft defences in exactly those three weeks is worth more
than his season projection says, and ADP cannot price that because ADP is
format-blind -- the leagues it is sampled from have different playoff windows.

Note on method: the obvious measure would be the Vegas implied total for those
weeks, but sportsbooks only post lines a few weeks out, so weeks 15-17 are empty
in August when the draft happens. Opponents, however, are known for all 272 games
from the day the schedule is released. So difficulty is measured by how many
points those specific opponents allowed last season.
"""

from __future__ import annotations

import polars as pl

from src.config import SCHEDULE, SEASON
from src.dst import weekly_dst_points
from src.ingest import nflverse as nv

PLAYOFF_WEEKS = list(SCHEDULE["playoff_weeks"])
REGULAR_WEEKS = list(range(1, SCHEDULE["regular_season_weeks"] + 1))


def defensive_strength(season: int) -> pl.DataFrame:
    """Points allowed per game by each defence in a completed season.

    Higher means a softer defence, and therefore a better matchup to face.
    """
    weekly = weekly_dst_points([season])
    return (
        weekly.group_by("team")
        .agg(pl.col("points_allowed").mean().alias("pa_per_game"))
        .rename({"team": "opponent"})
    )


def _matchups(season: int) -> pl.DataFrame:
    """Every team's opponent in every week of a season."""
    sched = nv.schedules([season]).filter(pl.col("season") == season)
    home = sched.select(
        pl.col("week").cast(pl.Int32),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
    )
    away = sched.select(
        pl.col("week").cast(pl.Int32),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
    )
    return pl.concat([home, away]).drop_nulls("opponent")


def team_schedule_strength(season: int = SEASON) -> pl.DataFrame:
    """Matchup difficulty for each team, split by season phase."""
    matchups = _matchups(season)
    strength = defensive_strength(season - 1)
    joined = matchups.join(strength, on="opponent", how="left").drop_nulls("pa_per_game")

    regular = (
        joined.filter(pl.col("week").is_in(REGULAR_WEEKS))
        .group_by("team").agg(pl.col("pa_per_game").mean().alias("sos_regular"))
    )
    playoff = (
        joined.filter(pl.col("week").is_in(PLAYOFF_WEEKS))
        .group_by("team").agg(pl.col("pa_per_game").mean().alias("sos_playoff"))
    )
    return (
        regular.join(playoff, on="team", how="left")
        .with_columns(
            # Positive means the weeks that decide the title are softer than the
            # rest of this team's season.
            (pl.col("sos_playoff") - pl.col("sos_regular")).alias("playoff_lift")
        )
        .sort("playoff_lift", descending=True)
    )


def add_playoff_lift(board: pl.DataFrame, season: int = SEASON) -> pl.DataFrame:
    """Attach each player's team playoff-schedule lift to the board."""
    strength = team_schedule_strength(season).rename({"team": "tm"})
    return board.join(strength, on="tm", how="left").with_columns(
        pl.col("playoff_lift").fill_null(0.0)
    )
