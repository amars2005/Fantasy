"""Team defence scoring and projection.

D/ST is a required starter in this league and had gone entirely unexamined. The
scoring is unusually detailed -- banded on both points and yards allowed -- so it
is worth computing exactly rather than approximating, then asking the only
question that matters for the draft: does any of it persist year to year?

Counting stats come from nflverse team stats. Points and yards allowed are not
in that feed directly; both are derived from the opponent's own row.
"""

from __future__ import annotations

import polars as pl

# Per the league settings.
DST_EVENT_SCORING = {
    "def_sacks": 1.0,
    "def_interceptions": 2.0,
    "def_fumbles": 2.0,          # fumble recoveries
    "def_safeties": 2.0,
    "def_punt_blocks": 2.0,
    "def_pat_blocks": 2.0,
    "def_fg_blocks": 2.0,
    "def_tds": 6.0,
}

# Two defensive scores nflverse's team frame does not count, and this league
# does not pay -- but ESPN prices both, so a league that imports from it needs
# the columns to exist. Derived from the same play-by-play pass that produces
# the long-touchdown bands; see `scripts/export_v2_bundle.py`.
#
# They are rare to the point of being almost theoretical: two defensive
# two-point returns league-wide across 2022-24, and no one-point safety at all.
# Carrying them costs a column each and means the import screen can stop
# warning about a bonus whose true contribution is zero.
DST_PBP_EVENTS = ("def_two_point_returns", "def_one_point_safeties")

POINTS_ALLOWED_BANDS = [
    (0, 0, 5.0), (1, 6, 4.0), (7, 13, 3.0), (14, 17, 1.0),
    (18, 27, 0.0), (28, 34, -1.0), (35, 45, -3.0), (46, 999, -5.0),
]
YARDS_ALLOWED_BANDS = [
    (0, 99, 5.0), (100, 199, 3.0), (200, 299, 2.0), (300, 349, 0.0),
    (350, 399, -1.0), (400, 449, -3.0), (450, 499, -5.0),
    (500, 549, -6.0), (550, 99999, -7.0),
]


def _band_expr(column: str, bands: list[tuple[int, int, float]]) -> pl.Expr:
    expr = pl.lit(0.0)
    for low, high, points in reversed(bands):
        expr = (
            pl.when((pl.col(column) >= low) & (pl.col(column) <= high))
            .then(pl.lit(points))
            .otherwise(expr)
        )
    return expr


def weekly_dst_points(seasons: list[int]) -> pl.DataFrame:
    """Weekly D/ST fantasy points under this league's rules."""
    import nflreadpy as nfl

    from src.ingest.cache import cached

    def load() -> pl.DataFrame:
        team = nfl.load_team_stats(seasons=seasons, summary_level="week")
        sched = nfl.load_schedules(seasons=seasons)

        # Points allowed: what the opponent scored in that game.
        home = sched.select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            pl.col("home_team").alias("team"),
            pl.col("away_score").cast(pl.Float64).alias("points_allowed"),
        )
        away = sched.select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            pl.col("away_team").alias("team"),
            pl.col("home_score").cast(pl.Float64).alias("points_allowed"),
        )
        allowed = pl.concat([home, away]).drop_nulls("points_allowed")

        # Yards allowed: the opponent's own offensive yardage that week.
        yard_cols = [c for c in ("passing_yards", "rushing_yards") if c in team.columns]
        offense = team.select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            pl.col("team").alias("opponent_team"),
            pl.sum_horizontal([pl.col(c).fill_null(0) for c in yard_cols])
            .cast(pl.Float64).alias("yards_allowed"),
        )

        events = [
            (pl.col(stat).fill_null(0) * pts)
            for stat, pts in DST_EVENT_SCORING.items()
            if stat in team.columns
        ]
        base = team.select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            "team", "opponent_team",
            pl.sum_horizontal(events).cast(pl.Float64).alias("event_points"),
        )

        return (
            base.join(allowed, on=["season", "week", "team"], how="left")
            .join(offense, on=["season", "week", "opponent_team"], how="left")
            .drop_nulls(["points_allowed", "yards_allowed"])
            .with_columns(
                _band_expr("points_allowed", POINTS_ALLOWED_BANDS).alias("pa_points"),
                _band_expr("yards_allowed", YARDS_ALLOWED_BANDS).alias("ya_points"),
            )
            .with_columns(
                (pl.col("event_points") + pl.col("pa_points") + pl.col("ya_points"))
                .alias("dst_points")
            )
        )

    return cached(f"dst_weekly_{min(seasons)}_{max(seasons)}", load)


def season_dst_points(seasons: list[int]) -> pl.DataFrame:
    weekly = weekly_dst_points(seasons)
    return (
        weekly.group_by(["team", "season"])
        .agg(
            pl.col("dst_points").sum().alias("points"),
            pl.col("event_points").sum().alias("event_points"),
            pl.col("pa_points").sum().alias("pa_points"),
            pl.col("ya_points").sum().alias("ya_points"),
            pl.len().alias("games"),
        )
        .sort("points", descending=True)
    )
