"""Historical season-total fantasy points, in this league's scoring.

Used to fit the ADP-rank -> expected-points curve and to backtest anything that
claims to project points.
"""

from __future__ import annotations

import polars as pl

from src.config import HISTORY_SEASONS
from src.ingest import nflverse as nv
from src.scoring import add_fantasy_points

SKILL = ["QB", "RB", "WR", "TE"]


def season_totals(seasons: list[int] | None = None) -> pl.DataFrame:
    """Regular-season fantasy point totals per player-season.

    Weekly rows are scored with our league rules and summed, so a player who
    missed games carries the cost of those games -- which is exactly what a
    draft-day expectation should reflect.
    """
    seasons = seasons or HISTORY_SEASONS
    weekly = nv.player_stats(seasons=seasons, level="week")

    # Regular season only; playoff weeks are irrelevant to fantasy scoring.
    if "season_type" in weekly.columns:
        weekly = weekly.filter(pl.col("season_type") == "REG")

    scored = add_fantasy_points(weekly)

    pos_col = "position" if "position" in scored.columns else "position_group"
    return (
        scored.group_by(["player_id", "season"])
        .agg(
            pl.col("player_display_name").first().alias("name"),
            pl.col(pos_col).first().alias("pos"),
            pl.col("team").last().alias("team"),
            pl.col("fantasy_points_league").sum().alias("points"),
            pl.len().alias("games"),
        )
        .filter(pl.col("pos").is_in(SKILL))
    )
