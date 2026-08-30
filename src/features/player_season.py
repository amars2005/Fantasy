"""Season-level player production, opportunity and context.

One row per player-season, carrying everything we know about what a player did
and how much work he was given. This is the substrate the model's lagged features
are built from.

The distinction that matters here is between *production* (fantasy points, which
are noisy) and *opportunity* (targets, carries, expected points, which are far
more stable year to year). Keeping expected points alongside actual points also
gives us the single most useful regression signal in fantasy: how much a player
outscored the work he was actually given.

Everything is processed and cached one season at a time. The weekly frames are
the memory hog in this pipeline -- ffopportunity alone is 159 columns -- and
holding a decade of them at once does not fit in a typical laptop's free RAM.
"""

from __future__ import annotations

import gc

import polars as pl

from src.config import HISTORY_SEASONS
from src.ingest import nflverse as nv
from src.ingest.cache import cache_path, cached
from src.scoring import add_fantasy_points

SKILL = ["QB", "RB", "WR", "TE"]

OPPORTUNITY_KEEP = (
    "total_fantasy_points_exp",
    "total_yards_gained_exp",
    "total_touchdown_exp",
    "receptions_exp",
)


def _opportunity_season(season: int) -> pl.DataFrame:
    """Expected fantasy points for one season, aggregated then discarded."""
    import nflreadpy as nfl

    def load() -> pl.DataFrame:
        weekly = nfl.load_ff_opportunity(seasons=[season], stat_type="weekly")
        keep = [c for c in OPPORTUNITY_KEEP if c in weekly.columns]
        out = (
            weekly.filter(pl.col("position").is_in(SKILL))
            # ffopportunity types season as a string; nflverse stats use an int.
            .with_columns(pl.col("season").cast(pl.Int32))
            .group_by(["player_id", "season"])
            .agg([pl.col(c).sum().alias(c) for c in keep])
        )
        del weekly
        gc.collect()
        return out

    return cached(f"opportunity_season_{season}", load)


def _season(season: int) -> pl.DataFrame:
    """Aggregate one season of weekly stats into player-season rows."""

    def load() -> pl.DataFrame:
        weekly = nv.player_stats(seasons=[season], level="week")
        if "season_type" in weekly.columns:
            weekly = weekly.filter(pl.col("season_type") == "REG")
        scored = add_fantasy_points(weekly)
        pos_col = "position" if "position" in scored.columns else "position_group"

        out = (
            scored.filter(pl.col(pos_col).is_in(SKILL))
            .group_by(["player_id", "season"])
            .agg(
                pl.col("player_display_name").first().alias("name"),
                pl.col(pos_col).first().alias("pos"),
                pl.col("team").last().alias("team"),
                pl.col("fantasy_points_league").sum().alias("points"),
                pl.len().alias("games"),
                pl.col("targets").sum().alias("targets"),
                pl.col("carries").sum().alias("carries"),
                pl.col("receptions").sum().alias("receptions"),
                pl.col("attempts").sum().alias("pass_attempts"),
                pl.col("receiving_yards").sum().alias("rec_yards"),
                pl.col("rushing_yards").sum().alias("rush_yards"),
                pl.col("target_share").mean().alias("target_share"),
                pl.col("air_yards_share").mean().alias("air_yards_share"),
            )
            .with_columns(pl.col("season").cast(pl.Int32))
        )
        del weekly, scored
        gc.collect()
        return out

    base = cached(f"player_season_{season}", load)

    base = base.with_columns(
        (pl.col("points") / pl.col("games")).alias("ppg"),
        (pl.col("targets") / pl.col("games")).alias("targets_pg"),
        (pl.col("carries") / pl.col("games")).alias("carries_pg"),
        ((pl.col("targets") + pl.col("carries")) / pl.col("games")).alias("touches_pg"),
        # Weighted opportunity rating: the standard blend of volume and depth.
        (1.5 * pl.col("target_share").fill_null(0)
         + 0.7 * pl.col("air_yards_share").fill_null(0)).alias("wopr"),
    )

    opp = _opportunity_season(season)
    base = base.join(opp, on=["player_id", "season"], how="left")
    if "total_fantasy_points_exp" in base.columns:
        base = base.with_columns(
            # Points scored above the opportunity actually given. Large positive
            # values are the classic regression candidates: touchdown luck that
            # the following season does not repeat.
            (pl.col("points") - pl.col("total_fantasy_points_exp")).alias("points_oe"),
            (pl.col("total_fantasy_points_exp") / pl.col("games")).alias("exp_ppg"),
        )
    return base


def build(seasons: list[int] | None = None) -> pl.DataFrame:
    """One row per player-season with production, opportunity and team context."""
    seasons = seasons or HISTORY_SEASONS
    frames = []
    for season in seasons:
        frames.append(_season(season))
        gc.collect()
    return pl.concat(frames, how="diagonal_relaxed")


def team_volume(seasons: list[int] | None = None) -> pl.DataFrame:
    """Team-season passing and rushing volume, for computing vacated opportunity."""
    ps = build(seasons or HISTORY_SEASONS)
    return ps.group_by(["team", "season"]).agg(
        pl.col("targets").sum().alias("team_targets"),
        pl.col("carries").sum().alias("team_carries"),
    )
