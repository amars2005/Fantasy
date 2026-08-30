"""Kicker and defence projections.

These two are 2 of the 15 roster spots and were missing from the board entirely,
which caused a live bug rather than merely an omission: roughly 28 kickers and
defences come off the board in a 210-pick draft, and a pick that cannot be marked
makes the pick counter lag reality. Everything keyed to "how many picks until my
next turn" -- which is the whole basis of VONA -- then drifts.

Projections use the same method as the skill positions: fit positional ADP rank
against what players at that rank actually scored, per this league's rules. The
resulting numbers are honest but nearly flat, which is the correct answer.
Kicker points have a year-over-year correlation of 0.109; defences reach 0.276,
which is why a defence is worth a real pick and a kicker is not.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from src.config import KICKER_SCORING, SEASON
from src.dst import season_dst_points
from src.ingest import nflverse as nv
from src.ingest.adp import load_adp
from src.ingest.ids import normalise_team
from src.scoring import add_fantasy_points

FIT_SEASONS = list(range(2017, 2026))


def _kicker_season_points(seasons: list[int]) -> pl.DataFrame:
    frames = []
    for season in seasons:
        weekly = nv.player_stats(seasons=[season], level="week")
        if "season_type" in weekly.columns:
            weekly = weekly.filter(pl.col("season_type") == "REG")
        scored = add_fantasy_points(
            weekly.filter(pl.col("position") == "K"), scoring=KICKER_SCORING
        )
        frames.append(
            scored.group_by("player_id")
            .agg(
                pl.col("player_display_name").first().alias("name"),
                pl.col("team").last().alias("team"),
                pl.col("fantasy_points_league").sum().alias("points"),
                pl.len().alias("games"),
            )
            .with_columns(pl.lit(season).cast(pl.Int32).alias("season"))
        )
    return pl.concat(frames)


def _fit_curve(ranks: np.ndarray, points: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(ranks, points)
    return model


def _curve_for(history: pl.DataFrame, rank_col: str = "rank") -> dict:
    ranked = history.with_columns(
        pl.col("points").rank("ordinal", descending=True).over("season").alias(rank_col)
    )
    ranks = ranked[rank_col].to_numpy().astype(float)
    points = ranked["points"].to_numpy().astype(float)
    model = _fit_curve(ranks, points)
    grid = np.arange(1, int(ranks.max()) + 1, dtype=float)
    mean = model.predict(grid)
    resid = points - model.predict(ranks)
    return {"grid": grid, "mean": mean, "sd": float(resid.std())}


def project_kdst(season: int = SEASON) -> pl.DataFrame:
    """K and DST rows for the draft board, in the projections contract."""
    adp = load_adp(year=season)
    kdst = adp.filter(pl.col("position").is_in(["PK", "DEF"])).with_columns(
        pl.when(pl.col("position") == "PK").then(pl.lit("K")).otherwise(pl.lit("DST")).alias("pos"),
        pl.col("team").map_elements(normalise_team, return_dtype=pl.Utf8).alias("tm"),
    )
    if kdst.height == 0:
        return pl.DataFrame()

    k_curve = _curve_for(_kicker_season_points(FIT_SEASONS))
    d_hist = season_dst_points(FIT_SEASONS).select("points", "season")
    d_curve = _curve_for(d_hist)

    frames = []
    for pos, curve in (("K", k_curve), ("DST", d_curve)):
        sub = kdst.filter(pl.col("pos") == pos)
        if sub.height == 0:
            continue
        sub = sub.with_columns(pl.col("adp").rank("ordinal").alias("pos_rank"))
        idx = np.clip(sub["pos_rank"].to_numpy().astype(int) - 1, 0, len(curve["grid"]) - 1)
        frames.append(
            sub.select(
                # Team defences have no player id, so key them on position and team.
                (pl.lit(f"{pos}-") + pl.col("tm")).alias("player_id"),
                "name", "pos", "tm", "adp",
                pl.col("stdev"), pl.col("bye"), "pos_rank",
            ).with_columns(
                pl.Series("proj_points", curve["mean"][idx]).round(1),
                pl.lit(curve["sd"]).round(1).alias("sd"),
                pl.lit(16.0).alias("games"),
                pl.lit("consensus").alias("source"),
            )
        )
    return pl.concat(frames) if frames else pl.DataFrame()
