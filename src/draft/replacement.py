"""Replacement level for a specific league shape.

Replacement level is the whole ballgame in a 14-team league: it is what converts
"projected points" (which says QBs are the most valuable players in football)
into "value over replacement" (which says they are not, because the 14th QB is
nearly as good as the 3rd).

We derive it rather than hardcoding it. Base starters are known from the league
config; the FLEX slots are allocated to whichever RB/WR/TE actually deserve them
given the projections, so replacement level responds to the shape of this year's
player pool instead of a rule of thumb.
"""

from __future__ import annotations

import polars as pl

from src.config import LEAGUE


def allocate_flex(proj: pl.DataFrame, league: dict | None = None) -> dict[str, int]:
    """Return the number of *starters* at each position, flex included.

    Base starters are filled first. The remaining flex slots go to the best
    players left across all flex-eligible positions, which is what managers
    collectively do.
    """
    league = league or LEAGUE
    teams = league["teams"]
    starters = league["starters"]
    eligible = league["flex_eligible"]

    counts = {pos: starters.get(pos, 0) * teams for pos in proj["pos"].unique()}

    flex_slots = starters.get("FLEX", 0) * teams
    if flex_slots:
        # Everyone at a flex-eligible position who is not already a base starter,
        # ranked by projection; the top `flex_slots` of them take the flex spots.
        pool = (
            proj.filter(pl.col("pos").is_in(eligible))
            .with_columns(
                pl.col("proj_points").rank("ordinal", descending=True).over("pos").alias("r")
            )
            .filter(pl.col("r") > pl.col("pos").replace_strict(counts, default=0))
            .sort("proj_points", descending=True)
            .head(flex_slots)
        )
        for pos, n in pool.group_by("pos").len().iter_rows():
            counts[pos] = counts.get(pos, 0) + n

    return counts


def replacement_levels(proj: pl.DataFrame, league: dict | None = None) -> dict[str, float]:
    """Projected points of the best non-starter at each position."""
    counts = allocate_flex(proj, league)
    levels: dict[str, float] = {}

    for pos in proj["pos"].unique().to_list():
        ranked = proj.filter(pl.col("pos") == pos).sort("proj_points", descending=True)
        idx = counts.get(pos, 0)  # 0-based index of the first non-starter
        if ranked.height == 0:
            continue
        if idx >= ranked.height:
            # Pool is shallower than the number of starting slots: replacement is
            # the worst rostered player, i.e. effectively nothing is free here.
            levels[pos] = float(ranked["proj_points"][-1])
        else:
            levels[pos] = float(ranked["proj_points"][idx])
    return levels


def add_vor(proj: pl.DataFrame, league: dict | None = None) -> pl.DataFrame:
    """Attach value over replacement, and rank the board by it."""
    levels = replacement_levels(proj, league)
    return (
        proj.with_columns(
            pl.col("pos").replace_strict(levels, default=0.0).alias("replacement")
        )
        .with_columns((pl.col("proj_points") - pl.col("replacement")).alias("vor"))
        .with_columns(pl.col("vor").rank("ordinal", descending=True).alias("vor_rank"))
        .sort("vor", descending=True)
    )
