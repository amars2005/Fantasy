"""Fantasy scoring.

This module is load-bearing: every projection, replacement level and VOR number
downstream is denominated in the points this file produces. It is deliberately a
pure function of a stat line with no I/O, so it can be tested exhaustively.
"""

from __future__ import annotations

import polars as pl

from src.config import FUMBLE_LOST_COLUMNS, SCORING


def score_line(stats: dict, scoring: dict | None = None) -> float:
    """Score a single stat line (a dict of stat -> value).

    Missing stats count as zero, so partial lines (a WR with no passing stats)
    score correctly without the caller having to pad them.
    """
    rules = scoring or SCORING
    total = 0.0
    for stat, points in rules.items():
        total += float(stats.get(stat, 0) or 0) * points
    return total


def add_fantasy_points(
    df: pl.DataFrame,
    scoring: dict | None = None,
    column: str = "fantasy_points_league",
) -> pl.DataFrame:
    """Add a league-scored fantasy points column to an nflverse stats frame.

    Handles the two nflverse quirks that bite here: fumbles lost are split
    across three columns, and any given stats frame may be missing columns
    entirely (older seasons, or position-filtered subsets).
    """
    rules = dict(scoring or SCORING)
    present = set(df.columns)

    # Collapse nflverse's three fumble-lost columns into the single stat the
    # scoring rules name.
    fumble_cols = [c for c in FUMBLE_LOST_COLUMNS if c in present]
    if fumble_cols:
        fumbles = pl.sum_horizontal(
            [pl.col(c).fill_null(0) for c in fumble_cols]
        )
    else:
        fumbles = pl.lit(0.0)

    terms = []
    for stat, points in rules.items():
        if stat == "fumbles_lost":
            terms.append(fumbles * points)
        elif stat in present:
            terms.append(pl.col(stat).fill_null(0) * points)
        # A stat absent from the frame contributes nothing.

    expr = pl.sum_horizontal(terms) if terms else pl.lit(0.0)
    return df.with_columns(expr.cast(pl.Float64).alias(column))
