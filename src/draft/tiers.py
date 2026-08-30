"""Tiering.

At the table you rarely need to know whether Nacua outranks Chase; you need to
know whether the player you want is the last of his kind. Tiers answer that, and
they are what tells you when to reach and when to wait.

The tier boundaries come for free from the projection method. Isotonic regression
pools adjacent ranks it cannot statistically distinguish, so a plateau in the
fitted curve *is* a set of players the historical data says are interchangeable.
We use those plateaus directly rather than imposing a clustering on top.
"""

from __future__ import annotations

import polars as pl


def add_tiers(proj: pl.DataFrame, value_col: str = "proj_points") -> pl.DataFrame:
    """Assign within-position tiers from plateaus in the projection curve."""
    ranked = proj.sort(["pos", value_col], descending=[False, True])
    return (
        ranked.with_columns(
            # A new tier starts wherever projected points actually change.
            (pl.col(value_col) != pl.col(value_col).shift(1).over("pos"))
            .fill_null(True)
            .cast(pl.Int32)
            .cum_sum()
            .over("pos")
            .alias("tier")
        )
        .with_columns(
            pl.len().over(["pos", "tier"]).alias("tier_size"),
            pl.col(value_col).max().over(["pos", "tier"]).alias("tier_points"),
        )
    )


def tier_summary(proj: pl.DataFrame) -> pl.DataFrame:
    """One row per tier: who is in it and where it runs out."""
    return (
        proj.group_by(["pos", "tier"])
        .agg(
            pl.col("proj_points").first().alias("points"),
            pl.len().alias("n"),
            pl.col("adp").min().round(1).alias("first_adp"),
            pl.col("adp").max().round(1).alias("last_adp"),
            pl.col("name").sort_by("adp").alias("players"),
        )
        .sort(["pos", "tier"])
    )


def players_left_in_tier(proj: pl.DataFrame, pos: str, tier: int) -> int:
    return proj.filter((pl.col("pos") == pos) & (pl.col("tier") == tier)).height
