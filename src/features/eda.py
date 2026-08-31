"""Correlation machinery for the year-over-year study.

Three measurements, kept apart on purpose because conflating them is how
fantasy analysis usually goes wrong:

  * **predictive**  -- does last season's value of a stat order next season's
    points?
  * **repeatable**  -- does the stat predict *itself* next season?
  * **incremental** -- does it still order next season's points once you already
    know how many fantasy points the player scored last year?

A stat can be strongly predictive and carry no information at all: `points_lag1`
predicts `points`, and anything correlated with `points_lag1` inherits that
correlation for free. The third column is the one that decides whether a feature
belongs in a model.

Every correlation is computed **within (season, position)** and pooled by group
size. Pooling across positions instead would measure the fact that running backs
carry the ball and receivers do not, which we already know.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

MIN_GROUP = 12  # below this a within-group correlation is noise


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so the result is a Spearman rather than a Pearson."""
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Ties get the mean of the ranks they span.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares residual of y on x, with an intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def grouped_spearman(
    df: pl.DataFrame,
    x: str,
    y: str,
    control: str | None = None,
    groups: tuple[str, ...] = ("season", "pos"),
) -> dict:
    """Spearman of `x` against `y`, pooled over (season, position) groups.

    With `control` set, this is a partial Spearman: both variables are stripped
    of their linear-in-ranks dependence on the control before correlating, which
    answers "what does this add on top of what we already knew".

    The standard error is computed across *seasons*, not across players. Players
    within a season share a scoring environment and an injury year, so treating
    them as independent draws would understate the spread by a factor of three.
    """
    cols = [x, y, *groups] + ([control] if control else [])
    sub = df.select([c for c in dict.fromkeys(cols)]).drop_nulls()

    per_group, weights, per_season = [], [], {}
    for key, chunk in sub.group_by(list(groups)):
        if chunk.height < MIN_GROUP:
            continue
        xv = chunk[x].to_numpy().astype(float)
        yv = chunk[y].to_numpy().astype(float)
        if np.ptp(xv) == 0 or np.ptp(yv) == 0:
            continue

        if control:
            cv = _rank(chunk[control].to_numpy().astype(float))
            if np.ptp(cv) == 0:
                continue
            rx = _residualise(_rank(xv), cv)
            ry = _residualise(_rank(yv), cv)
            # A variable regressed on itself leaves a residual of floating-point
            # dust, and correlating dust with dust returns a confident-looking
            # number. Refuse the group instead.
            if rx.std() < 1e-9 or ry.std() < 1e-9:
                continue
            rho = float(np.corrcoef(rx, ry)[0, 1])
        else:
            rho = float(spearmanr(xv, yv).statistic)
        if np.isnan(rho):
            continue

        per_group.append(rho)
        weights.append(chunk.height)
        season = key[groups.index("season")] if "season" in groups else 0
        per_season.setdefault(season, []).append((rho, chunk.height))

    if not per_group:
        return {"rho": float("nan"), "se": float("nan"), "n": 0, "groups": 0}

    rhos = np.array(per_group)
    w = np.array(weights, dtype=float)
    pooled = float((rhos * w).sum() / w.sum())

    season_means = np.array([
        float(np.average([r for r, _ in v], weights=[n for _, n in v]))
        for v in per_season.values()
    ])
    se = (
        float(season_means.std(ddof=1) / np.sqrt(len(season_means)))
        if len(season_means) > 1 else float("nan")
    )
    return {"rho": pooled, "se": se, "n": int(w.sum()), "groups": len(rhos)}


def yoy_frame(
    stats: pl.DataFrame,
    rosters: pl.DataFrame | None = None,
    min_prior_games: int = 8,
) -> pl.DataFrame:
    """Join each player-season to the season that follows it.

    Two survivorship traps live in this join, and both inflate every correlation
    downstream if you fall into them:

      * Filtering on the *target* season deletes the outcomes the exercise is
        about -- the players who got hurt or lost the job. `min_prior_games`
        therefore filters the prior season only.
      * Inner-joining to the next season's stat line does the same thing more
        quietly: a player who was rostered all year and never dressed simply
        vanishes instead of scoring the zero he actually scored. Pass `rosters`
        and those players are kept at zero; only players who left the league
        entirely are dropped, because "he retired" is not a fantasy outcome.
    """
    derived = stats.with_columns(
        (pl.col("rec_yards") / pl.col("targets").replace(0, None)).alias("yds_per_target"),
        (pl.col("rush_yards") / pl.col("carries").replace(0, None)).alias("yds_per_carry"),
        (
            pl.col("points")
            / (pl.col("targets") + pl.col("carries")).replace(0, None)
        ).alias("points_per_touch"),
        (pl.col("points_oe") / pl.col("games")).alias("points_oe_pg"),
    )

    prior = derived.filter(pl.col("games") >= min_prior_games)
    nxt = derived.select(
        "player_id", "season",
        pl.col("points").alias("y_points"),
        pl.col("ppg").alias("y_ppg"),
        pl.col("games").alias("y_games"),
    ).with_columns((pl.col("season") - 1).cast(pl.Int32).alias("season"))

    if rosters is None:
        return prior.join(nxt, on=["player_id", "season"], how="inner")

    # Still in the league the following season, by roster membership.
    survived = (
        rosters.select("player_id", "season")
        .unique()
        .with_columns(
            (pl.col("season") - 1).cast(pl.Int32).alias("season"),
            pl.lit(True).alias("_rostered_next"),
        )
    )
    return (
        prior.join(survived, on=["player_id", "season"], how="inner")
        .join(nxt, on=["player_id", "season"], how="left")
        .with_columns(
            pl.col("y_points").fill_null(0.0),
            pl.col("y_ppg").fill_null(0.0),
            pl.col("y_games").fill_null(0),
        )
        .drop("_rostered_next")
    )


def grouped_slope(
    df: pl.DataFrame,
    x: str,
    y: str,
    control: str | None = None,
    groups: tuple[str, ...] = ("season", "pos"),
) -> dict:
    """OLS slope of `y` on `x` within (season, position), pooled by group size.

    A correlation says whether an effect exists; a slope says how big it is, in
    the units the reader cares about -- points per game gained per point of
    quarterback upgrade, say. `control` is added to the regression rather than
    partialled out beforehand, which is the same thing and one fewer step to get
    wrong.

    The standard error is across seasons. That treats a season as the unit of
    replication, which is right for the season-level shocks (scoring
    environment, injury luck) but does not account for team-level clustering
    within a season: a team-level regressor like a quarterback change gives 32
    values, not 300, and the true error is wider than this reports. Read these
    slopes as directional.
    """
    cols = [x, y, *groups] + ([control] if control else [])
    sub = df.select([c for c in dict.fromkeys(cols)]).drop_nulls()

    slopes, weights, per_season = [], [], {}
    for key, chunk in sub.group_by(list(groups)):
        if chunk.height < MIN_GROUP:
            continue
        xv = chunk[x].to_numpy().astype(float)
        yv = chunk[y].to_numpy().astype(float)
        if np.ptp(xv) == 0:
            continue

        columns = [np.ones(len(xv)), xv]
        if control:
            cv = chunk[control].to_numpy().astype(float)
            if np.ptp(cv) == 0:
                continue
            columns.append(cv)
        design = np.column_stack(columns)
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        coef, *_ = np.linalg.lstsq(design, yv, rcond=None)

        slopes.append(float(coef[1]))
        weights.append(chunk.height)
        season = key[groups.index("season")] if "season" in groups else 0
        per_season.setdefault(season, []).append((float(coef[1]), chunk.height))

    if not slopes:
        return {"slope": float("nan"), "se": float("nan"), "n": 0}

    w = np.array(weights, dtype=float)
    pooled = float((np.array(slopes) * w).sum() / w.sum())
    season_means = np.array([
        float(np.average([v for v, _ in vals], weights=[n for _, n in vals]))
        for vals in per_season.values()
    ])
    se = (
        float(season_means.std(ddof=1) / np.sqrt(len(season_means)))
        if len(season_means) > 1 else float("nan")
    )
    return {"slope": pooled, "se": se, "n": int(w.sum())}


def bucket_effect(
    df: pl.DataFrame,
    column: str,
    outcome: str,
    control: str | None = None,
    n_buckets: int = 5,
    groups: tuple[str, ...] = ("season", "pos"),
) -> pl.DataFrame:
    """Mean outcome by within-group quantile of `column`, with a standard error.

    With `control` given, the reported `effect` is the mean *residual* outcome
    after removing what the control alone predicts, computed within each group.
    That is what separates "these players scored more" from "these players
    scored more than their own prior season said they would".
    """
    cols = [column, outcome, *groups] + ([control] if control else [])
    sub = df.select([c for c in dict.fromkeys(cols)]).drop_nulls()

    frames = []
    for _, chunk in sub.group_by(list(groups)):
        if chunk.height < MIN_GROUP:
            continue
        values = chunk[outcome].to_numpy().astype(float)
        if control:
            cv = chunk[control].to_numpy().astype(float)
            if np.ptp(cv) == 0:
                continue
            values = _residualise(values, cv)
        frames.append(chunk.with_columns(pl.Series("_effect", values)))

    if not frames:
        return pl.DataFrame()

    pooled = pl.concat(frames)
    return (
        pooled.with_columns(
            (
                (pl.col(column).rank("ordinal").over(list(groups)) - 1) * n_buckets
                // pl.len().over(list(groups))
            ).alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.col(column).mean().alias("level"),
            pl.col("_effect").mean().alias("effect"),
            (pl.col("_effect").std() / pl.len().sqrt()).alias("se"),
            pl.col(outcome).mean().alias("raw"),
            pl.len().alias("n"),
        )
        .sort("bucket")
    )
