"""Market-implied projections: positional ADP rank -> expected fantasy points.

The market tells us the *order* players are drafted in. History tells us what a
player drafted at a given positional rank actually goes on to score. Composing
the two gives a projection denominated in points -- which is what value over
replacement needs -- without inventing a single number of our own.

Two properties make this the right baseline rather than a placeholder:

  * It is honest about the top of the draft. Fitting to *preseason rank* rather
    than *end-of-season finish* avoids the winner's-curse inflation you get from
    assuming this year's RB1 will score what last year's RB1 scored. The RB
    drafted first has averaged ~220 PPR points, not the ~370 the eventual RB1
    scores.
  * It yields empirically calibrated uncertainty. The spread of outcomes at each
    rank is measured, not assumed, and it is enormous (SD ~150 at the top of the
    RB board). Everything downstream that reasons about risk needs that number.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from src.config import SEASON, SKILL_POSITIONS
from src.ingest.adp import load_adp, load_adp_history
from src.ingest.ids import resolve
from src.project.actuals import season_totals

FIT_SEASONS = list(range(2016, 2026))
SD_WINDOW = 8  # +/- ranks pooled when estimating outcome spread


def build_training_frame(seasons: list[int] | None = None) -> pl.DataFrame:
    """Historical (positional ADP rank -> actual points) observations."""
    seasons = seasons or FIT_SEASONS
    adp = resolve(load_adp_history(seasons)).filter(
        pl.col("pos").is_in(SKILL_POSITIONS) & pl.col("gsis_id").is_not_null()
    )
    actual = season_totals(seasons).rename({"player_id": "gsis_id", "points": "actual"})

    joined = adp.join(
        actual.select(["gsis_id", "season", "actual", "games"]),
        on=["gsis_id", "season"],
        how="left",
    ).with_columns(
        # A drafted player with no stat line scored zero. That is a real draft
        # outcome (holdout, injury, cut) and must not be dropped.
        pl.col("actual").fill_null(0.0),
        pl.col("games").fill_null(0),
    )
    return joined.with_columns(
        pl.col("adp").rank("ordinal").over(["season", "pos"]).alias("pos_rank")
    )


def _fit_position(ranks: np.ndarray, points: np.ndarray) -> IsotonicRegression:
    """Monotone-decreasing fit of points on rank.

    Raw per-rank means are non-monotone pure noise at the top (only ~10
    observations per rank). Isotonic regression imposes the one thing we are
    confident about -- that being drafted earlier should not predict fewer
    points -- without assuming a functional form.
    """
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(ranks, points)
    return model


def _rolling_sd(ranks: np.ndarray, resid: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Outcome spread at each rank, pooled over a window of neighbouring ranks."""
    out = np.empty(len(grid))
    for i, r in enumerate(grid):
        mask = np.abs(ranks - r) <= SD_WINDOW
        sample = resid[mask]
        out[i] = sample.std() if sample.size >= 5 else resid.std()
    return out


def fit_curves(seasons: list[int] | None = None) -> dict:
    """Fit per-position rank -> (expected points, sd) curves."""
    train = build_training_frame(seasons)
    curves = {}
    for pos in SKILL_POSITIONS:
        sub = train.filter(pl.col("pos") == pos)
        ranks = sub["pos_rank"].to_numpy().astype(float)
        points = sub["actual"].to_numpy().astype(float)
        if len(ranks) < 30:
            continue

        model = _fit_position(ranks, points)
        grid = np.arange(1, int(ranks.max()) + 1, dtype=float)
        mean = model.predict(grid)
        resid = points - model.predict(ranks)
        sd = _rolling_sd(ranks, resid, grid)

        games = sub["games"].to_numpy().astype(float)
        games_model = _fit_position(ranks, games)

        curves[pos] = {
            "grid": grid,
            "mean": mean,
            "sd": sd,
            "games": np.clip(games_model.predict(grid), 0, 17),
            "n": len(ranks),
        }
    return curves


def _lookup(curve: dict, ranks: np.ndarray, field: str) -> np.ndarray:
    idx = np.clip(ranks.astype(int) - 1, 0, len(curve["grid"]) - 1)
    return curve[field][idx]


def project(season: int = SEASON, curves: dict | None = None) -> pl.DataFrame:
    """Produce the projections contract for a season from live ADP."""
    curves = curves or fit_curves()
    adp = resolve(load_adp(year=season)).filter(pl.col("pos").is_in(SKILL_POSITIONS))
    adp = adp.with_columns(
        pl.col("adp").rank("ordinal").over("pos").alias("pos_rank")
    )

    frames = []
    for pos, curve in curves.items():
        sub = adp.filter(pl.col("pos") == pos)
        if sub.height == 0:
            continue
        ranks = sub["pos_rank"].to_numpy()
        frames.append(
            sub.select(
                pl.col("gsis_id").alias("player_id"),
                "name", "pos", "tm", "adp", "stdev", "bye", "pos_rank",
            ).with_columns(
                pl.Series("proj_points", _lookup(curve, ranks, "mean")).round(1),
                pl.Series("sd", _lookup(curve, ranks, "sd")).round(1),
                pl.Series("games", _lookup(curve, ranks, "games")).round(1),
                pl.lit("consensus").alias("source"),
            )
        )

    return pl.concat(frames).sort("adp")
