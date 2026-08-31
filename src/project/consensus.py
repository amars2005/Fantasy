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

Two different uncertainties come out of this fit and they answer different
questions, so they are kept in separate columns:

  ``sd``       how far a player's *season* lands from the curve. This is outcome
               spread, and it is what a risk or upside calculation needs.
  ``proj_se``  how far the *curve itself* would move if the ten seasons it is
               fitted on had come out differently, from a bootstrap of the fit.
               This is estimation error, and it is what decides whether a
               recommendation is trustworthy -- a coin flip between two players
               with identical projections stays a coin flip no matter how wide
               their outcome distributions are.

`sd` is much the larger of the two. Using it where `proj_se` belongs would make
every pick look like a lottery; using `proj_se` where `sd` belongs would make a
boom-or-bust player look safe.
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
N_BOOTSTRAP = 200  # refits of the rank -> points curve, for the standard error


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


def _bootstrap_curves(
    ranks: np.ndarray, points: np.ndarray, grid: np.ndarray,
    n_boot: int = N_BOOTSTRAP, seed: int = 0,
) -> np.ndarray:
    """Refits of the curve on resampled observations: (n_boot, len(grid)).

    The whole matrix is kept rather than only its standard deviation. Isotonic
    refits move in correlated blocks -- a bootstrap that pushes RB4 down pushes
    RB5 and RB6 down with it -- and a per-rank standard error throws that
    structure away. Anything reasoning about *which player to pick* needs it,
    because a shift common to a whole position changes nothing about the choice
    within that position.

    Resampling *seasons* would not work: the training frame is built by joining
    on season, so a ten-draw resample yields six or seven unique years and
    measures the effect of having less data. Resampling rows is the honest
    version, and it is the same procedure `rank_sensitivity.py` runs end to end.

    The seed is fixed so the board does not change between two runs on the same
    data -- a recommendation that moves when you reload it is not a
    recommendation.
    """
    rng = np.random.default_rng(seed)
    n = len(ranks)
    draws = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = _fit_position(ranks[idx], points[idx]).predict(grid)
    return draws


def fit_curves(seasons: list[int] | None = None, with_se: bool = True) -> dict:
    """Fit per-position rank -> (expected points, sd, proj_se) curves.

    `with_se=False` skips the bootstrap. Callers that refit the curve inside a
    loop -- the backtest does it once per season -- do not need it and should not
    pay for it.
    """
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

        boot = (
            _bootstrap_curves(ranks, points, grid) if with_se
            else np.empty((0, len(grid)))
        )
        curves[pos] = {
            "grid": grid,
            "mean": mean,
            "sd": sd,
            "boot": boot,
            "proj_se": boot.std(axis=0) if boot.size else np.zeros_like(grid),
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
                pl.Series("proj_se", _lookup(curve, ranks, "proj_se")).round(2),
                pl.Series("games", _lookup(curve, ranks, "games")).round(1),
                pl.lit("consensus").alias("source"),
            )
        )

    return pl.concat(frames).sort("adp")

def draw_projections(
    board: pl.DataFrame,
    curves: dict,
    n_draws: int = 200,
    rng: np.random.Generator | None = None,
    rank_noise: bool = True,
) -> np.ndarray:
    """Plausible alternative projections for the whole board: (n_draws, players).

    Two sources of error, both measured rather than assumed:

      * **the curve** -- one bootstrap refit per draw, taken whole so the
        correlation between neighbouring ranks survives.
      * **the ordering** -- each player's ADP is perturbed by his own observed
        `stdev` and the board re-ranked, because a player the room disagrees
        about does not have a well-determined positional rank to look up.

    What it does *not* cover is the market being collectively wrong about a
    player, which is the largest error of all and is not measurable from this
    data. Anything built on these draws is a lower bound on how uncertain a
    recommendation really is.

    Rows the fitted curves do not cover -- kickers and defences -- fall back to
    Gaussian noise on their own `proj_se`.
    """
    rng = rng or np.random.default_rng(0)
    n = board.height
    pos = np.array(board["pos"].to_list())
    adp = board["adp"].to_numpy().astype(float)
    stdev = np.maximum(
        board["stdev"].fill_null(1.0).to_numpy().astype(float), 0.5
    )
    base = board["proj_points"].to_numpy().astype(float)
    se = (
        board["proj_se"].fill_null(0.0).to_numpy().astype(float)
        if "proj_se" in board.columns else np.zeros(n)
    )

    out = np.empty((n_draws, n))
    for pos_name in np.unique(pos):
        mask = pos == pos_name
        curve = curves.get(str(pos_name))
        if curve is None or not curve.get("boot", np.empty(0)).size:
            out[:, mask] = base[mask][None, :] + rng.normal(
                0.0, 1.0, size=(n_draws, int(mask.sum()))
            ) * se[mask][None, :]
            continue

        boot, grid = curve["boot"], curve["grid"]
        rows = rng.integers(0, boot.shape[0], n_draws)
        if rank_noise:
            noisy = adp[mask][None, :] + rng.normal(
                0.0, 1.0, size=(n_draws, int(mask.sum()))
            ) * stdev[mask][None, :]
            ranks = noisy.argsort(axis=1).argsort(axis=1)
        else:
            fixed = np.argsort(np.argsort(adp[mask]))
            ranks = np.repeat(fixed[None, :], n_draws, axis=0)
        idx = np.clip(ranks, 0, len(grid) - 1)
        out[:, mask] = boot[rows[:, None], idx]

    return np.clip(out, 0.0, None)
