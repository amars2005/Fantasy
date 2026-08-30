"""Monte Carlo season simulation for a roster.

Point projections cannot rank draft strategies, because two rosters with the same
projected total are not equally good. Weekly lineup choice means depth has real
option value: a third startable running back is worth more than his season total
suggests, because some weeks he outscores your starters.

So we simulate weeks, not seasons. Each player's weekly score is drawn from a
gamma distribution whose dispersion was measured from 2018-2025 (elite players
CV ~0.45, low-volume players ~0.9), he is unavailable in some weeks according to
his projected games played, and the best legal lineup is chosen each week with
hindsight-free selection.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import polars as pl

from src.config import LEAGUE, REGULAR_SEASON_WEEKS
from src.ingest import nflverse as nv
from src.scoring import add_fantasy_points

CV_FIT_SEASONS = list(range(2018, 2026))
CV_BOUNDS = (0.35, 1.10)


@lru_cache(maxsize=1)
def fit_weekly_cv() -> dict[str, tuple[float, float]]:
    """Fit weekly coefficient of variation as cv = a * ppg^-b, per position.

    Volatility falls with volume: a 20-point-per-game receiver is far steadier
    than a 5-point one, and treating them alike would wildly misprice depth.
    """
    weekly = add_fantasy_points(nv.player_stats(seasons=CV_FIT_SEASONS, level="week"))
    if "season_type" in weekly.columns:
        weekly = weekly.filter(pl.col("season_type") == "REG")
    pos_col = "position" if "position" in weekly.columns else "position_group"

    agg = (
        weekly.filter(pl.col(pos_col).is_in(["QB", "RB", "WR", "TE"]))
        .group_by(["player_id", "season", pos_col])
        .agg(
            pl.col("fantasy_points_league").mean().alias("ppg"),
            pl.col("fantasy_points_league").std().alias("wsd"),
            pl.len().alias("g"),
        )
        .filter((pl.col("g") >= 8) & (pl.col("ppg") > 3))
        .with_columns((pl.col("wsd") / pl.col("ppg")).alias("cv"))
    )

    fits: dict[str, tuple[float, float]] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        sub = agg.filter(pl.col(pos_col) == pos)
        ppg = np.log(sub["ppg"].to_numpy())
        cv = np.log(np.clip(sub["cv"].to_numpy(), 1e-3, None))
        b, log_a = np.polyfit(ppg, cv, 1)
        fits[pos] = (float(np.exp(log_a)), float(-b))
    return fits


def weekly_cv(positions: np.ndarray, ppg: np.ndarray) -> np.ndarray:
    fits = fit_weekly_cv()
    out = np.empty(len(ppg))
    for i, (pos, rate) in enumerate(zip(positions, ppg)):
        a, b = fits.get(str(pos), fits["WR"])
        out[i] = a * max(rate, 1.0) ** (-b)
    return np.clip(out, *CV_BOUNDS)


def simulate_weekly_points(
    roster: pl.DataFrame,
    n_sims: int,
    rng: np.random.Generator,
    weeks: int = REGULAR_SEASON_WEEKS,
) -> np.ndarray:
    """(n_sims, weeks, n_players) array of simulated weekly fantasy points."""
    pos = np.array(roster["pos"].to_list())
    season_points = roster["proj_points"].to_numpy().astype(float)
    games = np.clip(roster["games"].to_numpy().astype(float), 1.0, weeks)

    # proj_points already carries the cost of missed games, so the per-game rate
    # is points-when-playing and availability is modelled separately. Combining
    # them reproduces the season projection in expectation without double-counting.
    rate = season_points / games
    cv = weekly_cv(pos, rate)

    shape = np.clip(1.0 / cv**2, 0.5, 50.0)
    scale = rate / shape
    points = rng.gamma(
        shape[None, None, :], scale[None, None, :], size=(n_sims, weeks, len(rate))
    )

    available = rng.random((n_sims, weeks, len(rate))) < (games / weeks)[None, None, :]
    return points * available


def best_lineup_points(
    weekly: np.ndarray, positions: np.ndarray, league: dict | None = None,
    per_week: bool = False,
) -> np.ndarray:
    """Points from the best legal lineup each week.

    Returns (n_sims,) season totals, or (n_sims, weeks) when `per_week` is set --
    head-to-head scoring needs the weekly series, not the sum.
    """
    league = league or LEAGUE
    starters = league["starters"]
    eligible = league["flex_eligible"]

    def top(pos: str) -> np.ndarray:
        idx = np.flatnonzero(positions == pos)
        if idx.size == 0:
            return np.zeros(weekly.shape[:2] + (0,))
        return -np.sort(-weekly[:, :, idx], axis=2)

    sorted_by_pos = {p: top(p) for p in ("QB", "RB", "WR", "TE")}

    total = np.zeros(weekly.shape[:2])
    used: dict[str, int] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        slots = starters.get(pos, 0)
        block = sorted_by_pos[pos][:, :, :slots]
        total += block.sum(axis=2)
        used[pos] = block.shape[2]

    flex_slots = starters.get("FLEX", 0)
    if flex_slots:
        leftovers = [sorted_by_pos[p][:, :, used.get(p, 0):] for p in eligible]
        pool = np.concatenate([x for x in leftovers if x.shape[2] > 0], axis=2)
        if pool.shape[2]:
            best = -np.sort(-pool, axis=2)[:, :, :flex_slots]
            total += best.sum(axis=2)

    return total if per_week else total.sum(axis=1)


def simulate_roster(
    roster: pl.DataFrame,
    n_sims: int = 400,
    rng: np.random.Generator | None = None,
    league: dict | None = None,
    per_week: bool = False,
) -> np.ndarray:
    """Distribution of starting-lineup points for a roster.

    Season totals by default; weekly series when `per_week` is set.
    """
    if roster.height == 0:
        return np.zeros((n_sims, REGULAR_SEASON_WEEKS)) if per_week else np.zeros(n_sims)
    rng = rng or np.random.default_rng(0)
    weekly = simulate_weekly_points(roster, n_sims, rng)
    return best_lineup_points(
        weekly, np.array(roster["pos"].to_list()), league, per_week=per_week
    )
