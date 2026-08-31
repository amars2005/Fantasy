"""How much should you trust the pick the board just recommended?

The board ranks players by VONA computed from a single set of projections. Those
projections are a fitted curve evaluated at a positional rank, and both the curve
and the rank are estimates. Reporting the resulting order without saying how
firmly it holds is the one place this tool claims precision it does not have.

So: re-run the whole ranking a few hundred times, each time on a different
plausible set of projections (`consensus.draw_projections`) and a different
resample of the draft simulation, and report what survives.

    p_best     share of draws where this player has the highest VONA
    p_top3     share of draws where he is in the top three
    regret     expected VONA given up by taking him instead of the draw's own
               best choice -- in projected points, so it is directly comparable
               to everything else on the board

`regret` is the number to read. `p_best` tells you how often a pick wins; regret
tells you what losing costs, and those are very different when the top of the
board is a plateau. A 22% `p_best` with 0.4 points of regret is not a difficult
decision -- it is a decision that does not matter.

None of this makes the projections better. It makes the board honest about when
it is guessing, which is most of the time after round three.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import LEAGUE
from src.draft.sim_draft import board_arrays, simulate_draft_orders
from src.draft.vona import optimal_lineup_points

DEFAULT_DRAWS = 200
# Draft orders are resampled from one larger simulation rather than re-simulated
# per draw: the survival matrix depends only on ADP, not on the projections, so
# simulating it again for every draw would burn the entire time budget
# reproducing the same numbers.
DEFAULT_SIMS = 1500
SIM_SUBSAMPLE = 300


def _marginal_interpolators(
    roster: list[dict], positions: list[str], league: dict, ceiling: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Exact piecewise-linear lineup gain as a function of a candidate's points.

    Adding one player to a fixed roster changes the optimal lineup only at points
    where he displaces somebody, so the gain is piecewise linear with kinks
    exactly at the roster's own projections. Evaluating it there and
    interpolating is exact, and it turns 250 lineup solves per draw into about
    twenty.
    """
    base = optimal_lineup_points(roster, league)
    knots = sorted({0.0, ceiling * 1.05} | {float(p["proj_points"]) for p in roster})
    xs = np.array(knots, dtype=float)

    out = {}
    for pos in positions:
        ys = np.array([
            optimal_lineup_points(
                roster + [{"pos": pos, "proj_points": float(x)}], league
            ) - base
            for x in xs
        ])
        out[pos] = (xs, ys)
    return out


def _marginals(
    draw: np.ndarray, pos: np.ndarray,
    interp: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    marginal = np.zeros(len(draw))
    for name, (xs, ys) in interp.items():
        mask = pos == name
        if mask.any():
            marginal[mask] = np.interp(draw[mask], xs, ys)
    return marginal


def pick_reliability(
    board: pl.DataFrame,
    roster: list[dict],
    picks_until_next: int,
    draws: np.ndarray,
    league: dict | None = None,
    n_sims: int = DEFAULT_SIMS,
    sim_subsample: int = SIM_SUBSAMPLE,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Per-player pick confidence and expected regret under projection error.

    `draws` is an (n_draws, board.height) matrix of alternative projections, in
    the board's own row order -- see `consensus.draw_projections`.

    The roster is held at its point projections. Noise on players already drafted
    shifts every candidate's marginal value in nearly the same direction, so it
    moves the level far more than the ordering, and carrying it would double the
    cost of this function for very little.
    """
    league = league or LEAGUE
    if draws.shape[1] != board.height:
        raise ValueError(
            f"draws has {draws.shape[1]} columns for a board of {board.height}"
        )

    pos = np.array(board["pos"].to_list())
    positions = sorted(set(pos.tolist()))
    interp = _marginal_interpolators(
        roster, positions, league, float(draws.max())
    )

    rng = rng or np.random.default_rng(11)
    mu, stdev = board_arrays(board)
    survives = simulate_draft_orders(mu, stdev, n_sims, rng) >= (picks_until_next + 1)

    n_draws, n_players = draws.shape
    vona_sum = np.zeros(n_players)
    vona_sq = np.zeros(n_players)
    best_count = np.zeros(n_players)
    top3_count = np.zeros(n_players)
    regret_sum = np.zeros(n_players)

    masks = {p: pos == p for p in positions}
    for d in range(n_draws):
        marginal = _marginals(draws[d], pos, interp)

        rows = rng.integers(0, n_sims, min(sim_subsample, n_sims))
        available = survives[rows]

        next_best = np.zeros(n_players)
        for p, mask in masks.items():
            if not mask.any():
                continue
            vals = np.where(available[:, mask], marginal[mask][None, :], -np.inf)
            best = vals.max(axis=1)
            # Draws where nothing at this position survives contribute zero:
            # there is no "next" player to compare against.
            next_best[mask] = float(np.where(np.isfinite(best), best, 0.0).mean())

        vona = marginal - next_best
        vona_sum += vona
        vona_sq += vona * vona

        order = np.argsort(-vona)
        best_count[order[0]] += 1
        top3_count[order[:3]] += 1
        regret_sum += vona[order[0]] - vona

    mean = vona_sum / n_draws
    var = np.maximum(vona_sq / n_draws - mean**2, 0.0)
    return board.select("player_id", "name", "pos").with_columns(
        pl.Series("vona_mean", mean),
        pl.Series("vona_sd", np.sqrt(var)),
        pl.Series("p_best", best_count / n_draws),
        pl.Series("p_top3", top3_count / n_draws),
        pl.Series("regret", regret_sum / n_draws),
    )


def confidence_summary(table: pl.DataFrame, threshold: float = 2.0) -> dict:
    """One line about whether the top of the board is a decision or a formality.

    `threshold` is in projected points: below it, the choice between the top two
    is not distinguishable given how well the curve is determined, and the tie
    should be broken on something the projection cannot see -- injury news,
    schedule, or which player you would rather own.
    """
    ranked = table.sort("vona_mean", descending=True)
    if ranked.height == 0:
        return {}
    top = ranked.row(0, named=True)
    runner_up = ranked.row(1, named=True) if ranked.height > 1 else None
    gap = float(top["vona_mean"] - runner_up["vona_mean"]) if runner_up else 0.0

    if runner_up is None:
        verdict = "only one player left"
    elif gap >= threshold and top["p_best"] >= 0.5:
        verdict = "clear"
    elif gap >= threshold:
        verdict = "leaning"
    elif float(runner_up["regret"]) < threshold / 2:
        verdict = "coin flip -- take either"
    else:
        verdict = "close"

    return {
        "top": top["name"],
        "top_pos": top["pos"],
        "p_best": round(float(top["p_best"]), 3),
        "runner_up": runner_up["name"] if runner_up else None,
        "gap": round(gap, 2),
        "runner_up_regret": round(float(runner_up["regret"]), 2) if runner_up else 0.0,
        "verdict": verdict,
    }
