"""Guards for the pick-confidence layer.

The failure this exists for is silent over-confidence. Every part of this
pipeline degrades quietly: a board with no error bars produces draws that never
move, which produces `p_best = 1.0`, which reads on the dashboard as certainty
about a decision nothing was measured for. The tests below pin the two things
that stop that -- the interpolation must be exact, and a board with no
uncertainty must report none.
"""

import numpy as np
import polars as pl
import pytest

from src.draft.reliability import (
    _marginal_interpolators, confidence_summary, pick_reliability,
)
from src.draft.sim_draft import add_calibrated_adp
from src.draft.vona import optimal_lineup_points
from src.project.consensus import draw_projections

LEAGUE = {
    "teams": 14,
    "rounds": 15,
    "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1},
    "flex_eligible": ("RB", "WR", "TE"),
}
POSITIONS = ["QB", "RB", "WR", "TE"]


def _board(n: int = 80, seed: int = 0, proj_se: float = 12.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    adp = np.sort(rng.uniform(1, 180, n))
    return add_calibrated_adp(pl.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "name": [f"Player {i}" for i in range(n)],
        "pos": rng.choice(POSITIONS, n).tolist(),
        "adp": adp,
        "stdev": np.clip(adp * 0.2, 1.0, 30.0),
        "proj_points": np.clip(300 - adp * 1.2 + rng.normal(0, 20, n), 5, None),
        "sd": np.full(n, 60.0),
        "proj_se": np.full(n, proj_se),
    }))


# --- the interpolation shortcut ---------------------------------------------
@pytest.mark.parametrize("roster", [
    [],
    [{"pos": "RB", "proj_points": 210.0}, {"pos": "WR", "proj_points": 190.0}],
    [{"pos": "QB", "proj_points": 300.0}, {"pos": "RB", "proj_points": 210.0},
     {"pos": "RB", "proj_points": 140.0}, {"pos": "WR", "proj_points": 190.0},
     {"pos": "WR", "proj_points": 165.0}, {"pos": "TE", "proj_points": 120.0}],
])
def test_marginal_interpolation_is_exact(roster):
    """Marginal lineup value is piecewise linear with kinks at roster values.

    The whole reliability pass rests on evaluating it at those kinks and
    interpolating -- 20 lineup solves per draw instead of 250. If that is not
    exact, every confidence number is quietly wrong.
    """
    interp = _marginal_interpolators(roster, POSITIONS, LEAGUE, 350.0)
    base = optimal_lineup_points(roster, LEAGUE)
    rng = np.random.default_rng(1)

    for pos in POSITIONS:
        for value in rng.uniform(0, 350, 150):
            exact = optimal_lineup_points(
                roster + [{"pos": pos, "proj_points": float(value)}], LEAGUE
            ) - base
            assert float(np.interp(value, *interp[pos])) == pytest.approx(exact)


# --- the reliability table ---------------------------------------------------
def test_probabilities_and_regret_are_coherent():
    board = _board()
    rng = np.random.default_rng(2)
    draws = (
        board["proj_points"].to_numpy()[None, :]
        + rng.normal(0, 1, (120, board.height)) * 12.0
    )
    table = pick_reliability(board, [], 13, draws, LEAGUE,
                             rng=np.random.default_rng(3))

    assert table["p_best"].sum() == pytest.approx(1.0)
    assert table["p_top3"].sum() == pytest.approx(3.0)
    assert (table["p_top3"] >= table["p_best"]).all()
    # Regret is measured against the best choice in each draw, so nobody can
    # have negative expected regret and the leader's must be the smallest.
    assert (table["regret"] >= -1e-9).all()
    best = table.sort("vona_mean", descending=True).row(0, named=True)
    assert best["regret"] == pytest.approx(table["regret"].min())


def test_identical_seeds_give_identical_answers():
    """A recommendation that changes when you reload it is not a recommendation."""
    board = _board()
    draws = np.repeat(board["proj_points"].to_numpy()[None, :], 40, axis=0)
    draws = draws + np.random.default_rng(9).normal(0, 10, draws.shape)
    a = pick_reliability(board, [], 13, draws, LEAGUE, rng=np.random.default_rng(5))
    b = pick_reliability(board, [], 13, draws, LEAGUE, rng=np.random.default_rng(5))
    assert a["p_best"].to_list() == b["p_best"].to_list()
    assert a["regret"].to_list() == b["regret"].to_list()


def test_draw_width_mismatch_is_rejected():
    board = _board(n=20)
    with pytest.raises(ValueError):
        pick_reliability(board, [], 5, np.zeros((10, 19)), LEAGUE)


# --- where the draws come from -----------------------------------------------
def test_draws_fall_back_to_proj_se_without_curves():
    board = _board(proj_se=15.0)
    draws = draw_projections(board, {}, n_draws=200, rng=np.random.default_rng(6))
    assert draws.shape == (200, board.height)
    assert (draws >= 0).all(), "a projection cannot be negative"
    spread = draws.std(axis=0)
    assert spread.mean() == pytest.approx(15.0, rel=0.2)


def test_a_board_without_error_bars_produces_draws_that_do_not_move():
    """The signal the board's guard reads before reporting any confidence."""
    board = _board(proj_se=0.0)
    draws = draw_projections(board, {}, n_draws=50, rng=np.random.default_rng(7))
    assert float((draws.std(axis=0) > 1e-9).mean()) == 0.0


def test_bootstrap_curves_drive_the_draws_when_present():
    board = _board(n=40, proj_se=0.0)
    grid = np.arange(1, 60, dtype=float)
    rng = np.random.default_rng(8)
    curves = {
        pos: {
            "grid": grid,
            # Descending curves with run-to-run wobble, like a real refit.
            "boot": np.maximum(
                300 - 4 * grid[None, :] + rng.normal(0, 8, (50, len(grid))), 0
            ),
        }
        for pos in POSITIONS
    }
    draws = draw_projections(board, curves, n_draws=100,
                             rng=np.random.default_rng(10))
    assert draws.std(axis=0).min() > 0, "every player should move between draws"


# --- the summary -------------------------------------------------------------
def _summary_frame(rows):
    return pl.DataFrame(
        rows, orient="row",
        schema={"player_id": pl.Utf8, "name": pl.Utf8, "pos": pl.Utf8,
                "vona_mean": pl.Float64, "vona_sd": pl.Float64,
                "p_best": pl.Float64, "p_top3": pl.Float64,
                "regret": pl.Float64},
    )


def test_summary_calls_a_plateau_a_coin_flip():
    table = _summary_frame([
        ("a", "A", "RB", 10.0, 5.0, 0.34, 0.9, 0.3),
        ("b", "B", "WR", 9.6, 5.0, 0.31, 0.9, 0.6),
    ])
    got = confidence_summary(table, threshold=2.0)
    assert got["top"] == "A"
    assert got["verdict"] == "coin flip -- take either"


def test_summary_calls_a_real_gap_clear():
    table = _summary_frame([
        ("a", "A", "RB", 30.0, 4.0, 0.80, 1.0, 0.2),
        ("b", "B", "WR", 12.0, 4.0, 0.10, 0.8, 18.0),
    ])
    assert confidence_summary(table, threshold=2.0)["verdict"] == "clear"


def test_summary_of_an_empty_board_is_empty():
    assert confidence_summary(_summary_frame([])) == {}
