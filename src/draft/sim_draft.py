"""Monte Carlo draft simulation.

Everything that matters on draft day is a question about *what will still be
there later*, not about who is best now. This module answers that by simulating
the rest of the draft many times from the current board state.

Bots are modelled by perturbing each player's ADP by his own observed standard
deviation and drafting in the resulting order. That is deliberately simple: ADP
is itself the aggregate of thousands of real drafts, so reproducing it (with its
real dispersion) reproduces the room. The calibration test in tests/ checks that
simulated draft positions actually recover the input ADP -- if they don't, every
survival probability built on top is wrong.
"""

from __future__ import annotations

import numpy as np
import polars as pl

DEFAULT_SIMS = 4000
# ADP dispersion widens for late picks; a floor keeps early studs from being
# treated as perfectly predictable.
MIN_STDEV = 0.5


def snake_picks(slot: int, teams: int, rounds: int) -> list[int]:
    """Overall pick numbers for a given draft slot in a snake draft."""
    picks = []
    for rnd in range(rounds):
        if rnd % 2 == 0:
            picks.append(rnd * teams + slot)
        else:
            picks.append(rnd * teams + (teams - slot + 1))
    return picks


def simulate_draft_orders(
    adp: np.ndarray,
    stdev: np.ndarray,
    n_sims: int = DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return an (n_sims, n_players) array of simulated draft positions.

    Position 1 = first player off the board in that simulation.
    """
    rng = rng or np.random.default_rng(0)
    sd = np.maximum(stdev, MIN_STDEV)
    scores = adp[None, :] + rng.normal(0.0, 1.0, size=(n_sims, adp.size)) * sd[None, :]
    order = np.argsort(scores, axis=1)
    positions = np.empty_like(order)
    np.put_along_axis(
        positions, order, np.arange(1, adp.size + 1)[None, :].repeat(n_sims, axis=0), axis=1
    )
    return positions


def calibrate(
    adp: np.ndarray,
    stdev: np.ndarray,
    n_sims: int = DEFAULT_SIMS,
    iters: int = 15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Solve for the latent means that make simulated draft position match ADP.

    Perturbing ADP and re-ranking does not return ADP: ranking is a nonlinear
    transform, so players with large dispersion get pulled toward the middle of
    the board and the ends of the list compress. Left uncorrected this biases
    late-round survival by ~10 picks. A few rounds of fixed-point iteration on
    the latent mean removes it.
    """
    rng = rng or np.random.default_rng(0)
    target = np.argsort(np.argsort(adp)).astype(float) + 1.0
    mu = adp.astype(float).copy()
    for _ in range(iters):
        positions = simulate_draft_orders(mu, stdev, n_sims, rng)
        mu -= 0.7 * (positions.mean(axis=0) - target)
    return mu


def add_calibrated_adp(board: pl.DataFrame, column: str = "adp_mu") -> pl.DataFrame:
    """Attach calibrated latent draft means once, so live calls stay fast.

    Calibration is a property of the player pool, not of who is still available,
    so it is computed against the full board and reused as players come off it.
    """
    adp = board["adp"].to_numpy().astype(float)
    stdev = np.maximum(board["stdev"].fill_null(MIN_STDEV).to_numpy().astype(float), MIN_STDEV)
    return board.with_columns(pl.Series(column, calibrate(adp, stdev)))


def board_arrays(board: pl.DataFrame, calibrated: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Latent draft means and dispersions for a board.

    Uses a precomputed `adp_mu` column when present; only falls back to running
    calibration inline if the caller never attached one.
    """
    stdev = np.maximum(board["stdev"].fill_null(MIN_STDEV).to_numpy().astype(float), MIN_STDEV)
    if "adp_mu" in board.columns:
        return board["adp_mu"].to_numpy().astype(float), stdev
    adp = board["adp"].to_numpy().astype(float)
    mu = calibrate(adp, stdev) if calibrated else adp
    return mu, stdev


def survival_probability(
    board: pl.DataFrame,
    pick: int,
    n_sims: int = DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """P(player is still available at `pick`) for every player on the board.

    `board` must already exclude drafted players, and `pick` is counted in
    *remaining* picks from now (1 = the very next selection in the room).
    """
    mu, stdev = board_arrays(board)
    positions = simulate_draft_orders(mu, stdev, n_sims, rng)
    prob = (positions >= pick).mean(axis=0)
    return board.with_columns(pl.Series("p_available", prob))


def expected_best_available(
    board: pl.DataFrame,
    pick: int,
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
    value_col: str = "vor",
    n_sims: int = DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Expected value of the best player left at each position at `pick`.

    This is the number VONA is measured against: not "what is a replacement-level
    player worth", but "what will I actually be able to get if I wait".
    """
    mu, stdev = board_arrays(board)
    value = board[value_col].to_numpy().astype(float)
    pos = np.array(board["pos"].to_list())

    sim_positions = simulate_draft_orders(mu, stdev, n_sims, rng)
    available = sim_positions >= pick

    out: dict[str, float] = {}
    for p in positions:
        mask = pos == p
        if not mask.any():
            out[p] = 0.0
            continue
        vals = np.where(available[:, mask], value[mask][None, :], -np.inf)
        best = vals.max(axis=1)
        # Simulations where nothing at this position survives contribute zero.
        best = np.where(np.isfinite(best), best, 0.0)
        out[p] = float(best.mean())
    return out
