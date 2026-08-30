"""Value Over Next Available -- the pick criterion.

Value over replacement answers "how much better is this player than a waiver
pickup". That is the wrong question at the table. The right question is "how much
better is this player than what I could still get at this position if I wait one
round" -- because the alternative to drafting a WR now is not a replacement-level
WR, it is the best WR still on the board at your next pick.

Player value here is *lineup-marginal*: how much a player adds to your optimal
starting lineup given the roster you already have. That makes the tool refuse to
recommend a third quarterback without needing a hand-written rule, and it prices
the FLEX slot correctly.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import LEAGUE
from src.draft.sim_draft import DEFAULT_SIMS, board_arrays, simulate_draft_orders

# A player who does not crack your starting lineup still has worth: bye cover,
# injury insurance, and the chance he outperforms a starter. Valuing him at zero
# would make the tool refuse to build any depth at all.
BENCH_WEIGHT = 0.20


def optimal_lineup_points(roster: list[dict], league: dict | None = None) -> float:
    """Points from the best legal starting lineup, plus discounted bench.

    Greedy fill is exact for this roster structure: required slots are filled
    best-first, then the single FLEX takes the best flex-eligible player left.
    """
    league = league or LEAGUE
    starters = league["starters"]
    eligible = set(league["flex_eligible"])

    by_pos: dict[str, list[float]] = {}
    for p in roster:
        by_pos.setdefault(p["pos"], []).append(p["proj_points"])
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    total = 0.0
    used: dict[str, int] = {pos: 0 for pos in by_pos}
    for pos, slots in starters.items():
        if pos == "FLEX":
            continue
        take = by_pos.get(pos, [])[:slots]
        total += sum(take)
        used[pos] = len(take)

    flex_slots = starters.get("FLEX", 0)
    if flex_slots:
        # Candidates for FLEX are whoever is left at a flex-eligible position.
        candidates = [
            (pts, pos)
            for pos in eligible
            for pts in by_pos.get(pos, [])[used.get(pos, 0):]
        ]
        candidates.sort(reverse=True)
        for pts, pos in candidates[:flex_slots]:
            total += pts
            used[pos] += 1

    # Everyone still unused is bench -- at *any* position. Counting only
    # flex-eligible leftovers here would price a backup quarterback at exactly
    # zero, which is wrong: he covers a bye and an injury.
    bench = [
        pts
        for pos, pts_list in by_pos.items()
        for pts in pts_list[used.get(pos, 0):]
    ]

    return total + BENCH_WEIGHT * sum(bench)


def marginal_value(
    board: pl.DataFrame, roster: list[dict], league: dict | None = None
) -> pl.DataFrame:
    """How much each available player would add to your optimal lineup."""
    base = optimal_lineup_points(roster, league)
    gains = [
        optimal_lineup_points(roster + [{"pos": pos, "proj_points": pts}], league) - base
        for pos, pts in zip(board["pos"].to_list(), board["proj_points"].to_list())
    ]
    return board.with_columns(pl.Series("marginal", gains))


def vona_table(
    board: pl.DataFrame,
    roster: list[dict],
    picks_until_next: int,
    league: dict | None = None,
    n_sims: int = DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Rank available players by value over what survives to your next pick.

    `picks_until_next` is how many players come off the board before you choose
    again -- for a snake draft, the gap between your consecutive picks.
    """
    league = league or LEAGUE
    board = marginal_value(board, roster, league)

    mu, stdev = board_arrays(board)
    positions = simulate_draft_orders(mu, stdev, n_sims, rng)
    survives = positions >= (picks_until_next + 1)

    marginal = board["marginal"].to_numpy().astype(float)
    pos_arr = np.array(board["pos"].to_list())

    # For each position: the expected best marginal value still on the board
    # when we next pick.
    expected_next: dict[str, float] = {}
    for p in np.unique(pos_arr):
        mask = pos_arr == p
        vals = np.where(survives[:, mask], marginal[mask][None, :], -np.inf)
        best = vals.max(axis=1)
        expected_next[str(p)] = float(np.where(np.isfinite(best), best, 0.0).mean())

    return (
        board.with_columns(
            pl.col("pos").replace_strict(expected_next, default=0.0).alias("next_best"),
            pl.Series("p_survives", survives.mean(axis=0)),
        )
        .with_columns((pl.col("marginal") - pl.col("next_best")).alias("vona"))
        .sort("vona", descending=True)
    )


def position_dropoff(vona: pl.DataFrame) -> pl.DataFrame:
    """Per-position urgency: what you lose by waiting a round at each position."""
    return (
        vona.group_by("pos")
        .agg(
            pl.col("marginal").max().alias("best_now"),
            pl.col("next_best").first(),
            pl.col("name").sort_by("marginal", descending=True).first().alias("best_player"),
        )
        .with_columns((pl.col("best_now") - pl.col("next_best")).alias("dropoff"))
        .sort("dropoff", descending=True)
    )
