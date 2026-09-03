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

The same question has to be answered for a player who does *not* start, and
answering it in raw points is what used to put a QB2 in round nine: a flat share
of projected points pays a quarterback more for being a quarterback. A bench
player is insurance, so he is priced over what you could stream at his position
and by how many starting slots he stands behind -- see `bench_model`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import polars as pl

from src.config import LEAGUE
from src.draft.replacement import allocate_flex, replacement_levels
from src.draft.sim_draft import DEFAULT_SIMS, board_arrays, simulate_draft_orders

# What one starting slot loses to byes and injuries over a season, and so how
# much of a season a bench player behind that slot actually plays: one bye plus
# a couple of missed weeks out of seventeen.
BENCH_WEIGHT = 0.20


class BenchValue(NamedTuple):
    """How to price a player at this position who does not crack the lineup.

    `vacancies` is the expected number of this position's starting slots open
    in a given week -- `BENCH_WEIGHT` per slot he could be promoted into, flex
    share included. `baseline` is what he is promoted *over*: the best player
    at his position you could still have for nothing.
    """

    vacancies: float
    baseline: float


def promotion_weights(vacancies: float, depth: int) -> list[float]:
    """Share of weeks the 1st, 2nd, ... `depth`-th backup actually starts.

    Slots go vacant independently, so the number open in a given week is
    Poisson with mean `vacancies`, and your k-th backup starts in the weeks at
    least k of them are open. The weights sum back to `vacancies`, so a
    position's cover is split between however many backups you own instead of
    each being paid in full for it. That is what a hand-written "no more than
    two quarterbacks" rule is really trying to say: behind one starting slot
    the third quarterback comes out at 0.001 of a season on his own.
    """
    out: list[float] = []
    pmf = math.exp(-vacancies)  # P(X = k - 1)
    tail = 1.0  # P(X >= k - 1)
    for k in range(1, depth + 1):
        tail = max(tail - pmf, 0.0)
        out.append(tail)
        pmf *= vacancies / k
    return out


def bench_model(board: pl.DataFrame, league: dict | None = None) -> dict[str, BenchValue]:
    """Per-position bench pricing, derived from the pool still on the board.

    Two corrections, both of which the flat weight gets wrong:

    * **Baseline.** A backup is insurance, and insurance is worth what it saves
      you over the claim you would otherwise make -- the best player at that
      position you could still get for free. A QB2 projecting 249 behind a
      replacement level of 235 is insuring fourteen points, not 249.
    * **Weight.** How often that cover gets used scales with how many starting
      slots it stands behind. A bench running back backs up two starters and a
      share of the FLEX; a QB2 backs up one quarterback. `allocate_flex` already
      works out how the flex slots land across positions for this pool, so the
      slot count is derived rather than assumed. `promotion_weights` then
      splits that cover across however many backups you already own, which is
      what stops a third quarterback being worth as much as the second.

    Both are computed against the *available* board, so the baseline falls as
    the draft empties: in the last rounds "what you could get for free" really
    is a waiver-wire body, and late fliers separate again.
    """
    league = league or LEAGUE
    teams = league["teams"]
    counts = allocate_flex(board, league)
    levels = replacement_levels(board, league)
    return {
        pos: BenchValue(BENCH_WEIGHT * counts.get(pos, 0) / teams, level)
        for pos, level in levels.items()
    }


def optimal_lineup_points(
    roster: list[dict],
    league: dict | None = None,
    bench: dict[str, BenchValue] | None = None,
) -> float:
    """Points from the best legal starting lineup, plus discounted bench.

    Greedy fill is exact for this roster structure: required slots are filled
    best-first, then the single FLEX takes the best flex-eligible player left.

    `bench` prices whoever is left over, per position; see `bench_model`.
    Omitting it falls back to a flat share of raw projected points, which is
    only right when every player being compared plays the same position.
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
    # zero, which is wrong: he covers a bye and an injury. But he covers it only
    # as well as he beats the man you would stream instead, which is what the
    # baseline subtracts; below it there is nothing to insure and the term is
    # zero rather than negative.
    for pos, pts_list in by_pos.items():
        leftovers = pts_list[used.get(pos, 0):]
        if not leftovers:
            continue
        if bench is None or pos not in bench:
            total += BENCH_WEIGHT * sum(leftovers)
            continue
        vacancies, baseline = bench[pos]
        for weight, pts in zip(promotion_weights(vacancies, len(leftovers)), leftovers):
            total += weight * max(0.0, pts - baseline)

    return total


def marginal_value(
    board: pl.DataFrame,
    roster: list[dict],
    league: dict | None = None,
    bench: dict[str, BenchValue] | None = None,
) -> pl.DataFrame:
    """How much each available player would add to your optimal lineup."""
    bench = bench_model(board, league) if bench is None else bench
    base = optimal_lineup_points(roster, league, bench)
    gains = [
        optimal_lineup_points(roster + [{"pos": pos, "proj_points": pts}], league, bench) - base
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
