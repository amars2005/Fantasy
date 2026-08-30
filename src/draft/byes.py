"""Bye-week aware roster valuation.

Bye weeks were displayed on the board but never entered a decision. In a
fourteen-week head-to-head season that is a real omission: three starters sharing
a bye is close to a guaranteed loss, and one loss matters far more for seeding in
a six-of-fourteen playoff race than the equivalent points would in a
total-points league.

The honest way to price this is not a penalty term. It is to value a roster the
way it is actually used -- week by week, with players on bye unavailable -- and
let the collision cost fall out. A roster whose byes are spread has more startable
weeks than one whose byes stack, even when both project identically over a season.
"""

from __future__ import annotations

import numpy as np

from src.config import LEAGUE, SCHEDULE

# Fantasy weeks that decide seeding. Byes land inside this window; playoff
# weeks 15-17 have no byes, so they cannot collide.
REGULAR_WEEKS = tuple(range(1, SCHEDULE["regular_season_weeks"] + 1))


def weekly_available(byes: np.ndarray, week: int) -> np.ndarray:
    """Boolean mask of players active in a given week."""
    return byes != week


def roster_weekly_value(
    points: np.ndarray, positions: np.ndarray, byes: np.ndarray,
    league: dict | None = None, bench_weight: float = 0.20,
) -> float:
    """Season value of a roster, summed over weeks with byes enforced.

    `points` is per-season projection; it is spread evenly across the weeks a
    player is actually available, which is what makes a stacked bye cost more
    than a spread one.
    """
    league = league or LEAGUE
    starters = league["starters"]
    eligible = set(league["flex_eligible"])
    n_weeks = len(REGULAR_WEEKS)
    if points.size == 0:
        return 0.0

    per_week = points / max(n_weeks, 1)
    total = 0.0

    for week in REGULAR_WEEKS:
        active = weekly_available(byes, week)
        if not active.any():
            continue
        week_points = per_week[active]
        week_pos = positions[active]

        used: dict[str, int] = {}
        ranked: dict[str, np.ndarray] = {}
        for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
            block = np.sort(week_points[week_pos == pos])[::-1]
            ranked[pos] = block
            slots = starters.get(pos, 0)
            take = block[:slots]
            total += take.sum()
            used[pos] = take.size

        flex_slots = starters.get("FLEX", 0)
        leftovers = np.concatenate(
            [ranked[p][used.get(p, 0):] for p in eligible if ranked[p].size > used.get(p, 0)]
        ) if any(ranked[p].size > used.get(p, 0) for p in eligible) else np.array([])
        if flex_slots and leftovers.size:
            leftovers = np.sort(leftovers)[::-1]
            total += leftovers[:flex_slots].sum()
            bench = leftovers[flex_slots:]
        else:
            bench = leftovers
        total += bench_weight * bench.sum()

    return float(total)


def bye_collision_cost(
    roster_points: np.ndarray, roster_positions: np.ndarray, roster_byes: np.ndarray,
    league: dict | None = None,
) -> float:
    """How much this roster loses purely to its bye distribution.

    Measured against the same roster with byes spread perfectly, so the number is
    the avoidable part -- not the unavoidable cost of byes existing at all.
    """
    actual = roster_weekly_value(roster_points, roster_positions, roster_byes, league)
    if roster_points.size == 0:
        return 0.0
    # Counterfactual: same players, byes dealt round-robin across the season.
    spread = np.array(
        [REGULAR_WEEKS[i % len(REGULAR_WEEKS)] for i in range(roster_points.size)]
    )
    ideal = roster_weekly_value(roster_points, roster_positions, spread, league)
    return float(ideal - actual)
