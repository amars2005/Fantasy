"""Draft strategy evaluation.

"Zero-RB vs Robust-RB" is usually argued from anecdote or from generic articles
written for 12-team leagues. This answers it for *this* league -- 14 teams, full
PPR, 2WR+1FLEX -- by playing the season out.

For each draft slot and each policy we run whole drafts against ADP-driven
opponents, simulate every team's season with weekly lineup decisions, and record
how often our roster finishes top of the league. Win rate is the metric, not
projected points: points ignore that you only have to beat thirteen other teams.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import LEAGUE, REGULAR_SEASON_WEEKS
from src.draft.h2h import simulate_season
from src.draft.sim_draft import board_arrays, snake_picks
from src.project.simulate import simulate_roster

# Bots will not hoard a position beyond what a sane manager rosters.
ROSTER_CAPS = {"QB": 2, "TE": 2, "RB": 6, "WR": 7}

# How many of the best-available players a bot weighs before picking. Bots follow
# ADP but fill obvious roster holes, which is what real managers do. Letting them
# draft on raw ADP alone would make every policy look brilliant by comparison and
# turn this study into "our optimiser beats naive bots" rather than a comparison
# between strategies.
BOT_CONSIDERS = 8

# A policy maps a round number (1-based) to the positions we are willing to take.
POLICIES: dict[str, dict] = {
    "BPA": {},
    "Robust-RB": {1: ("RB",), 2: ("RB",)},
    "Hero-RB": {1: ("RB",), 2: ("WR", "TE"), 3: ("WR", "TE"), 4: ("WR", "TE"), 5: ("WR", "TE")},
    "Zero-RB": {1: ("WR", "TE"), 2: ("WR", "TE"), 3: ("WR", "TE"), 4: ("WR", "TE")},
    "Elite-TE": {1: ("WR", "RB"), 2: ("TE",)},
    "Early-QB": {1: ("RB", "WR"), 2: ("QB",)},
    # Static VOR undervalues waiting at quarterback, because QB replacement level
    # barely moves for rounds at a time. This policy tests that directly.
    "Late-QB": {r: ("RB", "WR", "TE") for r in range(1, 9)},
}


def _lineup_value(counts: dict[str, int], pos: str, league: dict) -> float:
    """Crude marginal weight: starting slots first, then depth, then surplus."""
    starters = league["starters"]
    have = counts.get(pos, 0)
    need = starters.get(pos, 0)
    if have < need:
        return 1.0
    if pos in league["flex_eligible"] and have < need + 1:
        return 0.85
    if have < need + 2:
        return 0.45
    return 0.15


def simulate_one_draft(
    board: pl.DataFrame,
    slot: int,
    policy: dict,
    rng: np.random.Generator,
    league: dict | None = None,
) -> dict[int, list[int]]:
    """Run a full snake draft. Returns team index -> list of board row indices."""
    league = league or LEAGUE
    teams, rounds = league["teams"], league["rounds"]

    mu, stdev = board_arrays(board)
    positions = np.array(board["pos"].to_list())
    # Selection currency is value over replacement, not raw points. Scoring on
    # raw points makes every policy open with a quarterback -- he outscores every
    # running back on the board -- which is the exact error VOR exists to prevent.
    value = np.maximum(board["vor"].to_numpy().astype(float), 0.1)

    # Everyone drafts off the same noisy ordering; roster needs then filter it.
    preference = np.argsort(mu + rng.normal(0.0, 1.0, mu.size) * stdev)

    taken = np.zeros(mu.size, dtype=bool)
    rosters: dict[int, list[int]] = {t: [] for t in range(1, teams + 1)}
    counts: dict[int, dict[str, int]] = {t: {} for t in range(1, teams + 1)}

    for rnd in range(1, rounds + 1):
        order = range(1, teams + 1) if rnd % 2 else range(teams, 0, -1)
        for team in order:
            allowed = policy.get(rnd) if team == slot else None
            choice = None
            best_score = -np.inf
            considered = 0

            for idx in preference:
                if taken[idx]:
                    continue
                pos = str(positions[idx])
                if counts[team].get(pos, 0) >= ROSTER_CAPS.get(pos, 99):
                    continue
                if allowed is not None and pos not in allowed:
                    continue
                # Both we and the bots pick on lineup-aware value; bots simply
                # look at a shorter list of candidates, so they stay close to ADP.
                score = value[idx] * _lineup_value(counts[team], pos, league)
                if score > best_score:
                    best_score, choice = score, idx
                considered += 1
                if team != slot and considered >= BOT_CONSIDERS:
                    break

            if choice is None:  # policy left nothing legal; fall back to ADP
                remaining = preference[~taken[preference]]
                if remaining.size == 0:
                    continue
                choice = remaining[0]

            taken[choice] = True
            rosters[team].append(int(choice))
            counts[team][str(positions[choice])] = counts[team].get(str(positions[choice]), 0) + 1

    return rosters


def evaluate_policy(
    board: pl.DataFrame,
    slot: int,
    policy_name: str,
    n_drafts: int = 120,
    n_seasons: int = 30,
    seed: int = 0,
    league: dict | None = None,
) -> dict:
    """Playoff and title rates for one (slot, policy) pair.

    Scored the way the league is actually decided: fourteen head-to-head weeks,
    top six seeded into a three-week bracket. Total points is reported too, but
    it is not the objective -- a team can lead the league in scoring and lose in
    the semi-final, which is precisely what makes upside worth paying for.
    """
    league = league or LEAGUE
    policy = POLICIES[policy_name]
    rng = np.random.default_rng(seed)
    teams = league["teams"]

    playoff, titles, points = [], [], []
    for _ in range(n_drafts):
        rosters = simulate_one_draft(board, slot, policy, rng, league)

        weekly = np.zeros((teams, n_seasons, REGULAR_SEASON_WEEKS))
        for team, idx in rosters.items():
            weekly[team - 1] = simulate_roster(
                board[idx], n_sims=n_seasons, rng=rng, league=league, per_week=True
            )

        result = simulate_season(weekly, rng)
        playoff.append(result["playoff_rate"][slot - 1])
        titles.append(result["title_rate"][slot - 1])
        points.append(result["points_for"][slot - 1])

    title_rate = float(np.mean(titles))
    return {
        "slot": slot,
        "policy": policy_name,
        "title_rate": title_rate,
        "title_se": float(np.std(titles) / np.sqrt(n_drafts)),
        "playoff_rate": float(np.mean(playoff)),
        "playoff_se": float(np.std(playoff) / np.sqrt(n_drafts)),
        "mean_points": float(np.mean(points)),
        "n_drafts": n_drafts,
    }


def strategy_table(
    board: pl.DataFrame,
    slots: list[int] | None = None,
    policies: list[str] | None = None,
    n_drafts: int = 120,
    n_seasons: int = 30,
    league: dict | None = None,
) -> pl.DataFrame:
    league = league or LEAGUE
    slots = slots or list(range(1, league["teams"] + 1))
    policies = policies or list(POLICIES)

    rows = [
        evaluate_policy(board, slot, name, n_drafts, n_seasons, seed=1000 * slot + i, league=league)
        for slot in slots
        for i, name in enumerate(policies)
    ]
    return pl.DataFrame(rows).sort(["slot", "title_rate"], descending=[False, True])
