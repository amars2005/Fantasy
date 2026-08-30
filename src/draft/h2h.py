"""Head-to-head season and playoff simulation.

This league is not won by scoring the most points. It is won by finishing in the
top six of fourteen over fourteen weekly matchups, then winning three
single-elimination games. Those objectives differ in a way that matters for draft
strategy: weekly consistency wins matchups, while upside wins brackets, and a
team optimised purely for season-long total points is optimised for neither.

Seeds one and two receive first-round byes, which is worth a great deal in a
three-week bracket and makes the regular season count for more than it first
appears.
"""

from __future__ import annotations

import numpy as np

from src.config import SCHEDULE


def round_robin(teams: int, weeks: int) -> list[list[tuple[int, int]]]:
    """A balanced weekly schedule via the circle method.

    With fourteen teams a full round robin is thirteen weeks, so the fourteenth
    week repeats the first -- which is what real leagues do.
    """
    order = list(range(teams))
    schedule = []
    for _ in range(teams - 1):
        pairs = [
            (order[i], order[teams - 1 - i])
            for i in range(teams // 2)
        ]
        schedule.append(pairs)
        order = [order[0]] + [order[-1]] + order[1:-1]
    while len(schedule) < weeks:
        schedule.append(schedule[len(schedule) % (teams - 1)])
    return schedule[:weeks]


def simulate_season(weekly_scores: np.ndarray, rng: np.random.Generator) -> dict:
    """Play out the H2H season and playoffs.

    `weekly_scores` is (teams, n_sims, weeks). Returns per-team probabilities of
    making the playoffs and of winning the title.
    """
    teams, n_sims, weeks = weekly_scores.shape
    regular = SCHEDULE["regular_season_weeks"]
    playoff_weeks = SCHEDULE["playoff_weeks"]
    n_playoff = SCHEDULE["playoff_teams"]

    schedule = round_robin(teams, regular)
    wins = np.zeros((teams, n_sims))
    for week, pairs in enumerate(schedule):
        for a, b in pairs:
            a_won = weekly_scores[a, :, week] > weekly_scores[b, :, week]
            wins[a] += a_won
            wins[b] += ~a_won

    # Seed on record, breaking ties on total points as the league specifies.
    points_for = weekly_scores[:, :, :regular].sum(axis=2)
    seed_key = wins + points_for / (points_for.max() + 1.0) * 0.5
    # argsort descending -> seeds[0] is the top seed in each simulation
    seeds = np.argsort(-seed_key, axis=0)

    made_playoffs = np.zeros((teams, n_sims))
    for slot in range(n_playoff):
        made_playoffs[seeds[slot], np.arange(n_sims)] += 1

    # Bracket: seeds 1-2 bye; 3v6 and 4v5 in round one. Weeks are 1-based in
    # the league settings but 0-based as array indices.
    pw = [w - 1 for w in playoff_weeks]
    rows = np.arange(n_sims)

    def score(team_idx: np.ndarray, week_idx: int) -> np.ndarray:
        return weekly_scores[team_idx, rows, week_idx]

    # Track playoff *slots* (0 = top seed) rather than team ids, so reseeding can
    # compare who is the lower seed rather than the lower team number.
    def play(slot_a: np.ndarray, slot_b: np.ndarray, week_idx: int) -> np.ndarray:
        team_a, team_b = seeds[slot_a, rows], seeds[slot_b, rows]
        a_wins = score(team_a, week_idx) > score(team_b, week_idx)
        return np.where(a_wins, slot_a, slot_b)

    slot = lambda i: np.full(n_sims, i)
    w1 = play(slot(2), slot(5), pw[0])   # 3 vs 6
    w2 = play(slot(3), slot(4), pw[0])   # 4 vs 5

    # The top seed draws the worst surviving seed; higher slot index = worse seed.
    lower = np.maximum(w1, w2)
    higher = np.minimum(w1, w2)

    f1 = play(slot(0), lower, pw[1])
    f2 = play(slot(1), higher, pw[1])
    champ_slot = play(f1, f2, pw[2])
    champ = seeds[champ_slot, rows]

    titles = np.zeros((teams, n_sims))
    titles[champ, np.arange(n_sims)] = 1

    return {
        "playoff_rate": made_playoffs.mean(axis=1),
        "title_rate": titles.mean(axis=1),
        "points_for": points_for.mean(axis=1),
        "wins": wins.mean(axis=1),
    }
