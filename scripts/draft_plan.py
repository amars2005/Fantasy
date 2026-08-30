"""Round-by-round draft plan for one slot.

Answers the question you actually prepare with: at each of my picks, who is
realistically going to be there, and which position should I take?

The ranking is by **cost of waiting**, not by raw value over replacement. Those
two disagree constantly and the difference matters. Josh Allen carries a large
static VOR, so a VOR-ranked sheet says take him in round 3 -- but the quarterback
curve is flat, and an essentially identical quarterback is there two rounds
later. What should drive the pick is how much the position degrades before you
choose again, which is what this prints.

Survival probabilities come from the calibrated draft simulator, so "% there"
means what it says rather than a guess read off ADP.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import LEAGUE, SEASON
from src.draft.replacement import add_vor
from src.draft.sim_draft import (
    add_calibrated_adp,
    board_arrays,
    simulate_draft_orders,
    snake_picks,
)
from src.draft.tiers import add_tiers
from src.project.consensus import project

POSITIONS = ("RB", "WR", "TE", "QB")
URGENT = 5.0  # points of drop-off below which waiting is essentially free


def plan(slot: int, season: int = SEASON, rounds: int = 10, n_sims: int = 6000) -> None:
    board = add_calibrated_adp(add_tiers(add_vor(project(season))))
    picks = snake_picks(slot, LEAGUE["teams"], LEAGUE["rounds"])

    mu, stdev = board_arrays(board)
    positions = simulate_draft_orders(mu, stdev, n_sims, np.random.default_rng(11))

    names = board["name"].to_list()
    tiers = board["tier"].to_list()
    pos_arr = np.array(board["pos"].to_list())
    vor = board["vor"].to_numpy()
    adp = board["adp"].to_numpy()

    def expected_best(pick_no: int, position: str) -> float:
        """Expected VOR of the best player at `position` still there at a pick."""
        mask = pos_arr == position
        if not mask.any():
            return 0.0
        vals = np.where(positions[:, mask] >= pick_no, vor[mask][None, :], -np.inf)
        best = vals.max(axis=1)
        return float(np.where(np.isfinite(best), best, 0.0).mean())

    print(f"DRAFT PLAN -- slot {slot} of {LEAGUE['teams']}, full PPR\n")
    print(f"Your picks: {', '.join(str(p) for p in picks)}\n")

    for rnd, pick in enumerate(picks[:rounds], 1):
        following = picks[rnd] if rnd < len(picks) else None
        p_avail = (positions >= pick).mean(axis=0)

        rows = []
        for p in POSITIONS:
            options = [
                (names[i], adp[i], vor[i], tiers[i], p_avail[i])
                for i in np.flatnonzero(pos_arr == p)
                if p_avail[i] >= 0.40
            ]
            if not options:
                continue
            name, a, v, t, avail = max(options, key=lambda r: r[2])
            waiting_cost = v - (expected_best(following, p) if following else 0.0)
            rows.append((p, name, a, v, t, avail, waiting_cost))

        print(f"--- ROUND {rnd}  (pick {pick}) ---")
        if not rows:
            print("   nothing meaningful projected available\n")
            continue

        rows.sort(key=lambda r: -r[6])
        for p, name, a, v, t, avail, cost in rows:
            print(f"   {p}: {name:22s} adp {a:5.1f}  vor {v:6.1f}  T{t:<2d} "
                  f"{avail * 100:3.0f}% there   waiting costs {cost:6.1f}")

        top = rows[0]
        if top[6] > URGENT:
            print(f"   -> {top[0]} is urgent: take {top[1]} "
                  f"(waiting costs {top[6]:.1f} pts)\n")
        else:
            print(f"   -> no position is urgent (max {top[6]:.1f} pts) "
                  f"-- take the best player available\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()
    plan(args.slot, args.season, args.rounds)


if __name__ == "__main__":
    main()
