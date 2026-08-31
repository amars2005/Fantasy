"""How firm is the board's recommendation, round by round?

`rank_sensitivity.py` already asks whether the *first* pick survives refitting
the projection curve. This asks the same question at every pick in the draft,
and reports the two numbers that decide whether a recommendation is worth
following:

    p_best   how often this player tops the board across plausible projections
    regret   what taking the runner-up instead costs, in projected points

The pattern to look for is that regret collapses long before p_best does. Once
the top of the board is a plateau, "best" becomes a lottery between players who
are worth the same, and a 25% confidence with half a point of regret is not a
hard decision -- it is a decision that does not matter. Rounds where regret is
still large are the ones worth thinking about.

    python scripts/pick_reliability.py --slot 8
    python scripts/pick_reliability.py --slot 8 --draws 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import LEAGUE
from src.draft.board import DraftBoard
from src.draft.reliability import DEFAULT_DRAWS


def _autodraft(board: DraftBoard, upto: int, rng: np.random.Generator) -> None:
    """Advance the room to `upto` by taking players in simulated ADP order.

    Rough, and deliberately so: this is about how firm the board is at a typical
    round-N state, not about predicting a specific draft.
    """
    while board.pick_number < upto:
        left = board.available
        if left.height == 0:
            return
        noisy = (
            left["adp"].to_numpy().astype(float)
            + rng.normal(0, 1, left.height)
            * np.maximum(left["stdev"].fill_null(1.0).to_numpy().astype(float), 0.5)
        )
        taken = left["player_id"][int(np.argmin(noisy))]
        board.draft(taken, "me" if board.pick_number in board.picks else "other")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=8)
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(4)
    board = DraftBoard(slot=args.slot, resume=False)
    picks = board.picks[: args.rounds]

    print(f"Pick reliability, slot {args.slot}, {args.draws} projection draws\n")
    print(f"  {'rnd':>3s} {'pick':>4s}  {'recommendation':22s} {'pos':3s} "
          f"{'vona':>7s} {'p_best':>7s} {'regret':>7s}  runner-up")
    print("  " + "-" * 82)

    rows = []
    for rnd, pick in enumerate(picks, start=1):
        _autodraft(board, pick, rng)
        rec = board.recommend(n=3, n_draws=args.draws)
        conf = rec.get("confidence") or {}
        if not rec["recommendations"]:
            break
        top = rec["recommendations"][0]
        if conf.get("verdict", "unavailable") == "unavailable":
            print(f"  {rnd:3d} {pick:4d}  {top['name'][:22]:22s} {top['pos']:3s} "
                  f"{top['vona']:7.1f} {'-':>7s} {'-':>7s}  "
                  "(no uncertainty on this board)")
            continue
        print(f"  {rnd:3d} {pick:4d}  {top['name'][:22]:22s} {top['pos']:3s} "
              f"{top['vona']:7.1f} {conf['p_best']:7.1%} "
              f"{conf['runner_up_regret']:7.1f}  {conf['runner_up']} "
              f"[{conf['verdict']}]")
        rows.append({
            "round": rnd, "pick": pick, "player": top["name"], "pos": top["pos"],
            "vona": top["vona"], "p_best": conf["p_best"],
            "runner_up_regret": conf["runner_up_regret"],
            "verdict": conf["verdict"],
            "agrees_with_point_estimate": conf.get("agrees_with_point_estimate"),
        })
        # Take our own recommendation so the next round starts from a roster the
        # tool actually built.
        board.draft(top["player_id"], "me")

    if not rows:
        raise SystemExit("no rounds produced a reliability estimate")

    table = pl.DataFrame(rows)
    coin = table.filter(pl.col("runner_up_regret") < LEAGUE["teams"] / 14.0)
    disagree = table.filter(pl.col("agrees_with_point_estimate") == False)  # noqa: E712

    print("\n" + "=" * 84)
    print(f"  Median confidence in the top pick : {table['p_best'].median():.0%}")
    print(f"  Median cost of taking the runner-up: "
          f"{table['runner_up_regret'].median():.2f} projected points")
    print(f"  Rounds where the top two are effectively tied: "
          f"{coin.height}/{table.height}")
    if disagree.height:
        print(f"  Rounds where averaging over projection error changes the pick: "
              f"{disagree.height}  ({', '.join(disagree['player'].to_list())})")
    else:
        print("  Averaging over projection error never changed the pick.")
    print("=" * 84)


if __name__ == "__main__":
    main()
