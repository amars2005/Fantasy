"""Which draft strategy wins this league, from which slot.

Runs full drafts against roster-aware ADP opponents and plays out every team's
season with weekly lineup decisions. Offline analysis: takes a few minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src.config import DATA_PROCESSED, LEAGUE
from src.draft.replacement import add_vor
from src.draft.sim_draft import add_calibrated_adp
from src.draft.strategy import POLICIES, strategy_table
from src.project.consensus import project


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=int, default=150)
    ap.add_argument("--seasons", type=int, default=25)
    ap.add_argument("--slots", type=int, nargs="*", default=None)
    args = ap.parse_args()

    board = add_calibrated_adp(add_vor(project()))
    slots = args.slots or list(range(1, LEAGUE["teams"] + 1))

    print(f"Evaluating {len(POLICIES)} policies x {len(slots)} slots "
          f"x {args.drafts} drafts...")
    table = strategy_table(board, slots=slots, n_drafts=args.drafts, n_seasons=args.seasons)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "strategy_table.parquet"
    table.write_parquet(out)

    print(f"\nSaved {out}\n")
    overall = (
        table.group_by("policy")
        .agg(
            pl.col("title_rate").mean().alias("title"),
            pl.col("title_se").mean().alias("title_se"),
            pl.col("playoff_rate").mean().alias("playoff"),
            pl.col("mean_points").mean().alias("points"),
        )
        .sort("title", descending=True)
    )
    print("=== BY POLICY (baselines: title 0.071, playoff 0.429) ===")
    for r in overall.iter_rows(named=True):
        print(f"  {r['policy']:11s}  title {r['title']:.3f} +/- {r['title_se']:.3f}   "
              f"playoff {r['playoff']:.3f}   pts {r['points']:.0f}")

    print("\n=== BEST POLICY BY SLOT ===")
    best = (
        table.sort(["slot", "title_rate"], descending=[False, True])
        .group_by("slot").first()
    )
    for r in best.sort("slot").iter_rows(named=True):
        print(f"  slot {r['slot']:2d}: {r['policy']:11s} "
              f"title {r['title_rate']:.3f} +/- {r['title_se']:.3f}  "
              f"playoff {r['playoff_rate']:.3f}")


if __name__ == "__main__":
    main()
