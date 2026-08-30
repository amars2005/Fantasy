"""Build the projections contract for a season.

Writes data/processed/projections.parquet with the columns every downstream
consumer relies on, plus a report of where the projection disagrees most sharply
with the market -- those are the picks worth thinking hardest about.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src.config import DATA_PROCESSED, SEASON
from src.draft.replacement import add_vor, replacement_levels
from src.draft.tiers import add_tiers
from src.project.consensus import project

CONTRACT = ["player_id", "name", "pos", "tm", "proj_points", "sd", "games", "source"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=SEASON)
    args = ap.parse_args()

    proj = project(args.season)
    board = add_tiers(add_vor(proj))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "projections.parquet"
    board.write_parquet(out)

    missing = [c for c in CONTRACT if c not in board.columns]
    if missing:
        raise SystemExit(f"contract violation: missing {missing}")

    print(f"Saved {out}  ({board.height} players)\n")
    print("Replacement levels:")
    for pos, level in sorted(replacement_levels(board).items()):
        print(f"  {pos}: {level:.1f}")

    print("\n=== TOP 25 BY VALUE OVER REPLACEMENT ===")
    top = board.sort("vor", descending=True).head(25)
    for i, r in enumerate(top.iter_rows(named=True), 1):
        print(f"  {i:2d}. {r['name']:24s} {r['pos']:3s} {r['tm']:3s}  "
              f"adp {r['adp']:5.1f}  proj {r['proj_points']:6.1f}  vor {r['vor']:6.1f}  T{r['tier']}")

    # Where our value ranking and the market's draft order disagree most.
    gap = (
        board.with_columns(
            # rank() returns UInt32; subtracting without a cast underflows.
            pl.col("vor").rank("ordinal", descending=True).cast(pl.Int64).alias("value_rank"),
            pl.col("adp").rank("ordinal").cast(pl.Int64).alias("market_rank"),
        )
        .with_columns((pl.col("market_rank") - pl.col("value_rank")).alias("gap"))
    )
    print("\n=== BIGGEST VALUES (market drafts them later than their value) ===")
    for r in gap.sort("gap", descending=True).head(10).iter_rows(named=True):
        print(f"  {r['name']:24s} {r['pos']:3s}  adp {r['adp']:5.1f}  "
              f"value rank {r['value_rank']:3d}  (+{r['gap']} slots)")

    print("\n=== BIGGEST REACHES (market drafts them earlier than their value) ===")
    for r in gap.sort("gap").head(10).iter_rows(named=True):
        print(f"  {r['name']:24s} {r['pos']:3s}  adp {r['adp']:5.1f}  "
              f"value rank {r['value_rank']:3d}  ({r['gap']} slots)")


if __name__ == "__main__":
    main()
