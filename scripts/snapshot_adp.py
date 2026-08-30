"""Archive today's ADP so future seasons have velocity data.

Fantasy Football Calculator serves only a rolling seven-day window and ignores
every date parameter, so ADP movement cannot be reconstructed after the fact --
it has to be captured as it happens. Run this daily through preseason. One year
of snapshots unlocks both ADP velocity as a feature and the closing-line test as
an evaluation method, neither of which is possible from the live API alone.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src.config import DATA_RAW, LEAGUE, SEASON
from src.ingest.adp import load_adp

ARCHIVE = DATA_RAW.parent / "adp_history"


def snapshot(season: int = SEASON, scoring: str = "ppr") -> Path:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = ARCHIVE / f"adp_{scoring}_{season}_{today}.parquet"
    if path.exists():
        print(f"Already captured today: {path.name}")
        return path

    frame = load_adp(year=season, scoring=scoring, refresh=True).with_columns(
        pl.lit(today).alias("snapshot_date")
    )
    frame.write_parquet(path)
    print(f"Captured {frame.height} players -> {path.name}")
    print(f"  FFC window: {frame['adp_as_of'][0]}  drafts: {frame['total_drafts'][0]}")
    return path


def history(scoring: str = "ppr") -> pl.DataFrame:
    """All archived snapshots, for computing ADP velocity."""
    files = sorted(ARCHIVE.glob(f"adp_{scoring}_*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--scoring", default="ppr")
    args = ap.parse_args()
    snapshot(args.season, args.scoring)

    archive = history(args.scoring)
    if archive.height:
        dates = sorted(archive["snapshot_date"].unique().to_list())
        print(f"\nArchive: {len(dates)} snapshot(s), {dates[0]} -> {dates[-1]}")
        if len(dates) > 1:
            print("  Velocity features are now computable.")
        else:
            print("  Run daily; velocity needs at least two snapshots.")


if __name__ == "__main__":
    main()
