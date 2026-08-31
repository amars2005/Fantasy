"""Does drafting for upside win more titles than drafting for expected points?

The head-to-head analysis established that a three-week bracket is mostly
variance: a team scoring 25 points a week more than everyone else still only wins
55% of titles. If that is true, then a player's *spread* should be worth
something at the draft table over and above his mean -- and the current board
ignores spread entirely when picking.

This tests it directly. The pick criterion becomes `proj_points + lambda * sd`,
and lambda is swept from 0 (today's behaviour) upward. Everything is scored on
real historical outcomes through the real bracket, so the answer comes back in
titles rather than in rank correlation.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import DATA_PROCESSED, LEAGUE
from src.draft.h2h import simulate_season
from src.project.consensus import fit_curves

sys.path.insert(0, str(Path(__file__).resolve().parent))
from title_eval import (  # noqa: E402
    actual_weekly, board_for, consensus_projection, draft_with_vona,
    lineup_scores, weekly_matrix,
)

from src.config import REGULAR_SEASON_WEEKS  # noqa: E402


def projection_with_sd(season: int, curves: dict) -> pl.DataFrame:
    """Consensus projection plus the empirically measured spread at each rank."""
    from src.ingest.adp import load_adp
    from src.ingest.ids import resolve

    adp = resolve(load_adp(year=season)).filter(
        pl.col("pos").is_in(["QB", "RB", "WR", "TE"]) & pl.col("gsis_id").is_not_null()
    ).with_columns(pl.col("adp").rank("ordinal").over("pos").alias("r"))

    frames = []
    for pos, curve in curves.items():
        sub = adp.filter(pl.col("pos") == pos)
        if sub.height == 0:
            continue
        idx = np.clip(sub["r"].to_numpy().astype(int) - 1, 0, len(curve["grid"]) - 1)
        frames.append(sub.select(
            pl.col("gsis_id").alias("player_id"),
            pl.Series("proj_points", curve["mean"][idx]),
            pl.Series("sd", curve["sd"][idx]),
        ))
    return pl.concat(frames).unique(subset=["player_id"], keep="first")


def run(season: int, lam: float, slot: int, n_drafts: int, rng, curves) -> dict:
    projection = projection_with_sd(season, curves)
    # The upside tilt enters as the value the draft logic optimises.
    tilted = projection.with_columns(
        (pl.col("proj_points") + lam * pl.col("sd")).alias("proj_points")
    )
    board = board_for(season, tilted)
    if board.height < 120:
        return {}

    weekly = weekly_matrix(board, actual_weekly(season))
    positions = np.array(board["pos"].to_list())

    titles, playoffs, points = [], [], []
    for _ in range(n_drafts):
        rosters = draft_with_vona(board, slot, rng, n_sims=500)
        scores = np.zeros((LEAGUE["teams"], 1, REGULAR_SEASON_WEEKS))
        for team, idx in rosters.items():
            scores[team - 1, 0] = lineup_scores(idx, weekly, positions)
        result = simulate_season(scores, rng)
        titles.append(result["title_rate"][slot - 1])
        playoffs.append(result["playoff_rate"][slot - 1])
        points.append(result["points_for"][slot - 1])
    return {
        "season": season, "lambda": lam,
        "title_rate": float(np.mean(titles)),
        "playoff_rate": float(np.mean(playoffs)),
        "points_for": float(np.mean(points)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=8)
    ap.add_argument("--drafts", type=int, default=24)
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.0, 0.15, 0.3, 0.5])
    args = ap.parse_args()

    seasons = list(range(2021, 2026))
    span = list(range(2017, 2026))
    rng = np.random.default_rng(23)

    rows = []
    for lam in args.lambdas:
        for season in seasons:
            curves = fit_curves([s for s in span if s < season], with_se=False)
            got = run(season, lam, args.slot, args.drafts, rng, curves)
            if got:
                rows.append(got)
            gc.collect()
        recent = [r for r in rows if r["lambda"] == lam]
        print(f"  lambda={lam:<5.2f}  title {np.mean([r['title_rate'] for r in recent]):.3f}"
              f"   playoff {np.mean([r['playoff_rate'] for r in recent]):.3f}"
              f"   pts {np.mean([r['points_for'] for r in recent]):.0f}")

    table = pl.DataFrame(rows)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.write_parquet(DATA_PROCESSED / "upside_sweep.parquet")

    summary = (
        table.group_by("lambda")
        .agg(
            pl.col("title_rate").mean().alias("title"),
            pl.col("title_rate").std().alias("title_sd"),
            pl.col("playoff_rate").mean().alias("playoff"),
            pl.col("playoff_rate").std().alias("playoff_sd"),
        )
        .sort("lambda")
    )
    print("\n" + "=" * 62)
    print(f"  Upside tilt sweep, slot {args.slot}, {len(seasons)} seasons "
          f"x {args.drafts} drafts")
    print(f"  Baselines: title 0.071   playoff 0.429")
    print("-" * 62)
    for r in summary.iter_rows(named=True):
        marker = "  <- today" if r["lambda"] == 0.0 else ""
        print(f"  lambda {r['lambda']:<5.2f} title {r['title']:.3f} "
              f"(sd {r['title_sd']:.3f})  playoff {r['playoff']:.3f}{marker}")
    print("=" * 62)

    best = summary.sort("title", descending=True).row(0, named=True)
    base = summary.filter(pl.col("lambda") == 0.0).row(0, named=True)
    delta = best["title"] - base["title"]
    se = float(np.sqrt(best["title_sd"] ** 2 + base["title_sd"] ** 2) / np.sqrt(len(seasons)))
    verdict = "REAL" if abs(delta) > 2 * se else "within noise"
    print(f"\n  Best lambda {best['lambda']}: {delta:+.4f} title rate vs today "
          f"(+/- {se:.4f})  {verdict}")


if __name__ == "__main__":
    main()
