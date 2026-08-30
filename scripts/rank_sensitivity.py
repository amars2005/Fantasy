"""How much does the board depend on the ranks being right?

The whole chain is rank-derived: ADP positional rank feeds an isotonic curve,
which produces projected points, which produce VOR, which produce VONA. So the
fair objection is that if the ranks are wrong, everything downstream is wrong.

Two things are worth separating, because they have different answers.

The *level* at each rank is badly determined. Each rank has only ten historical
observations and the spread is enormous -- the running back drafted first has
scored between 13 and 408 points. So any single projected total should be read as
a wide distribution, not a number.

The *decision* may still be stable, because VONA compares players against each
other rather than against an absolute. This resamples the seasons the curve is
fitted on, refits, and re-runs the board each time. If the recommendation holds
up under that, rank noise is not what should worry you. If it flips constantly,
the board is reporting false precision.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import LEAGUE, SEASON
from src.draft.replacement import add_vor
from src.draft.sim_draft import add_calibrated_adp, snake_picks
from src.draft.vona import vona_table
from src.ingest.adp import load_adp
from src.ingest.ids import resolve
from sklearn.isotonic import IsotonicRegression

from src.project.consensus import (
    SD_WINDOW, SKILL_POSITIONS, _rolling_sd, build_training_frame, fit_curves,
)

FIT_SEASONS = list(range(2016, 2026))


def curves_from_frame(train: pl.DataFrame) -> dict:
    """Fit the rank -> points curves from an explicit set of observations.

    Resampling *seasons* does not work as a bootstrap here: the training frame is
    built by joining on season, so duplicate seasons collapse and a ten-draw
    resample yields six or seven unique years. That measures the effect of having
    less data, not the uncertainty in the curve. Resampling rows is the honest
    version.
    """
    curves = {}
    for pos in SKILL_POSITIONS:
        sub = train.filter(pl.col("pos") == pos)
        if sub.height < 30:
            continue
        ranks = sub["pos_rank"].to_numpy().astype(float)
        points = sub["actual"].to_numpy().astype(float)
        model = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(ranks, points)
        grid = np.arange(1, int(ranks.max()) + 1, dtype=float)
        resid = points - model.predict(ranks)
        games = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            ranks, sub["games"].to_numpy().astype(float)
        )
        curves[pos] = {
            "grid": grid,
            "mean": model.predict(grid),
            "sd": _rolling_sd(ranks, resid, grid),
            "games": np.clip(games.predict(grid), 0, 17),
            "n": len(ranks),
        }
    return curves


def board_from_curves(curves: dict, season: int = SEASON) -> pl.DataFrame:
    adp = resolve(load_adp(year=season)).filter(pl.col("pos").is_in(SKILL_POSITIONS))
    adp = adp.with_columns(pl.col("adp").rank("ordinal").over("pos").alias("pos_rank"))

    frames = []
    for pos, curve in curves.items():
        sub = adp.filter(pl.col("pos") == pos)
        if sub.height == 0:
            continue
        idx = np.clip(sub["pos_rank"].to_numpy().astype(int) - 1, 0, len(curve["grid"]) - 1)
        frames.append(
            sub.select(
                pl.col("gsis_id").alias("player_id"), "name", "pos", "tm",
                "adp", "stdev", "bye",
            ).with_columns(
                pl.Series("proj_points", curve["mean"][idx]),
                pl.Series("sd", curve["sd"][idx]),
                pl.Series("games", curve["games"][idx]),
            )
        )
    board = pl.concat(frames).sort("adp")
    return add_calibrated_adp(add_vor(board, LEAGUE))


def top_pick(board: pl.DataFrame, gap: int, rng: np.random.Generator) -> tuple[str, str]:
    table = vona_table(board, [], gap, LEAGUE, n_sims=600, rng=rng)
    row = table.row(0, named=True)
    return row["name"], row["pos"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=8)
    ap.add_argument("--draws", type=int, default=40)
    args = ap.parse_args()

    picks = snake_picks(args.slot, LEAGUE["teams"], LEAGUE["rounds"])
    gap = picks[1] - picks[0]
    rng = np.random.default_rng(5)

    train = build_training_frame(FIT_SEASONS)
    baseline_curves = fit_curves(FIT_SEASONS)
    base_board = board_from_curves(baseline_curves)
    base_name, base_pos = top_pick(base_board, gap, rng)
    print(f"Baseline recommendation at pick {picks[0]}: {base_name} ({base_pos})\n")

    names, positions, spreads = Counter(), Counter(), []
    for draw in range(args.draws):
        # Proper bootstrap: resample observations, keeping the sample size.
        idx = rng.integers(0, train.height, train.height)
        try:
            curves = curves_from_frame(train[idx])
            board = board_from_curves(curves)
            name, pos = top_pick(board, gap, rng)
        except Exception:
            continue
        names[name] += 1
        positions[pos] += 1
        # How far does the projection for the very top of the board move?
        spreads.append(float(board["proj_points"].max()))

    total = sum(names.values())
    if not total:
        raise SystemExit("no successful bootstrap draws")

    print(f"Across {total} bootstrap refits of the rank->points curve:\n")
    print("  TOP RECOMMENDATION")
    for name, count in names.most_common(6):
        bar = "#" * round(30 * count / total)
        print(f"    {name:22s} {count / total:5.0%}  {bar}")
    print("\n  TOP RECOMMENDATION'S POSITION")
    for pos, count in positions.most_common():
        print(f"    {pos:4s} {count / total:5.0%}")

    agree = names[base_name] / total
    print(f"\n  Agreement with the shipped board: {agree:.0%}")
    print(f"  Top-of-board projection ranged {min(spreads):.0f} to {max(spreads):.0f} "
          f"points across refits")

    print()
    if positions.most_common(1)[0][1] / total > 0.9:
        print("  The POSITION is stable even though the points are not.")
        print("  Rank noise moves the numbers, not the decision.")
    else:
        print("  The recommendation is NOT stable under rank uncertainty.")
        print("  The board is reporting more precision than it has.")


if __name__ == "__main__":
    main()
