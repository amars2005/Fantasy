"""Does college data help price rookies?

Rookies are the one group where the market has least to go on: no NFL snaps, no
box score, nothing but draft capital and scouting reports. If a model is ever
going to beat ADP, this is where it should happen.

So this asks the narrow question directly. For each season, train on prior
seasons and rank that year's drafted rookies three ways -- by ADP, by the model
without college features, and by the model with them -- then score all three by
within-position rank correlation against what the rookies actually did.

Pooled across seasons, because a single year has only twenty or so rookies worth
drafting and any one year's answer is noise.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.config import DATA_PROCESSED, SEASON
from src.features.build import add_targets, build
from src.features.player_season import build as build_player_season
from src.models.points import _matrix, train

SKILL = ["QB", "RB", "WR", "TE"]


def _rank_within_pos(df: pl.DataFrame, col: str, descending: bool) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col).rank("ordinal", descending=descending).over("pos").alias(f"{col}_rank")
    )


def _pooled_spearman(frames: list[pl.DataFrame], col: str) -> float:
    """Rank correlation pooled across seasons, computed within position-season."""
    xs, ys = [], []
    for df in frames:
        for pos in SKILL:
            sub = df.filter(pl.col("pos") == pos)
            if sub.height < 4:
                continue
            # Rank within this season-position, so seasons are comparable.
            pred = sub[col].rank(descending=False).to_numpy()
            actual = sub["y_points"].rank(descending=False).to_numpy()
            xs.extend(pred)
            ys.extend(actual)
    if len(xs) < 20:
        return float("nan")
    rho, _ = spearmanr(xs, ys)
    return float(rho)


def run(seasons: list[int]) -> None:
    span = list(range(min(seasons) - 3, max(seasons) + 1))
    stats = build_player_season(span)
    features = add_targets(build(span, stats=stats), stats)
    del stats
    gc.collect()

    plain_frames, college_frames, adp_frames = [], [], []
    for year in seasons:
        train_df = features.filter(pl.col("season") < year)
        rookies = features.filter(
            (pl.col("season") == year)
            & (pl.col("is_rookie") == 1)
            & (pl.col("adp") < 400)
            & (pl.col("pos").is_in(SKILL))
        )
        if train_df.height < 300 or rookies.height < 6:
            continue

        fit = train_df.filter(pl.col("y_games") > 0).with_columns(
            (pl.col("y_points") / pl.col("y_games")).alias("y_ppg")
        )

        for use_college, bucket in ((False, plain_frames), (True, college_frames)):
            ppg_model = train(fit, "y_ppg", use_market=True, use_college=use_college)
            games_model = train(train_df, "y_games", use_market=True, use_college=use_college)
            x, _ = _matrix(rookies, use_market=True, use_college=use_college)
            pred = np.clip(ppg_model.predict(x), 0, None) * np.clip(games_model.predict(x), 0, 17)
            bucket.append(rookies.with_columns(pl.Series("pred", pred)))

        adp_frames.append(rookies.with_columns((-pl.col("adp")).alias("pred")))
        print(f"  {year}: {rookies.height:2d} drafted rookies with an ADP")
        gc.collect()

    if not adp_frames:
        raise SystemExit("no seasons with enough rookies")

    adp_rho = _pooled_spearman(adp_frames, "pred")
    plain_rho = _pooled_spearman(plain_frames, "pred")
    college_rho = _pooled_spearman(college_frames, "pred")
    n = sum(f.height for f in adp_frames)

    print(f"\n{'=' * 62}")
    print(f"  ROOKIES ONLY -- pooled over {len(adp_frames)} seasons, n={n}")
    print("-" * 62)
    print(f"  ADP                        rho : {adp_rho:+.4f}")
    print(f"  model without college data rho : {plain_rho:+.4f}  ({plain_rho - adp_rho:+.4f})")
    print(f"  model with college data    rho : {college_rho:+.4f}  ({college_rho - adp_rho:+.4f})")
    print("=" * 62)

    if college_rho > adp_rho and college_rho > plain_rho:
        print("\n  College data helps, and the model beats ADP on rookies.")
    elif college_rho > plain_rho:
        print("\n  College data improves the model, but it still trails ADP on rookies.")
    else:
        print("\n  College data does not improve rookie projections.")
    print("\n  Sample is small; treat the sign as weak evidence, not a result.")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{
        "n": n, "seasons": len(adp_frames),
        "adp_rho": adp_rho, "plain_rho": plain_rho, "college_rho": college_rho,
    }]).write_parquet(DATA_PROCESSED / "rookie_backtest.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2020)
    ap.add_argument("--to-season", type=int, default=2025)
    args = ap.parse_args()
    print(f"Rookie evaluation {args.from_season}-{args.to_season}\n")
    run(list(range(args.from_season, args.to_season + 1)))


if __name__ == "__main__":
    main()
