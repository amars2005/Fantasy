"""Is there a weekly signal worth building on?

The draft is a single decision against a brutally efficient market. The other
fourteen weeks are not: there is no ADP for a week 9 start/sit call, so the
baseline is whatever heuristic a manager would use unaided rather than the
aggregated wisdom of thousands of drafters.

This probes whether that softer target is real before committing to building an
in-season system. It predicts each player's *next* week from information
available at the time, and scores it against the heuristics a person would
actually use: season-to-date average, and a recent-form average.

If the model cannot beat "just use his average", there is no in-season project
worth starting. If it can, the margin here is the honest estimate of what a full
build would be worth.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.config import DATA_PROCESSED
from src.ingest import nflverse as nv
from src.scoring import add_fantasy_points

SKILL = ["QB", "RB", "WR", "TE"]
MIN_WEEK = 5      # need some history before a rolling feature means anything
MAX_WEEK = 16     # predicting week+1, so stop before the last week

PARAMS = dict(
    objective="regression", metric="l2", learning_rate=0.05, num_leaves=15,
    min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, verbosity=-1, num_threads=2,
    seed=0, bagging_seed=0, feature_fraction_seed=0, deterministic=True,
)
FEATURES = [
    "ppg_to_date", "last3", "last1", "trend", "games_played",
    "targets_pg", "carries_pg", "target_share_avg", "snap_trend", "week",
]


def build_panel(seasons: list[int]) -> pl.DataFrame:
    """Player-week rows with rolling features and next week's outcome."""
    frames = []
    for season in seasons:
        weekly = nv.player_stats(seasons=[season], level="week")
        if "season_type" in weekly.columns:
            weekly = weekly.filter(pl.col("season_type") == "REG")
        scored = add_fantasy_points(weekly)
        pos_col = "position" if "position" in scored.columns else "position_group"

        base = (
            scored.filter(pl.col(pos_col).is_in(SKILL))
            .select(
                pl.col("player_id"), pl.col(pos_col).alias("pos"),
                pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
                pl.col("fantasy_points_league").alias("pts"),
                pl.col("targets").fill_null(0).cast(pl.Float64),
                pl.col("carries").fill_null(0).cast(pl.Float64),
                pl.col("target_share").fill_null(0).cast(pl.Float64),
            )
            .sort(["player_id", "week"])
        )

        # All features are cumulative or trailing, so nothing sees the future.
        base = base.with_columns(
            pl.col("pts").cum_sum().over("player_id").alias("_cum"),
            pl.int_range(1, pl.len() + 1).over("player_id").cast(pl.Float64).alias("games_played"),
        ).with_columns(
            (pl.col("_cum") / pl.col("games_played")).alias("ppg_to_date"),
            pl.col("pts").rolling_mean(3, min_samples=1).over("player_id").alias("last3"),
            pl.col("pts").alias("last1"),
            pl.col("targets").rolling_mean(3, min_samples=1).over("player_id").alias("targets_pg"),
            pl.col("carries").rolling_mean(3, min_samples=1).over("player_id").alias("carries_pg"),
            pl.col("target_share").rolling_mean(3, min_samples=1).over("player_id").alias("target_share_avg"),
        ).with_columns(
            (pl.col("last3") - pl.col("ppg_to_date")).alias("trend"),
            (pl.col("targets") + pl.col("carries")
             - pl.col("targets_pg") - pl.col("carries_pg")).alias("snap_trend"),
            # The label: what he scores the following week.
            pl.col("pts").shift(-1).over("player_id").alias("y_next"),
            pl.col("week").shift(-1).over("player_id").alias("_next_week"),
        )

        # Only keep rows where the next row really is the next week.
        frames.append(
            base.filter(
                (pl.col("_next_week") == pl.col("week") + 1)
                & pl.col("y_next").is_not_null()
                & (pl.col("week") >= MIN_WEEK)
                & (pl.col("week") <= MAX_WEEK)
            ).drop("_cum", "_next_week")
        )
        gc.collect()
    return pl.concat(frames)


def _matrix(df: pl.DataFrame) -> np.ndarray:
    blocks = [df[c].cast(pl.Float64).fill_null(0).to_numpy().astype(np.float32)
              for c in FEATURES]
    pos = df["pos"].to_numpy()
    for p in SKILL:
        blocks.append((pos == p).astype(np.float32))
    return np.column_stack(blocks)


def _rho(df: pl.DataFrame, col: str) -> float:
    """Within position-week rank correlation against next week's points."""
    total, weight = 0.0, 0.0
    for (_, _), sub in df.group_by(["pos", "week"], maintain_order=True):
        if sub.height < 8:
            continue
        r, _ = spearmanr(sub[col].to_numpy(), sub["y_next"].to_numpy())
        if not np.isnan(r):
            total += r * sub.height
            weight += sub.height
    return total / weight if weight else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-through", type=int, default=2023)
    ap.add_argument("--test-seasons", type=int, nargs="*", default=[2024, 2025])
    args = ap.parse_args()

    seasons = list(range(2019, max(args.test_seasons) + 1))
    panel = build_panel(seasons)
    train = panel.filter(pl.col("season") <= args.train_through)
    test = panel.filter(pl.col("season").is_in(args.test_seasons))
    print(f"train rows {train.height}   test rows {test.height}\n")

    model = lgb.train(
        PARAMS,
        lgb.Dataset(_matrix(train), label=train["y_next"].to_numpy().astype(float)),
        num_boost_round=300,
    )
    test = test.with_columns(pl.Series("model", model.predict(_matrix(test))))

    results = {
        "season-to-date average": _rho(test, "ppg_to_date"),
        "last 3 weeks":           _rho(test, "last3"),
        "last week only":         _rho(test, "last1"),
        "model":                  _rho(test, "model"),
    }
    best_heuristic = max(
        v for k, v in results.items() if k != "model"
    )

    print("=" * 58)
    print(f"  Predicting NEXT week, {args.test_seasons}, n={test.height}")
    print("-" * 58)
    for label, rho in sorted(results.items(), key=lambda kv: -kv[1]):
        mark = "  <- model" if label == "model" else ""
        print(f"  {label:24s} rho {rho:+.4f}{mark}")
    print("=" * 58)

    edge = results["model"] - best_heuristic
    print(f"\n  Model vs best heuristic: {edge:+.4f}")
    if edge > 0.02:
        print("  Meaningful weekly signal. An in-season system is worth building.")
    elif edge > 0:
        print("  Positive but thin. Weigh the build cost against this margin.")
    else:
        print("  No weekly edge over simple averaging. Do not build the in-season system.")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{"metric": k, "rho": v} for k, v in results.items()]).write_parquet(
        DATA_PROCESSED / "inseason_probe.parquet"
    )


if __name__ == "__main__":
    main()
