"""Can a model beat ADP at ranking defences?

Worth asking here when it was not worth asking for skill positions. Defences
persist year to year at r = 0.276 against the kicker's 0.109, and defence ADP is
thin and lightly contested -- most drafters pick a defence in the last two rounds
on reputation. That is a much weaker opponent than the aggregated consensus that
beat every skill-position model in this project.

Features are all knowable in August: last season's scoring split into its parts
(events versus the points- and yards-allowed bands), and the coming season's
schedule difficulty measured by what this defence's opponents scored last year.
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
from src.dst import season_dst_points
from src.features.schedule import team_schedule_strength
from src.ingest.adp import load_adp
from src.ingest.ids import normalise_team

FEATURES = [
    "points_lag1", "event_lag1", "pa_lag1", "ya_lag1",
    "points_lag2", "event_lag2",
    "sos_regular", "sos_playoff", "playoff_lift",
]

PARAMS = dict(
    objective="regression", metric="l2", learning_rate=0.05, num_leaves=7,
    min_data_in_leaf=15, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=8.0, verbosity=-1, num_threads=2,
    seed=0, bagging_seed=0, feature_fraction_seed=0, deterministic=True,
)


def build_panel(seasons: list[int]) -> pl.DataFrame:
    history = season_dst_points(seasons).select(
        "team", "season", "points", "event_points", "pa_points", "ya_points"
    )

    lagged = []
    for lag in (1, 2):
        lagged.append(
            history.select(
                "team",
                (pl.col("season") + lag).cast(pl.Int32).alias("season"),
                pl.col("points").alias(f"points_lag{lag}"),
                pl.col("event_points").alias(f"event_lag{lag}"),
                pl.col("pa_points").alias(f"pa_lag{lag}"),
                pl.col("ya_points").alias(f"ya_lag{lag}"),
            )
        )

    panel = history.select("team", pl.col("season").cast(pl.Int32), pl.col("points").alias("y"))
    for frame in lagged:
        panel = panel.join(frame, on=["team", "season"], how="left")

    # Schedule difficulty for the season being predicted.
    sched = []
    for season in sorted(panel["season"].unique().to_list()):
        try:
            got = team_schedule_strength(season).with_columns(
                pl.lit(season).cast(pl.Int32).alias("season")
            )
            sched.append(got)
        except Exception:
            continue
    if sched:
        panel = panel.join(pl.concat(sched), on=["team", "season"], how="left")
    return panel.drop_nulls("points_lag1")


def dst_adp(season: int) -> pl.DataFrame:
    adp = load_adp(year=season).filter(pl.col("position") == "DEF")
    return adp.select(
        pl.col("team").map_elements(normalise_team, return_dtype=pl.Utf8).alias("team"),
        pl.col("adp").cast(pl.Float64),
    ).unique(subset=["team"], keep="first")


def _matrix(df: pl.DataFrame) -> np.ndarray:
    return np.column_stack([
        df[c].cast(pl.Float64).fill_null(0).to_numpy().astype(np.float32)
        if c in df.columns else np.zeros(df.height, dtype=np.float32)
        for c in FEATURES
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2021)
    ap.add_argument("--to-season", type=int, default=2025)
    args = ap.parse_args()

    seasons = list(range(2016, args.to_season + 1))
    panel = build_panel(seasons)
    print(f"panel rows {panel.height}\n")

    rows = []
    for year in range(args.from_season, args.to_season + 1):
        train = panel.filter(pl.col("season") < year)
        test = panel.filter(pl.col("season") == year)
        if train.height < 60 or test.height < 20:
            continue

        model = lgb.train(
            PARAMS,
            lgb.Dataset(_matrix(train), label=train["y"].to_numpy().astype(float)),
            num_boost_round=250,
        )
        scored = test.with_columns(pl.Series("model", model.predict(_matrix(test))))
        merged = scored.join(dst_adp(year), on="team", how="inner")
        # Historical DEF coverage in ADP is thin -- 12 to 16 teams a year.
        if merged.height < 10:
            continue

        m_rho, _ = spearmanr(merged["model"].to_numpy(), merged["y"].to_numpy())
        a_rho, _ = spearmanr(-merged["adp"].to_numpy(), merged["y"].to_numpy())
        p_rho, _ = spearmanr(merged["points_lag1"].to_numpy(), merged["y"].to_numpy())
        rows.append({"season": year, "n": merged.height,
                     "model": m_rho, "adp": a_rho, "lastyear": p_rho})
        print(f"  {year}: n={merged.height:2d}  model {m_rho:+.3f}   "
              f"ADP {a_rho:+.3f}   last-year {p_rho:+.3f}")
        gc.collect()

    if not rows:
        raise SystemExit("no seasons evaluated")
    table = pl.DataFrame(rows)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.write_parquet(DATA_PROCESSED / "dst_model.parquet")

    n = table.height
    print("\n" + "=" * 60)
    print(f"  D/ST ranking, {n} seasons")
    print("-" * 60)
    for label in ("model", "adp", "lastyear"):
        mean = float(table[label].mean())
        sd = table[label].std()
        se = float(sd / np.sqrt(n)) if sd is not None and n > 1 else float("nan")
        print(f"  {label:10s} rho {mean:+.4f} +/- {se:.4f}")
    print("=" * 60)

    diff = table["model"] - table["adp"]
    edge = float(diff.mean())
    se = float(diff.std() / np.sqrt(n)) if n > 1 and diff.std() is not None else float("nan")
    wins = int((table["model"] > table["adp"]).sum())
    verdict = "PASSES" if edge > 2 * se else ("ahead, within noise" if edge > 0 else "FAILS")
    print(f"\n  model vs ADP: {edge:+.4f} +/- {se:.4f}   beat ADP {wins}/{n}   {verdict}")
    if edge > 0:
        print("  Unlike the skill positions, defence ADP is beatable territory.")


if __name__ == "__main__":
    main()
