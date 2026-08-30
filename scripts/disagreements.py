"""Is the model worth listening to where it disagrees with the market?

The backtest establishes that the model does not out-rank ADP on average. That
settles whether it should replace consensus -- it should not. It does not settle
a narrower and more useful question: when the model disagrees *loudly*, is it
right often enough to be worth a second look?

This tests exactly that. For each historical season we isolate the players where
model and market disagree most, and score both on that subset alone. If the model
is better there, it earns a place as a flag. If it isn't, it doesn't, and the
honest thing is to stop consulting it.

Then it prints the same disagreement report for the current season.
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
from src.project.consensus import fit_curves

SKILL = ["QB", "RB", "WR", "TE"]
SHRINK = 0.35
TOP_N = 25  # players per direction counted as a "loud" disagreement


def _adp_baseline(df: pl.DataFrame, curves: dict) -> np.ndarray:
    ranks = (
        df.with_columns(pl.col("adp").rank("ordinal").over(["season", "pos"]).alias("r"))
        ["r"].to_numpy()
    )
    pos = df["pos"].to_numpy()
    out = np.zeros(len(ranks))
    for p, curve in curves.items():
        mask = pos == p
        if mask.any():
            idx = np.clip(ranks[mask].astype(int) - 1, 0, len(curve["grid"]) - 1)
            out[mask] = curve["mean"][idx]
    return out


def _fit_and_score(features: pl.DataFrame, span: list[int], year: int) -> pl.DataFrame | None:
    """Residual-corrected projection for one season, trained on prior seasons."""
    train_df = features.filter((pl.col("season") < year) & (pl.col("adp") < 400))
    score_df = features.filter((pl.col("season") == year) & (pl.col("adp") < 400))
    if train_df.height < 200 or score_df.height < 40:
        return None

    curves = fit_curves([s for s in span if s < year])
    base_train = _adp_baseline(train_df, curves)
    base_score = _adp_baseline(score_df, curves)

    resid_train = train_df.with_columns(
        pl.Series("y_resid", train_df["y_points"].to_numpy() - base_train)
    )
    model = train(resid_train, "y_resid", use_market=True)
    x, _ = _matrix(score_df, use_market=True)
    correction = SHRINK * model.predict(x)

    return score_df.with_columns(
        pl.Series("adp_implied", base_score),
        pl.Series("correction", correction),
        pl.Series("model_points", base_score + correction),
    ).with_columns(
        pl.col("model_points").rank("ordinal", descending=True).over("pos")
        .cast(pl.Int32).alias("model_pos_rank"),
        pl.col("adp").rank("ordinal").over("pos").cast(pl.Int32).alias("adp_pos_rank"),
    ).with_columns(
        (pl.col("adp_pos_rank") - pl.col("model_pos_rank")).alias("rank_gap")
    )


def _spearman(df: pl.DataFrame, col: str) -> float:
    if df.height < 8:
        return float("nan")
    rho, _ = spearmanr(df[col].to_numpy(), df["y_points"].to_numpy())
    return float(rho)


def evaluate(seasons: list[int]) -> None:
    span = list(range(min(seasons) - 3, max(seasons) + 1))
    stats = build_player_season(span)
    features = add_targets(build(span, stats=stats), stats)
    del stats
    gc.collect()

    rows = []
    for year in seasons:
        scored = _fit_and_score(features, span, year)
        if scored is None:
            continue

        # The loudest disagreements, in both directions.
        loud = pl.concat([
            scored.sort("rank_gap", descending=True).head(TOP_N),
            scored.sort("rank_gap").head(TOP_N),
        ]).unique(subset=["player_id"])

        loud = loud.with_columns((-pl.col("adp")).alias("neg_adp"))
        rows.append({
            "season": year,
            "n_loud": loud.height,
            "model_rho": _spearman(loud, "model_points"),
            "adp_rho": _spearman(loud, "neg_adp"),
        })
        gc.collect()

    table = pl.DataFrame(rows).with_columns(
        (pl.col("model_rho") - pl.col("adp_rho")).alias("edge")
    )
    print("On the loudest disagreements only:\n")
    for r in table.iter_rows(named=True):
        print(f"  {r['season']}: n={r['n_loud']:3d}  model rho={r['model_rho']:+.3f}  "
              f"ADP rho={r['adp_rho']:+.3f}  edge={r['edge']:+.3f}")

    edge = float(table["edge"].mean())
    se = float(table["edge"].std() / np.sqrt(table.height))
    wins = int((table["edge"] > 0).sum())
    print("\n" + "=" * 58)
    print(f"  mean edge on disagreements: {edge:+.4f} +/- {se:.4f}")
    print(f"  seasons model was better  : {wins}/{table.height}")
    print("=" * 58)
    if edge > 2 * se:
        print("\n  The model IS worth consulting where it disagrees loudly.")
    elif edge > 0:
        print("\n  Model is nominally better on disagreements but inside the noise.")
        print("  Treat flags as prompts to think, not as signals to act.")
    else:
        print("\n  The model is NOT better even on its own loudest disagreements.")
        print("  It should not override the market in either direction.")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.write_parquet(DATA_PROCESSED / "disagreement_backtest.parquet")


def current(season: int) -> None:
    """Print this season's biggest model-vs-market disagreements."""
    span = list(range(season - 4, season + 1))
    stats = build_player_season([s for s in span if s < season])
    features = add_targets(build(span, stats=stats), stats)
    del stats
    gc.collect()

    scored = _fit_and_score(features, span, season)
    if scored is None:
        print(f"\nNo scoreable board for {season}.")
        return

    from src.ingest.adp import load_adp
    from src.ingest.ids import resolve

    lookup = resolve(load_adp(year=season)).select(
        pl.col("gsis_id").alias("player_id"), pl.col("name")
    ).unique(subset=["player_id"])
    scored = scored.join(lookup, on="player_id", how="left")

    print(f"\n\n{'=' * 70}")
    print(f"  {season} DISAGREEMENTS -- model vs market")
    print("=" * 70)
    print("\n  MODEL LIKES MORE THAN THE MARKET DOES")
    for r in scored.sort("rank_gap", descending=True).head(12).iter_rows(named=True):
        print(f"    {(r['name'] or r['player_id']):22s} {r['pos']:3s} "
              f"adp {r['adp']:6.1f}  {r['pos']}{r['adp_pos_rank']:<3d} -> "
              f"{r['pos']}{r['model_pos_rank']:<3d}  ({r['rank_gap']:+d})")
    print("\n  MODEL LIKES LESS THAN THE MARKET DOES")
    for r in scored.sort("rank_gap").head(12).iter_rows(named=True):
        print(f"    {(r['name'] or r['player_id']):22s} {r['pos']:3s} "
              f"adp {r['adp']:6.1f}  {r['pos']}{r['adp_pos_rank']:<3d} -> "
              f"{r['pos']}{r['model_pos_rank']:<3d}  ({r['rank_gap']:+d})")
    print("\n  These are prompts to think, not instructions. The backtest above")
    print("  says how much weight they have earned.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2021)
    ap.add_argument("--to-season", type=int, default=2025)
    ap.add_argument("--current", type=int, default=SEASON)
    ap.add_argument("--skip-current", action="store_true")
    args = ap.parse_args()

    evaluate(list(range(args.from_season, args.to_season + 1)))
    if not args.skip_current:
        current(args.current)


if __name__ == "__main__":
    main()
