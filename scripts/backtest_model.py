"""Backtest the model against the market.

The only question that matters: on a draft day that has already happened, would
this have ranked players better than ADP did?

Four approaches are compared, all scored by within-position rank correlation on
players who were actually drafted, because that is the decision the tool supports.
Being right about the 400th-best receiver is worth nothing.

  ADP        the market, as-is -- the benchmark
  pure       model trained with no knowledge of the market
  anchored   the same model, with ADP as one feature among many
  residual   model predicts only the *error* in the ADP-implied projection, and
             that correction is shrunk before being applied

The residual formulation is the one with a floor: if the correction carries no
signal, shrinkage drives it toward zero and the output collapses back to ADP. The
other two can, and do, add noise on top of a better baseline.

Everything is fitted on seasons strictly before the target season -- including
the ADP-to-points curve that forms the residual baseline.
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

from src.config import DATA_PROCESSED
from src.features.build import add_targets, build
from src.features.player_season import build as build_player_season
from src.models.points import (
    _matrix, importance, predict_points, ridge_residual_model, train,
)
from src.project.consensus import fit_curves

SKILL = ["QB", "RB", "WR", "TE"]
SHRINK = 0.35  # fraction of the predicted ADP-correction actually applied


def _within_position_spearman(df: pl.DataFrame, pred_col: str) -> float:
    """Average within-position rank correlation, weighted by position size."""
    total, weight = 0.0, 0.0
    for pos in SKILL:
        sub = df.filter(pl.col("pos") == pos)
        if sub.height < 8:
            continue
        rho, _ = spearmanr(sub[pred_col].to_numpy(), sub["y_points"].to_numpy())
        if np.isnan(rho):
            continue
        total += rho * sub.height
        weight += sub.height
    return total / weight if weight else float("nan")


def _adp_baseline(df: pl.DataFrame, curves: dict) -> np.ndarray:
    """ADP-implied points, from curves fitted only on training seasons."""
    ranks = (
        df.with_columns(pl.col("adp").rank("ordinal").over(["season", "pos"]).alias("r"))
        ["r"].to_numpy()
    )
    pos = df["pos"].to_numpy()
    out = np.zeros(len(ranks))
    for p, curve in curves.items():
        mask = pos == p
        if not mask.any():
            continue
        idx = np.clip(ranks[mask].astype(int) - 1, 0, len(curve["grid"]) - 1)
        out[mask] = curve["mean"][idx]
    return out


def backtest(seasons: list[int]) -> tuple[pl.DataFrame, dict]:
    span = list(range(min(seasons) - 3, max(seasons) + 1))
    stats = build_player_season(span)
    features = add_targets(build(span, stats=stats), stats)
    del stats
    gc.collect()

    rows, models = [], {}
    for year in seasons:
        train_df = features.filter(pl.col("season") < year)
        score_df = features.filter(pl.col("season") == year)
        if train_df.height < 300 or score_df.height == 0:
            continue

        # Only players the market actually priced.
        drafted_train = train_df.filter(pl.col("adp") < 400)
        evaluated = score_df.filter(pl.col("adp") < 400)
        if evaluated.height < 40 or drafted_train.height < 200:
            continue

        pure, _ = predict_points(train_df, evaluated, use_market=False)
        anchored, models = predict_points(train_df, evaluated, use_market=True)

        # --- residual model: correct the market rather than replace it --------
        curves = fit_curves([s for s in span if s < year])
        base_train = _adp_baseline(drafted_train, curves)
        base_score = _adp_baseline(evaluated, curves)

        resid_train = drafted_train.with_columns(
            pl.Series("y_resid", drafted_train["y_points"].to_numpy() - base_train)
        )
        resid_model = train(resid_train, "y_resid", use_market=True)
        x, _ = _matrix(evaluated, use_market=True)
        residual = base_score + SHRINK * resid_model.predict(x)

        # Same idea, but a regularised linear fit on eight features instead of a
        # sixty-feature booster.
        ridge = ridge_residual_model(resid_train, "y_resid")
        compact = base_score + SHRINK * ridge(evaluated)

        evaluated = evaluated.with_columns(
            pl.Series("pure_points", pure),
            pl.Series("anchored_points", anchored),
            pl.Series("residual_points", residual),
            pl.Series("compact_points", compact),
            (-pl.col("adp")).alias("neg_adp"),
        )

        adp_rho = _within_position_spearman(evaluated, "neg_adp")
        pure_rho = _within_position_spearman(evaluated, "pure_points")
        anchored_rho = _within_position_spearman(evaluated, "anchored_points")
        residual_rho = _within_position_spearman(evaluated, "residual_points")
        compact_rho = _within_position_spearman(evaluated, "compact_points")

        rows.append({
            "season": year, "n": evaluated.height,
            "adp_rho": adp_rho, "pure_rho": pure_rho,
            "anchored_rho": anchored_rho, "residual_rho": residual_rho,
            "compact_rho": compact_rho, "compact_edge": compact_rho - adp_rho,
            "pure_edge": pure_rho - adp_rho,
            "anchored_edge": anchored_rho - adp_rho,
            "residual_edge": residual_rho - adp_rho,
        })
        print(f"  {year}: n={evaluated.height:3d}  ADP={adp_rho:+.3f}   "
              f"pure={pure_rho:+.3f}({pure_rho - adp_rho:+.3f})  "
              f"anch={anchored_rho:+.3f}({anchored_rho - adp_rho:+.3f})  "
              f"resid={residual_rho:+.3f}({residual_rho - adp_rho:+.3f})  "
              f"compact={compact_rho:+.3f}({compact_rho - adp_rho:+.3f})")
        gc.collect()

    return pl.DataFrame(rows), models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2021)
    ap.add_argument("--to-season", type=int, default=2025)
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    print(f"Backtesting {seasons[0]}-{seasons[-1]} (train on prior seasons only)\n")
    table, models = backtest(seasons)
    if table.height == 0:
        raise SystemExit("no seasons evaluated")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.write_parquet(DATA_PROCESSED / "backtest.parquet")

    n = table.height
    print("\n" + "=" * 66)
    print(f"  ADP (benchmark)           rho : {table['adp_rho'].mean():+.4f}")
    print(f"  pure model                rho : {table['pure_rho'].mean():+.4f}")
    print(f"  market-anchored model     rho : {table['anchored_rho'].mean():+.4f}")
    print(f"  residual-correction model rho : {table['residual_rho'].mean():+.4f}")
    print(f"  compact ridge model       rho : {table['compact_rho'].mean():+.4f}")
    print("-" * 66)

    edges = {}
    for label in ("pure", "anchored", "residual", "compact"):
        col = f"{label}_edge"
        edge = float(table[col].mean())
        se = float(table[col].std() / np.sqrt(n))
        wins = int((table[col] > 0).sum())
        verdict = ("PASSES" if edge > 2 * se else
                   "ahead, within noise" if edge > 0 else "FAILS")
        edges[label] = (edge, se)
        print(f"  {label:9s} edge {edge:+.4f} +/- {se:.4f}   "
              f"beat ADP {wins}/{n}   {verdict}")
    print("=" * 66)

    best_label = max(edges, key=lambda k: edges[k][0])
    best, best_se = edges[best_label]

    if best > 2 * best_se:
        print(f"\n  GATE PASSED -- ship the {best_label} model.")
    elif best > 0:
        print(f"\n  GATE NOT PASSED -- {best_label} leads but sits inside the noise.")
        print("  Use it as a tie-breaker; keep consensus as the projection.")
    else:
        print("\n  GATE FAILED -- nothing beats ADP out of sample.")
        print("  Keep the consensus projections. The model's value is as a")
        print("  disagreement flag, not as a replacement ranking.")

    print("\nTop features (points-per-game model):")
    for name, pct in importance(models["ppg"], top=12):
        print(f"  {pct:5.1f}%  {name}")


if __name__ == "__main__":
    main()
