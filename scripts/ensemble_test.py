"""Does averaging predictions across seeds beat any single seed?

Inferred directly from the ablation: seed-to-seed spread in edge is 0.0070 while
the signal being chased is ~0.003. When variance dominates like that, averaging
the *predictions* (not merely reporting the mean of the metric, which is what the
ablation did) should recover accuracy roughly as sqrt(n_seeds).

This is the cheapest available improvement if it holds -- no new data, no new
features, just refitting the same model a few times.
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

from src.features.build import add_targets, build
from src.features.player_season import build as build_player_season
from src.models.points import PARAMS, _matrix, rank_target, train
from src.project.consensus import fit_curves

SKILL = ["QB", "RB", "WR", "TE"]
SHRINK = 0.35


def _rho(df: pl.DataFrame, col: str) -> float:
    total, weight = 0.0, 0.0
    for pos in SKILL:
        sub = df.filter(pl.col("pos") == pos)
        if sub.height < 8:
            continue
        r, _ = spearmanr(sub[col].to_numpy(), sub["y_points"].to_numpy())
        if not np.isnan(r):
            total += r * sub.height
            weight += sub.height
    return total / weight if weight else float("nan")


def _pct(df: pl.DataFrame, col: str, descending: bool) -> np.ndarray:
    return df.with_columns(
        (
            pl.col(col).rank("average", descending=descending).over(["season", "pos"])
            / pl.len().over(["season", "pos"])
        ).alias("_p")
    )["_p"].to_numpy().astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seeds", type=int, default=16)
    args = ap.parse_args()

    seasons = list(range(2021, 2026))
    span = list(range(2018, 2026))
    stats = build_player_season(span)
    features = add_targets(build(span, stats=stats), stats)
    del stats
    gc.collect()

    prepared = []
    for year in seasons:
        tr = features.filter((pl.col("season") < year) & (pl.col("adp") < 400))
        ev = features.filter((pl.col("season") == year) & (pl.col("adp") < 400))
        if tr.height < 200 or ev.height < 40:
            continue
        rt = tr.with_columns(
            pl.Series("y_rankresid", rank_target(tr) - _pct(tr, "adp", True))
        )
        ev = ev.with_columns(
            (-pl.col("adp")).alias("neg_adp"),
            pl.Series("adp_pct", _pct(ev, "adp", True)),
        )
        prepared.append((rt, ev, _rho(ev, "neg_adp")))

    # Collect per-seed predictions once, then evaluate growing ensembles.
    per_season_preds = []
    for rt, ev, adp_rho in prepared:
        x, _ = _matrix(ev, use_market=True, use_college=True)
        preds = []
        for seed in range(args.max_seeds):
            params = dict(PARAMS)
            params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            model = train(rt, "y_rankresid", params=params, use_market=True, use_college=True)
            preds.append(model.predict(x))
        per_season_preds.append((ev, adp_rho, np.array(preds)))
        gc.collect()

    print(f"Ensemble size vs edge (5 seasons, up to {args.max_seeds} seeds)\n")
    print(f"  {'n_seeds':>8s} {'edge vs ADP':>13s} {'sd across draws':>18s}")
    print("  " + "-" * 42)

    rng = np.random.default_rng(0)
    for n in (1, 2, 4, 8, 16):
        if n > args.max_seeds:
            continue
        draws = []
        for _ in range(24):  # resample which seeds go in the ensemble
            picks = rng.choice(args.max_seeds, size=n, replace=False)
            edges = []
            for ev, adp_rho, preds in per_season_preds:
                mean_pred = preds[picks].mean(axis=0)
                scored = ev.with_columns(
                    pl.Series("p", ev["adp_pct"].to_numpy() + SHRINK * mean_pred)
                )
                edges.append(_rho(scored, "p") - adp_rho)
            draws.append(float(np.mean(edges)))
        arr = np.array(draws)
        print(f"  {n:>8d} {arr.mean():+13.4f} {arr.std():>18.4f}")

    print("\n  If edge rises and spread falls with n, ensembling is free accuracy.")


if __name__ == "__main__":
    main()
