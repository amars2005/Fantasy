"""How much of a measured 'improvement' is just the random seed?

Refits the residual model across seeds on a fixed dataset. If the spread across
seeds is comparable to the difference between two feature sets, then any
before/after comparison run at a single seed is measuring noise.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from src.features.build import add_targets, build
from src.features.player_season import build as build_player_season
from src.models.points import PARAMS, _matrix, ridge_residual_model, train
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


def _baseline(df: pl.DataFrame, curves: dict) -> np.ndarray:
    ranks = df.with_columns(
        pl.col("adp").rank("ordinal").over(["season", "pos"]).alias("r")
    )["r"].to_numpy()
    pos = df["pos"].to_numpy()
    out = np.zeros(len(ranks))
    for p, c in curves.items():
        m = pos == p
        if m.any():
            out[m] = c["mean"][np.clip(ranks[m].astype(int) - 1, 0, len(c["grid"]) - 1)]
    return out


def main(seeds: int = 8) -> None:
    seasons = list(range(2021, 2026))
    span = list(range(2018, 2026))
    stats = build_player_season(span)
    features = add_targets(build(span, stats=stats), stats)
    del stats
    gc.collect()

    per_seed, ridge_edges = [], []
    for seed in range(seeds):
        edges = []
        for year in seasons:
            tr = features.filter((pl.col("season") < year) & (pl.col("adp") < 400))
            ev = features.filter((pl.col("season") == year) & (pl.col("adp") < 400))
            if tr.height < 200 or ev.height < 40:
                continue
            curves = fit_curves([s for s in span if s < year], with_se=False)
            bt, bs = _baseline(tr, curves), _baseline(ev, curves)
            rt = tr.with_columns(pl.Series("y_resid", tr["y_points"].to_numpy() - bt))

            params = dict(PARAMS)
            params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            model = train(rt, "y_resid", params=params, use_market=True)
            x, _ = _matrix(ev, use_market=True)
            scored = ev.with_columns(
                pl.Series("p", bs + SHRINK * model.predict(x)),
                (-pl.col("adp")).alias("neg_adp"),
            )
            edges.append(_rho(scored, "p") - _rho(scored, "neg_adp"))

            if seed == 0:  # ridge is deterministic; fit it once
                ridge = ridge_residual_model(rt, "y_resid")
                r_scored = ev.with_columns(
                    pl.Series("p", bs + SHRINK * ridge(ev)),
                    (-pl.col("adp")).alias("neg_adp"),
                )
                ridge_edges.append(_rho(r_scored, "p") - _rho(r_scored, "neg_adp"))
        per_seed.append(float(np.mean(edges)))
        print(f"  seed {seed}: residual-GBM edge {per_seed[-1]:+.4f}")
        gc.collect()

    arr = np.array(per_seed)
    print("\n" + "=" * 58)
    print(f"  GBM edge across {seeds} seeds: mean {arr.mean():+.4f}  sd {arr.std():.4f}")
    print(f"  range: {arr.min():+.4f} to {arr.max():+.4f}  (spread {arr.max()-arr.min():.4f})")
    print(f"  ridge (deterministic) edge : {np.mean(ridge_edges):+.4f}")
    print("=" * 58)
    print("\n  Any single-seed 'improvement' smaller than the spread above is noise.")


if __name__ == "__main__":
    main()
