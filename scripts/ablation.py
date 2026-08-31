"""Honest feature ablation, averaged over seeds.

Single-seed before/after comparisons on this dataset are worthless: the
run-to-run spread from LightGBM's bagging is about 0.007 in edge, which is larger
than most of the effects being tested. Every configuration here is therefore run
across several seeds and reported with a spread.
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
from src.models.points import (
    COLLEGE_FEATURES, DEPTH_FEATURES, INJURY_FEATURES, PARAMS, _matrix,
    rank_target, ridge_residual_model, train,
)
from src.project.consensus import fit_curves

SKILL = ["QB", "RB", "WR", "TE"]
SHRINK = 0.35

CONFIGS = {
    "base (no depth/injury/college)": tuple(DEPTH_FEATURES + INJURY_FEATURES + COLLEGE_FEATURES),
    "+ depth chart":                  tuple(INJURY_FEATURES + COLLEGE_FEATURES),
    "+ depth + injury":               tuple(COLLEGE_FEATURES),
    "+ depth + injury + college":     (),
}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    seasons = list(range(2021, 2026))
    span = list(range(2018, 2026))
    stats = build_player_season(span)
    features = add_targets(build(span, stats=stats), stats)
    del stats
    gc.collect()

    # Precompute the per-season split and ADP baseline once.
    prepared = []
    for year in seasons:
        tr = features.filter((pl.col("season") < year) & (pl.col("adp") < 400))
        ev = features.filter((pl.col("season") == year) & (pl.col("adp") < 400))
        if tr.height < 200 or ev.height < 40:
            continue
        curves = fit_curves([s for s in span if s < year], with_se=False)
        bt, bs = _baseline(tr, curves), _baseline(ev, curves)
        rt = tr.with_columns(pl.Series("y_resid", tr["y_points"].to_numpy() - bt))
        # Step 5: the same rows, but with the outcome as a within-position
        # percentile so the training objective matches the evaluation metric.
        # The target is the *gap* between where a player finished and where the
        # market had him, both as within-position percentiles. Predicting the raw
        # percentile instead throws away the ADP anchor, which is most of the
        # available signal.
        actual_pct = rank_target(tr)
        adp_pct = (
            tr.with_columns(
                (
                    pl.col("adp").rank("average", descending=True).over(["season", "pos"])
                    / pl.len().over(["season", "pos"])
                ).alias("_a")
            )["_a"].to_numpy().astype(float)
        )
        rt = rt.with_columns(pl.Series("y_rankresid", actual_pct - adp_pct))
        # Opportunity target for the hierarchical model: expected points per game
        # in the season being predicted, from ffopportunity.
        rt = rt.with_columns(
            (pl.col("y_points") / pl.col("y_games").clip(1, 17)).alias("y_exp_ppg")
        )
        ev_adp_pct = (
            ev.with_columns(
                (
                    pl.col("adp").rank("average", descending=True).over(["season", "pos"])
                    / pl.len().over(["season", "pos"])
                ).alias("_a")
            )["_a"].to_numpy().astype(float)
        )
        ev = ev.with_columns(
            (-pl.col("adp")).alias("neg_adp"),
            pl.Series("adp_pct", ev_adp_pct),
        )
        prepared.append((rt, ev, bs, _rho(ev, "neg_adp")))

    print(f"Ablation over {len(prepared)} seasons x {args.seeds} seeds\n")
    print(f"  {'configuration':32s} {'edge vs ADP':>13s}  {'sd':>7s}  {'range':>17s}")
    print("  " + "-" * 74)

    results = {}
    for label, drop in CONFIGS.items():
        per_seed = []
        for seed in range(args.seeds):
            params = dict(PARAMS)
            params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            edges = []
            for rt, ev, bs, adp_rho in prepared:
                model = train(rt, "y_resid", params=params, use_market=True,
                              use_college=True, drop=drop)
                x, _ = _matrix(ev, use_market=True, use_college=True, drop=drop)
                scored = ev.with_columns(pl.Series("p", bs + SHRINK * model.predict(x)))
                edges.append(_rho(scored, "p") - adp_rho)
            per_seed.append(float(np.mean(edges)))
            gc.collect()
        arr = np.array(per_seed)
        results[label] = arr
        print(f"  {label:32s} {arr.mean():+13.4f}  {arr.std():7.4f}  "
              f"{arr.min():+.4f}..{arr.max():+.4f}")

    # Step 5: rank-percentile objective, best feature set.
    per_seed = []
    for seed in range(args.seeds):
        params = dict(PARAMS)
        params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        edges = []
        for rt, ev, bs, adp_rho in prepared:
            model = train(rt, "y_rankresid", params=params, use_market=True, use_college=True)
            x, _ = _matrix(ev, use_market=True, use_college=True)
            # Market percentile plus a shrunk correction, mirroring the points
            # residual model but in rank space.
            scored = ev.with_columns(
                pl.Series("p", ev["adp_pct"].to_numpy() + SHRINK * model.predict(x))
            )
            edges.append(_rho(scored, "p") - adp_rho)
        per_seed.append(float(np.mean(edges)))
        gc.collect()
    arr = np.array(per_seed)
    results["rank-residual objective"] = arr
    print(f"  {'rank-residual objective':32s} {arr.mean():+13.4f}  {arr.std():7.4f}  "
          f"{arr.min():+.4f}..{arr.max():+.4f}")

    # Step 7: hierarchical. Rather than predicting the outcome directly, predict
    # the two things that produce it -- opportunity (expected points per game,
    # which is far more stable year to year than actual points) and availability
    # (games played) -- then compose. The claim being tested is that modelling the
    # predictable quantity beats modelling the noisy one.
    per_seed = []
    for seed in range(args.seeds):
        params = dict(PARAMS)
        params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        edges = []
        for rt, ev, bs, adp_rho in prepared:
            played = rt.filter(pl.col("y_games") > 0)
            opp_model = train(played, "y_exp_ppg", params=params,
                              use_market=True, use_college=True)
            games_model = train(rt, "y_games", params=params,
                                use_market=True, use_college=True)
            x, _ = _matrix(ev, use_market=True, use_college=True)
            composed = np.clip(opp_model.predict(x), 0, None) * np.clip(
                games_model.predict(x), 0, 17
            )
            scored = ev.with_columns(pl.Series("p", composed))
            edges.append(_rho(scored, "p") - adp_rho)
        per_seed.append(float(np.mean(edges)))
        gc.collect()
    arr = np.array(per_seed)
    results["hierarchical (opportunity x games)"] = arr
    print(f"  {'hierarchical (opp x games)':32s} {arr.mean():+13.4f}  {arr.std():7.4f}  "
          f"{arr.min():+.4f}..{arr.max():+.4f}")

    # Ridge is deterministic, so one pass is the whole answer.
    ridge_edges = []
    for rt, ev, bs, adp_rho in prepared:
        ridge = ridge_residual_model(rt, "y_resid")
        scored = ev.with_columns(pl.Series("p", bs + SHRINK * ridge(ev)))
        ridge_edges.append(_rho(scored, "p") - adp_rho)
    print(f"  {'compact ridge (8 features)':32s} {np.mean(ridge_edges):+13.4f}  "
          f"{0.0:7.4f}  deterministic")

    print("\n" + "=" * 78)
    base = results["base (no depth/injury/college)"]
    for label, arr in results.items():
        if label.startswith("base"):
            continue
        delta = arr.mean() - base.mean()
        pooled = np.sqrt(arr.std() ** 2 + base.std() ** 2) / np.sqrt(args.seeds)
        verdict = "REAL" if abs(delta) > 2 * pooled else "within seed noise"
        print(f"  {label:32s} delta {delta:+.4f} +/- {pooled:.4f}   {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
