"""Does knowing who a team signed make the projection better?

The roster-churn features (`src/features/roster_churn.py`) encode the half of an
offseason the feature table was missing: arrivals rather than only departures,
competition at the player's own position, incoming draft capital and money, and
whether the quarterback room improved. This measures whether any of that is
worth carrying.

Two gates, run in this order, because they answer different questions:

  **signal**  -- market-blind. Trained on prior seasons only, does adding the
  churn group improve within-position rank correlation with next-season points?
  A feature group that cannot clear this is certainly not going to beat ADP.

  **market**  -- the gate that decides whether it ships. Same rank-residual
  formulation the rest of the project settled on: predict the *correction* to
  ADP, and measure the edge against ADP itself. Needs historical ADP.

Both are averaged over seeds, because LightGBM's bagging moves the answer by
about as much as the feature group does. A single-seed comparison on this
dataset measures the seed.

A third mode, `--roster-source end-of-season`, reruns gate 1 against the leaking
roster snapshot the pipeline used before. The difference between the two runs is
the size of the leak, and it is larger than every honest effect this project has
ever measured.

    python scripts/signings_ablation.py --seeds 20
    python scripts/signings_ablation.py --seeds 20 --gate signal   # no ADP needed
    python scripts/signings_ablation.py --seeds 20 --gate signal \
        --roster-source end-of-season                              # the leak
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
    CHURN_FEATURES, PARAMS, _matrix, importance, rank_target, train,
)

SKILL = ["QB", "RB", "WR", "TE"]
SHRINK = 0.35
# Market-blind evaluation needs a decision-relevant population and cannot use
# ADP to define one. Prior-season top-N per position is the closest stand-in:
# roughly the players who would carry a draft price.
BLIND_TOP_N = {"QB": 32, "RB": 60, "WR": 80, "TE": 32}


def _rho(df: pl.DataFrame, col: str) -> float:
    """Within-position rank correlation with next-season points, size-weighted."""
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


def _adp_baseline(df: pl.DataFrame, curves: dict) -> np.ndarray:
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


def _adp_percentile(df: pl.DataFrame) -> np.ndarray:
    return (
        df.with_columns(
            (
                pl.col("adp").rank("average", descending=True).over(["season", "pos"])
                / pl.len().over(["season", "pos"])
            ).alias("_a")
        )["_a"].to_numpy().astype(float)
    )


def _has_market(features: pl.DataFrame, seasons: list[int]) -> bool:
    """True only if ADP is actually populated for the seasons being evaluated.

    `build` fills a missing ADP with 400, so a frame with no market data does not
    look empty -- it looks like a league where everybody went undrafted. Running
    the market gate on that would report a meaningless edge rather than fail.
    """
    priced = features.filter(
        pl.col("season").is_in(seasons) & (pl.col("adp") < 400)
    )
    return priced.height >= 40 * len(seasons)


def _blind_population(features: pl.DataFrame) -> pl.DataFrame:
    """Players with enough of a prior season to have a draft price."""
    return features.with_columns(
        pl.col("points_lag1").fill_null(0.0)
        .rank("ordinal", descending=True)
        .over(["season", "pos"])
        .alias("_prior_rank")
    ).filter(
        pl.col("_prior_rank")
        <= pl.col("pos").replace_strict(BLIND_TOP_N, default=0)
    )


def signal_gate(features: pl.DataFrame, seasons: list[int], seeds: int) -> dict:
    """Market-blind: does the churn group improve out-of-sample ordering?"""
    pool = _blind_population(features)
    prepared = []
    for year in seasons:
        tr = pool.filter(pl.col("season") < year)
        ev = pool.filter(pl.col("season") == year)
        if tr.height < 200 or ev.height < 40:
            continue
        prepared.append((tr, ev))
    if not prepared:
        raise SystemExit("no seasons with enough rows for the signal gate")

    results = {}
    for label, use_churn in (("without churn", False), ("with churn", True)):
        per_seed = []
        for seed in range(seeds):
            params = dict(PARAMS)
            params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            rhos = []
            for tr, ev in prepared:
                model = train(tr, "y_points", params=params, use_churn=use_churn)
                x, _ = _matrix(ev, use_churn=use_churn)
                rhos.append(_rho(ev.with_columns(pl.Series("p", model.predict(x))), "p"))
            per_seed.append(float(np.mean(rhos)))
            gc.collect()
        results[label] = np.array(per_seed)
    results["_n_eval"] = sum(ev.height for _, ev in prepared)
    results["_seasons"] = len(prepared)
    return results


def market_gate(features: pl.DataFrame, seasons: list[int], span: list[int],
                seeds: int) -> dict:
    """The shipping formulation: correct ADP in rank space, and measure the edge."""
    from src.project.consensus import fit_curves

    prepared = []
    for year in seasons:
        tr = features.filter((pl.col("season") < year) & (pl.col("adp") < 400))
        ev = features.filter((pl.col("season") == year) & (pl.col("adp") < 400))
        if tr.height < 200 or ev.height < 40:
            continue
        curves = fit_curves([s for s in span if s < year], with_se=False)
        base_tr = _adp_baseline(tr, curves)
        tr = tr.with_columns(
            pl.Series("y_resid", tr["y_points"].to_numpy() - base_tr),
            pl.Series("y_rankresid", rank_target(tr) - _adp_percentile(tr)),
        )
        ev = ev.with_columns(
            (-pl.col("adp")).alias("neg_adp"),
            pl.Series("adp_pct", _adp_percentile(ev)),
        )
        prepared.append((tr, ev, _rho(ev, "neg_adp")))
    if not prepared:
        raise SystemExit("no seasons with usable ADP for the market gate")

    results = {}
    for label, use_churn in (("without churn", False), ("with churn", True)):
        per_seed = []
        for seed in range(seeds):
            params = dict(PARAMS)
            params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            edges = []
            for tr, ev, adp_rho in prepared:
                model = train(tr, "y_rankresid", params=params, use_market=True,
                              use_college=True, use_churn=use_churn)
                x, _ = _matrix(ev, use_market=True, use_college=True,
                               use_churn=use_churn)
                scored = ev.with_columns(
                    pl.Series("p", ev["adp_pct"].to_numpy() + SHRINK * model.predict(x))
                )
                edges.append(_rho(scored, "p") - adp_rho)
            per_seed.append(float(np.mean(edges)))
            gc.collect()
        results[label] = np.array(per_seed)
    results["_seasons"] = len(prepared)
    return results


def _verdict(results: dict, seeds: int, unit: str) -> tuple[float, float]:
    base, churn = results["without churn"], results["with churn"]
    delta = churn.mean() - base.mean()
    pooled = np.sqrt(base.std() ** 2 + churn.std() ** 2) / np.sqrt(seeds)
    print(f"  {'without churn':16s} {base.mean():+9.4f}   seed sd {base.std():.4f}   "
          f"{base.min():+.4f}..{base.max():+.4f}")
    print(f"  {'with churn':16s} {churn.mean():+9.4f}   seed sd {churn.std():.4f}   "
          f"{churn.min():+.4f}..{churn.max():+.4f}")
    verdict = (
        "REAL" if abs(delta) > 2 * pooled
        else "within seed noise -- do not ship on this"
    )
    print(f"\n  delta {delta:+.4f} +/- {pooled:.4f} {unit}   {verdict}")
    return float(delta), float(pooled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--from-season", type=int, default=2021)
    ap.add_argument("--to-season", type=int, default=2025)
    ap.add_argument("--gate", choices=("both", "signal", "market"), default="both")
    ap.add_argument("--roster-source", choices=("draft-day", "end-of-season"),
                    default="draft-day",
                    help="end-of-season leaks in-season trades; for measurement only")
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    span = list(range(min(seasons) - 3, max(seasons) + 1))

    draft_day = args.roster_source == "draft-day"
    stats = build_player_season(list(range(min(span) - 1, max(span) + 1)))
    features = add_targets(
        build(span, stats=stats, draft_day_rosters=draft_day), stats
    )
    del stats
    gc.collect()

    print(f"Roster-churn ablation  {seasons[0]}-{seasons[-1]}, "
          f"{args.seeds} seeds, {len(CHURN_FEATURES)} added features")
    print(f"Roster snapshot: {args.roster_source}"
          + ("" if draft_day else "   <-- LEAKS IN-SEASON MOVES, measurement only")
          + "\n")

    rows = []
    if args.gate in ("both", "signal"):
        print("=" * 72)
        print("  GATE 1 -- SIGNAL (market-blind, within-position rho vs points)")
        print("=" * 72)
        res = signal_gate(features, seasons, args.seeds)
        print(f"  {res['_seasons']} seasons, {res['_n_eval']} evaluated player-seasons\n")
        delta, se = _verdict(res, args.seeds, "rho")
        rows.append({"gate": "signal", "delta": delta, "se": se,
                     "base": float(res["without churn"].mean()),
                     "churn": float(res["with churn"].mean())})
        print()

    if args.gate in ("both", "market"):
        print("=" * 72)
        print("  GATE 2 -- MARKET (rank-residual correction to ADP, edge vs ADP)")
        print("=" * 72)
        if not _has_market(features, seasons):
            print("  SKIPPED -- no historical ADP in the cache for these seasons.")
            print("  The FFC feed is the only source of it; this gate needs a run")
            print("  from a machine that can reach fantasyfootballcalculator.com.")
        else:
            res = market_gate(features, seasons, span, args.seeds)
            print(f"  {res['_seasons']} seasons\n")
            delta, se = _verdict(res, args.seeds, "edge")
            rows.append({"gate": "market", "delta": delta, "se": se,
                         "base": float(res["without churn"].mean()),
                         "churn": float(res["with churn"].mean())})

    # Which of the added features the booster actually leaned on -- useful even
    # when the group as a whole does not clear its gate.
    pool = _blind_population(features)
    fitted = train(pool.filter(pl.col("season") < max(seasons)), "y_points",
                   use_churn=True)
    ranked = [(n, p) for n, p in importance(fitted, top=200) if n in CHURN_FEATURES]
    if ranked:
        print("\n  Share of total gain taken by the churn features "
              f"({sum(p for _, p in ranked):.1f}% combined):")
        for name, pct in ranked[:8]:
            print(f"    {pct:5.2f}%  {name}")

    if rows:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out = DATA_PROCESSED / "signings_ablation.parquet"
        pl.DataFrame(rows).write_parquet(out)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
