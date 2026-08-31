"""Which feature groups actually earn their place?

The EDA has an opinion about most of the model's feature list, and an opinion is
not a measurement. This removes one group at a time and reports what the number
does, on the same multi-seed protocol everything else here uses.

A group whose removal *improves* the model was costing variance, not paying for
itself. With roughly a thousand decision-relevant rows and sixty-odd features,
that is the expected outcome for anything weak, and the EDA nominates several
candidates:

  lag2 production   a season two years old retains 93% of a one-year-old
                    season's predictive value, so seven lag2 features are mostly
                    a second, noisier copy of seven lag1 features
  contract year     measured at -0.006 percentiles once the prior season and
                    the one-year-deal confound are removed
  quarterback room  an eleven-point swing in quarterback quality moves a
                    returning pass-catcher by +0.04 +/- 0.15 ppg
  combine testing   six columns that are null for every established veteran
  role change       the one group the EDA argues *for*: promoted-to-starter
                    against still-a-backup is 2.1 ppg at running back

This runs the **market-blind** gate. It cannot tell you whether anything beats
ADP -- that needs historical ADP and `signings_ablation.py --gate market`. What
it can tell you is which groups carry signal at all, which is the necessary
condition.

    python scripts/feature_audit.py --seeds 20
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from signings_ablation import _blind_population, _rho
from src.config import DATA_PROCESSED
from src.features.build import add_targets, build
from src.features.player_season import build as build_player_season
from src.models.points import (
    CHURN_FEATURES, FEATURES, INJURY_FEATURES, PARAMS, ROLE_CHANGE_FEATURES,
    _matrix, train,
)

LAG2 = tuple(f for f in FEATURES if f.endswith("_lag2"))
CONTRACT = ("contract_cap_pct", "contract_years_left", "is_contract_year",
            "years_into_contract")
COMBINE = ("forty", "vertical", "broad_jump", "cone", "shuttle", "combine_wt")
QB_ROOM = tuple(f for f in CHURN_FEATURES if f.startswith("qb_"))
ROLE_CHANGE = tuple(ROLE_CHANGE_FEATURES)
OPPORTUNITY_CHURN = tuple(f for f in CHURN_FEATURES if not f.startswith("qb_"))

# (drop, extra). Most rows remove a group that is shipping; the last adds one
# that is not, because a feature the EDA argues for deserves the same test as a
# feature already in.
CONFIGS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "everything (current)": ((), ()),
    "- lag2 production": (LAG2, ()),
    "- contract year only": (("is_contract_year",), ()),
    "- all contract terms": (CONTRACT, ()),
    "- combine testing": (COMBINE, ()),
    "- quarterback room": (QB_ROOM, ()),
    "- opportunity churn": (OPPORTUNITY_CHURN, ()),
    "- injury history": (tuple(INJURY_FEATURES), ()),
    "+ role change": ((), ROLE_CHANGE),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--from-season", type=int, default=2021)
    ap.add_argument("--to-season", type=int, default=2025)
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    span = list(range(min(seasons) - 3, max(seasons) + 1))

    stats = build_player_season(list(range(min(span) - 1, max(span) + 1)))
    features = add_targets(build(span, stats=stats), stats)
    del stats
    gc.collect()

    pool = _blind_population(features)
    prepared = []
    for year in seasons:
        tr = pool.filter(pl.col("season") < year)
        ev = pool.filter(pl.col("season") == year)
        if tr.height >= 200 and ev.height >= 40:
            prepared.append((tr, ev))
    if not prepared:
        raise SystemExit("no seasons with enough rows")

    print(f"Feature audit  {seasons[0]}-{seasons[-1]}, {args.seeds} seeds, "
          f"{len(prepared)} seasons\n")
    print("  Market-blind: within-position rank correlation with next-season")
    print("  points, trained on prior seasons only. A group whose removal")
    print("  raises the number was costing more variance than it paid for.\n")
    print(f"  {'configuration':30s} {'rho':>8s} {'seed sd':>8s} "
          f"{'delta':>9s} {'+/-':>7s}")
    print("  " + "-" * 68)

    results = {}
    for label, (drop, extra) in CONFIGS.items():
        per_seed = []
        for seed in range(args.seeds):
            params = dict(PARAMS)
            params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
            rhos = []
            for tr, ev in prepared:
                model = train(tr, "y_points", params=params, use_churn=True,
                              drop=drop, extra=extra)
                x, _ = _matrix(ev, use_churn=True, drop=drop, extra=extra)
                rhos.append(_rho(ev.with_columns(pl.Series("p", model.predict(x))), "p"))
            per_seed.append(float(np.mean(rhos)))
            gc.collect()
        results[label] = np.array(per_seed)

        base = results["everything (current)"]
        arr = results[label]
        if label == "everything (current)":
            print(f"  {label:30s} {arr.mean():+8.4f} {arr.std():8.4f} "
                  f"{'--':>9s} {'':>7s}   baseline")
            continue
        delta = arr.mean() - base.mean()
        pooled = np.sqrt(arr.std() ** 2 + base.std() ** 2) / np.sqrt(args.seeds)
        if extra:
            verdict = ("EARNS ITS PLACE" if delta > 2 * pooled else
                       "makes it worse" if delta < -2 * pooled else "no effect")
            n = len(extra)
        else:
            verdict = ("HELPS to remove" if delta > 2 * pooled else
                       "HURTS to remove" if delta < -2 * pooled else "no effect")
            n = len(drop)
        print(f"  {label:30s} {arr.mean():+8.4f} {arr.std():8.4f} "
              f"{delta:+9.4f} {pooled:7.4f}   {verdict} ({n})")

    rows = []
    base = results["everything (current)"]
    for label, arr in results.items():
        delta = arr.mean() - base.mean()
        pooled = np.sqrt(arr.std() ** 2 + base.std() ** 2) / np.sqrt(args.seeds)
        drop, extra = CONFIGS[label]
        rows.append({"config": label, "rho": float(arr.mean()),
                     "seed_sd": float(arr.std()), "delta": float(delta),
                     "se": float(pooled), "n_dropped": len(drop),
                     "n_added": len(extra)})

    print("\n" + "=" * 70)
    print("  Read 'HELPS to remove' as: this group was noise on this sample.")
    print("  Read 'no effect' as: it is neither paying nor costing much -- keep")
    print("  it only if you have an outside reason to.")
    print("=" * 70)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "feature_audit.parquet"
    pl.DataFrame(rows).write_parquet(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
