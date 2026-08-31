"""EDA: what in a player's prior season predicts next season's fantasy points?

The projection engine is market-anchored by construction, so the only way a
model earns a place in it is by carrying information ADP does not already have.
This script measures where that information could come from, using nothing but
nflverse box scores -- no ADP required, so it runs even when the ADP feed does
not.

Three columns per stat, and the third is the one that matters:

    predictive    Spearman(prior stat, next-season points)
    repeatable    Spearman(prior stat, same stat next season)
    incremental   the same as predictive, but with last season's fantasy points
                  partialled out

A stat with a high predictive number and an incremental number near zero is not
telling you anything -- it is echoing last year's box score, which the market
has already read. Sorting by the third column is what the feature list should
have been built from.

    python scripts/eda_yoy.py                     # console report
    python scripts/eda_yoy.py --write-report      # also writes docs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import DATA_PROCESSED, ROOT
from src.features.eda import grouped_spearman, yoy_frame
from src.features.player_season import build as build_player_season

SKILL = ("QB", "RB", "WR", "TE")

# Grouped so the report reads as an argument rather than a dump.
STAT_GROUPS: dict[str, list[str]] = {
    "production": ["points", "ppg", "rec_yards", "rush_yards", "receptions"],
    "opportunity": [
        "targets", "targets_pg", "carries", "carries_pg", "touches_pg",
        "target_share", "air_yards_share", "wopr", "pass_attempts",
    ],
    "expected points": ["exp_ppg", "points_oe", "points_oe_pg"],
    "efficiency": ["yds_per_target", "yds_per_carry", "points_per_touch"],
    "availability": ["games"],
}
ALL_STATS = [s for group in STAT_GROUPS.values() for s in group]


def _table(frame: pl.DataFrame, stats: list[str], target: str) -> pl.DataFrame:
    """The three-column measurement for every stat, pooled across positions."""
    rows = []
    for stat in stats:
        if stat not in frame.columns:
            continue
        predictive = grouped_spearman(frame, stat, target)
        if predictive["n"] == 0:
            continue
        repeatable = (
            grouped_spearman(frame, stat, f"{stat}_next")
            if f"{stat}_next" in frame.columns
            else {"rho": float("nan")}
        )
        incremental = grouped_spearman(frame, stat, target, control="points")
        rows.append({
            "stat": stat,
            "predictive": predictive["rho"],
            "se": predictive["se"],
            "repeatable": repeatable["rho"],
            "incremental": incremental["rho"],
            "incremental_se": incremental["se"],
            "n": predictive["n"],
        })
    return pl.DataFrame(rows)


def _by_position(frame: pl.DataFrame, stats: list[str], target: str) -> pl.DataFrame:
    rows = []
    for pos in SKILL:
        sub = frame.filter(pl.col("pos") == pos)
        for stat in stats:
            if stat not in sub.columns:
                continue
            got = grouped_spearman(sub, stat, target)
            if got["n"] < 60:
                continue
            rows.append({"pos": pos, "stat": stat, "rho": got["rho"], "n": got["n"]})
    return pl.DataFrame(rows)


def _add_next_season_values(frame: pl.DataFrame, stats: pl.DataFrame,
                            names: list[str]) -> pl.DataFrame:
    """Attach next season's value of each stat, for the repeatability column."""
    have = [n for n in names if n in stats.columns]
    nxt = (
        stats.select(["player_id", "season"] + have)
        .rename({n: f"{n}_next" for n in have})
        .with_columns((pl.col("season") - 1).cast(pl.Int32).alias("season"))
    )
    return frame.join(nxt, on=["player_id", "season"], how="left")


def _bucketed(frame: pl.DataFrame, column: str, target: str,
              n_buckets: int = 5) -> pl.DataFrame:
    """Mean outcome by within-(season, position) quantile of `column`."""
    return (
        frame.drop_nulls([column, target])
        .with_columns(
            (
                (pl.col(column).rank("ordinal").over(["season", "pos"]) - 1)
                * n_buckets
                // pl.len().over(["season", "pos"])
            ).alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.col(column).mean().alias("bucket_mean"),
            pl.col(target).mean().alias("outcome"),
            pl.col("ppg").mean().alias("prior_ppg"),
            pl.len().alias("n"),
        )
        .sort("bucket")
    )


def _print_table(table: pl.DataFrame) -> None:
    print(f"  {'stat':18s} {'predictive':>11s} {'+/-':>6s} "
          f"{'repeatable':>11s} {'incremental':>12s} {'+/-':>6s} {'n':>6s}")
    print("  " + "-" * 78)
    for group, names in STAT_GROUPS.items():
        rows = table.filter(pl.col("stat").is_in(names)).sort("predictive", descending=True)
        if rows.height == 0:
            continue
        print(f"  {group}")
        for r in rows.iter_rows(named=True):
            rep = f"{r['repeatable']:+.3f}" if not np.isnan(r["repeatable"]) else "     -"
            # Flag only the increments that clear twice their season-to-season
            # spread; everything else is a number, not a finding.
            if np.isnan(r["incremental"]):     # the control, against itself
                incr, mark, incr_se = "     -", " ", "     -"
            else:
                incr = f"{r['incremental']:+11.3f}"
                mark = "*" if abs(r["incremental"]) > 2 * r["incremental_se"] else " "
                incr_se = f"{r['incremental_se']:6.3f}"
            print(f"    {r['stat']:16s} {r['predictive']:+11.3f} {r['se']:6.3f} "
                  f"{rep:>11s} {incr:>11s}{mark} {incr_se:>6s} {r['n']:6d}")
    print("\n  * incremental correlation exceeds twice its across-season spread")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2015)
    ap.add_argument("--to-season", type=int, default=2025)
    ap.add_argument("--min-prior-games", type=int, default=8)
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    stats = build_player_season(seasons)

    from src.features.build import _rosters

    rosters = _rosters(seasons)
    frame = yoy_frame(stats, rosters=rosters, min_prior_games=args.min_prior_games)
    frame = _add_next_season_values(frame, stats, ALL_STATS)

    print(f"Year-over-year EDA  {seasons[0]}-{seasons[-1]}\n")
    print(f"  {frame.height} player-season pairs "
          f"(>= {args.min_prior_games} games in the prior season, "
          f"still rostered the next)")
    zeros = frame.filter(pl.col("y_games") == 0).height
    print(f"  {zeros} of them ({zeros / frame.height:.1%}) never played the "
          f"following season and are kept at zero")
    for pos in SKILL:
        n = frame.filter(pl.col("pos") == pos).height
        print(f"    {pos}: {n}")

    # --- 1. the main table -------------------------------------------------
    print("\n" + "=" * 72)
    print("  PRIOR SEASON -> NEXT SEASON POINTS  (within season x position)")
    print("=" * 72)
    table = _table(frame, ALL_STATS, "y_points")
    _print_table(table)

    # --- 2. rate vs availability ------------------------------------------
    print("\n" + "=" * 72)
    print("  THE SAME STATS AGAINST RATE AND AVAILABILITY SEPARATELY")
    print("=" * 72)
    print("  Season points are (points per game) x (games). Those two factors")
    print("  are predictable to very different degrees.\n")
    ppg_table = _table(frame, ALL_STATS, "y_ppg")
    games_table = _table(frame, ALL_STATS, "y_games")
    merged = (
        table.select("stat", pl.col("predictive").alias("vs_points"))
        .join(ppg_table.select("stat", pl.col("predictive").alias("vs_ppg")), on="stat")
        .join(games_table.select("stat", pl.col("predictive").alias("vs_games")), on="stat")
        .sort("vs_ppg", descending=True)
    )
    print(f"  {'stat':18s} {'vs points':>10s} {'vs ppg':>10s} {'vs games':>10s}")
    print("  " + "-" * 52)
    for r in merged.iter_rows(named=True):
        print(f"  {r['stat']:18s} {r['vs_points']:+10.3f} {r['vs_ppg']:+10.3f} "
              f"{r['vs_games']:+10.3f}")

    # --- 3. per position ---------------------------------------------------
    print("\n" + "=" * 72)
    print("  BEST PREDICTOR OF NEXT-SEASON POINTS, BY POSITION")
    print("=" * 72)
    per_pos = _by_position(frame, ALL_STATS, "y_points")
    for pos in SKILL:
        rows = per_pos.filter(pl.col("pos") == pos).sort("rho", descending=True).head(6)
        if rows.height == 0:
            continue
        best = "  ".join(f"{r['stat']} {r['rho']:+.3f}" for r in rows.iter_rows(named=True))
        print(f"  {pos}: {best}")

    # --- 4. regression to the mean ----------------------------------------
    print("\n" + "=" * 72)
    print("  DOES SCORING ABOVE YOUR OPPORTUNITY REPEAT?")
    print("=" * 72)
    print("  Quintiles of prior-season points-over-expected per game. If the")
    print("  outcome column does not rise with the bucket, it was luck.\n")
    buckets = _bucketed(frame, "points_oe_pg", "y_ppg")
    print(f"  {'quintile':>9s} {'points_oe/g':>12s} {'prior ppg':>10s} "
          f"{'next ppg':>10s} {'n':>6s}")
    for r in buckets.iter_rows(named=True):
        print(f"  {r['bucket'] + 1:9d} {r['bucket_mean']:+12.2f} {r['prior_ppg']:10.2f} "
              f"{r['outcome']:10.2f} {r['n']:6d}")
    oe_incr = grouped_spearman(frame, "points_oe_pg", "y_ppg", control="ppg")
    print(f"\n  Partial correlation with next-season ppg, holding prior ppg fixed:"
          f" {oe_incr['rho']:+.3f}")

    # --- 5. age ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  AGE")
    print("=" * 72)
    ages = (
        rosters.select("player_id", "season", "age")
        .drop_nulls()
        .unique(subset=["player_id", "season"])
    )
    aged = frame.join(ages, on=["player_id", "season"], how="inner").with_columns(
        (pl.col("y_ppg") - pl.col("ppg")).alias("ppg_delta")
    )
    band = (
        aged.with_columns(
            pl.when(pl.col("age") < 24).then(pl.lit("<24"))
            .when(pl.col("age") < 27).then(pl.lit("24-26"))
            .when(pl.col("age") < 30).then(pl.lit("27-29"))
            .otherwise(pl.lit("30+")).alias("band")
        )
        .group_by(["pos", "band"])
        .agg(
            pl.col("ppg_delta").mean().alias("delta"),
            pl.col("y_games").mean().alias("games"),
            pl.len().alias("n"),
        )
    )
    print("  Change in points per game from one season to the next.\n")
    print(f"  {'pos':4s} {'<24':>10s} {'24-26':>10s} {'27-29':>10s} {'30+':>10s}")
    for pos in SKILL:
        cells = []
        for b in ("<24", "24-26", "27-29", "30+"):
            row = band.filter((pl.col("pos") == pos) & (pl.col("band") == b))
            cells.append(f"{row['delta'][0]:+10.2f}" if row.height else f"{'-':>10s}")
        print(f"  {pos:4s} " + " ".join(cells))

    # --- 6. how much decays --------------------------------------------------
    print("\n" + "=" * 72)
    print("  HOW FAST DOES A SEASON GO STALE?")
    print("=" * 72)
    two = _add_next_season_values(
        yoy_frame(stats, rosters=rosters, min_prior_games=args.min_prior_games),
        stats, ALL_STATS,
    )
    lag2 = (
        stats.select("player_id", "season", pl.col("points").alias("y_points_2"))
        .with_columns((pl.col("season") - 2).cast(pl.Int32).alias("season"))
    )
    two = two.join(lag2, on=["player_id", "season"], how="inner")
    print("  Restricted to players who were still producing two seasons later,")
    print("  so both columns are survivors -- the comparison between them is")
    print("  fair, the levels are optimistic.\n")
    decay = []
    for stat in ("ppg", "exp_ppg", "touches_pg", "target_share"):
        if stat not in two.columns:
            continue
        one = grouped_spearman(two, stat, "y_points")
        far = grouped_spearman(two, stat, "y_points_2")
        decay.append({"stat": stat, "one": one["rho"], "two": far["rho"],
                      "retained": far["rho"] / one["rho"]})
        print(f"  {stat:14s} next season {one['rho']:+.3f}   "
              f"two seasons out {far['rho']:+.3f}   "
              f"retained {far['rho'] / one['rho']:.0%}")
    decay = pl.DataFrame(decay)

    # --- 7. the spread that is left ---------------------------------------
    print("\n" + "=" * 72)
    print("  HOW WIDE IS THE OUTCOME ANYWAY?")
    print("=" * 72)
    print("  Prior-season points, in within-position quintiles, against the")
    print("  distribution of next-season points. This is the number the draft")
    print("  board has to live with: even the best-informed bucket is enormous.\n")
    spread = (
        frame.with_columns(
            (
                (pl.col("points").rank("ordinal").over(["season", "pos"]) - 1) * 5
                // pl.len().over(["season", "pos"])
            ).alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.col("points").mean().alias("prior"),
            pl.col("y_points").mean().alias("mean"),
            pl.col("y_points").std().alias("sd"),
            pl.col("y_points").quantile(0.10).alias("p10"),
            pl.col("y_points").quantile(0.90).alias("p90"),
            (pl.col("y_points") < 0.5 * pl.col("points")).mean().alias("halved"),
            pl.len().alias("n"),
        )
        .sort("bucket")
    )
    print(f"  {'quintile':>9s} {'prior':>8s} {'next':>8s} {'sd':>7s} "
          f"{'p10':>7s} {'p90':>7s} {'bust':>6s} {'n':>6s}")
    for r in spread.iter_rows(named=True):
        print(f"  {r['bucket'] + 1:9d} {r['prior']:8.0f} {r['mean']:8.0f} "
              f"{r['sd']:7.0f} {r['p10']:7.0f} {r['p90']:7.0f} "
              f"{r['halved']:5.0%} {r['n']:6d}")
    print("\n  'bust' = scored less than half of last season's total.")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.write_parquet(DATA_PROCESSED / "eda_yoy.parquet")
    print(f"\nWrote {DATA_PROCESSED / 'eda_yoy.parquet'}")

    if args.write_report:
        path = _write_report(
            args, frame, table, merged, per_pos, buckets, oe_incr, band, decay, spread
        )
        print(f"Wrote {path}")


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _write_report(args, frame, table, merged, per_pos, buckets, oe_incr, band,
                  decay, spread) -> Path:
    """Write the numbers to docs/ so the finding outlives the terminal buffer."""
    out = ROOT / "docs" / "eda_prior_season.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EDA: prior-season stats vs next-season fantasy points",
        "",
        f"Generated by `scripts/eda_yoy.py` over {args.from_season}-{args.to_season}. "
        f"{frame.height} player-season pairs, prior season filtered to "
        f"{args.min_prior_games}+ games, next season kept even when it is a zero.",
        "",
        "All correlations are Spearman, computed within (season, position) and "
        "pooled by group size. Standard errors are across seasons, not players.",
        "",
        "## Main table",
        "",
        "`incremental` partials out the player's prior-season fantasy points. "
        "It is the column that decides whether a feature carries information the "
        "market has not already read.",
        "",
        "Bold marks an incremental correlation larger than twice its "
        "across-season spread.",
        "",
        _md_row(["stat", "predictive", "±", "repeatable", "incremental", "n"]),
        _md_row(["---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in table.sort("incremental", descending=True, nulls_last=True).iter_rows(named=True):
        rep = f"{r['repeatable']:+.3f}" if not np.isnan(r["repeatable"]) else "—"
        if np.isnan(r["incremental"]):
            incr = "— *(the control)*"
        else:
            incr = f"{r['incremental']:+.3f} ± {r['incremental_se']:.3f}"
            if abs(r["incremental"]) > 2 * r["incremental_se"]:
                incr = f"**{incr}**"
        lines.append(_md_row([
            f"`{r['stat']}`", f"{r['predictive']:+.3f}", f"{r['se']:.3f}",
            rep, incr, str(r["n"]),
        ]))

    lines += [
        "",
        "## Rate vs availability",
        "",
        "Season points are (points per game) × (games played). The two factors "
        "are predictable to very different degrees.",
        "",
        _md_row(["stat", "vs points", "vs ppg", "vs games"]),
        _md_row(["---", "---:", "---:", "---:"]),
    ]
    for r in merged.iter_rows(named=True):
        lines.append(_md_row([
            f"`{r['stat']}`", f"{r['vs_points']:+.3f}",
            f"{r['vs_ppg']:+.3f}", f"{r['vs_games']:+.3f}",
        ]))

    lines += ["", "## By position", "",
              _md_row(["pos", "best predictors of next-season points"]),
              _md_row(["---", "---"])]
    for pos in SKILL:
        rows = per_pos.filter(pl.col("pos") == pos).sort("rho", descending=True).head(5)
        if rows.height == 0:
            continue
        lines.append(_md_row([pos, ", ".join(
            f"`{r['stat']}` {r['rho']:+.3f}" for r in rows.iter_rows(named=True)
        )]))

    lines += [
        "", "## Points over expected regresses", "",
        "Quintiles of prior-season points-over-expected per game.", "",
        _md_row(["quintile", "points_oe/g", "prior ppg", "next ppg", "n"]),
        _md_row(["---", "---:", "---:", "---:", "---:"]),
    ]
    for r in buckets.iter_rows(named=True):
        lines.append(_md_row([
            str(r["bucket"] + 1), f"{r['bucket_mean']:+.2f}", f"{r['prior_ppg']:.2f}",
            f"{r['outcome']:.2f}", str(r["n"]),
        ]))
    lines += [
        "",
        f"Partial correlation with next-season ppg holding prior ppg fixed: "
        f"**{oe_incr['rho']:+.3f}**.",
        "", "## Age", "",
        "Change in points per game from one season to the next.", "",
        _md_row(["pos", "<24", "24-26", "27-29", "30+"]),
        _md_row(["---", "---:", "---:", "---:", "---:"]),
    ]
    for pos in SKILL:
        cells = [pos]
        for b in ("<24", "24-26", "27-29", "30+"):
            row = band.filter((pl.col("pos") == pos) & (pl.col("band") == b))
            cells.append(f"{row['delta'][0]:+.2f}" if row.height else "—")
        lines.append(_md_row(cells))

    lines += [
        "", "## How fast a season goes stale", "",
        "Restricted to players still producing two seasons later, so both "
        "columns are survivors: the comparison between them is fair, the levels "
        "are optimistic.", "",
        _md_row(["stat", "→ next season", "→ two seasons out", "retained"]),
        _md_row(["---", "---:", "---:", "---:"]),
    ]
    for r in decay.iter_rows(named=True):
        lines.append(_md_row([
            f"`{r['stat']}`", f"{r['one']:+.3f}", f"{r['two']:+.3f}",
            f"{r['retained']:.0%}",
        ]))

    lines += [
        "", "## The spread that is left", "",
        "Prior-season points in within-position quintiles, against the "
        "distribution of next-season points. `bust` is the share who scored "
        "less than half of their prior season.", "",
        _md_row(["quintile", "prior", "next", "sd", "p10", "p90", "bust", "n"]),
        _md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in spread.iter_rows(named=True):
        lines.append(_md_row([
            str(r["bucket"] + 1), f"{r['prior']:.0f}", f"{r['mean']:.0f}",
            f"{r['sd']:.0f}", f"{r['p10']:.0f}", f"{r['p90']:.0f}",
            f"{r['halved']:.0%}", str(r["n"]),
        ]))

    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
