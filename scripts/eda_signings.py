"""EDA: what an offseason around a player does to his points.

The question the roster-churn features were built to answer, asked directly of
the data rather than through a model. Four things a team can do *to* a player
without him changing anything about himself:

  * upgrade or downgrade the quarterback throwing to him
  * promote or demote him on the depth chart
  * sign or draft competition at his own position
  * lose the players who were taking his touches

The population is **returning players**: same team both seasons, eight or more
games the season before. That is deliberate. Comparing players who moved to
players who stayed measures the move; holding the player and the team fixed and
varying what happened around him is the closest this data gets to the question.

Everything is reported net of the player's own prior season, because the
alternative is rediscovering that good players score more. `effect` columns are
residual points per game after removing what prior-season ppg already predicted,
within (season, position).

None of this is causal. Teams sign a better quarterback and a better receiver in
the same offseason, and no amount of controlling for last season separates them.
Read the effects as upper bounds on what the roster move is worth.

    python scripts/eda_signings.py
    python scripts/eda_signings.py --write-report
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src.config import DATA_PROCESSED, ROOT
from src.features.eda import bucket_effect, grouped_slope, grouped_spearman
from src.features.player_season import build as build_player_season

SKILL = ("QB", "RB", "WR", "TE")
PASS_CATCHERS = ("WR", "TE", "RB")

# Which churn feature is worth looking at for which position. Carries do not
# matter to a receiver and target competition does not matter to a quarterback,
# and printing every combination buries the four rows that mean something.
TREATMENTS = [
    ("qb_upgrade", "quarterback room, ppg better than last year's starter",
     PASS_CATCHERS),
    ("pos_in_targets", "targets arriving at his position", ("WR", "TE")),
    ("pos_in_carries", "carries arriving at his position", ("RB",)),
    ("pos_out_targets", "targets that left his position", ("WR", "TE")),
    ("pos_out_carries", "carries that left his position", ("RB",)),
    ("pos_rookie_overall", "draft slot spent on his position (300 = none)", SKILL),
    ("net_target_share", "share of team targets vacated, net of arrivals",
     ("WR", "TE", "RB")),
]


def _frame(seasons: list[int]) -> pl.DataFrame:
    """Returning players, their prior season, and what changed around them."""
    from src.features.build import _depth_chart, add_targets, build

    span = list(range(min(seasons) - 2, max(seasons) + 1))
    stats = build_player_season(span)
    feat = add_targets(build(seasons, stats=stats), stats)
    del stats
    gc.collect()

    # Prior-season role, so a depth-chart move can be seen as a move.
    depth_prior = (
        _depth_chart(span)
        .select(
            "player_id", "season",
            pl.col("is_starter").alias("was_starter"),
            pl.col("depth_schema").alias("prior_depth_schema"),
        )
        .with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))
    )

    return (
        feat.join(depth_prior, on=["player_id", "season"], how="left")
        .filter((pl.col("games_lag1") >= 8) & (pl.col("changed_team") == 0))
        .with_columns(
            (pl.col("y_points") / pl.col("y_games").clip(1, 17)).alias("y_ppg"),
        )
        .with_columns(
            pl.when(pl.col("was_starter").is_null()).then(None)
            .when((pl.col("was_starter") == 0) & (pl.col("is_starter") == 1))
            .then(pl.lit("promoted to starter"))
            .when((pl.col("was_starter") == 1) & (pl.col("is_starter") == 1))
            .then(pl.lit("starter, stayed"))
            .when((pl.col("was_starter") == 1) & (pl.col("is_starter") == 0))
            .then(pl.lit("demoted from starter"))
            .otherwise(pl.lit("backup, stayed"))
            .alias("role_move")
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2016)
    ap.add_argument("--to-season", type=int, default=2025)
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    frame = _frame(seasons)

    print(f"Signings and depth-chart moves  {seasons[0]}-{seasons[-1]}\n")
    print(f"  {frame.height} returning player-seasons "
          "(same team, 8+ games the year before)")
    for pos in SKILL:
        print(f"    {pos}: {frame.filter(pl.col('pos') == pos).height}")
    print("\n  'effect' is residual points per game after removing what the")
    print("  player's own prior-season ppg already predicted.")

    # --- 1. the quarterback room -------------------------------------------
    print("\n" + "=" * 78)
    print("  A BETTER QUARTERBACK")
    print("=" * 78)
    qb = frame.filter(pl.col("pos").is_in(PASS_CATCHERS))
    # Quintiles are the wrong split here: most teams keep their quarterback, so
    # three of the five buckets would be a solid block of zeros. The variable is
    # also asymmetric by construction -- the room's best prior season rarely
    # beats a starter you kept, so upgrades are small and downgrades are large.
    qb = qb.with_columns(
        pl.when(pl.col("qb_upgrade") <= -3.0).then(pl.lit("big downgrade"))
        .when(pl.col("qb_upgrade") < -0.5).then(pl.lit("small downgrade"))
        .when(pl.col("qb_upgrade") <= 0.5).then(pl.lit("unchanged"))
        .otherwise(pl.lit("upgrade"))
        .alias("qb_move")
    )
    rows = []
    print(f"  {'quarterback room':18s} {'mean':>7s} {'effect':>8s} {'+/-':>6s} "
          f"{'raw ppg':>8s} {'prior':>8s} {'n':>6s}")
    for move in ("big downgrade", "small downgrade", "unchanged", "upgrade"):
        flagged = qb.with_columns(
            (pl.col("qb_move") == move).cast(pl.Int8).alias("_flag")
        )
        block = flagged.filter(pl.col("_flag") == 1)
        if block.height < 40:
            continue
        got = bucket_effect(flagged, "_flag", "y_ppg", control="ppg_lag1",
                            n_buckets=2)
        if got.height != 2:
            continue
        effect = float(got["effect"][1] - got["effect"][0])
        se = float((got["se"][1] ** 2 + got["se"][0] ** 2) ** 0.5)
        rows.append({"move": move, "level": float(block["qb_upgrade"].mean()),
                     "effect": effect, "se": se,
                     "raw": float(block["y_ppg"].mean()),
                     "prior": float(block["ppg_lag1"].mean()),
                     "n": block.height})
        print(f"  {move:18s} {block['qb_upgrade'].mean():+7.2f} {effect:+8.2f} "
              f"{se:6.2f} {block['y_ppg'].mean():8.2f} "
              f"{block['ppg_lag1'].mean():8.2f} {block.height:6d}")
    buckets = pl.DataFrame(rows)

    print()
    qb_rows = []
    for pos in PASS_CATCHERS:
        sub = frame.filter(pl.col("pos") == pos)
        slope = grouped_slope(sub, "qb_upgrade", "y_ppg", control="ppg_lag1")
        rho = grouped_spearman(sub, "qb_upgrade", "y_ppg", control="ppg_lag1")
        qb_rows.append({"pos": pos, "slope": slope["slope"], "slope_se": slope["se"],
                        "rho": rho["rho"], "rho_se": rho["se"], "n": slope["n"]})
        print(f"  {pos}: {slope['slope']:+.3f} +/- {slope['se']:.3f} ppg per ppg of "
              f"quarterback upgrade   (rho {rho['rho']:+.3f} +/- {rho['se']:.3f}, "
              f"n={slope['n']})")

    changed = bucket_effect(qb, "qb_changed", "y_ppg", control="ppg_lag1",
                            n_buckets=2)
    qb_changed_gap = float("nan")
    if changed.height == 2:
        qb_changed_gap = float(changed["effect"][1] - changed["effect"][0])
        print(f"\n  Simply having a different quarterback is worth "
              f"{qb_changed_gap:+.2f} ppg on its own.")

    # --- 2. the depth chart -------------------------------------------------
    print("\n" + "=" * 78)
    print("  A DEPTH-CHART MOVE")
    print("=" * 78)
    print("  Starter status only. nflverse replaced the depth-chart feed in 2025")
    print("  and the raw rank is on a different scale either side of it; the")
    print("  starter flag is calibrated to survive the change, the rank is not.\n")
    role_rows = []
    print(f"  {'move':22s} {'pos':4s} {'effect':>8s} {'+/-':>6s} "
          f"{'raw ppg':>8s} {'prior':>8s} {'n':>6s}")
    for pos in SKILL:
        sub = frame.filter((pl.col("pos") == pos) & pl.col("role_move").is_not_null())
        for move in ("promoted to starter", "starter, stayed",
                     "demoted from starter", "backup, stayed"):
            flagged = sub.with_columns(
                (pl.col("role_move") == move).cast(pl.Int8).alias("_flag")
            )
            block = flagged.filter(pl.col("_flag") == 1)
            if block.height < 30:
                continue
            got = bucket_effect(flagged, "_flag", "y_ppg", control="ppg_lag1",
                                n_buckets=2)
            if got.height != 2:
                continue
            effect = float(got["effect"][1] - got["effect"][0])
            se = float((got["se"][1] ** 2 + got["se"][0] ** 2) ** 0.5)
            role_rows.append({"move": move, "pos": pos, "effect": effect, "se": se,
                              "raw": float(block["y_ppg"].mean()),
                              "prior": float(block["ppg_lag1"].mean()),
                              "n": block.height})
            print(f"  {move:22s} {pos:4s} {effect:+8.2f} {se:6.2f} "
                  f"{block['y_ppg'].mean():8.2f} {block['ppg_lag1'].mean():8.2f} "
                  f"{block.height:6d}")

    # --- 3. everything else -------------------------------------------------
    print("\n" + "=" * 78)
    print("  COMPETITION ARRIVING, AND OPPORTUNITY LEAVING")
    print("=" * 78)
    print("  Slope of residual points per game on each roster-churn quantity.\n")
    print(f"  {'feature':22s} {'pos':4s} {'slope':>10s} {'+/-':>8s} {'n':>6s}  "
          "per unit")
    effect_rows = []
    for feature, description, positions in TREATMENTS:
        if feature not in frame.columns:
            continue
        for pos in positions:
            sub = frame.filter(pl.col("pos") == pos)
            slope = grouped_slope(sub, feature, "y_ppg", control="ppg_lag1")
            if slope["n"] < 150:
                continue
            real = abs(slope["slope"]) > 2 * slope["se"]
            effect_rows.append({"feature": feature, "pos": pos,
                                "slope": slope["slope"], "se": slope["se"],
                                "n": slope["n"], "real": real,
                                "description": description})
            print(f"  {feature:22s} {pos:4s} {slope['slope']:+10.5f} "
                  f"{slope['se']:8.5f} {slope['n']:6d}  "
                  f"{'REAL' if real else ''}")

    # --- 4. moving teams yourself -------------------------------------------
    print("\n" + "=" * 78)
    print("  AND MOVING TEAMS YOURSELF")
    print("=" * 78)
    movers = _frame_with_movers(seasons)
    move_rows = []
    print(f"  {'pos':4s} {'stayed':>18s} {'moved':>18s} {'gap':>8s}")
    for pos in SKILL:
        sub = movers.filter(pl.col("pos") == pos)
        got = bucket_effect(sub, "changed_team", "y_ppg", control="ppg_lag1",
                            n_buckets=2)
        if got.height != 2:
            continue
        gap = float(got["effect"][1] - got["effect"][0])
        se = float((got["se"][1] ** 2 + got["se"][0] ** 2) ** 0.5)
        move_rows.append({"pos": pos, "gap": gap, "se": se,
                          "n_moved": int(got["n"][1]), "n_stayed": int(got["n"][0])})
        print(f"  {pos:4s} {got['raw'][0]:12.2f} ppg {got['raw'][1]:14.2f} ppg "
              f"{gap:+8.2f} +/- {se:.2f}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if effect_rows:
        pl.DataFrame(effect_rows).write_parquet(
            DATA_PROCESSED / "eda_signings.parquet"
        )
        print(f"\nWrote {DATA_PROCESSED / 'eda_signings.parquet'}")

    if args.write_report:
        path = _write_report(args, frame, buckets, qb_rows, role_rows,
                             effect_rows, move_rows, qb_changed_gap)
        print(f"Wrote {path}")


def _frame_with_movers(seasons: list[int]) -> pl.DataFrame:
    """Same construction, but keeping the players who changed teams."""
    from src.features.build import add_targets, build

    span = list(range(min(seasons) - 2, max(seasons) + 1))
    stats = build_player_season(span)
    feat = add_targets(build(seasons, stats=stats), stats)
    del stats
    gc.collect()
    return feat.filter(pl.col("games_lag1") >= 8).with_columns(
        (pl.col("y_points") / pl.col("y_games").clip(1, 17)).alias("y_ppg")
    )


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _write_report(args, frame, buckets, qb_rows, role_rows, effect_rows,
                  move_rows, qb_changed_gap) -> Path:
    out = ROOT / "docs" / "eda_signings.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EDA: what an offseason does to a returning player's points",
        "",
        f"Generated by `scripts/eda_signings.py` over {args.from_season}-"
        f"{args.to_season}. {frame.height} returning player-seasons — same team "
        "both years, eight or more games the season before.",
        "",
        "`effect` is residual points per game after removing what the player's "
        "own prior-season ppg already predicted, within (season, position). "
        "Standard errors are across seasons and do not account for team-level "
        "clustering, so they are optimistic.",
        "",
        "**None of this is causal.** A team that signs a better quarterback also "
        "signs better receivers, and controlling for last season does not "
        "separate them. Read every number here as an upper bound on what the "
        "roster move alone is worth.",
        "",
        "## A better quarterback",
        "",
        "`qb_upgrade` is the current room's best prior-season ppg minus what the "
        "quarterback who actually threw the ball here last year scored. It is "
        "asymmetric by construction: keep your starter and it is zero, so "
        "upgrades are small and downgrades are large.",
        "",
        _row(["quarterback room", "mean upgrade", "effect (ppg)", "±",
              "raw ppg", "prior ppg", "n"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in buckets.iter_rows(named=True):
        lines.append(_row([
            r["move"], f"{r['level']:+.2f}", f"**{r['effect']:+.2f}**",
            f"{r['se']:.2f}", f"{r['raw']:.2f}", f"{r['prior']:.2f}", str(r["n"]),
        ]))

    lines += ["",
              f"Simply having a different quarterback is worth "
              f"**{qb_changed_gap:+.2f} ppg** on its own.", "",
              _row(["pos", "ppg per ppg of QB upgrade", "partial rho", "n"]),
              _row(["---", "---:", "---:", "---:"])]
    for r in qb_rows:
        lines.append(_row([
            r["pos"], f"**{r['slope']:+.3f} ± {r['slope_se']:.3f}**",
            f"{r['rho']:+.3f} ± {r['rho_se']:.3f}", str(r["n"]),
        ]))

    lines += [
        "", "## A depth-chart move", "",
        "Starter status only. nflverse replaced the depth-chart feed in 2025 and "
        "the raw rank is on a different scale either side of it; the starter flag "
        "is calibrated to survive the change, the rank is not.",
        "",
        _row(["move", "pos", "effect (ppg)", "±", "raw ppg", "prior ppg", "n"]),
        _row(["---", "---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in role_rows:
        lines.append(_row([
            r["move"], r["pos"], f"**{r['effect']:+.2f}**", f"{r['se']:.2f}",
            f"{r['raw']:.2f}", f"{r['prior']:.2f}", str(r["n"]),
        ]))

    lines += [
        "", "## Competition arriving, opportunity leaving", "",
        "Slope of residual points per game on each roster-churn quantity. Bold "
        "clears twice its standard error.",
        "",
        _row(["feature", "pos", "slope (ppg per unit)", "±", "n", "meaning"]),
        _row(["---", "---", "---:", "---:", "---:", "---"]),
    ]
    for r in effect_rows:
        slope = f"{r['slope']:+.5f}"
        lines.append(_row([
            f"`{r['feature']}`", r["pos"],
            f"**{slope}**" if r["real"] else slope, f"{r['se']:.5f}",
            str(r["n"]), r["description"],
        ]))

    if move_rows:
        lines += ["", "## Changing teams yourself", "",
                  _row(["pos", "gap (ppg, moved − stayed)", "±", "n moved"]),
                  _row(["---", "---:", "---:", "---:"])]
        for r in move_rows:
            lines.append(_row([
                r["pos"], f"{r['gap']:+.2f}", f"{r['se']:.2f}", str(r["n_moved"]),
            ]))

    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
