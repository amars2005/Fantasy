"""EDA: fantasy points by position, by season, and by year of contract.

Two questions that the projection curve quietly assumes away.

**Does the scoring environment move?** The consensus curve is fitted on
2016-2025 pooled, which is only legitimate if a WR2 in 2017 was worth roughly
what a WR2 is worth now. Section one checks that directly, at the ranks this
league actually starts (QB14, RB28, WR42, TE14).

**Does a contract change what a player scores?** "Contract year" is one of the
oldest claims in fantasy, and it is exactly the kind of claim that survives on a
confound: players in the last year of a deal are established starters, so of
course they outscore the field. Section two controls for what the player did the
season before and asks whether anything is left.

    python scripts/eda_positions.py
    python scripts/eda_positions.py --write-report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import DATA_PROCESSED, LEAGUE, ROOT
from src.features.eda import bucket_effect, grouped_slope, grouped_spearman
from src.features.player_season import build as build_player_season

SKILL = ("QB", "RB", "WR", "TE")
# The NFL went from 16 games to 17 in 2021, which adds about 6% to every season
# total. A trend fitted on raw totals would find that and call it a change in the
# scoring environment, so trends are fitted on 17-game-equivalent points while
# the tables report what a manager actually banked.
SEASON_GAMES = {season: (16 if season < 2021 else 17) for season in range(2000, 2040)}
# Where this league's starters run out, which is where the scoring environment
# actually matters: QB14 is replacement level here, QB1 is not.
STARTER_RANK = {
    pos: LEAGUE["teams"] * LEAGUE["starters"].get(pos, 1) for pos in SKILL
}
# Flex is all WR in this league (see the README), so the receiver pool runs
# three deep per team rather than two.
STARTER_RANK["WR"] = LEAGUE["teams"] * 3


def _ranked(stats: pl.DataFrame) -> pl.DataFrame:
    return stats.with_columns(
        pl.col("points").rank("ordinal", descending=True)
        .over(["season", "pos"]).alias("pos_rank")
    )


def _environment(stats: pl.DataFrame) -> pl.DataFrame:
    """Points at the ranks that define each position's value, per season."""
    ranked = _ranked(stats)
    rows = []
    for pos in SKILL:
        cutoff = STARTER_RANK[pos]
        sub = ranked.filter(pl.col("pos") == pos)
        for season in sorted(sub["season"].unique().to_list()):
            year = sub.filter(pl.col("season") == season)
            if year.height < cutoff:
                continue
            top = year.filter(pl.col("pos_rank") <= cutoff)["points"]
            elite = year.filter(pl.col("pos_rank") <= 3)["points"].mean()
            last = year.filter(pl.col("pos_rank") == cutoff)["points"][0]
            scale = 17.0 / SEASON_GAMES.get(int(season), 17)
            rows.append({
                "pos": pos, "season": season,
                "elite": float(elite), "starter_mean": float(top.mean()),
                "replacement": float(last), "steepness": float(elite - last),
                "elite_17": float(elite) * scale,
                "replacement_17": float(last) * scale,
                "steepness_17": float(elite - last) * scale,
            })
    return pl.DataFrame(rows)


def _contract_frame(stats: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """Player-seasons with the contract in force and the prior season's output."""
    from src.features.build import _contract_features, _draft_capital, _rosters

    rosters = _rosters(seasons)
    spine = rosters.filter(pl.col("season").is_in(seasons))
    contracts = _contract_features(spine)

    prior = stats.select(
        "player_id", "season",
        pl.col("points").alias("prior_points"),
        pl.col("ppg").alias("prior_ppg"),
        pl.col("games").alias("prior_games"),
    ).with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))

    outcome = stats.select("player_id", "season", "points", "ppg", "games")

    return (
        spine.join(contracts, on=["player_id", "season"], how="inner")
        .join(prior, on=["player_id", "season"], how="left")
        .join(outcome, on=["player_id", "season"], how="left")
        .join(_draft_capital(), on="player_id", how="left")
        .with_columns(
            # A rostered player with no stat line scored zero, not nothing.
            pl.col("points").fill_null(0.0),
            pl.col("ppg").fill_null(0.0),
            pl.col("games").fill_null(0),
            pl.col("prior_points").fill_null(0.0),
            pl.col("prior_ppg").fill_null(0.0),
            pl.col("prior_games").fill_null(0),
            pl.col("draft_round").fill_null(8.0),
        )
        .with_columns(
            pl.col("years_into_contract").clip(0, 4).alias("contract_year_n"),
            (
                pl.col("points").rank("average").over(["season", "pos"])
                / pl.len().over(["season", "pos"])
            ).alias("points_pct"),
            (
                pl.col("prior_points").rank("average").over(["season", "pos"])
                / pl.len().over(["season", "pos"])
            ).alias("prior_points_pct"),
        )
    )


def _contract_table(frame: pl.DataFrame, key: str, labels: dict,
                    min_rows: int = 60) -> pl.DataFrame:
    """Mean outcome by contract bucket, raw and net of the prior season.

    `net` is the bucket's mean points percentile after removing what its own
    prior-season percentile already predicted, computed within (season,
    position). It is the difference between "these players scored more" and
    "these players scored more than they were on course to".
    """
    rows = []
    for value in sorted(frame[key].unique().drop_nulls().to_list()):
        sub = frame.filter(pl.col(key) == value)
        if sub.height < min_rows:
            continue
        got = bucket_effect(
            frame.with_columns((pl.col(key) == value).cast(pl.Int8).alias("_flag")),
            "_flag", "points_pct", control="prior_points_pct", n_buckets=2,
        )
        net = float(got["effect"][-1] - got["effect"][0]) if got.height == 2 else float("nan")
        rows.append({
            "bucket": labels.get(value, str(value)),
            "n": sub.height,
            "points": float(sub["points"].mean()),
            "games": float(sub["games"].mean()),
            "pct": float(sub["points_pct"].mean()),
            "prior_pct": float(sub["prior_points_pct"].mean()),
            "net": net,
        })
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2015)
    ap.add_argument("--to-season", type=int, default=2025)
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    stats = build_player_season(seasons)

    # --- 1. scoring environment -------------------------------------------
    env = _environment(stats)
    print(f"Position and contract EDA  {seasons[0]}-{seasons[-1]}\n")
    print("=" * 76)
    print("  THE SCORING ENVIRONMENT, BY SEASON")
    print("=" * 76)
    print("  Points at this league's starter cutoffs: QB14, RB28, WR42, TE14.")
    print("  'elite' is the mean of the top three, 'repl' the cutoff itself.\n")
    for pos in SKILL:
        sub = env.filter(pl.col("pos") == pos).sort("season")
        if sub.height == 0:
            continue
        print(f"  {pos}  (cutoff {STARTER_RANK[pos]})")
        print("    " + " ".join(f"{int(s):>6d}" for s in sub["season"]))
        print("    elite " + " ".join(f"{v:>6.0f}" for v in sub["elite"]))
        print("    repl  " + " ".join(f"{v:>6.0f}" for v in sub["replacement"]))
    print()
    for pos in SKILL:
        sub = env.filter(pl.col("pos") == pos).sort("season")
        if sub.height < 4:
            continue
        seasons_arr = sub["season"].to_numpy().astype(float)
        for field in ("replacement_17", "steepness_17"):
            slope = np.polyfit(seasons_arr, sub[field].to_numpy().astype(float), 1)[0]
            print(f"  {pos} {field:14s} trend {slope:+6.2f} points per season "
                  "(17-game equivalent)")

    # --- 2. contracts ------------------------------------------------------
    contracts = _contract_frame(stats, seasons)
    print("\n" + "=" * 76)
    print("  YEAR OF CONTRACT")
    print("=" * 76)
    print(f"  {contracts.height} player-seasons with a contract on file.")
    print("  'pct' is the within-position points percentile; 'net' is that")
    print("  percentile after removing what the prior season already predicted.\n")

    # The naive population is 70% players on one-year minimum deals -- camp
    # bodies who are, by construction, always in a "contract year". Showing that
    # before filtering it out is the point: it is where the folklore comes from.
    naive = _contract_table(
        contracts, "is_contract_year",
        {0: "not a contract year", 1: "contract year"},
    )
    print("  Every rostered player with a contract on file:\n")
    print(f"  {'bucket':20s} {'n':>6s} {'points':>8s} {'games':>6s} "
          f"{'pct':>6s} {'prior':>6s} {'net':>7s}")
    for r in naive.iter_rows(named=True):
        print(f"  {r['bucket']:20s} {r['n']:6d} {r['points']:8.1f} {r['games']:6.1f} "
              f"{r['pct']:6.3f} {r['prior_pct']:6.3f} {r['net']:+7.3f}")
    one_year = contracts.filter(
        (pl.col("is_contract_year") == 1) & (pl.col("years") == 1)
    ).height
    flagged = contracts.filter(pl.col("is_contract_year") == 1).height
    print(f"\n  {one_year}/{flagged} ({one_year / flagged:.0%}) of those "
          "'contract year' rows are one-year deals -- a player signed to a")
    print("  minimum contract is in the final year of it from the day he signs.")

    established = contracts.filter(pl.col("prior_games") >= 8)
    print(f"\n  Restricted to players with a real prior season "
          f"(8+ games): {established.height} rows\n")

    by_year = _contract_table(
        established, "contract_year_n",
        {0: "signed this year", 1: "year 2", 2: "year 3", 3: "year 4", 4: "year 5+"},
    )
    print(f"  {'bucket':20s} {'n':>6s} {'points':>8s} {'games':>6s} "
          f"{'pct':>6s} {'prior':>6s} {'net':>7s}")
    for r in by_year.iter_rows(named=True):
        print(f"  {r['bucket']:20s} {r['n']:6d} {r['points']:8.1f} {r['games']:6.1f} "
              f"{r['pct']:6.3f} {r['prior_pct']:6.3f} {r['net']:+7.3f}")

    # The folklore is about a starter in the last year of a real deal, not about
    # a journeyman on his fourth one-year contract. This is that population.
    multi = established.filter(pl.col("years") >= 3)
    final = _contract_table(
        multi, "is_contract_year",
        {0: "not a contract year", 1: "contract year"}, min_rows=40,
    )
    print(f"\n  Established players on deals of three years or more "
          f"({multi.height} rows):\n")
    for r in final.iter_rows(named=True):
        print(f"  {r['bucket']:20s} {r['n']:6d} {r['points']:8.1f} {r['games']:6.1f} "
              f"{r['pct']:6.3f} {r['prior_pct']:6.3f} {r['net']:+7.3f}")

    contract_year = grouped_spearman(
        multi, "is_contract_year", "points_pct", control="prior_points_pct"
    )
    contract_year_age = grouped_spearman(
        multi, "is_contract_year", "points_pct", control="age"
    )
    cap = grouped_spearman(
        established, "contract_cap_pct", "points_pct", control="prior_points_pct"
    )
    print("\n  Contract year on a multi-year deal, partial correlation with the")
    print(f"  points percentile, prior season held fixed: "
          f"{contract_year['rho']:+.3f} +/- {contract_year['se']:.3f}")
    print(f"  The same, holding age fixed instead:        "
          f"{contract_year_age['rho']:+.3f} +/- {contract_year_age['se']:.3f}")
    print(f"  Cap share, prior season held fixed:         "
          f"{cap['rho']:+.3f} +/- {cap['se']:.3f}")

    # --- 3. by position ----------------------------------------------------
    print("\n" + "=" * 76)
    print("  THE SAME TWO, BY POSITION")
    print("=" * 76)
    print(f"  {'pos':4s} {'contract year':>16s} {'cap share':>16s}")
    pos_rows = []
    for pos in SKILL:
        a = grouped_spearman(multi.filter(pl.col("pos") == pos), "is_contract_year",
                             "points_pct", control="prior_points_pct")
        b = grouped_spearman(established.filter(pl.col("pos") == pos),
                             "contract_cap_pct", "points_pct",
                             control="prior_points_pct")
        pos_rows.append({"pos": pos, "contract_year": a["rho"], "contract_year_se": a["se"],
                         "cap": b["rho"], "cap_se": b["se"]})
        print(f"  {pos:4s} {a['rho']:+9.3f} ±{a['se']:5.3f} "
              f"{b['rho']:+9.3f} ±{b['se']:5.3f}")

    # --- 4. the rookie deal ------------------------------------------------
    print("\n" + "=" * 76)
    print("  THE ROOKIE DEAL")
    print("=" * 76)
    print("  Points percentile by year of a first contract, for players drafted")
    print("  in the first three rounds. The 'year three breakout' claim.\n")
    rookies = contracts.filter(
        (pl.col("draft_round") <= 3)
        & (pl.col("years_exp") == pl.col("years_into_contract"))
    )
    rookie_rows = []
    print(f"  {'pos':4s} " + " ".join(f"{f'yr {y + 1}':>8s}" for y in range(4)))
    for pos in SKILL:
        cells, sub = [], rookies.filter(pl.col("pos") == pos)
        for year in range(4):
            block = sub.filter(pl.col("contract_year_n") == year)
            cells.append(f"{block['points_pct'].mean():8.3f}" if block.height >= 25
                         else f"{'-':>8s}")
            if block.height >= 25:
                rookie_rows.append({"pos": pos, "year": year + 1,
                                    "pct": float(block["points_pct"].mean()),
                                    "n": block.height})
        print(f"  {pos:4s} " + " ".join(cells))

    slope = grouped_slope(rookies, "contract_year_n", "points_pct")
    print(f"\n  Slope across rookie-deal years: {slope['slope']:+.4f} "
          f"+/- {slope['se']:.4f} percentile per year (n={slope['n']})")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    env.write_parquet(DATA_PROCESSED / "eda_environment.parquet")
    print(f"\nWrote {DATA_PROCESSED / 'eda_environment.parquet'}")

    if args.write_report:
        path = _write_report(args, env, by_year, final, naive, pos_rows,
                             rookie_rows, contract_year, contract_year_age,
                             cap, slope, contracts, established, multi)
        print(f"Wrote {path}")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _write_report(args, env, by_year, final, naive, pos_rows, rookie_rows,
                  contract_year, contract_year_age, cap, slope, contracts,
                  established, multi) -> Path:
    out = ROOT / "docs" / "eda_position_contract.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    seasons = sorted(env["season"].unique().to_list())

    lines = [
        "# EDA: points by position, season and year of contract",
        "",
        f"Generated by `scripts/eda_positions.py` over {args.from_season}-"
        f"{args.to_season}.",
        "",
        "## The scoring environment",
        "",
        "Points at this league's starter cutoffs (QB14, RB28, WR42, TE14). "
        "`elite` is the mean of the top three at the position; `replacement` is "
        "the cutoff itself.",
        "",
        _row(["pos", "metric"] + [str(int(s)) for s in seasons]),
        _row(["---", "---"] + ["---:"] * len(seasons)),
    ]
    for pos in SKILL:
        sub = env.filter(pl.col("pos") == pos).sort("season")
        if sub.height == 0:
            continue
        for field in ("elite", "replacement"):
            values = dict(zip(sub["season"].to_list(), sub[field].to_list()))
            lines.append(_row(
                [pos, field] + [f"{values[s]:.0f}" if s in values else "—"
                                for s in seasons]
            ))

    lines += ["",
              "Linear trend in 17-game-equivalent points per season. The league "
              "went from 16 games to 17 in 2021; fitting a trend to raw totals "
              "would find that and call it a change in the scoring environment.",
              "",
              _row(["pos", "replacement", "steepness"]),
              _row(["---", "---:", "---:"])]
    for pos in SKILL:
        sub = env.filter(pl.col("pos") == pos).sort("season")
        if sub.height < 4:
            continue
        x = sub["season"].to_numpy().astype(float)
        trends = [np.polyfit(x, sub[f].to_numpy().astype(float), 1)[0]
                  for f in ("replacement_17", "steepness_17")]
        lines.append(_row([pos] + [f"{t:+.2f}" for t in trends]))

    one_year = contracts.filter(
        (pl.col("is_contract_year") == 1) & (pl.col("years") == 1)
    ).height
    flagged = contracts.filter(pl.col("is_contract_year") == 1).height

    lines += [
        "", "## Year of contract", "",
        f"{contracts.height} player-seasons with a contract on file. `pct` is the "
        "within-position points percentile; **`net`** is that percentile after "
        "removing what the player's own prior season already predicted.",
        "",
        "### Where the folklore comes from",
        "",
        _row(["bucket", "n", "points", "games", "pct", "prior", "net"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in naive.iter_rows(named=True):
        lines.append(_row([
            r["bucket"], str(r["n"]), f"{r['points']:.1f}", f"{r['games']:.1f}",
            f"{r['pct']:.3f}", f"{r['prior_pct']:.3f}", f"{r['net']:+.3f}",
        ]))
    lines += [
        "",
        f"That table is not measuring motivation. **{one_year}/{flagged} "
        f"({one_year / flagged:.0%})** of those contract-year rows are one-year "
        "deals — a player on a minimum contract is in the final year of it from "
        "the day he signs, so the flag is mostly a marker for *fringe roster "
        "player*. Everything below restricts to players with a real prior season "
        f"({established.height} rows).",
        "", "### Year of the deal, established players", "",
        _row(["bucket", "n", "points", "games", "pct", "prior", "net"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in by_year.iter_rows(named=True):
        lines.append(_row([
            r["bucket"], str(r["n"]), f"{r['points']:.1f}", f"{r['games']:.1f}",
            f"{r['pct']:.3f}", f"{r['prior_pct']:.3f}", f"**{r['net']:+.3f}**",
        ]))

    lines += [
        "",
        f"### Contract year, on deals of three years or more ({multi.height} rows)",
        "", _row(["bucket", "n", "points", "games", "pct", "prior", "net"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in final.iter_rows(named=True):
        lines.append(_row([
            r["bucket"], str(r["n"]), f"{r['points']:.1f}", f"{r['games']:.1f}",
            f"{r['pct']:.3f}", f"{r['prior_pct']:.3f}", f"**{r['net']:+.3f}**",
        ]))

    lines += [
        "",
        f"Partial correlation of the contract-year flag with the points "
        f"percentile, prior season held fixed: **{contract_year['rho']:+.3f} ± "
        f"{contract_year['se']:.3f}**; holding age fixed instead: "
        f"**{contract_year_age['rho']:+.3f} ± {contract_year_age['se']:.3f}**. "
        f"Cap share, prior season held fixed: **{cap['rho']:+.3f} ± "
        f"{cap['se']:.3f}**.",
        "", "### By position", "",
        _row(["pos", "contract year", "cap share"]),
        _row(["---", "---:", "---:"]),
    ]
    for r in pos_rows:
        lines.append(_row([
            r["pos"], f"{r['contract_year']:+.3f} ± {r['contract_year_se']:.3f}",
            f"{r['cap']:+.3f} ± {r['cap_se']:.3f}",
        ]))

    if rookie_rows:
        years = sorted({r["year"] for r in rookie_rows})
        lines += ["", "## The rookie deal", "",
                  "Within-position points percentile by year of a first contract, "
                  "players drafted in rounds 1-3.", "",
                  _row(["pos"] + [f"year {y}" for y in years]),
                  _row(["---"] + ["---:"] * len(years))]
        for pos in SKILL:
            cells = {r["year"]: r["pct"] for r in rookie_rows if r["pos"] == pos}
            if not cells:
                continue
            lines.append(_row([pos] + [f"{cells[y]:.3f}" if y in cells else "—"
                                       for y in years]))
        lines += ["",
                  f"Slope across rookie-deal years: **{slope['slope']:+.4f} ± "
                  f"{slope['se']:.4f}** percentile per year."]

    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
