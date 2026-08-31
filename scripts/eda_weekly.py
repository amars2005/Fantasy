"""EDA: weekly scoring, consistency, and what head-to-head actually rewards.

This league is decided by fourteen weekly matchups and a three-week bracket, and
every other analysis in this project is denominated in season totals. Those are
not the same objective. Two players who both score 200 points are worth different
amounts if one does it evenly and the other does it in three explosions -- and
which one is worth more depends on whether you are the favourite.

Four questions, in the order they need answering:

  1. How volatile is a week, by position and by scoring level?
  2. Is being *unusually* volatile for your scoring level a repeatable trait, or
     is "he is inconsistent" a description of last season and nothing more?
  3. Do teammates' weeks move together? A quarterback and his receiver share a
     game script, and correlated starters make a lineup swingier than the sum of
     its parts.
  4. Given all that, does adding variance to a lineup help or hurt?

The fourth is the one that pays for the other three. If volatility is not a
draftable trait, the only lever on lineup variance is roster construction --
which players you pair, not which players you pick.

    python scripts/eda_weekly.py
    python scripts/eda_weekly.py --write-report
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import DATA_PROCESSED, LEAGUE, ROOT, SCHEDULE
from src.features.eda import grouped_spearman, residualise_within
from src.ingest import nflverse as nv
from src.scoring import add_fantasy_points

SKILL = ("QB", "RB", "WR", "TE")
MIN_WEEKS = 8            # below this a player-season's SD is not an estimate
BOOM, BUST = 20.0, 5.0   # a week that wins you a matchup, and one that loses it

# One team's skill starters in this league: QB, 2RB, 2WR, TE, and a FLEX that
# the README shows always goes to a receiver.
LINEUP = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
PARTITIONS = 30          # random league draws per season, for section four


def _weekly(seasons: list[int]) -> pl.DataFrame:
    """Every regular-season player-week, scored under this league's rules."""
    frames = []
    for season in seasons:
        raw = nv.player_stats(seasons=[season], level="week")
        if "season_type" in raw.columns:
            raw = raw.filter(pl.col("season_type") == "REG")
        scored = add_fantasy_points(raw)
        pos_col = "position" if "position" in scored.columns else "position_group"
        frames.append(
            scored.filter(pl.col(pos_col).is_in(SKILL))
            .select(
                "player_id", "week", "team",
                pl.col("season").cast(pl.Int32),
                pl.col(pos_col).alias("pos"),
                pl.col("fantasy_points_league").alias("points"),
            )
        )
        del raw, scored
        gc.collect()
    return pl.concat(frames, how="diagonal_relaxed")


def _player_seasons(weekly: pl.DataFrame) -> pl.DataFrame:
    """Per player-season: scoring level, weekly spread, and tail rates."""
    return (
        weekly.group_by(["player_id", "season", "pos"])
        .agg(
            pl.len().alias("weeks"),
            pl.col("points").mean().alias("ppg"),
            pl.col("points").std().alias("week_sd"),
            (pl.col("points") >= BOOM).mean().alias("boom_rate"),
            (pl.col("points") <= BUST).mean().alias("bust_rate"),
            pl.col("points").max().alias("best_week"),
        )
        .filter((pl.col("weeks") >= MIN_WEEKS) & pl.col("week_sd").is_not_null())
        .with_columns((pl.col("week_sd") / pl.col("ppg").clip(0.5, None)).alias("cv"))
    )


def _volatility_table(seasons_frame: pl.DataFrame) -> pl.DataFrame:
    """Weekly spread by position and scoring tier."""
    tiered = seasons_frame.with_columns(
        (
            (pl.col("ppg").rank("ordinal", descending=True).over(["season", "pos"]) - 1)
            * 3 // pl.len().over(["season", "pos"])
        ).alias("tier")
    )
    return (
        tiered.group_by(["pos", "tier"])
        .agg(
            pl.col("ppg").mean().alias("ppg"),
            pl.col("week_sd").mean().alias("week_sd"),
            pl.col("cv").mean().alias("cv"),
            pl.col("boom_rate").mean().alias("boom"),
            pl.col("bust_rate").mean().alias("bust"),
            pl.len().alias("n"),
        )
        .sort(["pos", "tier"])
    )


def _repeatability(seasons_frame: pl.DataFrame) -> pl.DataFrame:
    """Does a trait survive to next season once scoring level is removed?"""
    resid = residualise_within(seasons_frame, "week_sd", "ppg")
    resid = residualise_within(resid, "boom_rate", "ppg", alias="boom_resid")
    resid = residualise_within(resid, "bust_rate", "ppg", alias="bust_resid")

    nxt = resid.select(
        "player_id", "season",
        pl.col("ppg").alias("ppg_next"),
        pl.col("week_sd").alias("week_sd_next"),
        pl.col("week_sd_resid").alias("week_sd_resid_next"),
        pl.col("boom_resid").alias("boom_resid_next"),
        pl.col("bust_resid").alias("bust_resid_next"),
    ).with_columns((pl.col("season") - 1).cast(pl.Int32).alias("season"))

    joined = resid.join(nxt, on=["player_id", "season"], how="inner")
    rows = []
    for label, a, b in (
        ("points per game", "ppg", "ppg_next"),
        ("weekly SD, raw", "week_sd", "week_sd_next"),
        ("weekly SD, net of scoring level", "week_sd_resid", "week_sd_resid_next"),
        ("boom rate, net of level", "boom_resid", "boom_resid_next"),
        ("bust rate, net of level", "bust_resid", "bust_resid_next"),
    ):
        got = grouped_spearman(joined, a, b)
        rows.append({"trait": label, "rho": got["rho"], "se": got["se"], "n": got["n"]})
    return pl.DataFrame(rows)


def _teammate_correlations(weekly: pl.DataFrame) -> pl.DataFrame:
    """Do a team's own starters' weeks move together?"""
    ranked = (
        weekly.group_by(["season", "team", "pos", "player_id"])
        .agg(pl.col("points").sum().alias("total"), pl.len().alias("weeks"))
        .filter(pl.col("weeks") >= MIN_WEEKS)
        .with_columns(
            pl.col("total").rank("ordinal", descending=True)
            .over(["season", "team", "pos"]).alias("depth")
        )
    )
    labelled = weekly.join(
        ranked.select("season", "team", "pos", "player_id", "depth"),
        on=["season", "team", "pos", "player_id"], how="inner",
    ).with_columns(
        (pl.col("pos") + pl.col("depth").cast(pl.Utf8)).alias("slot")
    )

    pairs = [("QB1", "WR1"), ("QB1", "WR2"), ("QB1", "TE1"), ("QB1", "RB1"),
             ("WR1", "WR2"), ("WR1", "TE1"), ("RB1", "WR1")]
    rows = []
    for left, right in pairs:
        per_season: dict[int, list[float]] = {}
        wide = labelled.filter(pl.col("slot").is_in([left, right]))
        for (season, team), chunk in wide.group_by(["season", "team"]):
            pivot = chunk.pivot(values="points", index="week", on="slot",
                                aggregate_function="first")
            if left not in pivot.columns or right not in pivot.columns:
                continue
            both = pivot.select(left, right).drop_nulls()
            if both.height < MIN_WEEKS:
                continue
            a, b = both[left].to_numpy(), both[right].to_numpy()
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                continue
            per_season.setdefault(int(season), []).append(
                float(np.corrcoef(a, b)[0, 1])
            )
        if not per_season:
            continue
        season_means = np.array([float(np.mean(v)) for v in per_season.values()])
        rows.append({
            "pair": f"{left} / {right}",
            "r": float(np.mean([x for v in per_season.values() for x in v])),
            "se": float(season_means.std(ddof=1) / np.sqrt(len(season_means))),
            "n": sum(len(v) for v in per_season.values()),
        })
    return pl.DataFrame(rows)


def _replacement_week(weekly: pl.DataFrame) -> dict:
    """Median weekly score of the first man off the bench, per season-position.

    A starter who does not play is not a zero in a real lineup -- somebody gets
    started in his place. Substituting replacement level keeps bye weeks and
    inactives from inflating every variance number in section four.
    """
    counts = {pos: LEAGUE["teams"] * n for pos, n in LINEUP.items()}
    level = (
        weekly.group_by(["player_id", "season", "pos"])
        .agg(pl.col("points").mean().alias("ppg"), pl.len().alias("weeks"))
        .filter(pl.col("weeks") >= MIN_WEEKS)
        .with_columns(
            pl.col("ppg").rank("ordinal", descending=True)
            .over(["season", "pos"]).alias("rank")
        )
    )
    out = {}
    for pos, count in counts.items():
        band = level.filter(
            (pl.col("pos") == pos)
            & (pl.col("rank") > count)
            & (pl.col("rank") <= count + LEAGUE["teams"])
        )
        for season in band["season"].unique().to_list():
            sub = band.filter(pl.col("season") == season)
            out[(int(season), pos)] = float(sub["ppg"].median() or 0.0)
    return out


def _synthetic_leagues(weekly: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Deal starter-quality players into random leagues and play the season.

    The point is to get the joint distribution of weekly lineup totals from real
    player-weeks -- real correlations, real injuries, real blowout weeks -- rather
    than assuming a shape for it. Random assignment is what makes the mean and
    the spread vary independently enough to separate their effects.
    """
    weeks = list(range(1, SCHEDULE["regular_season_weeks"] + 1))
    replacement = _replacement_week(weekly)
    counts = {pos: LEAGUE["teams"] * n for pos, n in LINEUP.items()}
    rows = []

    for season in sorted(weekly["season"].unique().to_list()):
        year = weekly.filter(
            (pl.col("season") == season) & pl.col("week").is_in(weeks)
        )
        level = (
            year.group_by(["player_id", "pos"])
            .agg(pl.col("points").mean().alias("ppg"), pl.len().alias("weeks"))
            .filter(pl.col("weeks") >= MIN_WEEKS)
            .with_columns(
                pl.col("ppg").rank("ordinal", descending=True)
                .over("pos").alias("rank")
            )
        )

        blocks = {}
        for pos, count in counts.items():
            pool = level.filter((pl.col("pos") == pos) & (pl.col("rank") <= count))
            if pool.height < count:
                blocks = {}
                break
            grid = (
                pool.select("player_id")
                .join(pl.DataFrame({"week": weeks}), how="cross")
                .join(
                    year.filter(pl.col("pos") == pos).select("player_id", "week", "points"),
                    on=["player_id", "week"], how="left",
                )
                .with_columns(
                    pl.col("points").fill_null(replacement.get((int(season), pos), 0.0))
                )
                .sort(["player_id", "week"])
            )
            matrix = grid["points"].to_numpy().reshape(pool.height, len(weeks))
            blocks[pos] = matrix
        if not blocks:
            continue

        for draw in range(PARTITIONS):
            totals = np.zeros((LEAGUE["teams"], len(weeks)))
            for pos, slots in LINEUP.items():
                matrix = blocks[pos]
                order = rng.permutation(matrix.shape[0])
                dealt = matrix[order].reshape(LEAGUE["teams"], slots, len(weeks))
                totals += dealt.sum(axis=1)

            # Every team plays every other team, every week.
            wins = np.zeros(LEAGUE["teams"])
            for week in range(len(weeks)):
                scores = totals[:, week]
                wins += (scores[:, None] > scores[None, :]).sum(axis=1)
            played = (LEAGUE["teams"] - 1) * len(weeks)

            for team in range(LEAGUE["teams"]):
                rows.append({
                    "season": int(season), "draw": draw, "team": team,
                    "mean": float(totals[team].mean()),
                    "sd": float(totals[team].std(ddof=1)),
                    "win_rate": float(wins[team] / played),
                })
    return pl.DataFrame(rows)


def _variance_effect(leagues: pl.DataFrame) -> pl.DataFrame:
    """Effect of lineup spread on win rate, within bands of scoring level."""
    banded = leagues.with_columns(
        (pl.col("mean") - pl.col("mean").mean().over(["season", "draw"]))
        .alias("edge")
    ).with_columns(
        pl.when(pl.col("edge") < -8).then(pl.lit("well below average"))
        .when(pl.col("edge") < -2.5).then(pl.lit("below average"))
        .when(pl.col("edge") <= 2.5).then(pl.lit("average"))
        .when(pl.col("edge") <= 8).then(pl.lit("above average"))
        .otherwise(pl.lit("well above average"))
        .alias("band")
    )
    rows = []
    for band in ("well below average", "below average", "average",
                 "above average", "well above average"):
        sub = banded.filter(pl.col("band") == band)
        if sub.height < 200:
            continue
        # Slope of win rate on lineup SD, holding the scoring edge fixed.
        design = np.column_stack([
            np.ones(sub.height),
            sub["sd"].to_numpy().astype(float),
            sub["edge"].to_numpy().astype(float),
        ])
        y = sub["win_rate"].to_numpy().astype(float)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        dof = max(sub.height - design.shape[1], 1)
        cov = np.linalg.pinv(design.T @ design) * (resid @ resid) / dof
        rows.append({
            "band": band, "edge": float(sub["edge"].mean()),
            "sd": float(sub["sd"].mean()),
            "win_rate": float(sub["win_rate"].mean()),
            "slope": float(coef[1]), "se": float(np.sqrt(cov[1, 1])),
            "n": sub.height,
        })
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2015)
    ap.add_argument("--to-season", type=int, default=2025)
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    weekly = _weekly(seasons)
    frame = _player_seasons(weekly)

    print(f"Weekly scoring EDA  {seasons[0]}-{seasons[-1]}\n")
    print(f"  {weekly.height} player-weeks, {frame.height} player-seasons "
          f"with {MIN_WEEKS}+ games")

    # --- 1. how volatile is a week ----------------------------------------
    print("\n" + "=" * 76)
    print("  WEEKLY SPREAD, BY POSITION AND SCORING TIER")
    print("=" * 76)
    print(f"  boom = a week of {BOOM:.0f}+ points, bust = {BUST:.0f} or fewer.\n")
    volatility = _volatility_table(frame)
    print(f"  {'pos':4s} {'tier':>5s} {'ppg':>7s} {'week sd':>8s} {'sd/ppg':>7s} "
          f"{'boom':>6s} {'bust':>6s} {'n':>6s}")
    for r in volatility.iter_rows(named=True):
        print(f"  {r['pos']:4s} {r['tier'] + 1:5d} {r['ppg']:7.1f} "
              f"{r['week_sd']:8.1f} {r['cv']:7.2f} {r['boom']:6.0%} "
              f"{r['bust']:6.0%} {r['n']:6d}")

    # --- 2. does it repeat -------------------------------------------------
    print("\n" + "=" * 76)
    print("  DOES ANY OF IT REPEAT?")
    print("=" * 76)
    print("  Year-over-year correlation of each trait with itself. 'Net of")
    print("  scoring level' removes what points per game already explains.\n")
    repeat = _repeatability(frame)
    for r in repeat.iter_rows(named=True):
        print(f"  {r['trait']:34s} {r['rho']:+.3f} +/- {r['se']:.3f}  n={r['n']}")

    # --- 3. teammates ------------------------------------------------------
    print("\n" + "=" * 76)
    print("  DO TEAMMATES' WEEKS MOVE TOGETHER?")
    print("=" * 76)
    print("  Within-team correlation of weekly points, pooled over team-seasons.\n")
    pairs = _teammate_correlations(weekly)
    for r in pairs.iter_rows(named=True):
        print(f"  {r['pair']:14s} r = {r['r']:+.3f} +/- {r['se']:.3f}  "
              f"({r['n']} team-seasons)")

    # --- 4. what variance does --------------------------------------------
    print("\n" + "=" * 76)
    print("  WHAT VARIANCE DOES TO A WEEKLY RECORD")
    print("=" * 76)
    print(f"  {PARTITIONS} random leagues per season, dealt from the real")
    print("  starter pool and played out on real weekly scores. Slope is the")
    print("  change in win rate per extra point of weekly lineup SD, holding")
    print("  the team's scoring edge fixed.\n")
    leagues = _synthetic_leagues(weekly, np.random.default_rng(12))
    effect = _variance_effect(leagues)
    print(f"  {'band':22s} {'edge':>7s} {'sd':>7s} {'win rate':>9s} "
          f"{'slope':>9s} {'+/-':>7s}")
    for r in effect.iter_rows(named=True):
        print(f"  {r['band']:22s} {r['edge']:+7.1f} {r['sd']:7.1f} "
              f"{r['win_rate']:9.3f} {r['slope']:+9.5f} {r['se']:7.5f}")

    # --- 5. what that means for stacking -----------------------------------
    print("\n" + "=" * 76)
    print("  SO WHAT DOES STACKING COST?")
    print("=" * 76)
    stack = _stacking_cost(volatility, pairs, leagues, effect)
    if stack:
        print(f"  Pairing your quarterback with his own WR1 adds "
              f"{stack['extra_var']:.0f} to lineup variance")
        print(f"  ({2}rho x sd_QB x sd_WR, with rho = {stack['r']:+.3f}), taking "
              f"weekly SD from {stack['base_sd']:.1f} to {stack['stacked_sd']:.1f}.\n")
        print(f"  {'band':22s} {'win rate change':>16s} {'over 14 weeks':>15s}")
        for row in stack["bands"]:
            print(f"  {row['band']:22s} {row['delta']:+16.4f} "
                  f"{row['delta'] * 14:+15.2f} wins")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    volatility.write_parquet(DATA_PROCESSED / "eda_weekly.parquet")
    print(f"\nWrote {DATA_PROCESSED / 'eda_weekly.parquet'}")

    if args.write_report:
        path = _write_report(args, weekly, frame, volatility, repeat, pairs,
                             leagues, effect, stack)
        print(f"Wrote {path}")


def _stacking_cost(volatility: pl.DataFrame, pairs: pl.DataFrame,
                   leagues: pl.DataFrame, effect: pl.DataFrame) -> dict:
    """Turn the correlation measurement into a win-rate number.

    Two starters with weekly spreads sd_a and sd_b and correlation rho add
    2 x rho x sd_a x sd_b to the variance of their lineup's weekly total, over
    and above what they contribute independently. Everything on the right-hand
    side has been measured above, so the cost of a stack follows from the slopes
    in section four rather than from an opinion about upside.
    """
    row = pairs.filter(pl.col("pair") == "QB1 / WR1")
    if row.height == 0 or effect.height == 0:
        return {}
    r = float(row["r"][0])

    def top_tier_sd(pos: str) -> float:
        block = volatility.filter((pl.col("pos") == pos) & (pl.col("tier") == 0))
        return float(block["week_sd"][0]) if block.height else 0.0

    sd_qb, sd_wr = top_tier_sd("QB"), top_tier_sd("WR")
    extra_var = 2.0 * r * sd_qb * sd_wr
    base_sd = float(leagues["sd"].mean())
    stacked_sd = float(np.sqrt(base_sd**2 + extra_var))

    bands = [
        {"band": e["band"], "delta": e["slope"] * (stacked_sd - base_sd)}
        for e in effect.iter_rows(named=True)
    ]
    return {"r": r, "sd_qb": sd_qb, "sd_wr": sd_wr, "extra_var": extra_var,
            "base_sd": base_sd, "stacked_sd": stacked_sd, "bands": bands}


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _write_report(args, weekly, frame, volatility, repeat, pairs, leagues,
                  effect, stack) -> Path:
    out = ROOT / "docs" / "eda_weekly.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EDA: weekly scoring, consistency, and what head-to-head rewards",
        "",
        f"Generated by `scripts/eda_weekly.py` over {args.from_season}-"
        f"{args.to_season}. {weekly.height} player-weeks, {frame.height} "
        f"player-seasons with {MIN_WEEKS}+ games.",
        "",
        "## Weekly spread, by position and scoring tier",
        "",
        f"`boom` is a week of {BOOM:.0f}+ points, `bust` is {BUST:.0f} or fewer.",
        "",
        _row(["pos", "tier", "ppg", "week SD", "SD/ppg", "boom", "bust", "n"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in volatility.iter_rows(named=True):
        lines.append(_row([
            r["pos"], str(r["tier"] + 1), f"{r['ppg']:.1f}", f"{r['week_sd']:.1f}",
            f"{r['cv']:.2f}", f"{r['boom']:.0%}", f"{r['bust']:.0%}", str(r["n"]),
        ]))

    lines += [
        "", "## Does any of it repeat?", "",
        "Year-over-year correlation of each trait with itself. *Net of scoring "
        "level* removes what points per game already explains.",
        "", _row(["trait", "year-over-year rho", "n"]),
        _row(["---", "---:", "---:"]),
    ]
    for r in repeat.iter_rows(named=True):
        lines.append(_row([
            r["trait"], f"**{r['rho']:+.3f} ± {r['se']:.3f}**", str(r["n"]),
        ]))

    lines += [
        "", "## Do teammates' weeks move together?", "",
        "Within-team correlation of weekly points, pooled over team-seasons. "
        "Slots are by season points at the position, so `WR1` is the team's "
        "leading receiver that year.",
        "", _row(["pair", "r", "team-seasons"]),
        _row(["---", "---:", "---:"]),
    ]
    for r in pairs.iter_rows(named=True):
        lines.append(_row([
            r["pair"], f"**{r['r']:+.3f} ± {r['se']:.3f}**", str(r["n"]),
        ]))

    lines += [
        "", "## What variance does to a weekly record", "",
        f"{PARTITIONS} random leagues per season, dealt from the real "
        "starter pool and played out on real weekly scores — so the correlations "
        "and the blowout weeks are the ones that happened, not ones assumed. A "
        "starter with no stat line is replaced by that season's replacement-level "
        "weekly score, because a real manager starts somebody.",
        "",
        f"{leagues.height} synthetic team-seasons. `slope` is the change in win "
        "rate per extra point of weekly lineup SD, holding the team's scoring "
        "edge fixed.",
        "",
        _row(["band", "edge (ppw)", "lineup SD", "win rate", "slope", "±"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in effect.iter_rows(named=True):
        lines.append(_row([
            r["band"], f"{r['edge']:+.1f}", f"{r['sd']:.1f}",
            f"{r['win_rate']:.3f}", f"**{r['slope']:+.5f}**", f"{r['se']:.5f}",
        ]))

    if stack:
        lines += [
            "", "## So what does stacking cost?", "",
            "Two starters with weekly spreads `sd_a` and `sd_b` and correlation "
            "`r` add `2·r·sd_a·sd_b` to the variance of the lineup's weekly "
            "total. Every term on the right has been measured above, so the cost "
            "of a stack follows from the slopes rather than from an opinion "
            "about upside.",
            "",
            f"Pairing your quarterback with his own WR1 (r = {stack['r']:+.3f}, "
            f"top-tier weekly SDs {stack['sd_qb']:.1f} and {stack['sd_wr']:.1f}) "
            f"adds {stack['extra_var']:.0f} to lineup variance, taking weekly SD "
            f"from {stack['base_sd']:.1f} to {stack['stacked_sd']:.1f}.",
            "",
            _row(["band", "change in win rate", "over 14 weeks"]),
            _row(["---", "---:", "---:"]),
        ]
        for row in stack["bands"]:
            lines.append(_row([
                row["band"], f"**{row['delta']:+.4f}**",
                f"{row['delta'] * 14:+.2f} wins",
            ]))

    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
