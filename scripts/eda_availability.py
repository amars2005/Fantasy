"""EDA: what predicts how many games a player will actually play.

Season points are (points per game) x (games played), and the year-over-year
study found those two factors are predictable to very different degrees. Prior
production predicts next season's *rate* at +0.69 and next season's *games* at
+0.39; games played predicts games played at **+0.298**, the weakest
self-correlation in the whole table.

That is a strange result to leave alone, because availability is half the
outcome and the projection curve prices it implicitly. So:

  1. What does the games distribution actually look like, by position and age?
  2. Does last season's injury report add anything over last season's games?
     The model carries four injury features on the strength of one ablation;
     this is the direct measurement.
  3. Is a missed season a warning about the *rate* too, or only about the games?
  4. What does missed time cost, in points, at each position?

The third question is the one with a decision attached. "He's injury prone" is
used to fade a player entirely; if a lost season predicts only lost games and
not a worse player, the fade is priced wrong.

    python scripts/eda_availability.py
    python scripts/eda_availability.py --write-report
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src.config import DATA_PROCESSED, REGULAR_SEASON_WEEKS, ROOT
from src.features.eda import bucket_effect, grouped_spearman
from src.features.player_season import build as build_player_season

SKILL = ("QB", "RB", "WR", "TE")
INJURY_FEATURES = [
    "inj_weeks_out_lag1", "inj_weeks_questionable_lag1",
    "inj_weeks_dnp_lag1", "inj_weeks_on_report_lag1",
]
# A season with fewer than this many games is one the player mostly missed.
LOST_SEASON = 10


def _frame(seasons: list[int]) -> pl.DataFrame:
    """Player-seasons with prior production, injury history and the outcome."""
    from src.features.build import add_targets, build

    span = list(range(min(seasons) - 2, max(seasons) + 1))
    stats = build_player_season(span)
    feat = add_targets(build(seasons, stats=stats), stats)
    del stats
    gc.collect()

    return (
        feat.filter(pl.col("games_lag1") >= 1)
        .with_columns(
            (pl.col("y_points") / pl.col("y_games").clip(1, 17)).alias("y_ppg"),
            (pl.col("games_lag1") < LOST_SEASON).cast(pl.Int8).alias("lost_last_year"),
            (pl.col("y_games") >= 16).cast(pl.Int8).alias("played_full"),
        )
        .with_columns(
            [pl.col(c).fill_null(0.0) for c in INJURY_FEATURES
             if c in feat.columns]
        )
    )


def _by_age(frame: pl.DataFrame) -> pl.DataFrame:
    banded = frame.filter(pl.col("age").is_not_null()).with_columns(
        pl.when(pl.col("age") < 24).then(pl.lit("<24"))
        .when(pl.col("age") < 27).then(pl.lit("24-26"))
        .when(pl.col("age") < 30).then(pl.lit("27-29"))
        .otherwise(pl.lit("30+")).alias("band")
    )
    return (
        banded.group_by(["pos", "band"])
        .agg(
            pl.col("y_games").mean().alias("games"),
            pl.col("played_full").mean().alias("full"),
            pl.len().alias("n"),
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=2017)
    ap.add_argument("--to-season", type=int, default=2025)
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    frame = _frame(seasons)
    played = frame.filter(pl.col("games_lag1") >= 8)

    print(f"Availability EDA  {seasons[0]}-{seasons[-1]}\n")
    print(f"  {frame.height} player-seasons with a prior season; "
          f"{played.height} of them with 8+ prior games")

    # --- 1. the distribution ------------------------------------------------
    print("\n" + "=" * 76)
    print("  HOW MANY GAMES DOES A PLAYER ACTUALLY PLAY?")
    print("=" * 76)
    print("  Players who had a real prior season (8+ games).\n")
    dist = (
        played.group_by("pos")
        .agg(
            pl.col("y_games").mean().alias("mean"),
            pl.col("y_games").median().alias("median"),
            pl.col("played_full").mean().alias("full"),
            (pl.col("y_games") == 0).mean().alias("none"),
            (pl.col("y_games") < LOST_SEASON).mean().alias("lost"),
            pl.len().alias("n"),
        )
        .sort("pos")
    )
    print(f"  {'pos':4s} {'mean':>6s} {'median':>7s} {'16+':>6s} "
          f"{'<10':>6s} {'zero':>6s} {'n':>6s}")
    for r in dist.iter_rows(named=True):
        print(f"  {r['pos']:4s} {r['mean']:6.1f} {r['median']:7.0f} "
              f"{r['full']:6.0%} {r['lost']:6.0%} {r['none']:6.0%} {r['n']:6d}")

    ages = _by_age(played)
    print("\n  Games played by age band:\n")
    print(f"  {'pos':4s} {'<24':>10s} {'24-26':>10s} {'27-29':>10s} {'30+':>10s}")
    for pos in SKILL:
        cells = []
        for band in ("<24", "24-26", "27-29", "30+"):
            row = ages.filter((pl.col("pos") == pos) & (pl.col("band") == band))
            cells.append(f"{row['games'][0]:10.1f}" if row.height else f"{'-':>10s}")
        print(f"  {pos:4s} " + " ".join(cells))

    # --- 2. does the injury report add anything -----------------------------
    print("\n" + "=" * 76)
    print("  DOES LAST SEASON'S INJURY REPORT ADD TO LAST SEASON'S GAMES?")
    print("=" * 76)
    print("  Partial correlation with next-season games, holding prior games")
    print("  fixed. This is the direct test of a feature group the model")
    print("  carries on the strength of one ablation.\n")
    injury_rows = []
    baseline = grouped_spearman(played, "games_lag1", "y_games")
    print(f"  {'games_lag1 (the baseline)':32s} {baseline['rho']:+.3f} "
          f"+/- {baseline['se']:.3f}")
    for feature in INJURY_FEATURES:
        if feature not in played.columns:
            continue
        raw = grouped_spearman(played, feature, "y_games")
        partial = grouped_spearman(played, feature, "y_games", control="games_lag1")
        injury_rows.append({
            "feature": feature, "raw": raw["rho"], "raw_se": raw["se"],
            "partial": partial["rho"], "partial_se": partial["se"],
            "n": partial["n"],
        })
        mark = "*" if abs(partial["rho"]) > 2 * partial["se"] else " "
        print(f"  {feature:32s} {raw['rho']:+.3f}  ->  "
              f"{partial['rho']:+.3f}{mark} +/- {partial['se']:.3f}")
    print("\n  * clears twice its across-season spread. Left column is the raw")
    print("    correlation, right is what survives knowing prior games.")

    age_partial = grouped_spearman(played, "age", "y_games", control="games_lag1")
    print(f"\n  {'age':32s} {'':6s}     {age_partial['rho']:+.3f} "
          f"+/- {age_partial['se']:.3f}")

    # --- 3. is a lost season a warning about the rate too? ------------------
    print("\n" + "=" * 76)
    print("  IS A LOST SEASON A WARNING ABOUT THE PLAYER, OR ONLY HIS BODY?")
    print("=" * 76)
    print("  Players whose prior season was cut short (under 10 games) against")
    print("  those who played through, matched on prior points per game.\n")
    def contrast(sub: pl.DataFrame, outcome: str, control: str) -> tuple:
        got = bucket_effect(sub, "lost_last_year", outcome, control=control,
                            n_buckets=2)
        if got.height != 2:
            return float("nan"), float("nan"), 0
        effect = float(got["effect"][1] - got["effect"][0])
        se = float((got["se"][1] ** 2 + got["se"][0] ** 2) ** 0.5)
        return effect, se, int(got["n"][1])

    rate_rows = []
    print(f"  {'pos':4s} {'next games':>11s} {'+/-':>6s} {'next ppg':>10s} {'+/-':>6s}"
          f" {'n short':>8s}")
    for pos in SKILL:
        sub = frame.filter(pl.col("pos") == pos)
        g, g_se, n = contrast(sub, "y_games", "ppg_lag1")
        p, p_se, _ = contrast(sub, "y_ppg", "ppg_lag1")
        if not n:
            continue
        rate_rows.append({"pos": pos, "games": g, "games_se": g_se,
                          "ppg": p, "ppg_se": p_se, "n": n})
        print(f"  {pos:4s} {g:+11.2f} {g_se:6.2f} {p:+10.2f} {p_se:6.2f} {n:8d}")

    # The rate result has a confound: a short season measures ppg over few
    # games, so `ppg_lag1` is a noisier control for exactly the players being
    # tested, and a noisier control regresses them further toward the mean all
    # by itself. Matching on the season *before* the injury year instead uses a
    # control the injury cannot have contaminated.
    print("\n  Robustness: matched on the season before the injury year")
    print("  instead, which the injury cannot have contaminated.\n")
    clean = frame.filter(
        pl.col("ppg_lag2").is_not_null() & (pl.col("games_lag2") >= 8)
    )
    print(f"  {'pos':4s} {'next ppg':>10s} {'+/-':>6s} {'n short':>8s}")
    for row in rate_rows:
        sub = clean.filter(pl.col("pos") == row["pos"])
        p2, p2_se, n2 = contrast(sub, "y_ppg", "ppg_lag2")
        row["ppg_clean"], row["ppg_clean_se"], row["n_clean"] = p2, p2_se, n2
        if n2:
            print(f"  {row['pos']:4s} {p2:+10.2f} {p2_se:6.2f} {n2:8d}")

    # --- 4. what it costs ---------------------------------------------------
    print("\n" + "=" * 76)
    print("  WHAT MISSED TIME COSTS")
    print("=" * 76)
    starters = played.filter(pl.col("ppg_lag1") >= 10.0)
    cost_rows = []
    print(f"  {'pos':4s} {'ppg':>7s} {'games':>7s} {'expected':>9s} "
          f"{'if 17':>7s} {'lost':>7s}")
    for pos in SKILL:
        sub = starters.filter(pl.col("pos") == pos)
        if sub.height < 40:
            continue
        ppg = float(sub["y_ppg"].mean())
        games = float(sub["y_games"].mean())
        expected = ppg * games
        full = ppg * REGULAR_SEASON_WEEKS
        cost_rows.append({"pos": pos, "ppg": ppg, "games": games,
                          "expected": expected, "full": full,
                          "lost": full - expected, "n": sub.height})
        print(f"  {pos:4s} {ppg:7.2f} {games:7.1f} {expected:9.1f} "
              f"{full:7.1f} {full - expected:7.1f}")
    print("\n  For players who were startable last season (10+ ppg). 'lost' is")
    print("  what availability alone takes off a full-season projection.")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    dist.write_parquet(DATA_PROCESSED / "eda_availability.parquet")
    print(f"\nWrote {DATA_PROCESSED / 'eda_availability.parquet'}")

    if args.write_report:
        path = _write_report(args, frame, played, dist, ages, baseline,
                             injury_rows, age_partial, rate_rows, cost_rows)
        print(f"Wrote {path}")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _write_report(args, frame, played, dist, ages, baseline, injury_rows,
                  age_partial, rate_rows, cost_rows) -> Path:
    out = ROOT / "docs" / "eda_availability.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# EDA: what predicts how many games a player actually plays",
        "",
        f"Generated by `scripts/eda_availability.py` over {args.from_season}-"
        f"{args.to_season}. {frame.height} player-seasons with a prior season, "
        f"{played.height} of them with 8+ prior games.",
        "",
        "## How many games does a player actually play?",
        "",
        _row(["pos", "mean", "median", "16+ games", "under 10", "zero", "n"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in dist.iter_rows(named=True):
        lines.append(_row([
            r["pos"], f"{r['mean']:.1f}", f"{r['median']:.0f}", f"{r['full']:.0%}",
            f"{r['lost']:.0%}", f"{r['none']:.0%}", str(r["n"]),
        ]))

    lines += ["", "Games played by age band:", "",
              _row(["pos", "<24", "24-26", "27-29", "30+"]),
              _row(["---", "---:", "---:", "---:", "---:"])]
    for pos in SKILL:
        cells = [pos]
        for band in ("<24", "24-26", "27-29", "30+"):
            row = ages.filter((pl.col("pos") == pos) & (pl.col("band") == band))
            cells.append(f"{row['games'][0]:.1f}" if row.height else "—")
        lines.append(_row(cells))

    lines += [
        "", "## Does last season's injury report add to last season's games?", "",
        "Partial correlation with next-season games, holding prior games fixed. "
        f"The baseline — prior games against next games — is "
        f"**{baseline['rho']:+.3f} ± {baseline['se']:.3f}**.",
        "",
        _row(["feature", "raw", "holding prior games fixed", "n"]),
        _row(["---", "---:", "---:", "---:"]),
    ]
    for r in injury_rows:
        partial = f"{r['partial']:+.3f} ± {r['partial_se']:.3f}"
        real = abs(r["partial"]) > 2 * r["partial_se"]
        lines.append(_row([
            f"`{r['feature']}`", f"{r['raw']:+.3f}",
            f"**{partial}**" if real else partial, str(r["n"]),
        ]))
    lines.append(_row(["`age`", "—",
                       f"{age_partial['rho']:+.3f} ± {age_partial['se']:.3f}",
                       str(age_partial["n"])]))

    lines += [
        "", "## Is a lost season a warning about the player, or only his body?",
        "",
        "Players whose prior season was cut short (under 10 games) against those "
        "who played through, matched on prior points per game. Left column is the "
        "effect on next-season *games*, right on next-season *points per game*.",
        "",
        _row(["pos", "effect on games", "±", "effect on ppg", "±", "n short"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in rate_rows:
        lines.append(_row([
            r["pos"], f"**{r['games']:+.2f}**", f"{r['games_se']:.2f}",
            f"**{r['ppg']:+.2f}**", f"{r['ppg_se']:.2f}", str(r["n"]),
        ]))

    lines += [
        "",
        "The rate column has a confound. A short season measures points per game "
        "over few games, so `ppg_lag1` is a noisier control for exactly the "
        "players being tested, and a noisier control regresses them further "
        "toward the mean on its own. Matching on the season *before* the injury "
        "year instead uses a control the injury cannot have contaminated:",
        "",
        _row(["pos", "effect on ppg, clean control", "±", "n short"]),
        _row(["---", "---:", "---:", "---:"]),
    ]
    for r in rate_rows:
        if not r.get("n_clean"):
            continue
        lines.append(_row([
            r["pos"], f"**{r['ppg_clean']:+.2f}**", f"{r['ppg_clean_se']:.2f}",
            str(r["n_clean"]),
        ]))

    lines += [
        "", "## What missed time costs", "",
        "Players who were startable the season before (10+ ppg). `lost` is what "
        "availability alone takes off a full-season projection.",
        "",
        _row(["pos", "ppg", "games", "expected points", "if 17 games", "lost"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for r in cost_rows:
        lines.append(_row([
            r["pos"], f"{r['ppg']:.2f}", f"{r['games']:.1f}",
            f"{r['expected']:.1f}", f"{r['full']:.1f}", f"**{r['lost']:.1f}**",
        ]))

    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
