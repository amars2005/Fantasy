"""Evaluate projection sources by titles won, not by rank correlation.

Every model comparison in this project has been scored on within-position
Spearman. Nobody ever checked that Spearman is aligned with the thing the league
actually awards. It might not be: a projection that ranks slightly worse but
prices *uncertainty* better could win more titles, because a three-week bracket
rewards upside rather than expected points.

This closes that loop. For a historical season it drafts a team using a candidate
projection, fills the other thirteen rosters from that year's real ADP, then plays
the season out using **actual weekly scores from that season** and runs the real
playoff bracket. The output is a title rate, in the units the league pays out in.

Design note on fairness: every team sets its weekly lineup with hindsight. That
is generous to everyone equally, and it deliberately isolates the value of the
*draft* from the value of in-season lineup decisions. A projection that helped you
start the right players each week would score better still; this measures the
floor, not the ceiling.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from src.config import DATA_PROCESSED, LEAGUE, REGULAR_SEASON_WEEKS
from src.draft.h2h import simulate_season
from src.draft.replacement import add_vor
from src.draft.sim_draft import add_calibrated_adp
from src.draft.sim_draft import snake_picks
from src.draft.strategy import ROSTER_CAPS, _lineup_value
from src.draft.vona import vona_table
from src.ingest import nflverse as nv
from src.ingest.adp import load_adp
from src.ingest.ids import resolve
from src.project.consensus import fit_curves
from src.scoring import add_fantasy_points

SKILL = ["QB", "RB", "WR", "TE"]


def actual_weekly(season: int) -> pl.DataFrame:
    """Real weekly fantasy points per player for one season."""
    weekly = nv.player_stats(seasons=[season], level="week")
    if "season_type" in weekly.columns:
        weekly = weekly.filter(pl.col("season_type") == "REG")
    scored = add_fantasy_points(weekly)
    pos_col = "position" if "position" in scored.columns else "position_group"
    return (
        scored.filter(pl.col(pos_col).is_in(SKILL))
        .select(
            pl.col("player_id"),
            pl.col("week").cast(pl.Int32),
            pl.col("fantasy_points_league").alias("pts"),
        )
        .group_by(["player_id", "week"])
        .agg(pl.col("pts").sum())
    )


def board_for(season: int, projection: pl.DataFrame) -> pl.DataFrame:
    """Draftable board: that season's ADP joined to a candidate projection."""
    adp = resolve(load_adp(year=season)).filter(
        pl.col("pos").is_in(SKILL) & pl.col("gsis_id").is_not_null()
    ).select(
        pl.col("gsis_id").alias("player_id"), "name", "pos",
        pl.col("adp").cast(pl.Float64), pl.col("stdev").cast(pl.Float64),
    ).unique(subset=["player_id"], keep="first")

    board = adp.join(projection, on="player_id", how="inner")
    return add_calibrated_adp(add_vor(board, LEAGUE))


def weekly_matrix(board: pl.DataFrame, actual: pl.DataFrame) -> np.ndarray:
    """(n_players, weeks) of real weekly points, aligned to board row order."""
    weeks = list(range(1, REGULAR_SEASON_WEEKS + 1))
    wide = (
        actual.filter(pl.col("week").is_in(weeks))
        .pivot(values="pts", index="player_id", on="week", aggregate_function="sum")
    )
    joined = board.select("player_id").join(wide, on="player_id", how="left")
    cols = [str(w) for w in weeks if str(w) in joined.columns]
    mat = np.zeros((board.height, REGULAR_SEASON_WEEKS))
    for i, w in enumerate(weeks):
        if str(w) in joined.columns:
            mat[:, i] = joined[str(w)].fill_null(0.0).to_numpy()
    return mat


def lineup_scores(roster_idx: list[int], weekly: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Best legal lineup each week from real scores. Returns (weeks,)."""
    starters = LEAGUE["starters"]
    eligible = LEAGUE["flex_eligible"]
    rows = np.array(roster_idx, dtype=int)
    pts = weekly[rows]
    pos = positions[rows]

    total = np.zeros(weekly.shape[1])
    used: dict[str, int] = {}
    ranked: dict[str, np.ndarray] = {}
    for p in ("QB", "RB", "WR", "TE"):
        block = pts[pos == p]
        ranked[p] = -np.sort(-block, axis=0) if block.size else np.zeros((0, weekly.shape[1]))
        slots = starters.get(p, 0)
        take = ranked[p][:slots]
        total += take.sum(axis=0)
        used[p] = take.shape[0]

    flex_slots = starters.get("FLEX", 0)
    if flex_slots:
        pool = [ranked[p][used.get(p, 0):] for p in eligible if ranked[p].shape[0] > used.get(p, 0)]
        if pool:
            stacked = np.vstack(pool)
            total += (-np.sort(-stacked, axis=0))[:flex_slots].sum(axis=0)
    return total


def draft_with_vona(
    board: pl.DataFrame, slot: int, rng: np.random.Generator, n_sims: int = 800
) -> dict[int, list[int]]:
    """Run a draft where our picks use the real board logic, not a proxy.

    The strategy simulator picks greedily on value over replacement, which is
    fine for comparing positional policies against each other but is *not* what
    ships: static VOR over-rates quarterbacks and tight ends because it ignores
    that those curves are flat, and a VOR-greedy team takes an elite TE at pick 8
    and a quarterback at 21. The live board uses value over next available
    instead. Evaluating anything other than that measures a strawman.
    """
    teams, rounds = LEAGUE["teams"], LEAGUE["rounds"]
    picks = set(snake_picks(slot, teams, rounds))

    mu = board["adp_mu"].to_numpy().astype(float)
    stdev = np.maximum(board["stdev"].fill_null(0.5).to_numpy().astype(float), 0.5)
    positions = np.array(board["pos"].to_list())
    vor = np.maximum(board["vor"].to_numpy().astype(float), 0.1)
    preference = np.argsort(mu + rng.normal(0.0, 1.0, mu.size) * stdev)

    taken = np.zeros(mu.size, dtype=bool)
    rosters: dict[int, list[int]] = {t: [] for t in range(1, teams + 1)}
    counts: dict[int, dict[str, int]] = {t: {} for t in range(1, teams + 1)}
    row_ids = board["player_id"].to_list()
    index_of = {pid: i for i, pid in enumerate(row_ids)}

    overall = 0
    for rnd in range(1, rounds + 1):
        order = range(1, teams + 1) if rnd % 2 else range(teams, 0, -1)
        for team in order:
            overall += 1
            choice = None

            if overall in picks:
                # Our pick: value over what survives to our next selection.
                available = board.filter(
                    ~pl.col("player_id").is_in([row_ids[i] for i in np.flatnonzero(taken)])
                ) if taken.any() else board
                if available.height:
                    roster = [
                        {"pos": str(positions[i]), "proj_points": float(board["proj_points"][i])}
                        for i in rosters[team]
                    ]
                    following = [p for p in sorted(picks) if p > overall]
                    gap = (following[0] - overall) if following else available.height
                    table = vona_table(available, roster, max(gap, 1), LEAGUE,
                                       n_sims=n_sims, rng=rng)
                    for pid in table["player_id"].to_list():
                        idx = index_of[pid]
                        pos = str(positions[idx])
                        if counts[team].get(pos, 0) < ROSTER_CAPS.get(pos, 99):
                            choice = idx
                            break
            else:
                best = -np.inf
                considered = 0
                for idx in preference:
                    if taken[idx]:
                        continue
                    pos = str(positions[idx])
                    if counts[team].get(pos, 0) >= ROSTER_CAPS.get(pos, 99):
                        continue
                    score = vor[idx] * _lineup_value(counts[team], pos, LEAGUE)
                    if score > best:
                        best, choice = score, idx
                    considered += 1
                    if considered >= 8:
                        break

            if choice is None:
                remaining = preference[~taken[preference]]
                if remaining.size == 0:
                    continue
                choice = int(remaining[0])

            taken[choice] = True
            rosters[team].append(int(choice))
            pos = str(positions[choice])
            counts[team][pos] = counts[team].get(pos, 0) + 1

    return rosters


def evaluate(
    season: int,
    projection: pl.DataFrame,
    label: str,
    slot: int,
    n_drafts: int,
    rng: np.random.Generator,
) -> dict:
    board = board_for(season, projection)
    if board.height < 120:
        return {}
    actual = actual_weekly(season)
    weekly = weekly_matrix(board, actual)
    positions = np.array(board["pos"].to_list())

    titles, playoffs, points = [], [], []
    for _ in range(n_drafts):
        rosters = draft_with_vona(board, slot, rng)
        scores = np.zeros((LEAGUE["teams"], 1, REGULAR_SEASON_WEEKS))
        for team, idx in rosters.items():
            scores[team - 1, 0] = lineup_scores(idx, weekly, positions)
        result = simulate_season(scores, rng)
        titles.append(result["title_rate"][slot - 1])
        playoffs.append(result["playoff_rate"][slot - 1])
        points.append(result["points_for"][slot - 1])

    return {
        "season": season, "source": label,
        "title_rate": float(np.mean(titles)),
        "playoff_rate": float(np.mean(playoffs)),
        "points_for": float(np.mean(points)),
        "n": n_drafts,
    }


def consensus_projection(season: int, curves: dict) -> pl.DataFrame:
    """ADP-implied points, from curves fitted on prior seasons only."""
    adp = resolve(load_adp(year=season)).filter(
        pl.col("pos").is_in(SKILL) & pl.col("gsis_id").is_not_null()
    ).with_columns(pl.col("adp").rank("ordinal").over("pos").alias("r"))

    frames = []
    for pos, curve in curves.items():
        sub = adp.filter(pl.col("pos") == pos)
        if sub.height == 0:
            continue
        idx = np.clip(sub["r"].to_numpy().astype(int) - 1, 0, len(curve["grid"]) - 1)
        frames.append(sub.select(
            pl.col("gsis_id").alias("player_id"),
            pl.Series("proj_points", curve["mean"][idx]),
        ))
    return pl.concat(frames).unique(subset=["player_id"], keep="first")


def perfect_projection(season: int) -> pl.DataFrame:
    """What actually happened. The ceiling any projection is chasing."""
    actual = actual_weekly(season)
    return (
        actual.filter(pl.col("week") <= REGULAR_SEASON_WEEKS)
        .group_by("player_id")
        .agg(pl.col("pts").sum().alias("proj_points"))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=8)
    ap.add_argument("--drafts", type=int, default=60)
    ap.add_argument("--from-season", type=int, default=2021)
    ap.add_argument("--to-season", type=int, default=2025)
    args = ap.parse_args()

    seasons = list(range(args.from_season, args.to_season + 1))
    span = list(range(min(seasons) - 4, max(seasons) + 1))
    rng = np.random.default_rng(11)

    rows = []
    for season in seasons:
        curves = fit_curves([s for s in span if s < season], with_se=False)
        sources = {
            "consensus (ships today)": consensus_projection(season, curves),
            "perfect foresight": perfect_projection(season),
        }
        for label, projection in sources.items():
            result = evaluate(season, projection, label, args.slot, args.drafts, rng)
            if result:
                rows.append(result)
                print(f"  {season} {label:26s} title {result['title_rate']:.3f}  "
                      f"playoff {result['playoff_rate']:.3f}  pts {result['points_for']:.0f}")
        gc.collect()

    table = pl.DataFrame(rows)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.write_parquet(DATA_PROCESSED / "title_eval.parquet")

    print("\n" + "=" * 68)
    print(f"  Slot {args.slot}, {len(seasons)} seasons x {args.drafts} drafts, "
          f"real weekly outcomes")
    print(f"  Baselines: title {1 / LEAGUE['teams']:.3f}   playoff "
          f"{LEAGUE['starters'] and 6 / LEAGUE['teams']:.3f}")
    print("-" * 68)
    summary = (
        table.group_by("source")
        .agg(
            pl.col("title_rate").mean().alias("title"),
            pl.col("playoff_rate").mean().alias("playoff"),
            pl.col("points_for").mean().alias("points"),
        )
        .sort("title", descending=True)
    )
    for r in summary.iter_rows(named=True):
        print(f"  {r['source']:26s} title {r['title']:.3f}   "
              f"playoff {r['playoff']:.3f}   pts {r['points']:.0f}")
    print("=" * 68)

    ranked = summary.to_dicts()
    if len(ranked) >= 2:
        spread = ranked[0]["title"] - ranked[-1]["title"]
        print(f"\n  Perfect foresight beats shipping projections by "
              f"{spread:+.3f} title rate.")
        print("  That spread is the entire headroom available to projection work.")


if __name__ == "__main__":
    main()
