"""Emit golden fixtures: Python's answers, for the TypeScript port to match.

The v2 app reimplements the projection and decision maths in TypeScript. A hand
port is exactly where measured invariants quietly stop being true, so every
stage of it is pinned to a fixture generated here.

Two tolerance classes, because two kinds of maths are involved:

  * **Deterministic** -- scoring, isotonic fit, replacement, tiers, lineup,
    snake picks. Asserted to 1e-9. These must agree exactly.
  * **Stochastic** -- survival probabilities. numpy's PCG64 stream is not
    reproducible in JavaScript and reproducing it would buy nothing, so these
    are asserted distributionally instead. The fixture records the simulation
    count so the test can size its own tolerance.

The tier fixture deserves a note. `src/draft/tiers.py` starts a new tier
wherever projected points *change*, an exact float comparison, which works only
because isotonic regression assigns a bit-identical value across a pooled block.
So the fixture records tier **counts and boundaries**, not just values: a
tolerance check on values would pass happily while tiers were shattered into one
player each.

    python scripts/export_golden.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from src.config import (
    KICKER_SCORING,
    LEAGUE,
    ROOT,
    SCHEDULE,
    SCORING,
)
from src.draft.replacement import add_vor, replacement_levels
from src.draft.sim_draft import MIN_STDEV, survival_probability
from src.draft.tiers import add_tiers
from src.draft.vona import (
    BENCH_WEIGHT,
    BenchValue,
    bench_model,
    optimal_lineup_points,
    promotion_weights,
)
from src.dst import (
    DST_EVENT_SCORING,
    POINTS_ALLOWED_BANDS,
    YARDS_ALLOWED_BANDS,
)

BUNDLE = ROOT / "data" / "v2_export"
GOLDEN = ROOT / "v2" / "tests" / "golden"

SD_WINDOW = 8
MIN_OBSERVATIONS = 30
MAX_GAMES = 17
SURVIVAL_SIMS = 40000  # generous: the fixture is a reference, not a hot path


def _bands(rows) -> list[dict]:
    return [{"low": lo, "high": hi, "points": pts} for lo, hi, pts in rows]


def _dst_block() -> dict:
    return {
        "events": dict(DST_EVENT_SCORING),
        "pointsAllowedBands": _bands(POINTS_ALLOWED_BANDS),
        "yardsAllowedBands": _bands(YARDS_ALLOWED_BANDS),
    }


# --- the four configurations ------------------------------------------------
# Each carries the TypeScript-shaped config the fixture ships, plus the
# Python-shaped league dict `src/` functions expect. Keeping both here rather
# than translating in the test is deliberate: a translation bug would make a
# failing port look like a passing one.


def _configs() -> list[dict]:
    reference_ts = {
        "teams": LEAGUE["teams"],
        "rounds": LEAGUE["rounds"],
        "bench": LEAGUE["bench"],
        "irSlots": LEAGUE["ir_slots"],
        "starters": dict(LEAGUE["starters"]),
        "flexEligible": list(LEAGUE["flex_eligible"]),
        "positionMax": dict(LEAGUE["position_max"]),
        "schedule": {
            "regularSeasonWeeks": SCHEDULE["regular_season_weeks"],
            "playoffWeeks": list(SCHEDULE["playoff_weeks"]),
            "playoffTeams": SCHEDULE["playoff_teams"],
        },
        "scoring": dict(SCORING),
        "kickerScoring": dict(KICKER_SCORING),
        "dst": _dst_block(),
    }

    half_ppr = dict(SCORING)
    half_ppr["receptions"] = 0.5

    six_point = dict(SCORING)
    six_point["passing_tds"] = 6.0

    return [
        {
            "name": "reference_ppr_14",
            "format": "ppr",
            "ts": reference_ts,
            "py": LEAGUE,
        },
        {
            "name": "half_ppr_10",
            "format": "half-ppr",
            "ts": {
                **reference_ts,
                "teams": 10,
                "scoring": half_ppr,
                "starters": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "K": 1, "DST": 1},
            },
            "py": {
                **LEAGUE,
                "teams": 10,
                "starters": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "K": 1, "DST": 1},
            },
        },
        {
            # Superflex with six-point passing touchdowns: the case where the
            # borrowed ADP diverges most from what the room would really do.
            "name": "superflex_12",
            "format": "2qb",
            "ts": {
                **reference_ts,
                "teams": 12,
                "rounds": 16,
                "scoring": six_point,
                "starters": {
                    "QB": 1, "RB": 2, "WR": 2, "TE": 1,
                    "FLEX": 1, "SUPERFLEX": 1, "K": 1, "DST": 1,
                },
            },
            "py": {
                **LEAGUE,
                "teams": 12,
                "rounds": 16,
                "starters": {
                    "QB": 1, "RB": 2, "WR": 2, "TE": 1,
                    "FLEX": 1, "SUPERFLEX": 1, "K": 1, "DST": 1,
                },
            },
        },
        {
            # Degenerate on purpose: no flex at all, smallest supported league.
            "name": "no_flex_8",
            "format": "standard",
            "ts": {
                **reference_ts,
                "teams": 8,
                "scoring": {**SCORING, "receptions": 0.0},
                "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
            },
            "py": {
                **LEAGUE,
                "teams": 8,
                "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
            },
        },
    ]


# --- bundle loading ---------------------------------------------------------


def _table(name: str) -> pl.DataFrame:
    payload = json.loads((BUNDLE / f"{name}.json").read_text(encoding="utf-8"))
    return pl.DataFrame(
        {c: [r[i] for r in payload["rows"]] for i, c in enumerate(payload["columns"])}
    )


def _score(df: pl.DataFrame, rules: dict) -> np.ndarray:
    total = np.zeros(df.height)
    for stat, points in rules.items():
        if stat in df.columns:
            total += df[stat].fill_null(0).to_numpy().astype(float) * points
    return total


# --- the reference pipeline -------------------------------------------------


def _rolling_sd(ranks: np.ndarray, resid: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.empty(len(grid))
    for i, r in enumerate(grid):
        sample = resid[np.abs(ranks - r) <= SD_WINDOW]
        out[i] = sample.std() if sample.size >= 5 else resid.std()
    return out


def _fit(ranks: np.ndarray, values: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(ranks, values)
    return model


def fit_curves(training: pl.DataFrame, scoring: dict) -> dict:
    """Per-position rank -> (points, spread, games), under arbitrary scoring."""
    points = _score(training, scoring)
    curves = {}
    for pos in training["pos"].unique().to_list():
        mask = (training["pos"] == pos).to_numpy()
        if mask.sum() < MIN_OBSERVATIONS:
            continue
        ranks = training["pos_rank"].to_numpy().astype(float)[mask]
        pts = points[mask]
        games = training["games"].to_numpy().astype(float)[mask]

        model = _fit(ranks, pts)
        grid = np.arange(1, int(ranks.max()) + 1, dtype=float)
        resid = pts - model.predict(ranks)
        curves[pos] = {
            "grid": grid.tolist(),
            "mean": model.predict(grid).tolist(),
            "sd": _rolling_sd(ranks, resid, grid).tolist(),
            "games": np.clip(_fit(ranks, games).predict(grid), 0, MAX_GAMES).tolist(),
            "n": int(mask.sum()),
        }
    return curves


def _lookup(curve: dict, ranks: np.ndarray, field: str) -> np.ndarray:
    grid = np.asarray(curve["grid"])
    idx = np.clip(ranks.astype(int) - 1, 0, len(grid) - 1)
    return np.asarray(curve[field])[idx]


def project_skill(curves: dict, board: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for pos, curve in curves.items():
        sub = board.filter(pl.col("pos") == pos)
        if sub.height == 0:
            continue
        ranks = sub["pos_rank"].to_numpy()
        frames.append(
            sub.select("player_id", "name", "pos", "tm", "adp", "stdev", "bye",
                       "pos_rank", "adp_mu").with_columns(
                pl.Series("proj_raw", _lookup(curve, ranks, "mean")),
                pl.Series("sd_raw", _lookup(curve, ranks, "sd")),
                pl.Series("games_raw", _lookup(curve, ranks, "games")),
            ).with_columns(
                pl.col("proj_raw").round(1).alias("proj_points"),
                pl.col("sd_raw").round(1).alias("sd"),
                pl.col("games_raw").round(1).alias("games"),
            )
        )
    return pl.concat(frames) if frames else pl.DataFrame()


def _flat_curve(seasons: np.ndarray, points: np.ndarray) -> dict:
    """Finish-rank -> points, the way `src/project/kdst.py` builds it."""
    ranks = np.empty(len(points))
    for season in np.unique(seasons):
        mask = seasons == season
        order = np.argsort(-points[mask], kind="stable")
        r = np.empty(mask.sum())
        r[order] = np.arange(1, mask.sum() + 1)
        ranks[mask] = r

    model = _fit(ranks, points)
    grid = np.arange(1, int(ranks.max()) + 1, dtype=float)
    resid = points - model.predict(ranks)
    return {"grid": grid, "mean": model.predict(grid), "sd": float(resid.std())}


def project_kdst(board: pl.DataFrame, kicker: pl.DataFrame, dst: pl.DataFrame,
                 kicker_scoring: dict, dst_block: dict) -> pl.DataFrame:
    frames = []

    kickers = board.filter(pl.col("pos") == "K")
    if kickers.height:
        curve = _flat_curve(
            kicker["season"].to_numpy(), _score(kicker, kicker_scoring)
        )
        frames.append(_apply_flat(kickers, curve))

    defences = board.filter(pl.col("pos") == "DST")
    if defences.height:
        per_game = _score(dst, dst_block["events"])
        pa = dst["points_allowed"].to_numpy().astype(float)
        ya = dst["yards_allowed"].to_numpy().astype(float)
        per_game = per_game + _band(pa, dst_block["pointsAllowedBands"])
        per_game = per_game + _band(ya, dst_block["yardsAllowedBands"])

        totals = (
            dst.select("season", "team")
            .with_columns(pl.Series("pts", per_game))
            .group_by(["team", "season"])
            .agg(pl.col("pts").sum().alias("points"))
        )
        curve = _flat_curve(
            totals["season"].to_numpy(), totals["points"].to_numpy().astype(float)
        )
        frames.append(_apply_flat(defences, curve))

    return pl.concat(frames) if frames else pl.DataFrame()


def _band(values: np.ndarray, bands: list[dict]) -> np.ndarray:
    out = np.zeros(len(values))
    for b in bands:
        out = np.where((values >= b["low"]) & (values <= b["high"]), b["points"], out)
    return out


def _apply_flat(sub: pl.DataFrame, curve: dict) -> pl.DataFrame:
    ranks = sub["pos_rank"].to_numpy().astype(int)
    idx = np.clip(ranks - 1, 0, len(curve["grid"]) - 1)
    return sub.select("player_id", "name", "pos", "tm", "adp", "stdev", "bye",
                      "pos_rank", "adp_mu").with_columns(
        pl.Series("proj_raw", curve["mean"][idx]),
        pl.lit(curve["sd"]).alias("sd_raw"),
        pl.lit(16.0).alias("games_raw"),
    ).with_columns(
        pl.col("proj_raw").round(1).alias("proj_points"),
        pl.col("sd_raw").round(1).alias("sd"),
        pl.lit(16.0).alias("games"),
    )


# --- the SUPERFLEX extension ------------------------------------------------
# `src/` has no SUPERFLEX concept: `allocate_flex` reads only the FLEX slot, and
# a superflex league that failed to count that slot would misprice quarterbacks
# badly -- which is the single thing that format changes. v2 therefore *extends*
# the reference here rather than porting it.
#
# The extension is validated rather than asserted: `_check_extension_is_noop`
# below proves that for a league without a SUPERFLEX slot it reproduces `src/`
# exactly, so existing behaviour is untouched.


def _superflex_eligible(league: dict) -> list[str]:
    return list(dict.fromkeys([*league["flex_eligible"], "QB"]))


def allocate_flex_ext(proj: pl.DataFrame, league: dict) -> dict[str, int]:
    teams = league["teams"]
    starters = league["starters"]
    counts = {pos: starters.get(pos, 0) * teams for pos in proj["pos"].unique().to_list()}

    def fill(slot: str, eligible: list[str]) -> None:
        slots = starters.get(slot, 0) * teams
        if slots <= 0:
            return
        pool = (
            proj.filter(pl.col("pos").is_in(eligible))
            .with_columns(
                pl.col("proj_points").rank("ordinal", descending=True).over("pos").alias("r")
            )
            .filter(pl.col("r") > pl.col("pos").replace_strict(counts, default=0))
            .sort("proj_points", descending=True)
            .head(slots)
        )
        for pos, n in pool.group_by("pos").len().iter_rows():
            counts[pos] = counts.get(pos, 0) + n

    fill("FLEX", list(league["flex_eligible"]))
    fill("SUPERFLEX", _superflex_eligible(league))
    return counts


def replacement_levels_ext(proj: pl.DataFrame, league: dict) -> dict[str, float]:
    counts = allocate_flex_ext(proj, league)
    levels: dict[str, float] = {}
    for pos in proj["pos"].unique().to_list():
        ranked = proj.filter(pl.col("pos") == pos).sort("proj_points", descending=True)
        if ranked.height == 0:
            continue
        idx = counts.get(pos, 0)
        levels[pos] = float(
            ranked["proj_points"][-1] if idx >= ranked.height else ranked["proj_points"][idx]
        )
    return levels


def add_vor_ext(proj: pl.DataFrame, league: dict) -> pl.DataFrame:
    levels = replacement_levels_ext(proj, league)
    return (
        proj.with_columns(
            pl.col("pos").replace_strict(levels, default=0.0).alias("replacement")
        )
        .with_columns((pl.col("proj_points") - pl.col("replacement")).alias("vor"))
        .with_columns(pl.col("vor").rank("ordinal", descending=True).alias("vor_rank"))
        .sort("vor", descending=True)
    )


def bench_model_ext(proj: pl.DataFrame, league: dict) -> dict[str, BenchValue]:
    counts = allocate_flex_ext(proj, league)
    levels = replacement_levels_ext(proj, league)
    return {
        pos: BenchValue(BENCH_WEIGHT * counts.get(pos, 0) / league["teams"], level)
        for pos, level in levels.items()
    }


def optimal_lineup_ext(
    roster: list[dict], league: dict, bench: dict[str, BenchValue] | None = None
) -> float:
    if not league["starters"].get("SUPERFLEX"):
        return optimal_lineup_points(roster, league, bench)

    starters = league["starters"]
    by_pos: dict[str, list[float]] = {}
    for p in roster:
        by_pos.setdefault(p["pos"], []).append(p["proj_points"])
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    total = 0.0
    used: dict[str, int] = {pos: 0 for pos in by_pos}
    for pos, slots in starters.items():
        if pos in ("FLEX", "SUPERFLEX"):
            continue
        take = by_pos.get(pos, [])[:slots]
        total += sum(take)
        used[pos] = len(take)

    for slot, eligible in (
        ("FLEX", list(league["flex_eligible"])),
        ("SUPERFLEX", _superflex_eligible(league)),
    ):
        n = starters.get(slot, 0)
        if not n:
            continue
        candidates = [
            (pts, pos) for pos in eligible for pts in by_pos.get(pos, [])[used.get(pos, 0):]
        ]
        candidates.sort(reverse=True)
        for pts, pos in candidates[:n]:
            total += pts
            used[pos] = used.get(pos, 0) + 1

    for pos, lst in by_pos.items():
        leftovers = lst[used.get(pos, 0):]
        if not leftovers:
            continue
        if bench is None or pos not in bench:
            total += BENCH_WEIGHT * sum(leftovers)
            continue
        vacancies, baseline = bench[pos]
        for weight, pts in zip(promotion_weights(vacancies, len(leftovers)), leftovers):
            total += weight * max(0.0, pts - baseline)
    return total


def _check_extension_is_noop(proj: pl.DataFrame, league: dict, name: str) -> None:
    """For a league with no SUPERFLEX slot, the extension must equal `src/`."""
    if league["starters"].get("SUPERFLEX"):
        return
    ref = replacement_levels(proj, league)
    ext = replacement_levels_ext(proj, league)
    if ref != ext:
        raise SystemExit(
            f"{name}: the SUPERFLEX extension changed replacement levels for a "
            f"league that has no SUPERFLEX slot.\n  src/: {ref}\n  ext : {ext}"
        )
    ref_vor = add_vor(proj, league).sort("player_id")["vor"].to_numpy()
    ext_vor = add_vor_ext(proj, league).sort("player_id")["vor"].to_numpy()
    if not np.allclose(ref_vor, ext_vor, atol=0, rtol=0):
        raise SystemExit(f"{name}: the SUPERFLEX extension changed VOR values")
    if bench_model(proj, league) != bench_model_ext(proj, league):
        raise SystemExit(f"{name}: the SUPERFLEX extension changed the bench model")


# --- fixture assembly -------------------------------------------------------


def build_fixture(cfg: dict) -> dict:
    fmt = cfg["format"]
    training = _table(f"training_{fmt}")
    board = _table(f"board_{fmt}")
    kicker = _table("kicker_components")
    dst = _table("dst_components")

    ts = cfg["ts"]
    curves = fit_curves(training, ts["scoring"])
    skill = project_skill(curves, board)
    kd = project_kdst(board, kicker, dst, ts["kickerScoring"], ts["dst"])

    projected = pl.concat([skill, kd], how="diagonal_relaxed") if kd.height else skill
    _check_extension_is_noop(projected, cfg["py"], cfg["name"])
    valued = add_tiers(add_vor_ext(projected, cfg["py"]))

    tier_counts = {
        row["pos"]: row["n"]
        for row in valued.group_by("pos").agg(pl.col("tier").n_unique().alias("n")).to_dicts()
    }

    # Sample rosters exercising the lineup optimiser: empty, a partial build,
    # one with a surplus quarterback (which must not be valued as a starter),
    # and one deep enough to push players onto the bench.
    top = valued.sort("vor", descending=True)
    def roster_of(specs: list[tuple[str, float]]) -> list[dict]:
        return [{"pos": p, "proj_points": pts} for p, pts in specs]

    rosters = [
        [],
        roster_of([("RB", 220.0), ("WR", 210.0)]),
        roster_of([("QB", 310.0), ("QB", 300.0), ("RB", 220.0)]),
        roster_of([("RB", 220.0), ("RB", 180.0), ("WR", 210.0), ("WR", 190.0),
                   ("WR", 170.0), ("TE", 150.0), ("QB", 300.0), ("K", 120.0),
                   ("DST", 110.0), ("RB", 90.0)]),
    ]
    # Both bench branches: the flat fallback a caller gets with no board, and
    # the per-position model the board actually ranks by.
    bench = bench_model_ext(valued, cfg["py"])
    lineups = [
        {
            "roster": r,
            "points": optimal_lineup_ext(r, cfg["py"]),
            "bench_points": optimal_lineup_ext(r, cfg["py"], bench),
        }
        for r in rosters
    ]

    # Survival: stochastic, so the test asserts distributionally. Uses the same
    # calibrated means the client will.
    sim_board = valued.select("player_id", "adp", "stdev", "adp_mu", "pos", "vor")
    pick = 15
    surv = survival_probability(
        sim_board, pick, n_sims=SURVIVAL_SIMS, rng=np.random.default_rng(11)
    )

    return {
        "name": cfg["name"],
        "format": fmt,
        "league": ts,
        "curves": curves,
        "projections": valued.select(
            "player_id", "pos", "pos_rank", "proj_points", "sd", "games"
        ).to_dicts(),
        # Unrounded, because rounding to one decimal is cosmetic but sits on
        # exact ties often enough that a 1e-14 difference in float summation
        # order flips a value by 0.1. The raw agreement is the real invariant.
        "projections_raw": valued.select(
            "player_id", "proj_raw", "sd_raw", "games_raw"
        ).to_dicts(),
        "replacement": {
            k: float(v) for k, v in replacement_levels_ext(valued, cfg["py"]).items()
        },
        "bench": {
            pos: {"vacancies": value.vacancies, "baseline": value.baseline}
            for pos, value in bench.items()
        },
        "vor": valued.select("player_id", "vor", "vor_rank").to_dicts(),
        "tiers": valued.select(
            "player_id", "pos", "tier", "tier_size", "tier_points"
        ).to_dicts(),
        "tier_counts": tier_counts,
        "lineups": lineups,
        "survival": {
            "pick": pick,
            "n_sims": SURVIVAL_SIMS,
            "values": surv.select("player_id", "p_available").to_dicts(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=GOLDEN)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Writing golden fixtures to {args.out}")
    for cfg in _configs():
        fixture = build_fixture(cfg)
        path = args.out / f"{cfg['name']}.json"
        path.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
        print(f"  {path.name:26s} {len(fixture['projections']):4d} players  "
              f"{path.stat().st_size / 1e6:5.2f} MB  "
              f"tiers={fixture['tier_counts']}")


if __name__ == "__main__":
    main()
