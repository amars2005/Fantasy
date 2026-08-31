"""Emit the static data bundle the v2 web app consumes.

The v2 app runs its projection maths in TypeScript, in the browser, so it cannot
re-score ten seasons of nflverse weekly stats the way `src/project/actuals.py`
does. Instead this script exports *stat components* -- the raw per-player-season
sums of every stat the scoring rules reference -- and lets the client apply an
arbitrary scoring vector as a dot product.

That is the whole trick that makes per-league custom scoring possible without
shipping polars, scikit-learn and 98 MB of parquet into a serverless function.

Identity resolution stays here too. Joining FFC names to nflverse ids is the step
that silently breaks a board (see `src/ingest/ids.py`), it is already solved and
tested in Python, and the browser has no business redoing it. The training frame
is therefore exported *pre-joined*: each row is a drafted player-season with its
positional ADP rank alongside its stat components.

Nothing in `src/` is modified; this script is a read-only consumer of it.

    python scripts/export_v2_bundle.py                 # everything
    python scripts/export_v2_bundle.py --skip-adp      # offline, no FFC calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nflreadpy as nfl
import polars as pl

from src.config import (
    FUMBLE_LOST_COLUMNS,
    KICKER_SCORING,
    ROOT,
    SCORING,
    SEASON,
    SKILL_POSITIONS,
)
from src.draft.sim_draft import MIN_STDEV, calibrate
from src.dst import DST_EVENT_SCORING, weekly_dst_points
from src.ingest import nflverse as nv
from src.ingest.adp import load_adp
from src.ingest.cache import cache_path, cached
from src.ingest.ids import TEAM_NICKNAMES, normalise_team, resolve

OUT_DIR = ROOT / "data" / "v2_export"

# Seasons the historical rank -> points curve is fitted over. Mirrors
# consensus.FIT_SEASONS; kept local so this script never mutates that module.
FIT_SEASONS = list(range(2016, 2026))

# Fantasy Football Calculator's scoring formats, minus `dynasty`: that endpoint
# returns 11 players, which cannot produce a board. Every league we support is
# anchored to the nearest of these -- see the design spec.
FFC_SCORING = ("standard", "ppr", "half-ppr", "2qb")

# FFC ignores the `teams` query parameter. Verified 2026-08-30: fetching
# teams=12 and teams=14 in the same minute returns byte-identical ADP for all
# 271 players, on every scoring format. (The README already records that it
# likewise ignores every date parameter.) So ADP is exported once per scoring
# format, and team count -- which still drives replacement level, snake order
# and VONA -- is applied per league in the client instead.
FFC_TEAMS_PARAM = 12

# Politeness delay between uncached FFC requests.
FFC_DELAY_S = 1.0

# The stat components a scoring vector can reference. Everything in SCORING
# except `fumbles_lost`, which nflverse splits across three columns.
COMPONENT_STATS = tuple(k for k in SCORING if k != "fumbles_lost")
ALL_COMPONENTS = list(COMPONENT_STATS) + ["fumbles_lost"]


class ExportError(RuntimeError):
    """Raised when the upstream data no longer supports the scoring rules."""


def _guard(df: pl.DataFrame, keys, label: str) -> None:
    """Fail loudly when a scoring key no longer resolves to a real column.

    `src/scoring.py` deliberately treats an absent column as contributing zero,
    which is right for position-filtered frames but means an upstream rename
    would silently zero out a whole stat -- and the projections would still look
    entirely plausible. This is the guard that catches that, and it lives here
    rather than in `src/scoring.py` precisely because that tolerance is
    load-bearing for the existing pipeline.
    """
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ExportError(
            f"{label}: nflverse no longer provides {missing}. "
            "The scoring rules reference them, so the export would silently "
            "produce zeros. Fix the mapping before shipping this bundle."
        )


def _columnar(df: pl.DataFrame, name: str, **meta) -> dict:
    """Compact columnar JSON: a header plus a list of row arrays."""
    return {
        "name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columns": df.columns,
        "rows": [list(r) for r in df.iter_rows()],
        **meta,
    }


def _write(payload: dict, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    # Write-then-rename: a crash mid-write cannot leave a truncated bundle that
    # the app would happily parse.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)
    print(f"  {path.name:30s} {len(payload.get('rows', [])):6d} rows  "
          f"{path.stat().st_size / 1e6:6.2f} MB")
    return path


def _round_floats(df: pl.DataFrame, places: int = 2) -> pl.DataFrame:
    cols = [c for c, t in zip(df.columns, df.dtypes) if t.is_float()]
    return df.with_columns([pl.col(c).round(places) for c in cols]) if cols else df


# --- stat components --------------------------------------------------------


def build_stat_components(seasons: list[int]) -> pl.DataFrame:
    """Per player-season sums of every raw stat the scoring rules reference."""
    weekly = nv.player_stats(seasons=seasons, level="week")
    _guard(weekly, COMPONENT_STATS, "stat components")
    _guard(weekly, FUMBLE_LOST_COLUMNS, "fumbles lost")

    if "season_type" in weekly.columns:
        weekly = weekly.filter(pl.col("season_type") == "REG")

    pos_col = "position" if "position" in weekly.columns else "position_group"
    fumbles = pl.sum_horizontal(
        [pl.col(c).fill_null(0) for c in FUMBLE_LOST_COLUMNS]
    ).alias("fumbles_lost")

    return (
        weekly.with_columns(fumbles)
        .group_by(["player_id", "season"])
        .agg(
            pl.col(pos_col).first().alias("stat_pos"),
            pl.len().alias("games"),
            *[pl.col(c).fill_null(0).sum().alias(c) for c in COMPONENT_STATS],
            pl.col("fumbles_lost").sum().alias("fumbles_lost"),
        )
        .filter(pl.col("stat_pos").is_in(SKILL_POSITIONS))
    )


# --- training frames --------------------------------------------------------


def export_training(out: Path, components: pl.DataFrame, scoring: str,
                    seasons: list[int]) -> bool:
    """Historical drafted player-seasons: positional ADP rank + stat components.

    Pre-joined against nflverse ids here so the client never needs the
    crosswalk. A drafted player with no stat line is kept with zeroed
    components: that is a real draft outcome (holdout, injury, cut) and
    dropping it would bias the curve upward.
    """
    frames = []
    for year in seasons:
        df = _fetch_adp(year, scoring)
        if df is not None:
            frames.append(df)
    if not frames:
        print(f"    no history for {scoring}")
        return False

    history = pl.concat(frames, how="vertical_relaxed")
    resolved = resolve(history).filter(
        pl.col("pos").is_in(SKILL_POSITIONS) & pl.col("gsis_id").is_not_null()
    )

    joined = (
        resolved.join(
            components.rename({"player_id": "gsis_id"}),
            on=["gsis_id", "season"],
            how="left",
        )
        .with_columns(
            pl.col("games").fill_null(0),
            *[pl.col(c).fill_null(0.0) for c in ALL_COMPONENTS],
        )
        .with_columns(
            pl.col("adp").rank("ordinal").over(["season", "pos"]).alias("pos_rank")
        )
        .select(["season", "pos", "pos_rank", "games", *ALL_COMPONENTS])
        .sort(["season", "pos", "pos_rank"])
    )

    payload = _columnar(
        _round_floats(joined), f"training_{scoring}",
        scoring=scoring, seasons=seasons, components=ALL_COMPONENTS,
    )
    _write(payload, out, f"training_{scoring}")
    return True


# --- kickers and defences ---------------------------------------------------


def export_kicker_components(out: Path, seasons: list[int]) -> None:
    """Per kicker-season banded field-goal and PAT counts.

    Bands rather than points: this league pays a sixth point at 60+ yards and
    another league will not.
    """
    weekly = nv.player_stats(seasons=seasons, level="week")
    _guard(weekly, KICKER_SCORING.keys(), "kicker components")

    if "season_type" in weekly.columns:
        weekly = weekly.filter(pl.col("season_type") == "REG")

    frame = (
        weekly.filter(pl.col("position") == "K")
        .group_by(["player_id", "season"])
        .agg(
            pl.col("player_display_name").first().alias("name"),
            pl.col("team").last().alias("team"),
            pl.len().alias("games"),
            *[pl.col(c).fill_null(0).sum().alias(c) for c in KICKER_SCORING],
        )
        .sort(["season", "player_id"])
    )
    _write(_columnar(frame, "kicker_components", seasons=seasons,
                     stats=list(KICKER_SCORING)),
           out, "kicker_components")


def export_dst_components(out: Path, seasons: list[int]) -> None:
    """Per team-game defensive events, points allowed and yards allowed.

    Per *game* rather than per season: the points- and yards-allowed bands are
    non-linear, so a season total cannot be re-banded under different rules.
    """
    team = cached(
        f"team_stats_week_{min(seasons)}_{max(seasons)}",
        lambda: nfl.load_team_stats(seasons=seasons, summary_level="week"),
    )
    _guard(team, DST_EVENT_SCORING.keys(), "dst events")

    events = team.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "team",
        *[pl.col(c).fill_null(0).alias(c) for c in DST_EVENT_SCORING],
    )
    # `weekly_dst_points` already derives points/yards allowed from the
    # opponent's own row. Reuse it read-only rather than duplicating that join.
    allowed = weekly_dst_points(seasons).select(
        "season", "week", "team", "points_allowed", "yards_allowed"
    )
    frame = events.join(
        allowed, on=["season", "week", "team"], how="inner"
    ).sort(["season", "week", "team"])

    _write(_columnar(_round_floats(frame, 1), "dst_components", seasons=seasons,
                     events=list(DST_EVENT_SCORING)),
           out, "dst_components")


# --- current-season boards --------------------------------------------------


def _fetch_adp(year: int, scoring: str) -> pl.DataFrame | None:
    """One FFC format-year, with a polite delay on a genuine network call."""
    key = f"adp_{scoring}_{FFC_TEAMS_PARAM}_{year}"
    warm = cache_path(key).exists()
    try:
        df = load_adp(year=year, scoring=scoring, teams=FFC_TEAMS_PARAM)
    except Exception as exc:  # FFC has no data for this combination
        print(f"    skip {scoring}/{year}: {exc}")
        return None
    if not warm:
        time.sleep(FFC_DELAY_S)
    return df


def export_board(out: Path, scoring: str, season: int) -> dict | None:
    """Current-season draftable players: skill positions plus K and DST."""
    current = _fetch_adp(season, scoring)
    if current is None or current.height == 0:
        return None

    skill = (
        resolve(current)
        .filter(pl.col("pos").is_in(SKILL_POSITIONS) & pl.col("gsis_id").is_not_null())
        .select(
            pl.col("gsis_id").alias("player_id"),
            "name", "pos", "tm", "adp", "stdev", "bye",
        )
    )
    # K and DST must be on the board even though they are barely worth
    # projecting: a pick that cannot be marked desynchronises the pick counter,
    # and every "picks until my next turn" number drifts with it.
    kdst = (
        current.filter(pl.col("position").is_in(["PK", "DEF"]))
        .with_columns(
            pl.when(pl.col("position") == "PK")
            .then(pl.lit("K")).otherwise(pl.lit("DST")).alias("pos"),
            pl.col("team").map_elements(normalise_team, return_dtype=pl.Utf8).alias("tm"),
        )
        .select(
            # Team defences have no player id, so key them on position and team.
            (pl.col("pos") + pl.lit("-") + pl.col("tm")).alias("player_id"),
            "name", "pos", "tm", "adp", "stdev", "bye",
        )
    )

    board = pl.concat([skill, kdst], how="vertical_relaxed").sort("adp")
    board = board.with_columns(
        pl.col("adp").rank("ordinal").over("pos").cast(pl.Int64).alias("pos_rank")
    )

    # Calibrated latent draft means, over the whole board exactly as
    # `board.py` does it. Depends only on adp and stdev -- never on league
    # scoring -- so it is precomputed here rather than run in the browser.
    adp = board["adp"].to_numpy().astype(float)
    stdev = board["stdev"].fill_null(MIN_STDEV).to_numpy().astype(float)
    board = board.with_columns(pl.Series("adp_mu", calibrate(adp, stdev)).round(3))

    as_of = str(current["adp_as_of"][0]) if current.height else None
    payload = _columnar(_round_floats(board, 3), f"board_{scoring}",
                        season=season, scoring=scoring, adp_as_of=as_of)
    _write(payload, out, f"board_{scoring}")
    return {"scoring": scoring, "file": f"board_{scoring}.json",
            "players": board.height, "adp_as_of": as_of}


# --- schedule ---------------------------------------------------------------


def export_schedule(out: Path, season: int) -> None:
    """Opponents for every team-week, plus last season's defensive softness.

    Exported as inputs rather than a finished lift number, so a league that
    calls weeks 14-16 its playoffs gets a lift computed for those weeks.
    """
    sched = nv.schedules([season]).filter(pl.col("season") == season)
    home = sched.select(
        pl.col("week").cast(pl.Int32),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
    )
    away = sched.select(
        pl.col("week").cast(pl.Int32),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
    )
    matchups = pl.concat([home, away]).drop_nulls("opponent").sort(["team", "week"])

    strength = (
        weekly_dst_points([season - 1])
        .group_by("team")
        .agg(pl.col("points_allowed").mean().round(3).alias("pa_per_game"))
        .rename({"team": "opponent"})
        .sort("opponent")
    )

    payload = _columnar(matchups, "schedule_difficulty", season=season)
    payload["strength"] = _columnar(strength, "defensive_strength", season=season - 1)
    _write(payload, out, "schedule_difficulty")


# --- entry point ------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-adp", action="store_true", help="offline; no FFC calls")
    args = ap.parse_args()

    out: Path = args.out
    print(f"Exporting v2 bundle to {out}")
    print(f"Fit seasons {FIT_SEASONS[0]}-{FIT_SEASONS[-1]}, season {args.season}\n")

    print("kicker components")
    export_kicker_components(out, FIT_SEASONS)
    print("dst components")
    export_dst_components(out, FIT_SEASONS)
    print("schedule")
    export_schedule(out, args.season)

    formats: list[dict] = []
    trained: list[str] = []
    if args.skip_adp:
        print("adp: skipped")
    else:
        components = build_stat_components(FIT_SEASONS)
        print(f"training frames ({components.height} player-seasons available)")
        for scoring in FFC_SCORING:
            if export_training(out, components, scoring, FIT_SEASONS):
                trained.append(scoring)
        print("current boards")
        for scoring in trained:
            meta = export_board(out, scoring, args.season)
            if meta:
                formats.append(meta)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": args.season,
        "fit_seasons": FIT_SEASONS,
        "components": ALL_COMPONENTS,
        "kicker_stats": list(KICKER_SCORING),
        "dst_events": list(DST_EVENT_SCORING),
        "scoring_formats": trained,
        "boards": formats,
        # Shipped rather than duplicated in TypeScript: the board's search box
        # matches defences on team nickname, and a hand-copied table would
        # drift.
        "team_nicknames": TEAM_NICKNAMES,
        "notes": {
            "ffc_teams_param": "ignored by the API; ADP is per scoring format only",
            "dynasty": "excluded; the endpoint returns ~11 players",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nWrote {len(formats)} boards and {len(trained)} training frames.")


if __name__ == "__main__":
    main()
