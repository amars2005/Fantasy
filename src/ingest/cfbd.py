"""College football data, for the one group the NFL data cannot describe: rookies.

A first-year player has no NFL box score, so the veteran model has nothing to work
with beyond draft capital and a forty time. College production is the missing
evidence -- and college *usage share* especially, since it is the direct analogue
of the target-share features that carry the most signal on the NFL side.

Joining the two worlds is the fiddly part. nflverse stores a slug
("cameron-ward-1") while CFBD uses a numeric athlete id, so they do not join
directly. CFBD's own draft endpoint carries both the college athlete id and the
NFL draft slot, and (year, round, pick) is a unique key into nflverse's draft
table -- so the draft itself is the bridge.

Requires CFBD_API_KEY in the environment or in a .env file at the repo root.
Free key: https://collegefootballdata.com/key
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

from src.config import ROOT
from src.ingest.cache import cached

BASE_URL = "https://api.collegefootballdata.com"
RATE_LIMIT_SLEEP = 0.6  # be a polite client of a free API


class MissingKeyError(RuntimeError):
    pass


def _load_env() -> None:
    """Read .env into the environment if the key is not already set."""
    if os.environ.get("CFBD_API_KEY"):
        return
    env_path = Path(ROOT) / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _api_key() -> str:
    _load_env()
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise MissingKeyError(
            "CFBD_API_KEY not set. Put it in .env at the repo root "
            "(which is gitignored) or export it."
        )
    return key


def _get(endpoint: str, **params) -> list:
    """Authenticated GET. The key is only ever sent to api.collegefootballdata.com."""
    url = f"{BASE_URL}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise MissingKeyError("CFBD rejected the API key (401/403).") from exc
        raise
    time.sleep(RATE_LIMIT_SLEEP)
    return payload


# --- the bridge -------------------------------------------------------------
def draft_bridge(years: list[int]) -> pl.DataFrame:
    """Map NFL draft slot -> college athlete id, with pre-draft scouting grades.

    `preDraftGrade` is independently useful: it is the scouting consensus, which
    is not the same thing as where a team actually spent its pick. A high grade
    that fell in the draft is exactly the shape of an undervalued rookie.
    """

    def load() -> pl.DataFrame:
        rows = []
        for year in years:
            for pick in _get("draft/picks", year=year):
                rows.append({
                    "draft_year": year,
                    "round": pick.get("round"),
                    "pick": pick.get("pick"),
                    "overall": pick.get("overall"),
                    "college_athlete_id": pick.get("collegeAthleteId"),
                    "cfb_name": pick.get("name"),
                    "college_team": pick.get("collegeTeam"),
                    "college_conference": pick.get("collegeConference"),
                    "pre_draft_ranking": pick.get("preDraftRanking"),
                    "pre_draft_pos_ranking": pick.get("preDraftPositionRanking"),
                    "pre_draft_grade": pick.get("preDraftGrade"),
                })
        return pl.DataFrame(rows, schema_overrides={
            "college_athlete_id": pl.Int64, "round": pl.Int32, "pick": pl.Int32,
            "overall": pl.Int32, "pre_draft_ranking": pl.Float64,
            "pre_draft_pos_ranking": pl.Float64, "pre_draft_grade": pl.Float64,
        })

    return cached(f"cfbd_draft_{min(years)}_{max(years)}", load, max_age_hours=24 * 30)


# --- college production -----------------------------------------------------
def player_usage(years: list[int]) -> pl.DataFrame:
    """Share of the offence a player accounted for, per college season."""

    def load() -> pl.DataFrame:
        rows = []
        for year in years:
            for rec in _get("player/usage", year=year):
                usage = rec.get("usage") or {}
                rows.append({
                    "college_athlete_id": int(rec["id"]) if rec.get("id") else None,
                    "college_season": year,
                    "cfb_pos": rec.get("position"),
                    "usage_overall": usage.get("overall"),
                    "usage_pass": usage.get("pass"),
                    "usage_rush": usage.get("rush"),
                    "usage_third_down": usage.get("thirdDown"),
                })
        return pl.DataFrame(rows, schema_overrides={"college_athlete_id": pl.Int64})

    return cached(f"cfbd_usage_{min(years)}_{max(years)}", load, max_age_hours=24 * 30)


STAT_CATEGORIES = ("receiving", "rushing")
KEEP_STATS = {"REC", "YDS", "TD", "CAR", "YPR", "YPC"}


def season_stats(years: list[int]) -> pl.DataFrame:
    """Receiving and rushing production per college season, pivoted wide."""

    def load() -> pl.DataFrame:
        rows = []
        for year in years:
            for category in STAT_CATEGORIES:
                for rec in _get("stats/player/season", year=year, category=category):
                    if rec.get("statType") not in KEEP_STATS:
                        continue
                    try:
                        value = float(rec.get("stat"))
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "college_athlete_id": int(rec["playerId"]) if rec.get("playerId") else None,
                        "college_season": year,
                        "key": f"cfb_{category[:3]}_{rec['statType'].lower()}",
                        "value": value,
                    })
        frame = pl.DataFrame(rows, schema_overrides={"college_athlete_id": pl.Int64})
        return frame.pivot(
            values="value", index=["college_athlete_id", "college_season"],
            on="key", aggregate_function="first",
        )

    return cached(f"cfbd_stats_{min(years)}_{max(years)}", load, max_age_hours=24 * 30)


def final_college_season(years: list[int]) -> pl.DataFrame:
    """One row per college athlete: their last, and best, college seasons.

    The final season says what he was just before the NFL saw him; the peak says
    what he is capable of. Both matter, and they are often different players.
    """
    usage = player_usage(years)
    stats = season_stats(years)
    combined = usage.join(stats, on=["college_athlete_id", "college_season"], how="full", coalesce=True)
    combined = combined.filter(pl.col("college_athlete_id").is_not_null())

    last = (
        combined.sort("college_season")
        .group_by("college_athlete_id")
        .last()
        .rename(lambda c: c if c in ("college_athlete_id",) else f"{c}_final")
    )
    peak = combined.group_by("college_athlete_id").agg(
        pl.col("usage_overall").max().alias("cfb_usage_peak"),
        pl.col("cfb_rec_yds").max().alias("cfb_rec_yds_peak"),
        pl.col("cfb_rus_yds").max().alias("cfb_rus_yds_peak"),
        pl.col("cfb_rec_td").max().alias("cfb_rec_td_peak"),
        pl.col("cfb_rus_td").max().alias("cfb_rus_td_peak"),
        pl.len().alias("cfb_seasons"),
    )
    return last.join(peak, on="college_athlete_id", how="full", coalesce=True)
