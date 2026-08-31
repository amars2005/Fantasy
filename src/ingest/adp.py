"""Average Draft Position from Fantasy Football Calculator.

Data courtesy of Fantasy Football Calculator (https://fantasyfootballcalculator.com).
Their API is free for personal and commercial use and updates once daily; we
cache to disk and honour that cadence rather than polling.

The `stdev` / `high` / `low` fields are the reason we use this source over a
plain ranking list: they give the *distribution* of where a player goes, which is
what pick-survival probability -- and therefore VONA -- is built on.

One live source, one hard dependency: if FFC is unreachable on draft morning
there is no board at all. `snapshot_adp.py` already archives ADP to
`data/adp_history/`, and that archive is committed, so it is the natural
last-resort fallback -- a board built on last week's ADP is wrong at the margins
and vastly better than no board.
"""

from __future__ import annotations

import json
import urllib.request
import warnings

import polars as pl

from src.config import DATA_RAW, LEAGUE, SEASON
from src.ingest.cache import cached

ARCHIVE = DATA_RAW.parent / "adp_history"

BASE_URL = "https://fantasyfootballcalculator.com/api/v1/adp"
USER_AGENT = "fantasy-draft-tool/0.1 (personal use)"

SCHEMA = {
    "player_id": pl.Int64,
    "name": pl.Utf8,
    "position": pl.Utf8,
    "team": pl.Utf8,
    "adp": pl.Float64,
    "times_drafted": pl.Int64,
    "high": pl.Int64,
    "low": pl.Int64,
    "stdev": pl.Float64,
    "bye": pl.Int64,
}


def _fetch(scoring: str, teams: int, year: int) -> pl.DataFrame:
    url = f"{BASE_URL}/{scoring}?teams={teams}&year={year}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)

    if payload.get("status") != "Success":
        raise RuntimeError(f"FFC ADP request failed for {year}: {payload.get('status')}")

    players = payload.get("players") or []
    if not players:
        raise RuntimeError(f"FFC returned no players for {year}")

    rows = [{k: p.get(k) for k in SCHEMA} for p in players]
    df = pl.DataFrame(rows, schema=SCHEMA)

    meta = payload.get("meta", {})
    return df.with_columns(
        pl.lit(year).alias("season"),
        pl.lit(meta.get("total_drafts")).cast(pl.Int64).alias("total_drafts"),
        pl.lit(meta.get("end_date")).alias("adp_as_of"),
    )


def _from_archive(year: int, scoring: str) -> pl.DataFrame:
    """The most recent committed snapshot for a season, if one exists."""
    files = sorted(ARCHIVE.glob(f"adp_{scoring}_{year}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no archived ADP snapshot for {year}")
    frame = pl.read_parquet(files[-1])
    return frame.select([c for c in frame.columns if c != "snapshot_date"])


def load_adp(
    year: int = SEASON,
    scoring: str = "ppr",
    teams: int | None = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Load ADP for a season. Current season refreshes daily; past seasons never.

    Falls back through disk cache, then the committed snapshot archive. Only a
    season we have never seen at all raises.
    """
    teams = teams or LEAGUE["teams"]
    key = f"adp_{scoring}_{teams}_{year}"
    max_age = 12.0 if year >= SEASON else None
    try:
        return cached(
            key,
            lambda: _fetch(scoring, teams, year),
            max_age_hours=max_age,
            refresh=refresh,
        )
    except Exception as exc:
        try:
            frame = _from_archive(year, scoring)
        except FileNotFoundError:
            raise exc from None
        warnings.warn(
            f"ADP {year}: live fetch and disk cache both unavailable "
            f"({exc.__class__.__name__}); using archived snapshot "
            f"as of {frame['adp_as_of'][0]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return frame


def load_adp_history(years: list[int], scoring: str = "ppr") -> pl.DataFrame:
    """Historical ADP, for backtesting the model against the market."""
    frames = [load_adp(year=y, scoring=scoring) for y in years]
    return pl.concat(frames, how="vertical_relaxed")
