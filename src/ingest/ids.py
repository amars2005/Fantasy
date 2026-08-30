"""Player identity resolution.

Joining Fantasy Football Calculator's ADP to nflverse stats is the single most
likely source of silent, hard-to-spot bugs in this project: a player who fails to
match doesn't error, he just quietly vanishes from the draft board.

Three real hazards this module handles, all observed in the live data:
  * Suffix collisions -- "Marvin Harrison Jr." normalises to the same key as his
    father, who is also in the crosswalk.
  * Position collisions -- there are two "D.J. Moore"s, a WR and a CB.
  * Null-id decoys -- retired/duplicate rows carry no gsis_id and would win a
    naive join.

So we match on (name, position), prefer rows that actually have a gsis_id,
and break remaining ties by team. Anything left unresolved is reported, never
silently dropped.
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

from src.ingest import nflverse as nv

# Everything is normalised to nflverse convention. Note nflverse uses LA (not
# LAR) for the Rams, which neither of the other two sources does.
TEAM_MAP = {
    "LAR": "LA", "RAM": "LA", "STL": "LA",
    "JAC": "JAX",
    "KCC": "KC", "GBP": "GB", "NEP": "NE", "NOS": "NO",
    "SFO": "SF", "TBB": "TB", "LVR": "LV", "OAK": "LV",
    "SDC": "LAC",
    "FA*": "FA", "": "FA",
}

# Search aliases for defences, not identity: the draft room says "Ravens D/ST"
# while the ADP feed names that row "Baltimore Defense". Without these a defence
# is a pick you cannot type, and an unmarkable pick is exactly what
# desynchronises the counter. Spelling variants are space-separated because these
# are concatenated into one searchable haystack per player.
TEAM_NICKNAMES = {
    "ARI": "cardinals cards", "ATL": "falcons", "BAL": "ravens", "BUF": "bills",
    "CAR": "panthers", "CHI": "bears", "CIN": "bengals", "CLE": "browns",
    "DAL": "cowboys", "DEN": "broncos", "DET": "lions", "GB": "packers",
    "HOU": "texans", "IND": "colts", "JAX": "jaguars jags", "KC": "chiefs",
    "LA": "rams", "LAC": "chargers", "LV": "raiders", "MIA": "dolphins",
    "MIN": "vikings vikes", "NE": "patriots pats", "NO": "saints",
    "NYG": "giants", "NYJ": "jets", "PHI": "eagles", "PIT": "steelers",
    "SEA": "seahawks hawks", "SF": "49ers niners", "TB": "buccaneers bucs",
    "TEN": "titans", "WAS": "commanders",
}

# FFC labels defence and kicker differently from nflverse.
POSITION_MAP = {"DEF": "DST", "PK": "K", "DST": "DST", "K": "K"}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Nicknames the sources disagree on. Keyed by normalised FFC name -> normalised
# crosswalk name. Kept explicit rather than fuzzy-matched: a wrong fuzzy match is
# worse than a missing player, because it is invisible.
NAME_ALIASES = {
    "chig okonkwo": "chigoziem okonkwo",
    "kenny gainwell": "kenneth gainwell",
    "gabe davis": "gabriel davis",
    "josh palmer": "joshua palmer",
    "cam ward": "cameron ward",
    "mike evans": "michael evans",
    "will shipley": "william shipley",
    "tank bigsby": "cartavious bigsby",
    "hollywood brown": "marquise brown",
    "scotty miller": "scott miller",
    "nick westbrook-ikhine": "nicholas westbrookikhine",
}


def normalise_team(team: str | None) -> str:
    if not team:
        return "FA"
    t = team.strip().upper()
    return TEAM_MAP.get(t, t)


def normalise_position(pos: str | None) -> str:
    if not pos:
        return "UNK"
    p = pos.strip().upper()
    return POSITION_MAP.get(p, p)


def normalise_name(name: str | None) -> str:
    """Lowercase, strip accents, punctuation and generational suffixes.

    Mirrors nflverse's `merge_name` convention so the two sides of the join agree.
    """
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[.'`\-,]", "", n)
    parts = [p for p in n.split() if p not in SUFFIXES]
    cleaned = " ".join(parts).strip()
    return NAME_ALIASES.get(cleaned, cleaned)


def _prepare_crosswalk() -> pl.DataFrame:
    """The crosswalk, normalised and ranked so the best row per key wins."""
    xw = nv.ff_playerids()
    return (
        xw.with_columns(
            pl.col("name")
            .map_elements(normalise_name, return_dtype=pl.Utf8)
            .alias("key"),
            pl.col("position")
            .map_elements(normalise_position, return_dtype=pl.Utf8)
            .alias("pos"),
            pl.col("team")
            .map_elements(normalise_team, return_dtype=pl.Utf8)
            .alias("tm"),
            pl.col("gsis_id").is_not_null().alias("has_gsis"),
        )
        .filter(pl.col("key") != "")
        .select(
            "key", "pos", "tm", "has_gsis", "gsis_id", "sleeper_id",
            "fantasypros_id", "birthdate", "draft_year", "draft_round",
            "draft_pick", "draft_ovr", "college",
        )
    )


def resolve(
    df: pl.DataFrame,
    name_col: str = "name",
    pos_col: str = "position",
    team_col: str = "team",
) -> pl.DataFrame:
    """Attach gsis_id (and draft/birth metadata) to a frame of named players.

    Adds `key`, `pos`, `tm` normalised columns plus crosswalk fields. Rows that
    fail to resolve keep a null gsis_id -- the caller decides how loud to be.
    """
    left = df.with_columns(
        pl.col(name_col).map_elements(normalise_name, return_dtype=pl.Utf8).alias("key"),
        pl.col(pos_col).map_elements(normalise_position, return_dtype=pl.Utf8).alias("pos"),
        pl.col(team_col).map_elements(normalise_team, return_dtype=pl.Utf8).alias("tm"),
    )
    xw = _prepare_crosswalk()

    # Pass 1: name + position + team. Most specific, resolves the Harrison Jr./Sr.
    # and D.J. Moore cases outright.
    exact = (
        xw.filter(pl.col("has_gsis"))
        .unique(subset=["key", "pos", "tm"], keep="first")
        .drop("has_gsis")
    )
    out = left.join(exact, on=["key", "pos", "tm"], how="left")

    # Pass 2: name + position, for players whose team changed since the crosswalk
    # was last built. Restricted to rows with a gsis_id so decoys can't win.
    by_pos = (
        xw.filter(pl.col("has_gsis"))
        .unique(subset=["key", "pos"], keep="first")
        .drop("has_gsis", "tm")
    )
    fill_cols = [c for c in by_pos.columns if c not in ("key", "pos")]
    out = (
        out.join(by_pos, on=["key", "pos"], how="left", suffix="_p2")
        .with_columns(
            [pl.col(c).fill_null(pl.col(f"{c}_p2")).alias(c) for c in fill_cols]
        )
        .drop([f"{c}_p2" for c in fill_cols])
    )

    # Pass 3: surname + position + team, for nickname variants the alias table
    # doesn't cover ("Chig" vs "Chigoziem"). Position and team together make a
    # surname collision unlikely enough to accept.
    surname = pl.col("key").str.split(" ").list.last()
    by_surname = (
        xw.filter(pl.col("has_gsis"))
        .with_columns(surname.alias("surname"))
        .unique(subset=["surname", "pos", "tm"], keep="first")
        .drop("has_gsis", "key")
    )
    out = (
        out.with_columns(surname.alias("surname"))
        .join(by_surname, on=["surname", "pos", "tm"], how="left", suffix="_p3")
        .with_columns(
            [pl.col(c).fill_null(pl.col(f"{c}_p3")).alias(c) for c in fill_cols]
        )
        .drop([f"{c}_p3" for c in fill_cols] + ["surname"])
    )
    return out


def unresolved(df: pl.DataFrame, top_n: int | None = None, adp_col: str = "adp") -> pl.DataFrame:
    """Rows that failed to resolve, optionally limited to the top N by ADP."""
    miss = df.filter(pl.col("gsis_id").is_null())
    if top_n is not None and adp_col in df.columns:
        miss = miss.filter(pl.col(adp_col) <= top_n)
    return miss
