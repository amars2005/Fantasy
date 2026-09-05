"""League configuration and project-wide paths.

Every downstream calculation (replacement level, VOR, VONA, strategy) reads its
league shape from here. Change LEAGUE and the whole pipeline follows.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

SEASON = 2026
HISTORY_SEASONS = list(range(2015, 2026))  # 2015..2025 inclusive

# --- The league -------------------------------------------------------------
# Imperial Immortals 2026 -- ESPN, 14-team H2H points PPR, snake, no keepers.
LEAGUE = {
    "teams": 14,
    "scoring": "ppr",
    "rounds": 15,              # roster size: 9 starters + 6 bench (2 of which are IR)
    "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1},
    "flex_eligible": ("RB", "WR", "TE"),
    "bench": 6,
    "ir_slots": 2,
    # Hard roster maximums enforced by the platform.
    "position_max": {"QB": 4, "RB": 8, "WR": 8, "TE": 3, "K": 3, "DST": 3},
}

# Head-to-head, not total points. You play 14 weekly matchups, six of fourteen
# teams reach the playoffs, and the title is a three-week single-elimination
# bracket. That changes the objective: the goal is to make the top six and then
# survive three games, not to accumulate the most points over the season.
SCHEDULE = {
    "regular_season_weeks": 14,     # fantasy weeks 1-14 decide seeding
    "playoff_weeks": (15, 16, 17),  # one week per round
    "playoff_teams": 6,
    "matchups_per_week": 1,
}

# Positions we actually project. K and DST are near-random year to year and are
# drafted in the last two rounds; they get rank-based placeholders, not models.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
DRAFTABLE_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

REGULAR_SEASON_WEEKS = 17  # NFL games each player plays (18-week season, one bye)


# --- Scoring rules ----------------------------------------------------------
# Full PPR. Values are points per unit of the named stat.
SCORING = {
    "passing_yards": 0.04,          # 1 pt / 25 yds
    "passing_tds": 4.0,
    "passing_interceptions": -2.0,
    "passing_2pt_conversions": 2.0,
    "rushing_yards": 0.1,           # 1 pt / 10 yds
    "rushing_tds": 6.0,
    "rushing_2pt_conversions": 2.0,
    "receptions": 1.0,              # full PPR
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "receiving_2pt_conversions": 2.0,
    "fumbles_lost": -2.0,
    "special_teams_tds": 6.0,          # kickoff return TDs (KRTD)
    # League-specific extras from the Miscellaneous block. Rare -- six skill-player
    # occurrences across 2023-25 -- but each is worth a full touchdown, and
    # nflverse's own PPR column does not include them.
    "pt_return_tds": 6.0,              # punt return TDs (PRTD); disjoint from the above
    "fumble_recovery_tds": 6.0,        # fumble recovered for TD (FTD)
}

# --- components no league of ours scores, but some league does ---------------
#
# Everything above is *this* league's rules, and until now the bundle carried
# exactly the stat columns those rules name. That is why an imported league
# whose settings included a long-touchdown bonus was told the board could not
# score it: the column simply was not there to multiply.
#
# The vocabulary a bundle carries and the rules one league happens to pay are
# different things. These are the extra columns, carried for every player
# whether or not the reference league pays them, so that an imported league can.
#
# Touchdowns banded by the length of the scoring play. Both ESPN and Sleeper
# price these, and both treat them as *counters that sum*: a 55-yard touchdown
# trips "40+ yard TD" and "50+ yard TD" together, so a coarse rule pays every
# band it spans. The bands here are therefore the finest either platform
# offers, and a coarser rule is expanded across them on import -- the same
# treatment `KICKER_SCORING`'s distance bands already get.
#
# Unlike every other component these are not in nflverse's weekly frames; they
# are derived from play-by-play in `scripts/export_v2_bundle.py`, which is also
# where the derivation is reconciled against nflverse's own touchdown totals.
TD_LENGTH_BANDS = ((0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, None))


def td_band_key(kind: str, low: int, high: int | None) -> str:
    """`passing_td_40_49`, `receiving_td_50_` -- matching the kicker bands."""
    return f"{kind}_td_{low}_{'' if high is None else high}"


LONG_TD_COMPONENTS = tuple(
    td_band_key(kind, low, high)
    for kind in ("passing", "rushing", "receiving")
    for low, high in TD_LENGTH_BANDS
)

# The subset nflverse's `fantasy_points_ppr` also computes. Used to cross-check
# the scoring engine against a reference; the league extras above are deliberate
# divergences from it, not bugs.
STANDARD_SCORING = {
    k: v for k, v in SCORING.items()
    if k not in ("pt_return_tds", "fumble_recovery_tds")
}

# Kicker scoring is banded by distance and this league goes further than most:
# it pays a sixth point for 60+ yards, where the common default stops at 50+.
# That rewards leg strength (and dome/altitude kickers) more than usual.
KICKER_SCORING = {
    "fg_made_0_19": 3.0,
    "fg_made_20_29": 3.0,
    "fg_made_30_39": 3.0,     # FG0: everything inside 40 is worth 3
    "fg_made_40_49": 4.0,     # FG40
    "fg_made_50_59": 5.0,     # FG50
    "fg_made_60_": 6.0,       # FG60 -- the unusual tier
    "pat_made": 1.0,          # PAT
    # FGM: -1 for every miss, at any distance.
    "fg_missed_0_19": -1.0,
    "fg_missed_20_29": -1.0,
    "fg_missed_30_39": -1.0,
    "fg_missed_40_49": -1.0,
    "fg_missed_50_59": -1.0,
    "fg_missed_60_": -1.0,
}


# nflverse splits fumbles lost across three columns; we sum them into one stat.
FUMBLE_LOST_COLUMNS = (
    "sack_fumbles_lost",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
)
