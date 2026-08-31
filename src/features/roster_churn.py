"""Who arrived, who left, and whether the offense around a player got better.

The feature table already knows about *vacated* opportunity -- the targets and
carries belonging to players who are no longer on the roster. That is only half
of an offseason. A team that loses 120 targets and signs a receiver who saw 140
somewhere else has not opened anything up; a team that loses its quarterback and
replaces him with a better one has improved every pass-catcher on the roster
without a single one of them changing anything about himself.

So this module measures the other half:

  * **arrivals** -- prior-season volume walking in the door, at team level and at
    the player's own position, which is competition rather than opportunity
  * **net change** -- departures minus arrivals, the quantity a vacancy argument
    actually rests on
  * **incoming draft capital and money** -- a first-round rookie or an expensive
    free agent at your position is a demotion that has not happened yet
  * **the quarterback room** -- how the best arm on the roster compares to the
    one that actually threw the ball here last season

Every quantity is built from prior-season production and current-season roster
membership. Both are public well before week one, so none of it leaks.

The unit of "arrival" is a change of roster, not a transaction type. A free
agent, a trade, a waiver claim and a drafted rookie all arrive the same way, and
for the purpose of "who is going to get the ball" they are the same event.
"""

from __future__ import annotations

import polars as pl

# A team's starting quarterback last season is the one who actually threw the
# passes. Below this many attempts nobody counts as having held the job.
MIN_STARTER_ATTEMPTS = 100
# Prior-season ppg from a three-game sample is not evidence about a quarterback
# room; require a real sample before a backup's rate can define it.
MIN_QB_GAMES = 6
UNDRAFTED_OVERALL = 300.0

CHURN_FEATURES = [
    # team-level flow
    "arrived_targets", "arrived_carries",
    "net_targets", "net_carries",
    "arrived_target_share", "net_target_share",
    # the player's own position group on his own team
    "pos_in_targets", "pos_in_carries", "pos_in_pass_att",
    "pos_out_targets", "pos_out_carries",
    "pos_net_targets", "pos_net_carries",
    # what the team spent on the competition
    "pos_in_cap_pct", "pos_rookie_overall",
    # the quarterback room
    "qb_room_ppg", "qb_room_att", "qb_prev_ppg", "qb_upgrade", "qb_changed",
    "qb_room_unknown",
]


def _prior_volume(stats: pl.DataFrame, year: int) -> pl.DataFrame:
    """What each player produced in the season before `year`."""
    cols = ["targets", "carries", "pass_attempts", "points", "ppg", "games"]
    have = [c for c in cols if c in stats.columns]
    return (
        stats.filter(pl.col("season") == year - 1)
        .select(["player_id", "team", "pos"] + have)
        .rename({c: f"prior_{c}" for c in have})
        .rename({"team": "prior_stat_team", "pos": "prior_stat_pos"})
        .unique(subset=["player_id"], keep="first")
    )


def _memberships(rosters: pl.DataFrame, year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(roster in `year`, roster in `year - 1`), one row per player."""
    current = (
        rosters.filter(pl.col("season") == year)
        .select("player_id", "team", "pos")
        .unique(subset=["player_id"], keep="first")
    )
    previous = (
        rosters.filter(pl.col("season") == year - 1)
        .select("player_id", pl.col("team").alias("prev_team"))
        .unique(subset=["player_id"], keep="first")
    )
    return current, previous


def _flows(
    stats: pl.DataFrame, rosters: pl.DataFrame, year: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Arrivals and departures for `year`, each carrying prior-season volume.

    A player's team last season is taken from the roster where there is one and
    from his stat line otherwise, so someone who was signed off a practice squad
    mid-season is not counted as arriving twice.
    """
    volume = _prior_volume(stats, year)
    current, previous = _memberships(rosters, year)

    labelled = (
        current.join(previous, on="player_id", how="left")
        .join(volume, on="player_id", how="left")
        .with_columns(
            pl.coalesce(["prev_team", "prior_stat_team"]).alias("prev_team")
        )
        .with_columns(
            [pl.col(f"prior_{c}").fill_null(0.0)
             for c in ("targets", "carries", "pass_attempts", "points", "games")]
        )
    )
    arrivals = labelled.filter(
        pl.col("prev_team").is_null() | (pl.col("prev_team") != pl.col("team"))
    )

    # Departures: on the roster last season, not on it now. Position comes from
    # where they played, since that is the role the team has to replace.
    gone = (
        rosters.filter(pl.col("season") == year - 1)
        .select("player_id", "team", "pos")
        .unique(subset=["player_id"], keep="first")
        .join(
            current.select("player_id", pl.col("team").alias("team_now")),
            on="player_id", how="left",
        )
        .filter(pl.col("team_now").is_null() | (pl.col("team_now") != pl.col("team")))
        .join(volume, on="player_id", how="left")
        .with_columns(
            [pl.col(f"prior_{c}").fill_null(0.0)
             for c in ("targets", "carries", "pass_attempts", "points", "games")]
        )
    )
    return arrivals, gone


def _team_totals(stats: pl.DataFrame, year: int) -> pl.DataFrame:
    """Prior-season team volume, the denominator for every share."""
    return (
        stats.filter(pl.col("season") == year - 1)
        .group_by("team")
        .agg(
            pl.col("targets").sum().alias("team_prior_targets"),
            pl.col("carries").sum().alias("team_prior_carries"),
        )
    )


def _quarterback_room(
    stats: pl.DataFrame, rosters: pl.DataFrame, year: int
) -> pl.DataFrame:
    """How this year's quarterback room compares to last year's actual starter.

    `qb_prev_ppg` is what the offense got last season, measured on the passer who
    actually threw the ball. `qb_room_ppg` is the best prior-season rate on the
    current roster. Their difference is the closest thing available to "did the
    quarterback situation improve", and it is knowable in March.

    A room with nobody who has played a real NFL season scores zero, which is the
    same number a room full of genuinely bad quarterbacks scores. Those are not
    the same situation -- a first-round rookie is an unknown, not a disaster --
    so `qb_room_unknown` flags the difference rather than leaving the model to
    guess which zero it is looking at.
    """
    prior_qbs = stats.filter(
        (pl.col("season") == year - 1) & (pl.col("pos") == "QB")
    )

    prev_starter = (
        prior_qbs.filter(pl.col("pass_attempts") >= MIN_STARTER_ATTEMPTS)
        .sort("pass_attempts", descending=True)
        .group_by("team")
        .first()
        .select(
            "team",
            pl.col("player_id").alias("prev_starter_id"),
            pl.col("ppg").alias("qb_prev_ppg"),
        )
    )

    current, _ = _memberships(rosters, year)
    room = (
        current.filter(pl.col("pos") == "QB")
        .join(
            prior_qbs.select(
                "player_id",
                pl.col("ppg").alias("prior_ppg"),
                pl.col("games").alias("prior_games"),
                pl.col("pass_attempts").alias("prior_att"),
            ),
            on="player_id", how="left",
        )
        .group_by("team")
        .agg(
            pl.col("prior_ppg")
            .filter(pl.col("prior_games") >= MIN_QB_GAMES)
            .max().alias("qb_room_ppg"),
            pl.col("prior_att").max().alias("qb_room_att"),
            pl.col("player_id").alias("_room_ids"),
        )
    )

    return (
        room.join(prev_starter, on="team", how="left")
        .with_columns(
            pl.col("qb_room_ppg").is_null().cast(pl.Int8).alias("qb_room_unknown"),
        )
        .with_columns(
            pl.col("qb_room_ppg").fill_null(0.0),
            pl.col("qb_room_att").fill_null(0.0),
            pl.col("qb_prev_ppg").fill_null(0.0),
        )
        .with_columns(
            (pl.col("qb_room_ppg") - pl.col("qb_prev_ppg")).alias("qb_upgrade"),
            (
                pl.col("prev_starter_id").is_null()
                | ~pl.col("_room_ids").list.contains(pl.col("prev_starter_id"))
            ).cast(pl.Int8).alias("qb_changed"),
            pl.lit(year).cast(pl.Int32).alias("season"),
        )
        .drop("_room_ids", "prev_starter_id")
    )


def _incoming_capital(
    draft_picks: pl.DataFrame | None, current: pl.DataFrame, year: int
) -> pl.DataFrame:
    """Best draft slot the team spent on this position in this year's draft."""
    empty = pl.DataFrame(
        schema={"team": pl.Utf8, "pos": pl.Utf8, "pos_rookie_overall": pl.Float64}
    )
    if draft_picks is None:
        return empty
    rookies = draft_picks.filter(pl.col("draft_season") == year).select(
        "player_id", "draft_overall"
    )
    if rookies.height == 0:
        return empty
    return (
        current.join(rookies, on="player_id", how="inner")
        .group_by(["team", "pos"])
        .agg(pl.col("draft_overall").min().alias("pos_rookie_overall"))
    )


def build_churn(
    stats: pl.DataFrame,
    rosters: pl.DataFrame,
    target_seasons: list[int],
    contract_caps: pl.DataFrame | None = None,
    draft_picks: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per (player_id, season) carrying every roster-churn feature.

    `contract_caps` is (player_id, season, contract_cap_pct); `draft_picks` is
    (player_id, draft_season, draft_overall). Both are optional -- the features
    that need them come back null rather than blocking the rest.
    """
    frames = []
    for year in target_seasons:
        arrivals, gone = _flows(stats, rosters, year)
        current, _ = _memberships(rosters, year)
        if current.height == 0:
            continue

        team_in = arrivals.group_by("team").agg(
            pl.col("prior_targets").sum().alias("arrived_targets"),
            pl.col("prior_carries").sum().alias("arrived_carries"),
        )
        team_out = gone.group_by("team").agg(
            pl.col("prior_targets").sum().alias("departed_targets"),
            pl.col("prior_carries").sum().alias("departed_carries"),
        )

        pos_in = arrivals.group_by(["team", "pos"]).agg(
            pl.col("prior_targets").sum().alias("pos_in_targets"),
            pl.col("prior_carries").sum().alias("pos_in_carries"),
            pl.col("prior_pass_attempts").sum().alias("pos_in_pass_att"),
        )
        pos_out = gone.group_by(["team", "pos"]).agg(
            pl.col("prior_targets").sum().alias("pos_out_targets"),
            pl.col("prior_carries").sum().alias("pos_out_carries"),
        )

        if contract_caps is not None:
            caps = contract_caps.filter(pl.col("season") == year).select(
                "player_id", "contract_cap_pct"
            )
            pos_money = (
                arrivals.join(caps, on="player_id", how="inner")
                .group_by(["team", "pos"])
                .agg(pl.col("contract_cap_pct").max().alias("pos_in_cap_pct"))
            )
        else:
            pos_money = pl.DataFrame(
                schema={"team": pl.Utf8, "pos": pl.Utf8, "pos_in_cap_pct": pl.Float64}
            )

        # A player's own arrival is not competition for himself.
        own = arrivals.select(
            "player_id",
            pl.col("prior_targets").alias("_own_targets"),
            pl.col("prior_carries").alias("_own_carries"),
            pl.col("prior_pass_attempts").alias("_own_att"),
        )

        year_frame = (
            current.join(team_in, on="team", how="left")
            .join(team_out, on="team", how="left")
            .join(pos_in, on=["team", "pos"], how="left")
            .join(pos_out, on=["team", "pos"], how="left")
            .join(pos_money, on=["team", "pos"], how="left")
            .join(_incoming_capital(draft_picks, current, year),
                  on=["team", "pos"], how="left")
            .join(_team_totals(stats, year), on="team", how="left")
            .join(own, on="player_id", how="left")
            .with_columns(pl.lit(year).cast(pl.Int32).alias("season"))
        )
        year_frame = year_frame.join(
            _quarterback_room(stats, rosters, year),
            on=["team", "season"], how="left",
        )
        frames.append(year_frame)

    if not frames:
        return pl.DataFrame(schema={"player_id": pl.Utf8, "season": pl.Int32})

    out = pl.concat(frames, how="diagonal_relaxed")
    zero_fill = [
        "arrived_targets", "arrived_carries", "departed_targets", "departed_carries",
        "pos_in_targets", "pos_in_carries", "pos_in_pass_att",
        "pos_out_targets", "pos_out_carries",
        "_own_targets", "_own_carries", "_own_att",
    ]
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in zero_fill])

    out = (
        out.with_columns(
            # Remove the player's own volume from his position's incoming total.
            (pl.col("pos_in_targets") - pl.col("_own_targets")).alias("pos_in_targets"),
            (pl.col("pos_in_carries") - pl.col("_own_carries")).alias("pos_in_carries"),
            (pl.col("pos_in_pass_att") - pl.col("_own_att")).alias("pos_in_pass_att"),
            # No rookie taken at this position is information, not missingness.
            pl.col("pos_rookie_overall").fill_null(UNDRAFTED_OVERALL),
        )
        .with_columns(
            (pl.col("departed_targets") - pl.col("arrived_targets")).alias("net_targets"),
            (pl.col("departed_carries") - pl.col("arrived_carries")).alias("net_carries"),
            (pl.col("pos_out_targets") - pl.col("pos_in_targets")).alias("pos_net_targets"),
            (pl.col("pos_out_carries") - pl.col("pos_in_carries")).alias("pos_net_carries"),
        )
        .with_columns(
            (pl.col("arrived_targets") / pl.col("team_prior_targets"))
            .alias("arrived_target_share"),
            (pl.col("net_targets") / pl.col("team_prior_targets"))
            .alias("net_target_share"),
        )
    )
    return out.select(
        ["player_id", "season"] + [c for c in CHURN_FEATURES if c in out.columns]
    )
