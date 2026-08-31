"""Assemble the model's training table: one row per (player, target season).

Every feature must be knowable on draft day of the target season. That rule is
the whole game here -- it is trivially easy to build a model that looks superb in
backtest because a feature leaked the season it was predicting. So production
features come strictly from prior seasons, while roster, contract and Vegas
context come from the target season, all of which are public before week 1.

Feature groups:
  * production/opportunity, lagged one and two seasons
  * expected points and points-over-expected (the regression signal)
  * player attributes: age, NFL experience, draft capital
  * contract: years remaining, contract year, cap share, guaranteed money
  * team context: team change, vacated targets and carries, Vegas implied total
  * roster churn: who arrived, what they brought, and whether the quarterback
    room improved (see src/features/roster_churn.py)
"""

from __future__ import annotations

import polars as pl

from src.features.player_season import build as build_player_season
from src.features.roster_churn import build_churn
from src.ingest import nflverse as nv
from src.ingest.cache import cached
from src.ingest.nflverse import CURRENT_MAX_AGE

LAG_STATS = [
    "points", "ppg", "games", "targets", "carries", "receptions",
    "targets_pg", "carries_pg", "touches_pg", "target_share",
    "air_yards_share", "wopr", "rec_yards", "rush_yards",
    "exp_ppg", "points_oe",
]


def _roster_columns(frame: pl.DataFrame, season: int) -> pl.DataFrame:
    return (
        frame.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.lit(season).cast(pl.Int32).alias("season"),
            pl.col("team"),
            pl.col("position").alias("pos"),
            pl.col("birth_date"),
            pl.col("years_exp").cast(pl.Float64),
            pl.col("entry_year").cast(pl.Float64),
        )
        .filter(pl.col("player_id").is_not_null())
        .unique(subset=["player_id", "season"], keep="first")
    )


def _season_roster(season: int, draft_day: bool = True) -> pl.DataFrame:
    """The roster as it stood at the start of `season`.

    This distinction is not pedantry. `load_rosters` returns a season-level
    snapshot taken at the *end* of the year, so a player traded in October is
    listed with the team that acquired him -- 12% of skill players in 2024 sit on
    a different team there than they did in week one. Every feature derived from
    roster membership (team context, vacated opportunity, and everything in
    roster_churn) would then be reading the season it is supposed to predict.

    Weekly rosters give the week-one snapshot instead, which is what a drafter
    could actually see. They stop at the last completed season, so the season
    currently being drafted falls back to the live roster -- which for a season
    that has not started yet is the same thing.

    `draft_day=False` restores the leaking end-of-season snapshot. It exists so
    the size of the leak can be measured rather than asserted; nothing that ships
    should use it.
    """
    import nflreadpy as nfl

    if not draft_day:
        return cached(
            f"roster_endofseason_{season}",
            lambda: _roster_columns(nfl.load_rosters(seasons=[season]), season),
            max_age_hours=CURRENT_MAX_AGE,
        )

    def week_one() -> pl.DataFrame:
        weekly = nfl.load_rosters_weekly(seasons=[season])
        return _roster_columns(
            weekly.filter(pl.col("week") == pl.col("week").min()), season
        )

    try:
        return cached(f"roster_week1_{season}", week_one, max_age_hours=None)
    except Exception:
        # No weekly file yet: the season has not been played.
        return cached(
            f"roster_live_{season}",
            lambda: _roster_columns(nfl.load_rosters(seasons=[season]), season),
            max_age_hours=CURRENT_MAX_AGE,
        )


def _rosters(seasons: list[int], draft_day: bool = True) -> pl.DataFrame:
    """Draft-day rosters with age and experience, one row per player-season."""
    frames = [_season_roster(season, draft_day) for season in seasons]
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(
            (pl.col("season") - pl.col("birth_date").dt.year().cast(pl.Float64))
            .alias("age")
        )
        .drop("birth_date")
    )


def _contracts() -> pl.DataFrame:
    """Every contract, expanded so we can find the one active in a given season."""
    import nflreadpy as nfl

    con = cached("contracts", nfl.load_contracts, max_age_hours=24.0)
    return (
        con.filter(pl.col("gsis_id").is_not_null() & pl.col("year_signed").is_not_null())
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("year_signed").cast(pl.Int32),
            pl.col("years").cast(pl.Int32),
            pl.col("apy").cast(pl.Float64),
            pl.col("guaranteed").cast(pl.Float64),
            pl.col("apy_cap_pct").cast(pl.Float64),
        )
    )


def _draft_capital() -> pl.DataFrame:
    """Where a player was drafted, from the draft-pick record itself.

    Taking this from the contracts table instead only covers players who have an
    OverTheCap record, which drops well over half of all rookies -- exactly the
    group draft capital matters most for.
    """
    import nflreadpy as nfl

    picks = cached(
        "draft_picks_all", lambda: nfl.load_draft_picks(seasons=True), max_age_hours=24 * 7
    )
    return (
        picks.filter(pl.col("gsis_id").is_not_null())
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("round").cast(pl.Float64).alias("draft_round"),
            pl.col("pick").cast(pl.Float64).alias("draft_overall"),
        )
        .unique(subset=["player_id"], keep="first")
    )


def _contract_features(players: pl.DataFrame) -> pl.DataFrame:
    """Attach the contract in force during each player's target season.

    Cap share is the useful signal, more than raw dollars: a team that has
    committed real money to a player tends to give him the ball, and a player in
    the last year of his deal has every incentive to produce.
    """
    con = _contracts()
    joined = (
        players.select("player_id", "season")
        .join(con, on="player_id", how="left")
        .filter(
            (pl.col("year_signed") <= pl.col("season"))
            & (pl.col("season") < pl.col("year_signed") + pl.col("years"))
        )
        # If deals overlap, the most recently signed one is in force.
        .sort("year_signed")
        .group_by(["player_id", "season"])
        .last()
    )
    return joined.with_columns(
        (pl.col("year_signed") + pl.col("years") - 1 - pl.col("season")).alias("contract_years_left"),
        ((pl.col("year_signed") + pl.col("years") - 1) == pl.col("season"))
        .cast(pl.Int8)
        .alias("is_contract_year"),
        (pl.col("season") - pl.col("year_signed")).alias("years_into_contract"),
    ).rename({"apy": "contract_apy", "guaranteed": "contract_guaranteed",
              "apy_cap_pct": "contract_cap_pct"})


def _vacated(season_stats: pl.DataFrame, rosters: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """Targets and carries a team must replace, per team-season.

    Computed as prior-season volume belonging to players no longer on the roster.
    This is how a model sees an opening before it shows up in any box score --
    and it is the main lever for valuing a player who changed teams.
    """
    frames = []
    for year in seasons:
        prior = season_stats.filter(pl.col("season") == year - 1)
        current = rosters.filter(pl.col("season") == year).select("player_id", "team")
        if prior.height == 0 or current.height == 0:
            continue
        # Volume from last year whose owner is not on that same team now.
        stayed = current.rename({"team": "team_now"})
        merged = prior.join(stayed, on="player_id", how="left").with_columns(
            (pl.col("team_now") == pl.col("team")).fill_null(False).alias("stayed")
        )
        frames.append(
            merged.filter(~pl.col("stayed"))
            .group_by("team")
            .agg(
                pl.col("targets").sum().alias("vacated_targets"),
                pl.col("carries").sum().alias("vacated_carries"),
            )
            .with_columns(pl.lit(year).cast(pl.Int32).alias("season"))
        )
    return pl.concat(frames) if frames else pl.DataFrame(
        schema={"team": pl.Utf8, "vacated_targets": pl.Float64,
                "vacated_carries": pl.Float64, "season": pl.Int32}
    )


def _combine() -> pl.DataFrame:
    """Athletic testing, keyed by gsis_id.

    Matters almost entirely for rookies: for a player with no NFL snaps, draft
    capital plus athletic profile is most of what is knowable about him.
    """
    import nflreadpy as nfl

    cb = cached("combine_all", lambda: nfl.load_combine(seasons=True), max_age_hours=24 * 7)
    ids = nv.ff_playerids().select(
        pl.col("gsis_id").alias("player_id"), pl.col("pfr_id")
    ).filter(pl.col("player_id").is_not_null() & pl.col("pfr_id").is_not_null())

    return (
        cb.filter(pl.col("pfr_id").is_not_null())
        .select(
            "pfr_id",
            pl.col("forty").cast(pl.Float64),
            pl.col("vertical").cast(pl.Float64),
            pl.col("broad_jump").cast(pl.Float64),
            pl.col("cone").cast(pl.Float64),
            pl.col("shuttle").cast(pl.Float64),
            pl.col("wt").cast(pl.Float64).alias("combine_wt"),
        )
        .join(ids, on="pfr_id", how="inner")
        .unique(subset=["player_id"], keep="first")
        .drop("pfr_id")
    )


def _vegas(seasons: list[int]) -> pl.DataFrame:
    """Season implied points per team from posted Vegas totals and spreads."""
    sched = nv.schedules(seasons)
    home = sched.select(
        pl.col("season").cast(pl.Int32),
        pl.col("home_team").alias("team"),
        # Favourite carries a negative spread, so subtracting lifts their total.
        (pl.col("total_line") / 2 - pl.col("spread_line") / 2).alias("implied"),
    )
    away = sched.select(
        pl.col("season").cast(pl.Int32),
        pl.col("away_team").alias("team"),
        (pl.col("total_line") / 2 + pl.col("spread_line") / 2).alias("implied"),
    )
    return (
        pl.concat([home, away])
        .drop_nulls("implied")
        .group_by(["team", "season"])
        .agg(pl.col("implied").mean().alias("vegas_implied_ppg"))
    )


def _injury_history(seasons: list[int]) -> pl.DataFrame:
    """Durability signal from prior-season injury reports.

    Games played already captures time missed, but the injury report captures
    something games do not: a player who was listed every week and played through
    it is carrying risk that has not yet shown up in his availability. Counts are
    lagged one season, so nothing here leaks the season being predicted.
    """
    import nflreadpy as nfl

    frames = []
    for season in seasons:
        def load(season: int = season) -> pl.DataFrame:
            raw = nfl.load_injuries(seasons=[season])
            return (
                raw.filter(pl.col("gsis_id").is_not_null())
                .group_by("gsis_id")
                .agg(
                    (pl.col("report_status") == "Out").sum().alias("weeks_out"),
                    (pl.col("report_status") == "Questionable").sum().alias("weeks_questionable"),
                    (pl.col("practice_status") == "Did Not Participate In Practice")
                    .sum().alias("weeks_dnp"),
                    pl.len().alias("weeks_on_report"),
                )
                .select(
                    pl.col("gsis_id").alias("player_id"),
                    pl.lit(season).cast(pl.Int32).alias("season"),
                    pl.col("weeks_out").cast(pl.Float64),
                    pl.col("weeks_questionable").cast(pl.Float64),
                    pl.col("weeks_dnp").cast(pl.Float64),
                    pl.col("weeks_on_report").cast(pl.Float64),
                )
            )

        try:
            frames.append(cached(f"injuries_{season}", load, max_age_hours=24 * 7))
        except Exception:
            continue

    if not frames:
        return pl.DataFrame(schema={"player_id": pl.Utf8, "season": pl.Int32})
    # Lag by one season: last year's report history predicts this year's health.
    return pl.concat(frames).with_columns(
        (pl.col("season") + 1).cast(pl.Int32).alias("season")
    ).rename({
        "weeks_out": "inj_weeks_out_lag1",
        "weeks_questionable": "inj_weeks_questionable_lag1",
        "weeks_dnp": "inj_weeks_dnp_lag1",
        "weeks_on_report": "inj_weeks_on_report_lag1",
    })


# Per-team players the pre-2025 feed lists at `depth_team == 1`, averaged over
# 2015-2024 and stable to about +/-0.1 across those seasons. The old feed ranks
# within a *formation slot*, so all three starting receivers are a "1".
LEGACY_STARTER_RATE = {"QB": 1.01, "RB": 1.48, "WR": 3.04, "TE": 1.24}
# 2025+ ranks the whole position group instead, 1..15. These cutoffs are the ones
# whose per-team counts land nearest the rates above, so `is_starter` means the
# same thing on both sides of the change. Running back is the loose one: the old
# feed called 1.48 backs per team a starter, which is a committee fudge no cutoff
# on a strict ordering can reproduce.
STARTER_RANK = {"QB": 1, "RB": 1, "WR": 3, "TE": 1}


def _depth_chart(seasons: list[int]) -> pl.DataFrame:
    """Preseason depth-chart role per player-season.

    Role is the strongest available proxy for opportunity, and opportunity is the
    most predictable input in fantasy football -- a WR1 designation is worth far
    more than any amount of prior-season efficiency. This is published before
    week one, so it is legitimately knowable on draft day.

    nflverse changed this feed's schema in 2025, and the two versions are not on
    the same scale. Through 2024 `depth_team` is the player's slot within a
    *formation*, capped at 3, so a team fields about three receivers at rank 1.
    From 2025 `pos_rank` is a strict ordering of the whole position group and
    runs to 15, so exactly one receiver is rank 1 and the mean jumps from 1.75 to
    3.59.

    `is_starter` is therefore calibrated per position (see STARTER_RANK) and is
    the only column here that means the same thing on both sides of the break.
    `depth_rank` is passed through as the source reports it and is **not**
    comparable across 2025 -- it is kept for inspection, and `depth_schema` says
    which feed produced it, but it is deliberately not a model feature.
    """
    import nflreadpy as nfl

    frames = []
    for season in seasons:
        def load(season: int = season) -> pl.DataFrame:
            raw = nfl.load_depth_charts(seasons=[season])
            if "pos_rank" in raw.columns:      # 2025+ schema
                out = (
                    raw.filter(pl.col("pos_abb").is_in(["QB", "RB", "WR", "TE"]))
                    .filter(pl.col("gsis_id").is_not_null())
                    .with_columns(pl.col("dt").cast(pl.Utf8))
                    # Earliest snapshot = the preseason chart.
                    .sort("dt")
                    .group_by(["gsis_id"])
                    .first()
                    .select(
                        pl.col("gsis_id").alias("player_id"),
                        pl.lit(season).cast(pl.Int32).alias("season"),
                        pl.col("pos_rank").cast(pl.Float64).alias("depth_rank"),
                        (
                            pl.col("pos_rank")
                            <= pl.col("pos_abb").replace_strict(STARTER_RANK, default=1)
                        ).cast(pl.Int8).alias("is_starter"),
                        pl.lit("pos_rank").alias("depth_schema"),
                    )
                )
            else:                               # 2020-2024 schema
                out = (
                    raw.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
                    .filter(pl.col("gsis_id").is_not_null())
                    .filter(pl.col("week") == pl.col("week").min())
                    .group_by("gsis_id")
                    .agg(pl.col("depth_team").cast(pl.Float64).min().alias("depth_rank"))
                    .select(
                        pl.col("gsis_id").alias("player_id"),
                        pl.lit(season).cast(pl.Int32).alias("season"),
                        "depth_rank",
                        (pl.col("depth_rank") == 1).cast(pl.Int8).alias("is_starter"),
                        pl.lit("depth_team").alias("depth_schema"),
                    )
                )
            return out.unique(subset=["player_id", "season"], keep="first")

        try:
            # Key carries the schema version: the columns this returns changed
            # when the feed did, and a cache written by the old code would come
            # back missing them rather than erroring somewhere obvious.
            frames.append(cached(f"depth_role_v2_{season}", load, max_age_hours=12.0))
        except Exception:
            continue

    if not frames:
        return pl.DataFrame(schema={"player_id": pl.Utf8, "season": pl.Int32,
                                    "depth_rank": pl.Float64, "is_starter": pl.Int8,
                                    "depth_schema": pl.Utf8})
    return pl.concat(frames).with_columns(pl.col("depth_rank").clip(1, 5))


def _college_features(draft_years: list[int]) -> pl.DataFrame:
    """College production and usage, keyed by gsis_id.

    Only rookies have anything at stake here: by a player's second NFL season his
    own NFL usage tells you far more than what he did in college. But for a first
    year player it is the only production evidence that exists, and college usage
    share is the direct analogue of the NFL target-share features that carry the
    most signal.

    nflverse stores a slug where CFBD stores a numeric id, so the two do not join
    directly -- the draft slot itself is the bridge.
    """
    from src.ingest.cfbd import MissingKeyError, draft_bridge, final_college_season

    try:
        bridge = draft_bridge(draft_years)
        college = final_college_season(list(range(min(draft_years) - 6, max(draft_years))))
    except MissingKeyError:
        # No key configured: proceed without college data rather than failing.
        return pl.DataFrame(schema={"player_id": pl.Utf8})

    # nflverse's `pick` is the *overall* selection number (round 2 starts at 33),
    # while CFBD's `pick` counts within the round and its `overall` is the global
    # number. Joining the two `pick` columns silently matches round one only.
    picks = _draft_picks_raw().select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("season").cast(pl.Int32).alias("draft_year"),
        pl.col("pick").cast(pl.Int32).alias("overall"),
    ).filter(pl.col("player_id").is_not_null())

    linked = picks.join(
        bridge.with_columns(pl.col("draft_year").cast(pl.Int32)),
        on=["draft_year", "overall"],
        how="inner",
    )
    return (
        linked.join(college, on="college_athlete_id", how="left")
        .select(
            "player_id", "pre_draft_grade", "pre_draft_pos_ranking",
            "usage_overall_final", "usage_pass_final", "usage_rush_final",
            "cfb_rec_rec_final", "cfb_rec_yds_final", "cfb_rec_td_final",
            "cfb_rus_car_final", "cfb_rus_yds_final", "cfb_rus_td_final",
            "cfb_usage_peak", "cfb_rec_yds_peak", "cfb_rus_yds_peak", "cfb_seasons",
        )
        .unique(subset=["player_id"], keep="first")
    )


def _draft_picks_raw() -> pl.DataFrame:
    import nflreadpy as nfl

    return cached(
        "draft_picks_all_raw",
        lambda: nfl.load_draft_picks(seasons=True),
        max_age_hours=24 * 7,
    )


def _draft_class() -> pl.DataFrame:
    """Draft slot *and* the year it was spent, which draft capital alone drops.

    Roster churn needs to know that a team took a running back at pick 20 *this*
    April, not that the running back it already had was once taken at pick 20.
    """
    return (
        _draft_picks_raw()
        .filter(pl.col("gsis_id").is_not_null())
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("season").cast(pl.Int32).alias("draft_season"),
            pl.col("pick").cast(pl.Float64).alias("draft_overall"),
        )
        .unique(subset=["player_id"], keep="first")
    )


def _adp(seasons: list[int]) -> pl.DataFrame:
    """Historical ADP per player-season. Known on draft day, so usable."""
    from src.ingest.adp import load_adp
    from src.ingest.ids import resolve

    frames = []
    for year in seasons:
        try:
            got = resolve(load_adp(year=year))
        except Exception:
            continue
        frames.append(
            got.filter(pl.col("gsis_id").is_not_null())
            .select(
                pl.col("gsis_id").alias("player_id"),
                pl.lit(year).cast(pl.Int32).alias("season"),
                pl.col("adp").cast(pl.Float64),
                pl.col("stdev").cast(pl.Float64).alias("adp_stdev"),
            )
            .unique(subset=["player_id", "season"], keep="first")
        )
    if not frames:
        return pl.DataFrame(schema={"player_id": pl.Utf8, "season": pl.Int32,
                                    "adp": pl.Float64, "adp_stdev": pl.Float64})
    return pl.concat(frames)


def build(target_seasons: list[int], stats: pl.DataFrame | None = None,
          draft_day_rosters: bool = True) -> pl.DataFrame:
    """Training/scoring table for the given target seasons.

    `stats` may be passed in to avoid rebuilding the player-season frame, which
    is the largest object in this pipeline -- constructing it twice in one
    process is enough to exhaust a laptop's free memory.

    `draft_day_rosters=False` reverts to end-of-season roster membership, which
    leaks in-season trades into every team-context feature. It is here to be
    measured against, not used.
    """
    lo = min(target_seasons) - 3
    hi = max(target_seasons)
    stat_seasons = list(range(lo, hi + 1))

    if stats is None:
        stats = build_player_season([s for s in stat_seasons if s < hi] or [hi - 1])
    rosters = _rosters(list(range(lo, hi + 1)), draft_day=draft_day_rosters)
    vacated = _vacated(stats, rosters, target_seasons)
    vegas = _vegas(list(range(lo, hi + 1)))

    # Spine: every player on a roster in a target season.
    spine = rosters.filter(pl.col("season").is_in(target_seasons))

    for lag in (1, 2):
        lagged = stats.select(
            ["player_id", "season"] + [c for c in LAG_STATS if c in stats.columns]
        ).with_columns((pl.col("season") + lag).cast(pl.Int32).alias("season"))
        lagged = lagged.rename(
            {c: f"{c}_lag{lag}" for c in lagged.columns if c not in ("player_id", "season")}
        )
        spine = spine.join(lagged, on=["player_id", "season"], how="left")

    # Prior team, to detect a move.
    prev_team = (
        stats.select("player_id", "season", pl.col("team").alias("prev_team"))
        .with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))
    )
    contracts = _contract_features(spine)
    churn = build_churn(
        stats, rosters, target_seasons,
        contract_caps=contracts.select("player_id", "season", "contract_cap_pct"),
        draft_picks=_draft_class(),
    )
    spine = (
        spine.join(prev_team, on=["player_id", "season"], how="left")
        .join(contracts, on=["player_id", "season"], how="left")
        .join(churn, on=["player_id", "season"], how="left")
        .join(vacated, on=["team", "season"], how="left")
        .join(vegas, on=["team", "season"], how="left")
        .join(_combine(), on="player_id", how="left")
        .join(_draft_capital(), on="player_id", how="left")
        .join(_adp(target_seasons), on=["player_id", "season"], how="left")
        .join(_college_features(list(range(lo - 4, hi + 1))), on="player_id", how="left")
        .join(_depth_chart(target_seasons), on=["player_id", "season"], how="left")
        .join(
            _injury_history(list(range(lo, hi + 1))),
            on=["player_id", "season"], how="left",
        )
        .with_columns(
            # Undrafted is meaningful information, not missing data: encode it as
            # one pick past the end of the draft rather than leaving it null.
            pl.col("draft_round").fill_null(8.0),
            pl.col("draft_overall").fill_null(300.0),
            # Undrafted in fantasy terms: past the end of any realistic board.
            pl.col("adp").fill_null(400.0),
            pl.col("adp_stdev").fill_null(50.0),
        )
        .with_columns(
            (pl.col("prev_team").is_not_null() & (pl.col("prev_team") != pl.col("team")))
            .cast(pl.Int8)
            .alias("changed_team"),
            (pl.col("years_exp") == 0).cast(pl.Int8).alias("is_rookie"),
        )
    )
    return spine


def add_targets(features: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    """Attach the outcome we are predicting: points and games in the target season."""
    outcome = stats.select(
        "player_id", "season",
        pl.col("points").alias("y_points"),
        pl.col("games").alias("y_games"),
    )
    return features.join(outcome, on=["player_id", "season"], how="left").with_columns(
        # A rostered player with no stat line scored zero. Dropping those rows
        # would train the model only on players who panned out.
        pl.col("y_points").fill_null(0.0),
        pl.col("y_games").fill_null(0),
    )
