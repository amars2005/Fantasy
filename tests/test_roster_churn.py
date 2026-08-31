"""Guards for the roster-churn features.

Two failures are worth a test each.

The first is arithmetic: a receiver who signs somewhere new must not be counted
as his own competition. Without the self-exclusion, every player who changed
teams carries his own prior volume in `pos_in_targets`, which is a feature that
says "this player was good last year" wearing a disguise.

The second is leakage, and it is the one that actually bit. `load_rosters`
returns an end-of-season snapshot, so a player traded in October is listed with
the team that acquired him. Building arrivals from that lets the model see
midseason trades on draft day; measured, it inflated the feature group's
apparent value by roughly four times.
"""

import polars as pl
import pytest

from src.features.roster_churn import CHURN_FEATURES, build_churn


def _stats() -> pl.DataFrame:
    """One prior season (2023) of production for five players."""
    rows = [
        # player, team, pos, targets, carries, pass_attempts, points, ppg, games
        ("wr_stay", "AAA", "WR", 120, 0, 0, 200.0, 12.5, 16),
        ("wr_move", "BBB", "WR", 140, 0, 0, 210.0, 13.1, 16),
        ("rb_gone", "AAA", "RB", 40, 220, 0, 180.0, 11.3, 16),
        ("qb_aaa", "AAA", "QB", 0, 30, 560, 320.0, 20.0, 16),
        ("qb_bbb", "BBB", "QB", 0, 10, 480, 220.0, 13.8, 16),
    ]
    return pl.DataFrame(
        rows, orient="row",
        schema={"player_id": pl.Utf8, "team": pl.Utf8, "pos": pl.Utf8,
                "targets": pl.Float64, "carries": pl.Float64,
                "pass_attempts": pl.Float64, "points": pl.Float64,
                "ppg": pl.Float64, "games": pl.Float64},
    ).with_columns(pl.lit(2023).cast(pl.Int32).alias("season"))


def _rosters() -> pl.DataFrame:
    """2023 memberships, then 2024 after wr_move signs with AAA and rb_gone leaves."""
    prior = [("wr_stay", "AAA", "WR"), ("wr_move", "BBB", "WR"),
             ("rb_gone", "AAA", "RB"), ("qb_aaa", "AAA", "QB"),
             ("qb_bbb", "BBB", "QB")]
    now = [("wr_stay", "AAA", "WR"), ("wr_move", "AAA", "WR"),
           ("qb_bbb", "AAA", "QB"), ("qb_aaa", "BBB", "QB"),
           ("rookie_rb", "AAA", "RB")]
    frames = []
    for season, rows in ((2023, prior), (2024, now)):
        frames.append(
            pl.DataFrame(
                rows, orient="row",
                schema={"player_id": pl.Utf8, "team": pl.Utf8, "pos": pl.Utf8},
            ).with_columns(pl.lit(season).cast(pl.Int32).alias("season"))
        )
    return pl.concat(frames)


@pytest.fixture
def churn() -> pl.DataFrame:
    return build_churn(_stats(), _rosters(), [2024])


def test_every_declared_feature_is_produced(churn):
    for feature in CHURN_FEATURES:
        assert feature in churn.columns, f"{feature} is declared but never built"


def test_a_signing_is_not_his_own_competition(churn):
    """wr_move brings 140 targets to AAA; those are not competition for him."""
    moved = churn.filter(pl.col("player_id") == "wr_move").row(0, named=True)
    stayed = churn.filter(pl.col("player_id") == "wr_stay").row(0, named=True)
    assert moved["pos_in_targets"] == 0.0
    assert stayed["pos_in_targets"] == 140.0


def test_departures_free_up_opportunity_at_the_position(churn):
    """rb_gone's 220 carries left AAA, and only a rookie replaced them."""
    rookie = churn.filter(pl.col("player_id") == "rookie_rb").row(0, named=True)
    assert rookie["pos_out_carries"] == 220.0
    assert rookie["pos_in_carries"] == 0.0     # the rookie's own zero, excluded
    assert rookie["pos_net_carries"] == 220.0


def test_the_quarterback_room_is_compared_to_last_season_s_starter(churn):
    """AAA swapped a 20.0 ppg starter for a 13.8 ppg one; that is a downgrade."""
    aaa = churn.filter(pl.col("player_id") == "wr_stay").row(0, named=True)
    assert aaa["qb_prev_ppg"] == pytest.approx(20.0)
    assert aaa["qb_room_ppg"] == pytest.approx(13.8)
    assert aaa["qb_upgrade"] == pytest.approx(-6.2)
    assert aaa["qb_changed"] == 1

    bbb = churn.filter(pl.col("player_id") == "qb_aaa").row(0, named=True)
    assert bbb["qb_upgrade"] == pytest.approx(20.0 - 13.8)


def test_an_inexperienced_quarterback_room_is_flagged_not_scored_zero(churn):
    """A room with no qualifying prior season is unknown, not terrible.

    Both cases produce `qb_room_ppg == 0`, and a model given only that number
    cannot tell a first-round rookie from a disaster.
    """
    aaa = churn.filter(pl.col("player_id") == "wr_stay").row(0, named=True)
    assert aaa["qb_room_unknown"] == 0

    rookie_only = _rosters().filter(
        (pl.col("season") == 2023) | (pl.col("player_id") == "rookie_rb")
    ).vstack(pl.DataFrame(
        [("rookie_qb", "AAA", "QB")], orient="row",
        schema={"player_id": pl.Utf8, "team": pl.Utf8, "pos": pl.Utf8},
    ).with_columns(pl.lit(2024).cast(pl.Int32).alias("season")))
    got = build_churn(_stats(), rookie_only, [2024])
    row = got.filter(pl.col("player_id") == "rookie_qb").row(0, named=True)
    assert row["qb_room_ppg"] == 0.0
    assert row["qb_room_unknown"] == 1


def test_net_flow_nets_arrivals_against_departures(churn):
    """Net flow counts every position, including the quarterbacks who ran.

    AAA lost rb_gone (40 targets, 220 carries) and qb_aaa (30 carries), and
    gained wr_move (140 targets) and qb_bbb (10 carries).
    """
    aaa = churn.filter(pl.col("player_id") == "wr_stay").row(0, named=True)
    assert aaa["arrived_targets"] == 140.0
    assert aaa["arrived_carries"] == 10.0
    assert aaa["net_targets"] == 40.0 - 140.0
    assert aaa["net_carries"] == (220.0 + 30.0) - 10.0


def test_incoming_draft_capital_lands_on_the_right_position(churn):
    picks = pl.DataFrame({
        "player_id": ["rookie_rb"],
        "draft_season": pl.Series([2024], dtype=pl.Int32),
        "draft_overall": [12.0],
    })
    got = build_churn(_stats(), _rosters(), [2024], draft_picks=picks)
    by_id = {r["player_id"]: r for r in got.iter_rows(named=True)}
    assert by_id["rookie_rb"]["pos_rookie_overall"] == 12.0
    # A twelfth-overall running back is not news for the receivers.
    assert by_id["wr_stay"]["pos_rookie_overall"] == 300.0


def test_a_season_with_no_prior_data_does_not_explode():
    got = build_churn(_stats(), _rosters(), [2030])
    assert got.height == 0 or got["player_id"].is_null().all()


# --- the leak the features were nearly built on ------------------------------
@pytest.mark.slow
def test_rosters_are_the_week_one_snapshot_not_the_end_of_season_one():
    """A completed season's roster must match week one, not the final roster.

    This is the guard for the actual bug: `load_rosters` reports where a player
    finished the season, so an October trade shows up as if it were a March
    signing. Roughly one skill player in eight moves during a season, which is
    more than enough to manufacture a result.
    """
    import nflreadpy as nfl

    from src.features.build import _rosters as draft_day_rosters

    season = 2024
    ours = draft_day_rosters([season]).select("player_id", "team")
    weekly = nfl.load_rosters_weekly(seasons=[season])
    week_one = (
        weekly.filter(pl.col("week") == weekly["week"].min())
        .filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
        .select(pl.col("gsis_id").alias("player_id"),
                pl.col("team").alias("week_one_team"))
        .unique(subset=["player_id"])
    )

    joined = ours.join(week_one, on="player_id", how="inner")
    assert joined.height > 500, "not enough overlap to verify the snapshot"
    mismatched = joined.filter(pl.col("team") != pl.col("week_one_team"))
    assert mismatched.height == 0, (
        f"{mismatched.height} players carry a team they did not have in week one "
        "-- the roster snapshot is leaking in-season moves"
    )
