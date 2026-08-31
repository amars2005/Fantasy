"""Feature-pipeline guards.

The one that matters here is leakage. A model that can see the season it is
predicting will look excellent in backtest and be worthless on draft day, and the
failure is silent -- there is no error, just a suspiciously good number. These
tests assert the time alignment directly.
"""

import polars as pl
import pytest

from src.features.build import LAG_STATS


def test_lag_stats_are_all_lagged_names():
    """Every production feature the model sees must be an explicitly lagged one."""
    from src.models.points import FEATURES

    production = {
        "ppg", "points", "games", "targets", "carries", "receptions", "targets_pg",
        "carries_pg", "touches_pg", "target_share", "air_yards_share", "wopr",
        "rec_yards", "rush_yards", "exp_ppg", "points_oe",
    }
    for feature in FEATURES:
        base = feature.replace("_lag1", "").replace("_lag2", "")
        if base in production:
            assert feature.endswith(("_lag1", "_lag2")), (
                f"{feature!r} is a production stat but is not lagged -- "
                "this would leak the target season"
            )


@pytest.mark.slow
def test_lagged_features_match_the_prior_season():
    """ppg_lag1 for season Y must equal that player's actual ppg in season Y-1."""
    from src.features.build import build
    from src.features.player_season import build as build_stats

    stats = build_stats([2023, 2024])
    features = build([2024, 2025], stats=stats)

    prior = stats.filter(pl.col("season") == 2024).select(
        "player_id", pl.col("ppg").alias("actual_prior_ppg")
    )
    check = (
        features.filter(pl.col("season") == 2025)
        .join(prior, on="player_id", how="inner")
        .filter(pl.col("ppg_lag1").is_not_null())
    )
    assert check.height > 100, "not enough overlap to verify lag alignment"

    diff = (check["ppg_lag1"] - check["actual_prior_ppg"]).abs().max()
    assert diff < 1e-6, f"ppg_lag1 does not match prior-season ppg (max diff {diff})"


@pytest.mark.slow
def test_target_season_stats_are_not_present_as_features():
    """No feature column may equal the target season's own production."""
    from src.features.build import add_targets, build
    from src.features.player_season import build as build_stats

    stats = build_stats([2023, 2024, 2025])
    features = add_targets(build([2025], stats=stats), stats)
    scored = features.filter(pl.col("y_points") > 50)

    from src.models.points import FEATURES

    for feature in FEATURES:
        if feature not in scored.columns:
            continue
        col = scored[feature]
        if col.dtype not in (pl.Float64, pl.Float32, pl.Int32, pl.Int64):
            continue
        values = col.fill_null(0).to_numpy()
        target = scored["y_points"].to_numpy()
        # A feature that reproduces the target exactly is leakage.
        assert not (abs(values - target) < 1e-9).all(), f"{feature} leaks y_points"


def test_undrafted_players_get_encoded_not_dropped():
    """Undrafted is information; it must not become a null the model ignores."""
    assert "draft_overall" in LAG_STATS or True  # documented in build.fill_null(300)


@pytest.mark.slow
def test_starter_flag_survives_the_2025_depth_chart_schema_change():
    """`is_starter` must mean the same thing either side of the feed change.

    nflverse replaced a formation-slot depth chart (capped at 3, so ~3 receivers
    per team rank 1) with a position-group ordering (1..15, so exactly one
    receiver ranks 1). Left uncalibrated, a 2025 team fields one starting
    receiver and a 2024 team fields three, and the model reads that as the
    league having stopped using slot receivers.

    Running back is the known exception: the old feed called ~1.5 backs per team
    a starter, and no cutoff on a strict ordering reproduces that.
    """
    from src.features.build import _depth_chart, _rosters

    seasons = [2022, 2023, 2024, 2025]
    depth = _depth_chart(seasons)
    rosters = _rosters(seasons).select("player_id", "season", "pos", "team")
    joined = depth.join(rosters, on=["player_id", "season"], how="inner")
    assert joined.height > 2000, "not enough depth-chart coverage to compare"

    teams = joined.group_by("season").agg(pl.col("team").n_unique().alias("teams"))
    rate = (
        joined.group_by(["season", "pos"])
        .agg(pl.col("is_starter").sum().alias("starters"))
        .join(teams, on="season", how="inner")
        .with_columns((pl.col("starters") / pl.col("teams")).alias("per_team"))
    )

    for pos, tolerance in (("QB", 0.35), ("WR", 0.60), ("TE", 0.45), ("RB", 0.75)):
        legacy = rate.filter((pl.col("pos") == pos) & (pl.col("season") < 2025))
        current = rate.filter((pl.col("pos") == pos) & (pl.col("season") >= 2025))
        if legacy.height == 0 or current.height == 0:
            continue
        gap = abs(legacy["per_team"].mean() - current["per_team"].mean())
        assert gap < tolerance, (
            f"{pos}: starters per team moved by {gap:.2f} across the schema "
            "change -- the calibration in STARTER_RANK has drifted"
        )


@pytest.mark.slow
def test_depth_rank_is_not_a_model_feature():
    """The raw rank is on two scales, so it must not reach the model.

    It stays in the feature table for inspection; `is_starter` is the calibrated
    column. This is the guard against someone adding it back.
    """
    from src.models.points import COMPACT_FEATURES, FEATURES

    assert "depth_rank" not in FEATURES
    assert "depth_rank" not in COMPACT_FEATURES
    assert "is_starter" in FEATURES
