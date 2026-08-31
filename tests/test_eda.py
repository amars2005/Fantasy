"""Guards for the EDA measurement helpers.

Five analysis scripts now share these, and every number in `docs/` comes out of
them. The failure mode they exist for is a plausible-looking wrong answer: a
partial correlation that does not actually remove the control still returns a
number, and nothing downstream would notice.
"""

import numpy as np
import polars as pl
import pytest

from src.features.eda import (
    bucket_effect, grouped_slope, grouped_spearman, residualise_within,
    yoy_frame,
)


def _frame(n: int = 600, seed: int = 0, slope: float = 2.0,
           noise: float = 1.0) -> pl.DataFrame:
    """y = slope*x + control + noise, over two seasons and two positions."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    control = rng.normal(0, 1, n)
    return pl.DataFrame({
        "season": rng.choice([2022, 2023], n).tolist(),
        "pos": rng.choice(["RB", "WR"], n).tolist(),
        "x": x,
        "control": control,
        "y": slope * x + control + rng.normal(0, noise, n),
    })


# --- correlation -------------------------------------------------------------
def test_spearman_recovers_a_planted_relationship():
    got = grouped_spearman(_frame(), "x", "y")
    assert got["rho"] > 0.7
    assert got["n"] == 600


def test_partial_correlation_removes_the_control():
    """x is independent of the control, so partialling it out changes little."""
    frame = _frame()
    plain = grouped_spearman(frame, "x", "y")
    partial = grouped_spearman(frame, "x", "y", control="control")
    assert partial["rho"] > plain["rho"], (
        "removing an independent source of variance in y should sharpen, "
        "not blunt, the relationship"
    )


def test_a_relationship_that_is_only_the_control_vanishes():
    """The whole point of the incremental column: y = control, x = control."""
    rng = np.random.default_rng(3)
    n = 600
    control = rng.normal(0, 1, n)
    frame = pl.DataFrame({
        "season": rng.choice([2022, 2023], n).tolist(),
        "pos": rng.choice(["RB", "WR"], n).tolist(),
        "control": control,
        "x": control + rng.normal(0, 0.05, n),
        "y": control + rng.normal(0, 0.05, n),
    })
    assert grouped_spearman(frame, "x", "y")["rho"] > 0.9
    assert abs(grouped_spearman(frame, "x", "y", control="control")["rho"]) < 0.3


def test_a_variable_controlled_on_itself_is_refused():
    """Residual dust correlated with residual dust returns a confident lie."""
    got = grouped_spearman(_frame(), "x", "y", control="x")
    assert np.isnan(got["rho"])


def test_groups_too_small_are_skipped():
    tiny = _frame(n=8)
    assert grouped_spearman(tiny, "x", "y")["n"] == 0


# --- slope -------------------------------------------------------------------
def test_slope_recovers_the_planted_coefficient():
    got = grouped_slope(_frame(slope=2.0, noise=0.5), "x", "y", control="control")
    assert got["slope"] == pytest.approx(2.0, abs=0.1)


def test_slope_is_in_the_units_of_the_inputs():
    """Doubling x's scale must halve the slope; a correlation would not move."""
    frame = _frame(slope=2.0, noise=0.5).with_columns((pl.col("x") * 2).alias("x2"))
    base = grouped_slope(frame, "x", "y", control="control")["slope"]
    scaled = grouped_slope(frame, "x2", "y", control="control")["slope"]
    assert scaled == pytest.approx(base / 2, rel=0.05)


# --- residuals and buckets ---------------------------------------------------
def test_residualise_leaves_nothing_of_the_control():
    frame = residualise_within(_frame(), "y", "control")
    keep = frame.drop_nulls("y_resid")
    assert keep.height > 500
    left = np.corrcoef(keep["y_resid"].to_numpy(), keep["control"].to_numpy())[0, 1]
    assert abs(left) < 0.15, "the control still explains the residual"


def test_bucket_effect_orders_by_the_planted_slope():
    table = bucket_effect(_frame(slope=2.0, noise=0.5), "x", "y",
                          control="control")
    assert table.height == 5
    effects = table["effect"].to_list()
    assert effects == sorted(effects), "effect should rise with the bucket"
    assert effects[-1] - effects[0] > 3.0


def test_bucket_effect_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(5)
    n = 600
    frame = pl.DataFrame({
        "season": rng.choice([2022, 2023], n).tolist(),
        "pos": rng.choice(["RB", "WR"], n).tolist(),
        "x": rng.normal(0, 1, n),
        "control": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
    })
    table = bucket_effect(frame, "x", "y", control="control")
    spread = table["effect"].max() - table["effect"].min()
    assert spread < 0.6, f"found a {spread:.2f} effect in pure noise"


# --- the year-over-year join -------------------------------------------------
def _stats() -> pl.DataFrame:
    rows = []
    for season in (2022, 2023):
        for i in range(40):
            rows.append({
                "player_id": f"p{i}", "season": season, "pos": "WR",
                "team": "AAA", "points": 100.0 + i, "ppg": 8.0,
                "games": 16, "targets": 80.0, "carries": 0.0,
                "rec_yards": 700.0, "rush_yards": 0.0, "points_oe": 5.0,
            })
    # p0 played in 2022 and never again: a real zero, not a missing row.
    return pl.DataFrame(rows).filter(
        ~((pl.col("player_id") == "p0") & (pl.col("season") == 2023))
    ).with_columns(pl.col("season").cast(pl.Int32))


def test_yoy_keeps_a_rostered_player_who_never_played_again():
    rosters = pl.DataFrame({
        "player_id": [f"p{i}" for i in range(40)] * 2,
        "season": [2022] * 40 + [2023] * 40,
    }).with_columns(pl.col("season").cast(pl.Int32))

    without = yoy_frame(_stats(), min_prior_games=8)
    with_rosters = yoy_frame(_stats(), rosters=rosters, min_prior_games=8)

    assert without.filter(pl.col("player_id") == "p0").height == 0, (
        "an inner join silently drops the player who scored zero"
    )
    kept = with_rosters.filter(pl.col("player_id") == "p0")
    assert kept.height == 1
    assert kept["y_points"][0] == 0.0
    assert kept["y_games"][0] == 0
