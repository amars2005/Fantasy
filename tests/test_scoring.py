"""Scoring correctness.

Hand-computed stat lines first, then a cross-check against nflverse's own
fantasy_points_ppr over a full season. The second test is the real guard: it
catches column-name drift and sign errors across ~5000 real stat lines.
"""

import polars as pl
import pytest

from src.scoring import add_fantasy_points, score_line


def test_qb_line_hand_computed():
    # 300 pass yds (12) + 3 pass TD (12) - 1 INT (2) + 20 rush yds (2) + 1 rush TD (6)
    stats = {
        "passing_yards": 300,
        "passing_tds": 3,
        "passing_interceptions": 1,
        "rushing_yards": 20,
        "rushing_tds": 1,
    }
    assert score_line(stats) == pytest.approx(30.0)


def test_wr_line_hand_computed():
    # 8 rec (8) + 120 rec yds (12) + 1 rec TD (6)
    stats = {"receptions": 8, "receiving_yards": 120, "receiving_tds": 1}
    assert score_line(stats) == pytest.approx(26.0)


def test_rb_line_with_fumble_hand_computed():
    # 85 rush yds (8.5) + 1 rush TD (6) + 4 rec (4) + 30 rec yds (3) - 1 fumble (2)
    stats = {
        "rushing_yards": 85,
        "rushing_tds": 1,
        "receptions": 4,
        "receiving_yards": 30,
        "fumbles_lost": 1,
    }
    assert score_line(stats) == pytest.approx(19.5)


def test_league_extras_are_scored():
    """Punt-return and fumble-recovery TDs are worth 6 in this league."""
    assert score_line({"pt_return_tds": 1}) == pytest.approx(6.0)
    assert score_line({"fumble_recovery_tds": 1}) == pytest.approx(6.0)


def test_return_td_columns_do_not_double_count():
    """special_teams_tds and pt_return_tds are disjoint in nflverse, so both score."""
    df = pl.DataFrame({"special_teams_tds": [1], "pt_return_tds": [1]})
    assert add_fantasy_points(df)["fantasy_points_league"][0] == pytest.approx(12.0)


def test_empty_line_scores_zero():
    assert score_line({}) == 0.0


def test_missing_columns_are_treated_as_zero():
    # A frame with only receiving stats must still score, not raise.
    df = pl.DataFrame({"receptions": [5], "receiving_yards": [50]})
    out = add_fantasy_points(df)
    assert out["fantasy_points_league"][0] == pytest.approx(10.0)


def test_fumbles_split_across_nflverse_columns_are_summed():
    df = pl.DataFrame(
        {
            "rushing_yards": [100],
            "rushing_fumbles_lost": [1],
            "receiving_fumbles_lost": [1],
        }
    )
    out = add_fantasy_points(df)
    # 100 rush yds (10) - 2 fumbles (4)
    assert out["fantasy_points_league"][0] == pytest.approx(6.0)


@pytest.mark.slow
def test_matches_nflverse_ppr_on_2025_season():
    """Our PPR scoring should reproduce nflverse's fantasy_points_ppr.

    Any systematic gap means a rule is wrong; this is the test that would have
    caught a bad column name or a flipped sign.
    """
    import nflreadpy as nfl

    from src.config import STANDARD_SCORING

    df = nfl.load_player_stats(seasons=[2025], summary_level="week")
    # Compare on the standard rules only: our league adds punt-return and
    # fumble-recovery touchdowns, which nflverse's PPR column does not score.
    scored = add_fantasy_points(df, scoring=STANDARD_SCORING)
    diff = (
        scored.select(
            (pl.col("fantasy_points_league") - pl.col("fantasy_points_ppr")).abs()
        )
        .to_series()
        .drop_nulls()
    )
    assert diff.max() < 0.02, f"max divergence {diff.max()} from nflverse PPR"
