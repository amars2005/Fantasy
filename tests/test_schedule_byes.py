"""Bye collision and playoff-schedule guards."""

import numpy as np
import pytest

from src.draft.byes import bye_collision_cost, roster_weekly_value


def _roster():
    points = np.array([300.0, 250.0, 220.0, 260.0, 240.0, 200.0, 150.0, 140.0, 130.0])
    positions = np.array(["QB", "RB", "RB", "WR", "WR", "TE", "RB", "WR", "WR"])
    return points, positions


def test_stacked_byes_are_worth_less_than_spread_byes():
    points, positions = _roster()
    stacked = roster_weekly_value(points, positions, np.array([6, 6, 6, 6, 6, 6, 10, 11, 12]))
    spread = roster_weekly_value(points, positions, np.array([5, 6, 7, 8, 9, 10, 11, 12, 13]))
    assert spread > stacked, "stacking byes must cost something"


def test_collision_cost_is_zero_for_evenly_spread_byes():
    points, positions = _roster()
    spread = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert bye_collision_cost(points, positions, spread) == pytest.approx(0.0, abs=1e-6)


def test_collision_cost_is_positive_when_byes_stack():
    points, positions = _roster()
    stacked = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6])
    assert bye_collision_cost(points, positions, stacked) > 0


def test_empty_roster_has_no_collision_cost():
    empty = np.array([])
    assert bye_collision_cost(empty, np.array([]), np.array([])) == 0.0


@pytest.mark.slow
def test_playoff_schedule_uses_known_opponents_not_vegas():
    """Vegas posts no lines for weeks 15-17 in August; opponents are known."""
    from src.features.schedule import team_schedule_strength

    strength = team_schedule_strength(2026)
    assert strength.height >= 30
    assert strength["sos_playoff"].null_count() == 0, (
        "playoff SOS must be computable preseason"
    )
