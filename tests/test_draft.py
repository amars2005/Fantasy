"""Draft layer: snake order, lineup maths, replacement level, simulator calibration."""

import numpy as np
import polars as pl
import pytest

from src.draft.replacement import allocate_flex, replacement_levels
from src.draft.sim_draft import calibrate, simulate_draft_orders, snake_picks
from src.draft.vona import optimal_lineup_points

LEAGUE = {
    "teams": 14,
    "rounds": 15,
    "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1},
    "flex_eligible": ("RB", "WR", "TE"),
}


# --- snake order ------------------------------------------------------------
def test_snake_picks_first_slot():
    picks = snake_picks(1, 14, 4)
    assert picks == [1, 28, 29, 56]


def test_snake_picks_last_slot():
    picks = snake_picks(14, 14, 4)
    assert picks == [14, 15, 42, 43]


def test_snake_picks_middle_slot_gaps_alternate():
    picks = snake_picks(5, 14, 4)
    assert picks == [5, 24, 33, 52]


# --- lineup maths -----------------------------------------------------------
def _p(pos, pts):
    return {"pos": pos, "proj_points": pts}


def test_lineup_fills_required_slots_before_flex():
    roster = [_p("QB", 300), _p("RB", 200), _p("RB", 150), _p("WR", 180), _p("WR", 170), _p("TE", 120)]
    # All six are starters; no flex-eligible player left over.
    assert optimal_lineup_points(roster, LEAGUE) == pytest.approx(1120.0)


def test_flex_takes_the_best_leftover():
    roster = [_p("QB", 300), _p("RB", 200), _p("RB", 150), _p("WR", 180),
              _p("WR", 170), _p("TE", 120), _p("WR", 160)]
    # The spare WR (160) fills FLEX.
    assert optimal_lineup_points(roster, LEAGUE) == pytest.approx(1280.0)


def test_second_quarterback_is_worth_only_bench_value():
    """The tool must not want a QB2 as much as a startable skill player."""
    base = [_p("QB", 300), _p("RB", 200), _p("RB", 150), _p("WR", 180), _p("WR", 170), _p("TE", 120)]
    qb2 = optimal_lineup_points(base + [_p("QB", 290)], LEAGUE) - optimal_lineup_points(base, LEAGUE)
    wr3 = optimal_lineup_points(base + [_p("WR", 160)], LEAGUE) - optimal_lineup_points(base, LEAGUE)
    assert qb2 < wr3
    assert qb2 == pytest.approx(0.20 * 290)  # bench weight only


# --- replacement level ------------------------------------------------------
def _board():
    rows = []
    for pos, n, top in [("QB", 40, 320), ("RB", 80, 260), ("WR", 100, 270), ("TE", 30, 220)]:
        for i in range(n):
            rows.append({"pos": pos, "proj_points": top - i * 3.0, "name": f"{pos}{i+1}"})
    return pl.DataFrame(rows)


def test_flex_allocation_adds_up_to_the_flex_slots():
    counts = allocate_flex(_board(), LEAGUE)
    base = {"QB": 14, "RB": 28, "WR": 28, "TE": 14}
    extra = sum(counts[p] - base[p] for p in base)
    assert extra == LEAGUE["teams"] * LEAGUE["starters"]["FLEX"]


def test_quarterback_replacement_is_high_in_a_one_qb_league():
    """The point of VOR: QBs score most but replace cheaply."""
    levels = replacement_levels(_board(), LEAGUE)
    assert levels["QB"] > levels["RB"]
    assert levels["QB"] > levels["WR"]


# --- simulator calibration --------------------------------------------------
def test_simulated_draft_positions_recover_input_adp():
    """The gate: if bots don't reproduce ADP, every survival probability is wrong."""
    rng = np.random.default_rng(3)
    n = 180
    adp = np.arange(1, n + 1, dtype=float)
    stdev = np.linspace(1.0, 20.0, n)

    mu = calibrate(adp, stdev, n_sims=3000, rng=rng)
    positions = simulate_draft_orders(mu, stdev, 6000, rng)
    error = np.abs(positions.mean(axis=0) - adp)

    assert error.mean() < 2.0, f"mean calibration error {error.mean():.2f} picks"
    assert error[:56].mean() < 1.5, "early rounds must be tightly calibrated"


def test_uncalibrated_simulation_is_biased():
    """Documents why calibrate() exists rather than using raw ADP."""
    rng = np.random.default_rng(3)
    n = 180
    adp = np.arange(1, n + 1, dtype=float)
    stdev = np.linspace(1.0, 20.0, n)

    raw = np.abs(simulate_draft_orders(adp, stdev, 6000, rng).mean(axis=0) - adp).mean()
    mu = calibrate(adp, stdev, n_sims=3000, rng=rng)
    fixed = np.abs(simulate_draft_orders(mu, stdev, 6000, rng).mean(axis=0) - adp).mean()
    assert fixed < raw / 2
