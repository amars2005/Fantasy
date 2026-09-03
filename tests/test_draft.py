"""Draft layer: snake order, lineup maths, replacement level, simulator calibration."""

import numpy as np
import polars as pl
import pytest

from src.draft.replacement import allocate_flex, replacement_levels
from src.draft.sim_draft import calibrate, simulate_draft_orders, snake_picks
from src.draft.vona import (
    bench_model,
    marginal_value,
    optimal_lineup_points,
    promotion_weights,
)

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


# --- bench value ------------------------------------------------------------
# Every starting slot filled, so anything added lands on the bench.
STARTED = [
    _p("QB", 320), _p("RB", 260), _p("RB", 257), _p("WR", 250), _p("WR", 247),
    _p("TE", 220), _p("RB", 245), _p("K", 140), _p("DST", 110),
]


def _bench():
    return bench_model(_board(), LEAGUE)


def _bench_value(pos, pts, bench):
    return (
        optimal_lineup_points(STARTED + [_p(pos, pts)], LEAGUE, bench)
        - optimal_lineup_points(STARTED, LEAGUE, bench)
    )


def test_promotion_weights_split_a_position_s_cover():
    """P(at least k slots open) sums back to the expected number open."""
    for vacancies in (0.2, 0.48, 0.6):
        weights = promotion_weights(vacancies, 40)
        assert sum(weights) == pytest.approx(vacancies, abs=1e-9)
        assert weights[:6] == sorted(weights[:6], reverse=True)
        assert len(set(weights[:6])) == 6


def test_backup_is_priced_over_what_you_could_stream():
    bench = _bench()
    qb2 = 300.0
    weight = promotion_weights(bench["QB"].vacancies, 1)[0]
    assert _bench_value("QB", qb2, bench) == pytest.approx(
        weight * (qb2 - bench["QB"].baseline)
    )
    # A small fraction of what a flat share of his raw points used to pay him.
    assert _bench_value("QB", qb2, bench) < 0.5 * 0.20 * qb2


def test_bench_receiver_beats_a_replaceable_quarterback():
    """The complaint this exists to answer: a QB2 outranking every bench body.

    Both beat their own replacement level by the same margin, so all that
    separates them is how many starting slots they stand behind.
    """
    bench = _bench()
    edge = 20.0
    qb2 = bench["QB"].baseline + edge
    wr = bench["WR"].baseline + edge
    assert _bench_value("WR", wr, bench) > _bench_value("QB", qb2, bench)

    # Under the flat weight the ordering was the other way round, on nothing
    # but the quarterback's bigger raw total.
    assert _bench_value("QB", qb2, None) > _bench_value("WR", wr, None)


def test_third_quarterback_is_worth_almost_nothing():
    bench = _bench()
    two = STARTED + [_p("QB", 300)]
    base = optimal_lineup_points(two, LEAGUE, bench)
    qb3 = optimal_lineup_points(two + [_p("QB", 290)], LEAGUE, bench) - base
    wr = optimal_lineup_points(two + [_p("WR", 200)], LEAGUE, bench) - base
    assert qb3 < 1.0
    assert wr > 10 * qb3


def test_bench_player_below_the_waiver_wire_is_worth_zero():
    bench = _bench()
    assert _bench_value("WR", bench["WR"].baseline - 20, bench) == pytest.approx(0.0)


def test_marginal_value_prices_the_bench_off_the_available_board():
    """`marginal_value` derives the model itself; nobody has to remember to."""
    board = _board().with_columns(
        pl.col("name").alias("player_id"), pl.lit(1.0).alias("adp")
    )
    marginal = marginal_value(board, STARTED, LEAGUE)
    best = {
        pos: marginal.filter(pl.col("pos") == pos)["marginal"].max()
        for pos in ("QB", "RB", "WR", "TE")
    }
    # Nothing left at quarterback is a starter, so the best available QB is
    # worth less than the best available back or receiver, both of whom are
    # also backups here.
    assert best["QB"] < best["RB"]
    assert best["QB"] < best["WR"]


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
