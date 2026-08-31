"""Model-layer guards.

The determinism test is the important one. LightGBM's bagging and feature
sampling are stochastic, and on this dataset the run-to-run spread in measured
edge is about 0.007 -- larger than most of the feature effects being tested.
Without fixed seeds, every before/after comparison silently measures noise, and
the failure is invisible because each individual run looks fine.
"""

import numpy as np
import polars as pl
import pytest

from src.models.points import PARAMS, _matrix, rank_target, train


def _frame(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "pos": rng.choice(["QB", "RB", "WR", "TE"], n).tolist(),
        "season": rng.choice([2022, 2023], n).tolist(),
        "adp": rng.uniform(1, 200, n),
        "adp_stdev": rng.uniform(1, 30, n),
        "ppg_lag1": rng.uniform(0, 25, n),
        "exp_ppg_lag1": rng.uniform(0, 25, n),
        "depth_rank": rng.integers(1, 5, n).astype(float),
        "age": rng.uniform(21, 35, n),
        "y_points": rng.uniform(0, 350, n),
    })


def test_training_is_deterministic_across_runs():
    """Two fits on identical data must produce identical predictions."""
    df = _frame()
    x, _ = _matrix(df, use_market=True)
    a = train(df, "y_points", use_market=True).predict(x)
    b = train(df, "y_points", use_market=True).predict(x)
    assert np.allclose(a, b), "model output varies between identical runs"


def test_seed_is_pinned_in_params():
    """Guards the specific bug: unseeded bagging made comparisons meaningless."""
    for key in ("seed", "bagging_seed", "feature_fraction_seed"):
        assert key in PARAMS, f"{key} must be pinned for reproducible comparisons"


def test_different_seeds_do_change_output():
    """Sanity check that the seed is actually wired through, not ignored."""
    df = _frame()
    x, _ = _matrix(df, use_market=True)
    p0 = dict(PARAMS); p0.update(seed=0, bagging_seed=0, feature_fraction_seed=0)
    p1 = dict(PARAMS); p1.update(seed=99, bagging_seed=99, feature_fraction_seed=99)
    a = train(df, "y_points", params=p0, use_market=True).predict(x)
    b = train(df, "y_points", params=p1, use_market=True).predict(x)
    assert not np.allclose(a, b), "seed appears to have no effect"


def test_rank_target_is_a_within_group_percentile():
    df = _frame()
    pct = rank_target(df)
    assert pct.min() > 0.0 and pct.max() <= 1.0
    # Ordering within a season-position group must match the points ordering.
    check = df.with_columns(pl.Series("pct", pct)).filter(
        (pl.col("season") == 2022) & (pl.col("pos") == "WR")
    )
    pts = check["y_points"].to_numpy()
    pcts = check["pct"].to_numpy()
    assert np.all(np.argsort(pts) == np.argsort(pcts))


def test_dropping_features_changes_the_matrix_width():
    """Ablation depends on `drop` actually removing columns."""
    df = _frame()
    full, names_full = _matrix(df, use_market=True)
    less, names_less = _matrix(df, use_market=True, drop=("adp", "adp_stdev"))
    assert full.shape[1] - less.shape[1] == 2
    assert "adp" in names_full and "adp" not in names_less


def test_extra_features_can_be_added_for_a_fair_test():
    """A candidate feature must be testable on the same path as a shipping one.

    Without this, "we tried adding it" runs through different code than "we
    tried removing it", and the two answers are not comparable.
    """
    df = _frame().with_columns(pl.lit(1.0).alias("role_change"))
    base, names_base = _matrix(df)
    more, names_more = _matrix(df, extra=("role_change",))
    assert more.shape[1] - base.shape[1] == 1
    assert "role_change" in names_more and "role_change" not in names_base
