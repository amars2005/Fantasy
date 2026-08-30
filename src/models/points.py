"""Gradient-boosted projection of season fantasy points.

Trained with an expanding window: to predict season Y we use only seasons before
Y, which is the only honest way to measure whether this would have helped on a
draft day that had already happened.

The model predicts points-per-game and games-played separately rather than season
totals directly. Those are different questions -- how good is he, and how much
will he play -- with different drivers, and separating them stops a durable
mediocre player and a brilliant fragile one from looking identical.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

FEATURES = [
    # prior production and opportunity
    "ppg_lag1", "points_lag1", "games_lag1", "targets_pg_lag1", "carries_pg_lag1",
    "touches_pg_lag1", "target_share_lag1", "air_yards_share_lag1", "wopr_lag1",
    "rec_yards_lag1", "rush_yards_lag1", "exp_ppg_lag1", "points_oe_lag1",
    "ppg_lag2", "games_lag2", "touches_pg_lag2", "target_share_lag2",
    "wopr_lag2", "exp_ppg_lag2", "points_oe_lag2",
    # who the player is
    "age", "years_exp", "draft_round", "draft_overall",
    # what the team has committed to him. Cap share rather than raw dollars:
    # dollar amounts trend with the cap, which lets the model infer the season.
    "contract_cap_pct", "contract_years_left", "is_contract_year",
    "years_into_contract",
    # situation
    "vacated_targets", "vacated_carries", "vegas_implied_ppg",
    "changed_team", "is_rookie",
    # role, from the preseason depth chart -- published before week one
    "depth_rank", "is_starter",
    # durability, from last season's injury reports
    "inj_weeks_out_lag1", "inj_weeks_questionable_lag1",
    "inj_weeks_dnp_lag1", "inj_weeks_on_report_lag1",
    # athletic profile (mostly for rookies, null for most veterans)
    "forty", "vertical", "broad_jump", "cone", "shuttle", "combine_wt",
]

# College production. Null for undrafted players and for anyone whose college
# years predate the data, so these live behind a flag: they are the only
# production evidence a rookie has, and pure noise for an established veteran.
COLLEGE_FEATURES = [
    "pre_draft_grade", "pre_draft_pos_ranking",
    "usage_overall_final", "usage_pass_final", "usage_rush_final",
    "cfb_rec_rec_final", "cfb_rec_yds_final", "cfb_rec_td_final",
    "cfb_rus_car_final", "cfb_rus_yds_final", "cfb_rus_td_final",
    "cfb_usage_peak", "cfb_rec_yds_peak", "cfb_rus_yds_peak", "cfb_seasons",
]

PARAMS = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.04,
    num_leaves=15,
    min_data_in_leaf=30,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbosity=-1,
    num_threads=4,
    # Bagging and feature sampling are stochastic. Without a fixed seed the
    # run-to-run spread is about the same size as the effects being measured,
    # which makes every before/after comparison meaningless.
    seed=0,
    bagging_seed=0,
    feature_fraction_seed=0,
    deterministic=True,
)
N_ROUNDS = 350

# ADP is public before any draft, so using it is not leakage. Keeping it optional
# lets us measure the thing that actually matters: whether the model adds
# anything *on top of* the market, rather than whether it can replace it.
MARKET_FEATURES = ["adp", "adp_stdev"]


DEPTH_FEATURES = ["depth_rank", "is_starter"]
INJURY_FEATURES = [
    "inj_weeks_out_lag1", "inj_weeks_questionable_lag1",
    "inj_weeks_dnp_lag1", "inj_weeks_on_report_lag1",
]


def _matrix(df: pl.DataFrame, use_market: bool = False,
            use_college: bool = False,
            drop: tuple[str, ...] = ()) -> tuple[np.ndarray, list[str]]:
    """Feature matrix, built directly from polars to avoid a pandas copy.

    `drop` removes named features, which is what makes honest ablation possible:
    the same code path with and without a feature group.
    """
    feature_list = (
        FEATURES
        + (MARKET_FEATURES if use_market else [])
        + (COLLEGE_FEATURES if use_college else [])
    )
    feature_list = [f for f in feature_list if f not in drop]
    cols = [c for c in feature_list if c in df.columns]
    blocks = [
        df.select(pl.col(c).cast(pl.Float64)).to_series().to_numpy().astype(np.float32)
        for c in cols
    ]
    names = list(cols)
    # Position is the one categorical that matters; one-hot keeps it explicit.
    pos = df["pos"].to_numpy()
    for p in ("QB", "RB", "WR", "TE"):
        blocks.append((pos == p).astype(np.float32))
        names.append(f"pos_{p}")
    return np.column_stack(blocks), names


def train(train_df: pl.DataFrame, target: str, params: dict | None = None,
          use_market: bool = False, use_college: bool = False,
          drop: tuple[str, ...] = ()) -> lgb.Booster:
    x, names = _matrix(train_df, use_market, use_college, drop)
    y = train_df[target].to_numpy().astype(float)
    dataset = lgb.Dataset(x, label=y, feature_name=names)
    return lgb.train(params or PARAMS, dataset, num_boost_round=N_ROUNDS)


def predict_points(
    train_df: pl.DataFrame, score_df: pl.DataFrame, use_market: bool = False
) -> tuple[np.ndarray, dict[str, lgb.Booster]]:
    """Predict season points as (points per game) x (games played)."""
    fit = train_df.filter(pl.col("y_games") > 0).with_columns(
        (pl.col("y_points") / pl.col("y_games")).alias("y_ppg")
    )
    ppg_model = train(fit, "y_ppg", use_market=use_market)

    # Games is fitted on everyone, including the zeros: not playing is an outcome.
    games_model = train(train_df, "y_games", use_market=use_market)

    x, _ = _matrix(score_df, use_market)
    ppg = np.clip(ppg_model.predict(x), 0, None)
    games = np.clip(games_model.predict(x), 0, 17)
    return ppg * games, {"ppg": ppg_model, "games": games_model}


# --- compact model ----------------------------------------------------------
# With roughly 1,400 decision-relevant rows, a 60-feature booster is mostly
# variance. This is the short list the importance rankings actually support,
# fitted with a linear model so the estimator cannot invent structure the sample
# size does not license.
COMPACT_FEATURES = [
    "adp", "adp_stdev",          # the market, and how much it disagrees with itself
    "exp_ppg_lag1",              # opportunity-based expected points
    "ppg_lag1",                  # prior production
    "depth_rank",                # role
    "contract_cap_pct",          # what the team has invested
    "age",
    "draft_overall",
]


def rank_target(df: pl.DataFrame, points_col: str = "y_points") -> np.ndarray:
    """Outcome expressed as a within-position-season percentile.

    The model is graded on within-position rank correlation but trained on
    squared error in points, which is a mismatch: squared error spends its
    capacity on the few enormous seasons rather than on getting the ordering
    right. Training on the percentile optimises what is actually measured.
    """
    ranked = df.with_columns(
        (
            pl.col(points_col).rank("average").over(["season", "pos"])
            / pl.len().over(["season", "pos"])
        ).alias("_pct")
    )
    return ranked["_pct"].to_numpy().astype(float)


def ridge_residual_model(
    train_df: pl.DataFrame, target: str, alpha: float = 10.0
):
    """Fit a regularised linear model on the compact feature set.

    Returns a callable that scores a frame, with imputation and scaling baked in
    so the caller cannot accidentally apply them inconsistently.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def matrix(df: pl.DataFrame) -> np.ndarray:
        blocks = [
            df.select(pl.col(c).cast(pl.Float64)).to_series().to_numpy().astype(float)
            if c in df.columns else np.zeros(df.height)
            for c in COMPACT_FEATURES
        ]
        pos = df["pos"].to_numpy()
        for p in ("QB", "RB", "WR", "TE"):
            blocks.append((pos == p).astype(float))
        return np.column_stack(blocks)

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
    )
    pipeline.fit(matrix(train_df), train_df[target].to_numpy().astype(float))
    return lambda df: pipeline.predict(matrix(df))


def importance(model: lgb.Booster, top: int = 20) -> list[tuple[str, float]]:
    gains = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    ranked = sorted(zip(names, gains), key=lambda kv: -kv[1])
    total = sum(gains) or 1.0
    return [(n, 100 * g / total) for n, g in ranked[:top]]
