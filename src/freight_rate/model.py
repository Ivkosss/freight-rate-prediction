"""Models, label-outlier detection and metrics.

Final model = geometric mean of two additive log-rate models that share one
linear time trend:

1. ``HybridModel``  (backfitting)
       log(rate) = GBM_lane(lane/load features) + Linear(market, quarter-end ramp, trend)
   The market part is linear so it extrapolates sensibly to unseen months
   (Nov-Dec) instead of being clamped to the last seen tree leaf.

2. ``AdditiveGBM``
       log(rate) - trend = GBM with interaction constraints {lane group} / {market group}
   A non-linear market part, but trees are not allowed to mix market and lane
   features.  Without this constraint the model learnt spurious
   "equipment premium x market level" interactions from only ~300 distinct days,
   which cost 2 MAPE points on the August time fold.

Why the explicit trend?  After controlling for market_index and quote_signal
there is a steady ~+7%/year drift in rates.  Tree models cannot extrapolate a
trend, which showed up as a consistent -2% bias on every forward-in-time fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .features import LANE_FEATURES, MARKET_LINEAR_FEATURES, MARKET_TREE_FEATURES, make_features

SEED = 42
OUTLIER_LOG_RATIO = np.log(1.5)  # |log(actual / expected)| above this => corrupted label

GBM_PARAMS = dict(
    learning_rate=0.05,
    max_iter=1500,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    loss="absolute_error",  # robust to label noise that survives the outlier filter
    random_state=SEED,
)


def make_gbm(columns: list[str], **overrides) -> HistGradientBoostingRegressor:
    params = {**GBM_PARAMS, **overrides}
    categorical = [i for i, c in enumerate(columns) if c == "equipment"]
    return HistGradientBoostingRegressor(categorical_features=categorical or None, **params)


# ----------------------------------------------------------------------------- members
@dataclass
class PlainGBM:
    """Unconstrained GBM on all features (no trend).  Kept as an ablation baseline."""

    def fit(self, frame):
        X = make_features(frame)[LANE_FEATURES + MARKET_TREE_FEATURES]
        self.model_ = make_gbm(list(X.columns)).fit(X, np.log(frame["posted_rate"]))
        return self

    def predict(self, frame):
        X = make_features(frame)[LANE_FEATURES + MARKET_TREE_FEATURES]
        return np.exp(self.model_.predict(X))


@dataclass
class HybridModel:
    """GBM on lane features + linear market/trend component, fit by backfitting."""

    iterations: int = 3

    def fit(self, frame):
        X = make_features(frame)
        y = np.log(frame["posted_rate"].to_numpy())
        lane, market = X[LANE_FEATURES], X[MARKET_LINEAR_FEATURES]
        lane_fit = np.zeros(len(y))
        for i in range(self.iterations):
            self.linear_ = LinearRegression().fit(market, y - lane_fit)
            offset = self.linear_.predict(market)
            last = i == self.iterations - 1
            self.gbm_ = make_gbm(LANE_FEATURES, max_iter=1500 if last else 800).fit(lane, y - offset)
            lane_fit = self.gbm_.predict(lane)
        return self

    @property
    def trend_per_day(self) -> float:
        return float(self.linear_.coef_[MARKET_LINEAR_FEATURES.index("days")])

    def coefficients(self) -> dict:
        return dict(zip(MARKET_LINEAR_FEATURES, map(float, self.linear_.coef_)))

    def predict(self, frame):
        X = make_features(frame)
        return np.exp(self.gbm_.predict(X[LANE_FEATURES]) + self.linear_.predict(X[MARKET_LINEAR_FEATURES]))


@dataclass
class AdditiveGBM:
    """GBM with lane/market interaction constraints on a de-trended target."""

    trend_per_day: float = 0.0

    def fit(self, frame):
        X = make_features(frame)
        cols = LANE_FEATURES + MARKET_TREE_FEATURES
        groups = [list(range(len(LANE_FEATURES))), list(range(len(LANE_FEATURES), len(cols)))]
        y = np.log(frame["posted_rate"].to_numpy()) - self.trend_per_day * X["days"].to_numpy()
        self.model_ = make_gbm(cols, interaction_cst=groups).fit(X[cols], y)
        return self

    def predict(self, frame):
        X = make_features(frame)
        cols = LANE_FEATURES + MARKET_TREE_FEATURES
        return np.exp(self.model_.predict(X[cols]) + self.trend_per_day * X["days"].to_numpy())


@dataclass
class RateModel:
    """Final model: geometric mean of HybridModel and AdditiveGBM (shared trend)."""

    def fit(self, frame):
        self.hybrid_ = HybridModel().fit(frame)
        self.additive_ = AdditiveGBM(trend_per_day=self.hybrid_.trend_per_day).fit(frame)
        return self

    def predict(self, frame):
        return np.sqrt(self.hybrid_.predict(frame) * self.additive_.predict(frame))


# ----------------------------------------------------------------------------- data quality
def flag_label_outliers(frame: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    """Out-of-fold detection of corrupted posted_rate labels.

    A model trained on the other folds predicts each row; rows whose label is
    more than 1.5x away (either direction) from that prediction are flagged.
    Corrupted labels are 2-6x (or 0.15-0.5x) off, far outside the ~2% normal
    pricing noise, so the threshold is not sensitive (every cut-off from 1.3x to
    2x flags the same 677 training rows).
    """
    X = make_features(frame)[LANE_FEATURES + MARKET_TREE_FEATURES]
    y = np.log(frame["posted_rate"].to_numpy())
    oof = np.zeros(len(frame))
    for train_idx, test_idx in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        model = make_gbm(list(X.columns), max_iter=500)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])
    return pd.Series(np.abs(y - oof) > OUTLIER_LOG_RATIO, index=frame.index)


def metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    err = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(100 * np.mean(np.abs(err) / actual)),
        "median_APE_%": float(100 * np.median(np.abs(err) / actual)),
        "bias_%": float(100 * (predicted.sum() / actual.sum() - 1)),
        "n": int(len(actual)),
    }
