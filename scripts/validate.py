"""Validation of the modelling approach on train_test.csv.

Split strategy
--------------
validation.csv covers Nov-Dec 2025, i.e. it lies strictly *after* the
labelled data (Jan-Oct 2025).  A random K-fold split would let the model see
loads from the same days/market regime as the test rows and would overstate
accuracy.  The primary scheme is therefore **rolling-origin (expanding
window) time-series CV**:

    fold 1: train Jan-Jun  -> test Jul
    fold 2: train Jan-Jul  -> test Aug
    fold 3: train Jan-Aug  -> test Sep
    fold 4: train Jan-Sep  -> test Oct
    gap   : train Jan-Aug  -> test Oct (one-month gap, mimics predicting Dec from data ending Oct)

Two secondary checks:
    * random 5-fold CV - reported only to show how optimistic a leaky split is;
    * unseen-city CV - 8 cities held out entirely (pickup or delivery), because
      8 validation cities never occur in training.

Metrics are computed on test rows whose label is not a flagged corruption
(``clean``) and on all rows (``raw``).

Usage:  python scripts/validate.py
Writes: reports/validation_results.csv, reports/validation_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight_rate.data import attach_daily, clean, daily_market, read  # noqa: E402
from freight_rate.features import make_features  # noqa: E402
from freight_rate.model import AdditiveGBM, HybridModel, PlainGBM, RateModel, flag_label_outliers, metrics  # noqa: E402

REPORTS = ROOT / "reports"


# ----------------------------------------------------------------------------- baselines
class LaneBaseline:
    """Median rate-per-mile by equipment x distance band (what a broker's rate sheet does)."""

    bands = [0, 250, 500, 750, 1000, 1500, 2000, 2500, 10_000]

    def fit(self, frame):
        f = frame.assign(rpm=frame.posted_rate / frame.distance, band=pd.cut(frame.distance, self.bands))
        self.table_ = f.groupby(["equipment", "band"], observed=True).rpm.median()
        return self

    def predict(self, frame):
        band = pd.cut(frame.distance, self.bands)
        keys = list(zip(frame.equipment, band))
        return self.table_.reindex(keys).to_numpy() * frame.distance.to_numpy()


class LogLinear:
    """Ridge regression on log(rate) with the same engineered features (one-hot equipment)."""

    def fit(self, frame):
        X = self._x(frame)
        self.cols_ = X.columns
        self.mu_, self.sd_ = X.mean(), X.std().replace(0, 1)
        self.model_ = Ridge(alpha=1.0).fit((X - self.mu_) / self.sd_, np.log(frame.posted_rate))
        return self

    def _x(self, frame):
        X = make_features(frame).drop(columns="equipment")
        X["weight"] = X["weight"].fillna(X["weight"].median())
        for e in ("Reefer", "Flatbed"):
            X[e] = (frame.equipment == e).astype(int)
        return X

    def predict(self, frame):
        X = self._x(frame)[self.cols_]
        return np.exp(self.model_.predict((X - self.mu_) / self.sd_))


CANDIDATES = {
    "baseline_lane_rpm": lambda: LaneBaseline(),
    "ridge_log": lambda: LogLinear(),
    "gbm_plain": lambda: PlainGBM(),
    "hybrid_gbm_linear": lambda: HybridModel(),
    "additive_gbm_no_trend": lambda: AdditiveGBM(),
    "final_ensemble": lambda: RateModel(),
}


# ----------------------------------------------------------------------------- splits
def time_folds(frame):
    month = frame.date.dt.month
    for test_month in (7, 8, 9, 10):
        yield f"time_m{test_month:02d}", month < test_month, month == test_month
    yield "time_gap_oct", month <= 8, month == 10


def random_folds(frame):
    for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(frame)):
        a = np.zeros(len(frame), bool)
        a[tr] = True
        yield f"random_k{i}", a, ~a


def city_folds(frame):
    cities = np.array(sorted(frame.pickup.unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(cities)
    for i, held in enumerate(np.array_split(cities, 8)):
        touches = frame.pickup.isin(held) | frame.delivery.isin(held)
        yield f"city_k{i}", ~touches.to_numpy(), touches.to_numpy()


def run():
    raw = read("train_test.csv")
    frame = clean(raw)
    frame = attach_daily(frame, daily_market(frame))
    frame["is_outlier"] = flag_label_outliers(frame).to_numpy()
    print(f"flagged label outliers: {frame.is_outlier.sum()} ({frame.is_outlier.mean():.2%})")

    rows = []
    schemes = {"time": time_folds, "random": random_folds, "unseen_city": city_folds}
    for scheme, folds in schemes.items():
        names = CANDIDATES if scheme == "time" else ["baseline_lane_rpm", "final_ensemble"]
        for fold, tr, te in folds(frame):
            train = frame[tr & ~frame.is_outlier.to_numpy()]
            test = frame[te]
            for name in names:
                pred = CANDIDATES[name]().fit(train).predict(test)
                ok = ~test.is_outlier.to_numpy()
                rows.append({"scheme": scheme, "fold": fold, "model": name,
                             **{f"clean_{k}": v for k, v in metrics(test.posted_rate[ok], pred[ok]).items()},
                             **{f"raw_{k}": v for k, v in metrics(test.posted_rate, pred).items()}})
                print(scheme, fold, name, round(rows[-1]["clean_MAE"], 2), round(rows[-1]["clean_MAPE_%"], 3))

    # ablation: does dropping corrupted labels from TRAINING help? (time folds, final model)
    for fold, tr, te in time_folds(frame):
        test = frame[te]
        ok = ~test.is_outlier.to_numpy()
        pred = RateModel().fit(frame[tr]).predict(test)
        rows.append({"scheme": "time", "fold": fold, "model": "final_keep_outliers",
                     **{f"clean_{k}": v for k, v in metrics(test.posted_rate[ok], pred[ok]).items()},
                     **{f"raw_{k}": v for k, v in metrics(test.posted_rate, pred).items()}})

    results = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    results.to_csv(REPORTS / "validation_results.csv", index=False)
    cols = ["clean_MAE", "clean_RMSE", "clean_MAPE_%", "clean_bias_%", "raw_MAE"]
    summary = results.groupby(["scheme", "model"])[cols].mean().round(3)
    print(summary.to_string())
    (REPORTS / "validation_summary.json").write_text(
        json.dumps({f"{s}|{m}": r for (s, m), r in summary.to_dict("index").items()}, indent=2)
    )


if __name__ == "__main__":
    run()
