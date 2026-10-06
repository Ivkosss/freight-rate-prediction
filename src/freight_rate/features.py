"""Feature engineering.

The model is additive in log space:

    log(rate) = lane_part(lane & load features) + market_part(date & market features)

so features are split into two groups.  Lanes are described by *coordinates*,
not city IDs: the validation set contains 8 cities that never appear in
train_test.csv (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo,
Norfolk, San Diego), so city one-hot / target encodings would not generalise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]
REFERENCE_DATE = pd.Timestamp("2025-01-01")
QUARTER_END_WINDOW = 30  # days; quarter-end tightening ramp (chosen by time-series CV: 7/14/21/30/45)

LANE_FEATURES = [
    "distance", "log_distance", "equipment", "weight", "weight_missing",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "circuity", "delta_lat", "delta_lon",
]
MARKET_TREE_FEATURES = [
    "market_index", "quote_signal", "day_market_index", "day_quote_signal",
    "quote_signal_rel", "day_of_week", "days_to_quarter_end",
]
MARKET_LINEAR_FEATURES = ["day_market_index", "day_quote_signal", "quote_signal_rel", "quarter_end_ramp", "days"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def days_to_quarter_end(dates: pd.Series) -> pd.Series:
    quarter_end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
    return (quarter_end - dates).dt.days


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=frame.index)

    # --- lane / load ---------------------------------------------------------
    X["distance"] = frame["distance"]
    X["log_distance"] = np.log(frame["distance"])
    X["equipment"] = pd.Categorical(frame["equipment"], categories=EQUIPMENT).codes
    X["weight"] = frame["weight"]
    X["weight_missing"] = frame["weight_missing"]
    for side in ("pickup", "delivery"):
        X[f"{side}_lat"] = frame[f"{side}_lat"]
        X[f"{side}_lon"] = frame[f"{side}_lon"]
    straight = haversine_miles(frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"])
    X["circuity"] = frame["distance"] / np.maximum(straight, 1.0)
    X["delta_lat"] = frame["delivery_lat"] - frame["pickup_lat"]
    X["delta_lon"] = frame["delivery_lon"] - frame["pickup_lon"]

    # --- market / calendar ---------------------------------------------------
    X["market_index"] = frame["market_index"]
    X["quote_signal"] = frame["quote_signal"]
    X["day_market_index"] = frame["day_market_index"]
    X["day_quote_signal"] = frame["day_quote_signal"]
    X["quote_signal_rel"] = frame["quote_signal"] - frame["day_quote_signal"]
    X["day_of_week"] = frame["date"].dt.dayofweek
    X["days_to_quarter_end"] = days_to_quarter_end(frame["date"])
    X["quarter_end_ramp"] = np.clip(QUARTER_END_WINDOW - X["days_to_quarter_end"], 0, None)
    X["days"] = (frame["date"] - REFERENCE_DATE).dt.days
    return X
