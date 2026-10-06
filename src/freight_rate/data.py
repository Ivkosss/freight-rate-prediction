"""Loading and cleaning of the freight-rate data.

Every cleaning rule here was motivated by something found during EDA
(see scripts/eda.py and the report):

* ``weight`` has negative values (sign-entry errors, ~0.6%) -> absolute value.
* ``weight`` has missing values (~0.6%) -> left as NaN + ``weight_missing`` flag;
  the gradient-boosted model routes NaN natively.  (47,500 lb is the legal
  max and behaves like real heavy loads, so it is kept as-is.)
* ``market_index`` is missing for ~0.8% of rows.  It is a *date-level* market
  signal (within-date std ~0.025 vs. across-date std ~0.17), so a missing value
  is imputed with the median of the other loads on the same date.
* ``posted_rate`` contains corrupted labels (~1.4%): rates 2-6x or 0.15-0.5x
  what the lane/equipment/distance implies.  They are detected out-of-fold in
  ``model.flag_label_outliers`` and removed from training only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TARGET = "posted_rate"


def read(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / name)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that is safe to apply to train AND inference data."""
    out = frame.copy()
    out["weight_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)

    out["market_index_missing"] = out["market_index"].isna().astype(int)
    by_date = out.groupby("date")["market_index"].transform("median")
    out["market_index"] = out["market_index"].fillna(by_date)
    return out


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) lookup built from pickup and delivery columns."""
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            parts.append(
                frame[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
            )
    coords = pd.concat(parts).groupby("city")[["lat", "lon"]].median()
    return coords


def daily_market(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date market aggregates (mean market_index / quote_signal over all loads that day).

    These are market-level, label-free signals: on any given day they are
    known from that day's quotes, so they are legitimate at prediction time.
    """
    data = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    daily = data.groupby("date").agg(
        day_market_index=("market_index", "mean"),
        day_quote_signal=("quote_signal", "mean"),
        day_loads=("quote_signal", "size"),
    )
    return daily


def attach_daily(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=[c for c in daily.columns if c in frame.columns])
    return out.join(daily, on="date")


def build_december_frame(december: pd.DataFrame, coords: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Turn the 7-column December chart template into a full model input.

    The template only carries lane, distance, equipment, weight and date.  The
    missing columns are filled the same way they would be in production:
    coordinates from the city lookup, and market_index / quote_signal from the
    observed market on that date (daily mean across all validation loads).
    """
    out = december.copy()
    out["date"] = pd.to_datetime(out["date"])
    for side in ("pickup", "delivery"):
        out[f"{side}_lat"] = out[side].map(coords["lat"])
        out[f"{side}_lon"] = out[side].map(coords["lon"])
    out = out.join(daily, on="date")
    out["market_index"] = out["day_market_index"]
    out["quote_signal"] = out["day_quote_signal"]
    missing = out[["pickup_lat", "delivery_lat", "market_index"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"December rows without coordinates/market data: {out.loc[missing, 'date'].tolist()}")
    return clean(out)
