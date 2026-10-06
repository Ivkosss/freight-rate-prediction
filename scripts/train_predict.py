"""Train the final model on all labelled data and produce the deliverables.

Outputs
    validation_predictions.csv            load_id,predicted_rate for the 12,000 validation loads
    data/december_chart_inputs.csv        predicted_rate column filled (31 rows)
    models/rate_model.joblib              fitted model
    reports/final_model.json              outlier count, trend, linear market coefficients

Usage:  python scripts/train_predict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight_rate.data import (  # noqa: E402
    DATA_DIR, attach_daily, build_december_frame, city_coordinates, clean, daily_market, read,
)
from freight_rate.model import RateModel, flag_label_outliers  # noqa: E402


def main() -> None:
    train = clean(read("train_test.csv"))
    valid = clean(read("validation.csv"))

    # Daily market aggregates over every load we observe on a date (labels not used).
    daily = daily_market(train, valid)
    train = attach_daily(train, daily)
    valid = attach_daily(valid, daily)

    outliers = flag_label_outliers(train)
    fit_frame = train[~outliers]
    print(f"Training on {len(fit_frame):,} rows ({int(outliers.sum())} corrupted labels removed)")
    model = RateModel().fit(fit_frame)

    # ---- validation predictions -------------------------------------------------
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    pred = pd.Series(model.predict(valid), index=valid["load_id"])
    template["predicted_rate"] = template["load_id"].map(pred).round(2)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    template.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(template):,} rows)")

    # ---- December chart -----------------------------------------------------------
    december_path = DATA_DIR / "december_chart_inputs.csv"
    december = pd.read_csv(december_path)
    coords = city_coordinates(train, valid)
    dec_frame = build_december_frame(december.drop(columns="predicted_rate"), coords, daily)
    december["predicted_rate"] = np.round(model.predict(dec_frame), 2)
    december.to_csv(december_path, index=False)
    print(f"Wrote {december_path.relative_to(ROOT)}")
    print(december[["date", "predicted_rate"]].to_string(index=False))

    # ---- artefacts ------------------------------------------------------------------
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "rate_model.joblib")
    info = {
        "train_rows": int(len(train)),
        "label_outliers_removed": int(outliers.sum()),
        "trend_per_year_log": model.hybrid_.trend_per_day * 365,
        "linear_market_coefficients": model.hybrid_.coefficients(),
        "december_market_inputs": dec_frame[["date", "market_index", "quote_signal"]]
        .assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d"))
        .round(4)
        .to_dict("records"),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "final_model.json").write_text(json.dumps(info, indent=2))
    train.assign(is_outlier=outliers.to_numpy())[["load_id", "is_outlier"]].query("is_outlier").to_csv(
        ROOT / "reports" / "flagged_label_outliers.csv", index=False
    )


if __name__ == "__main__":
    main()
