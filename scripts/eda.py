"""Exploratory analysis and data-quality audit.

Writes figures to reports/figures/ and a data-quality summary to
reports/data_quality.json.  Run after scripts/validate.py if you also want the
validation-results figure.

Usage:  python scripts/eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight_rate.data import attach_daily, clean, daily_market, read  # noqa: E402
from freight_rate.features import EQUIPMENT  # noqa: E402

FIG = ROOT / "reports" / "figures"
INK, ACCENT, MUTED, GRID = "#1F2A30", "#064A56", "#8A9AA0", "#E3E8EA"
EQ_COLORS = {"Dry Van": "#064A56", "Reefer": "#D9822B", "Flatbed": "#6C8EBF"}
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    raw = read("train_test.csv")
    raw_valid = read("validation.csv")
    train = attach_daily(clean(raw), daily_market(clean(raw), clean(raw_valid)))
    train["rpm"] = train.posted_rate / train.distance

    # ---------------------------------------------------------------- data-quality audit
    unseen = sorted(set(raw_valid.pickup) - set(raw.pickup))
    within_date_std = raw.groupby("date").market_index.std().mean()
    across_date_std = raw.groupby("date").market_index.mean().std()
    flagged = pd.read_csv(ROOT / "reports" / "flagged_label_outliers.csv") if (
        ROOT / "reports" / "flagged_label_outliers.csv").exists() else None
    audit = {
        "rows": {"train_test": len(raw), "validation": len(raw_valid)},
        "date_range": {
            "train_test": [str(raw.date.min().date()), str(raw.date.max().date())],
            "validation": [str(raw_valid.date.min().date()), str(raw_valid.date.max().date())],
        },
        "weight_missing": {"train_test": int(raw.weight.isna().sum()), "validation": int(raw_valid.weight.isna().sum())},
        "weight_negative": {"train_test": int((raw.weight < 0).sum()), "validation": int((raw_valid.weight < 0).sum())},
        "weight_at_47500_cap": {"train_test": int((raw.weight == 47500).sum()), "validation": int((raw_valid.weight == 47500).sum())},
        "market_index_missing": {"train_test": int(raw.market_index.isna().sum()), "validation": int(raw_valid.market_index.isna().sum())},
        "market_index_within_date_std": round(float(within_date_std), 4),
        "market_index_across_date_std": round(float(across_date_std), 4),
        "label_outliers_flagged": None if flagged is None else int(len(flagged)),
        "duplicate_load_ids": int(raw.load_id.duplicated().sum()),
        "duplicate_feature_rows": int(raw.drop(columns="load_id").duplicated().sum()),
        "validation_cities_unseen_in_train": unseen,
        "validation_rows_touching_unseen_city": int((raw_valid.pickup.isin(unseen) | raw_valid.delivery.isin(unseen)).sum()),
    }
    (ROOT / "reports" / "data_quality.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

    # ---------------------------------------------------------------- 1. rate vs distance
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sample = train.sample(6000, random_state=0)
    for eq in EQUIPMENT:
        s = sample[sample.equipment == eq]
        ax.scatter(s.distance, s.rpm, s=4, alpha=0.35, color=EQ_COLORS[eq], label=eq, linewidths=0)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Rate per mile ($)")
    ax.set_title("Rate per mile falls with distance; Reefer > Flatbed > Dry Van")
    ax.legend(frameon=False, markerscale=4)
    save(fig, "01_rate_per_mile_vs_distance.png")

    # ---------------------------------------------------------------- 2. label outliers
    if flagged is not None:
        out = train.load_id.isin(flagged.load_id)
        med = train.groupby("equipment").rpm.transform("median")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.scatter(train.distance[~out], (train.rpm / med)[~out], s=3, alpha=0.15, color=MUTED, linewidths=0, label="kept")
        ax.scatter(train.distance[out], (train.rpm / med)[out], s=6, alpha=0.8, color="#C0392B", linewidths=0,
                   label=f"flagged corrupted label (n={out.sum()})")
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.5, 1, 2, 5])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Distance (miles)")
        ax.set_ylabel("Rate per mile / equipment median")
        ax.set_title("~1.4% of posted_rate labels are 2-6x too high or 0.15-0.5x too low")
        ax.legend(frameon=False, markerscale=3, loc="upper right")
        save(fig, "02_label_outliers.png")

    # ---------------------------------------------------------------- 3. weight issues
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(raw.weight.dropna(), bins=120, color=ACCENT)
    ax.set_xlabel("weight (lb), raw")
    ax.set_ylabel("loads")
    ax.set_title(f"weight: {audit['weight_negative']['train_test']} negative (sign errors), "
                 f"{audit['weight_missing']['train_test']} missing, spike at 47,500 lb legal max")
    save(fig, "03_weight_raw.png")

    # ---------------------------------------------------------------- 4. market over time
    daily = daily_market(clean(raw), clean(raw_valid))
    rate_daily = train[~train.load_id.isin(flagged.load_id if flagged is not None else [])]
    rate_daily = rate_daily.assign(lr=np.log(rate_daily.rpm)).groupby("date").lr.mean()
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True)
    axes[0].plot(daily.index, daily.day_market_index, color=ACCENT, lw=0.9)
    axes[0].plot(daily.index, daily.day_market_index.rolling(7, center=True).mean(), color=INK, lw=1.6)
    axes[0].set_title("Daily mean market_index (thin) and 7-day mean: strong weekly cycle + regimes")
    axes[1].plot(daily.index, daily.day_quote_signal, color="#D9822B", lw=1)
    axes[1].set_title("Daily mean quote_signal: steps up in the last month of Q1-Q3 (not in Dec)")
    axes[2].plot(rate_daily.index, rate_daily.values, color=INK, lw=1)
    axes[2].set_title("Daily mean log(rate per mile), labelled data")
    for a in axes:
        a.axvline(pd.Timestamp("2025-11-01"), color="#C0392B", ls="--", lw=1)
    axes[0].text(pd.Timestamp("2025-11-03"), daily.day_market_index.max() * 0.98, "validation\nNov-Dec", color="#C0392B", fontsize=8, va="top")
    save(fig, "04_market_over_time.png")

    # ---------------------------------------------------------------- 5. split diagram
    fig, ax = plt.subplots(figsize=(8, 2.6))
    folds = [("fold 1", 6, 7), ("fold 2", 7, 8), ("fold 3", 8, 9), ("fold 4", 9, 10), ("gap fold", 8, 10)]
    for i, (name, last_train, test) in enumerate(folds):
        y = len(folds) - i
        ax.barh(y, last_train, left=0.5, color=ACCENT, height=0.6)
        ax.barh(y, 1, left=test - 0.5, color="#D9822B", height=0.6)
        ax.text(0.3, y, name, ha="right", va="center", fontsize=8)
    ax.barh(0, 2, left=10.5, color="#C0392B", height=0.6)
    ax.text(0.3, 0, "final", ha="right", va="center", fontsize=8)
    ax.barh(0, 10, left=0.5, color=ACCENT, height=0.6)
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks([])
    ax.set_xlim(-1.2, 12.6)
    ax.grid(False)
    ax.set_title("Rolling-origin time-series CV (blue = train, orange = test); final model -> Nov-Dec (red)")
    save(fig, "05_time_split.png")

    # ---------------------------------------------------------------- 6. validation results
    results_path = ROOT / "reports" / "validation_results.csv"
    if results_path.exists():
        res = pd.read_csv(results_path)
        t = res[res.scheme == "time"].groupby("model")[["clean_MAPE_%", "clean_bias_%"]].mean()
        order = ["baseline_lane_rpm", "gbm_plain", "additive_gbm_no_trend", "ridge_log", "hybrid_gbm_linear", "final_ensemble"]
        t = t.reindex([o for o in order if o in t.index])
        fig, ax = plt.subplots(figsize=(7, 3.2))
        colors = [ACCENT if m == "final_ensemble" else MUTED for m in t.index]
        ax.barh(t.index, t["clean_MAPE_%"], color=colors)
        for y, (mape, bias) in enumerate(zip(t["clean_MAPE_%"], t["clean_bias_%"])):
            ax.text(mape + 0.05, y, f"{mape:.2f}%  (bias {bias:+.1f}%)", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, t["clean_MAPE_%"].max() * 1.45)
        ax.set_xlabel("MAPE on forward-in-time folds (%)")
        ax.set_title("Time-series CV: mean over 5 forward folds")
        save(fig, "06_time_cv_results.png")


if __name__ == "__main__":
    main()
