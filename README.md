# Freight Rate Prediction – Spotter ML Engineer Assessment

Predicts `posted_rate` for truck loads. Trained on `data/train_test.csv` (Jan–Oct 2025),
predicts `data/validation.csv` (Nov–Dec 2025) and the fixed December chart lane.

**Result (rolling-origin time-series CV, 5 forward folds, clean labels):**
MAPE **1.71%**, MAE **$37.8**, bias **+0.0%**. A lane rate-sheet baseline gets 3.74% and an
unconstrained gradient-boosting model gets 2.72%.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/train_predict.py     # ~40 s: trains the final model, writes the deliverables
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Optional (used for the report):

```bash
python scripts/validate.py          # ~7 min: time-series / random / unseen-city CV for all candidate models
python scripts/eda.py               # data-quality audit + figures in reports/figures/
```

Tested with Python 3.11+, pandas 2.3, numpy 2.x, scikit-learn 1.9 (any scikit-learn ≥ 1.4 works).

## Outputs

| File | What it is |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 validation loads |
| `data/december_chart_inputs.csv` | the 31 December rows with `predicted_rate` filled |
| `scorer_results/candidate_december.png` | chart produced by the provided `score.py` |
| `reports/validation_results.csv` | per-fold metrics for every model and split scheme |
| `reports/data_quality.json` | data-quality audit numbers |
| `reports/final_model.json` | trend and market coefficients of the final model, December market inputs |
| `reports/flagged_label_outliers.csv` | training rows whose labels were treated as corrupted |
| `reports/Freight_Rate_Report.docx` / `.pdf` | written report |

## Layout

```
src/freight_rate/
  data.py       loading, cleaning, daily market aggregates, December input builder
  features.py   lane/load features and market/calendar features
  model.py      HybridModel, AdditiveGBM, final RateModel ensemble, outlier detection, metrics
scripts/
  train_predict.py   final training + predictions
  validate.py        split strategy and model comparison
  eda.py             data-quality audit and figures
score.py        provided scorer (unchanged)
```

## Approach in brief

**Data-quality issues and how they were handled**

| Issue | Size | Handling |
|---|---|---|
| Corrupted `posted_rate` labels (2–6× too high or 0.15–0.5× too low) | 677 rows (1.4%) | flagged out-of-fold (label more than 1.5× from the prediction), removed from training. The robust absolute-error loss handles anything the filter misses |
| Negative `weight` (sign error) | 292 train / 145 valid | absolute value (the residuals of the corrected rows match the clean rows) |
| Missing `weight` | 300 / 165 | kept as NaN (the GBM routes it natively), plus a flag |
| `weight` piled up at 47,500 lb | 1,191 / 297 | kept: this is the legal maximum, and the rows price like real heavy loads |
| Missing `market_index` | 374 / 249 | it is a date-level signal (std within a date 0.025, across dates 0.17), so it is imputed with the median of that date |
| 8 validation cities never seen in training (1,447 rows) | — | lanes are described by coordinates, not city IDs |

**Key findings**
* Rate ≈ distance × per-mile rate. The per-mile rate falls with distance, and the equipment premium is Reefer +12%, Flatbed +7% over Dry Van. Heavier loads cost more, by about +3% per 10k lb.
* `market_index` has a strong 7-day cycle (peak Thursday, trough Sunday) and multi-month regimes. Daily rates track it.
* After controlling for the market there is a steady **~+7%/year upward trend** and a **quarter-end ramp** over the last ~30 days of each quarter.
* `quote_signal` is mostly a date-level regime signal: it steps up in the last month of Q1–Q3 but not in Nov–Dec. Per load it carries little information.

**Split.** The validation set lies strictly *after* the labelled data, so the main scheme is a
rolling-origin time-series CV: train on Jan–Jun and test on Jul, … , train on Jan–Sep and test on Oct, plus one fold that trains on Jan–Aug and tests on Oct
(a one-month gap, like going from October to December). A random K-fold is reported only to show how optimistic it is.
An unseen-city CV (8 city groups held out) checks generalisation to new cities.

**Model.** Two additive models in log space are combined with a geometric mean:
1. *Hybrid*: gradient-boosted trees on lane/load features, plus a **linear** market component
   (daily market index, daily quote signal, quarter-end ramp, time trend), fit by backfitting.
2. *Additive GBM*: gradient-boosted trees with interaction constraints that keep lane features and market features in separate trees,
   fit on a de-trended target.

Tree models alone cannot extrapolate the trend, which gave a −2% bias on every forward fold. They also
learnt spurious equipment × market interactions from only ~300 distinct days. The linear trend and the interaction
constraints remove both problems.

**December chart.** The template has no coordinates, `market_index` or `quote_signal`. Coordinates come
from the city lookup. The market fields are the observed daily means over all validation loads on each date,
which are known on that date and use no labels. The chart therefore shows the weekly market cycle on top of
the trend and the quarter-end ramp.
