# UAE Car Valuation Lab

An end-to-end data science project estimating **historical UAE used-car listing asking prices** from a public 2026 snapshot. It includes source documentation, a leakage-aware train/calibration/test workflow, a make/model median baseline, CatBoost model selection, split-conformal prediction intervals, a Streamlit exploration app, and exported evaluation evidence.

> This estimates listed asking prices in one snapshot. It does not predict completed sale prices or today's market value. A random holdout measures interpolation within this snapshot, not future-market performance.

## Project structure

- `fetch_data.py` downloads the source snapshot locally (the CSV is not redistributed here).
- `train.py` validates/cleans data, selects CatBoost settings on a validation subset, fits the selected model, calibrates a 90% interval on a separate split, and evaluates once on a held-out test set.
- `app.py` serves the interactive estimator, evaluation metrics, market distribution, and model feature importance.
- `data/README.md` records provenance and the license limitation.
- `artifacts/results.json` and `artifacts/test_predictions.csv` hold the exact metrics and test predictions produced by the run; `artifacts/price_model.cbm` is the local model artifact.

## Run locally

Python 3.9+ is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_data.py
python train.py
streamlit run app.py
```

The first model run benchmarks six CatBoost configurations and may take a few minutes. It uses seed 42. In addition to removing exact duplicates, it follows the source notebook's documented data-quality screen: asking prices below AED 5,000 are described there as wrong/fake and excluded. This leaves 7,296 eligible listings from the 9,953-row snapshot. The rule is recorded in `artifacts/results.json`; it does not remove ordinary high-price or mileage tails. `vehicle_age_years` is excluded because it is redundant with model year in this snapshot. Address, location labels below city, and coordinates are excluded to reduce identifier-like signals and improve generalization.

## Evaluation design

- 60% fit, 20% calibration, and 20% final test rows, using fixed seed 42.
- Candidate model settings are ranked on a validation slice inside the fit set; the test set is not used for tuning.
- Target is `log1p(price)`; outputs and errors are shown in AED.
- Baseline predicts the fit-set median price for each make/model, falling back to the overall fit-set median.
- Reported metrics: MAE, median absolute error, MAPE, median absolute percentage error, WAPE, RMSE, R², fraction within 10%/20% of asking price, and empirical held-out coverage of the nominal 90% split-conformal interval.
- Feature importance is predictive attribution, not causal explanation.

## Snapshot result

On the screened snapshot, the 1,460-row held-out test set produced **AED 33,355 MAE**, **AED 12,914 median absolute error**, **20.7% MAPE**, **14.5% median absolute percentage error**, and **64.0% of estimates within 20%** of the listed price. The model achieved R² 0.849. The training-set make/model median baseline had AED 76,590 MAE, so the model's MAE was **56.5% lower** on this same test set. The global 90% conformal interval covered 91.0% of the test set with a radius of AED 81,852 each way. Price-band interval radii and held-out coverage were AED 14,446 (89.5%, n=342) at or below AED 50k; AED 26,586 (89.2%, n=378) for AED 50–100k; AED 61,866 (90.7%, n=461) for AED 100–250k; AED 156,351 (90.4%, n=197) for AED 250–500k; and AED 258,947 (85.4%, n=82) above AED 500k. High-value and rare-vehicle estimates remain less certain.

These are held-out listing-price results after the documented below-AED-5,000 quality screen. They are not evidence of realized transaction-price accuracy or today's market performance. Exact metrics, split details, snapshot hash, candidate validation scores, and segment breakdowns are in `artifacts/results.json`.

## Reproducibility and data terms

Read [`data/README.md`](data/README.md) before downloading or republishing data. Upstream currently provides no explicit license in its repository README; the CSV is therefore kept out of version control. The source URL is on a mutable branch, so `train.py` saves the input SHA-256 in `artifacts/results.json` to identify the exact snapshot used. Re-running after an upstream file changes can produce different scores.

## Limitations and responsible use

The source gives no listing collection timestamps, outcomes, or detailed sampling methodology. A random split can contain similar vehicles across fit and test; it does not simulate a future-time deployment. Data is concentrated in Dubai and some makes/models are sparse. Price-band calibration gives group-level coverage under exchangeability, not an individual guarantee or protection against changed market conditions. The >AED 500k calibration group has only 55 rows. Prices can reflect errors, negotiation strategy, condition, accident history, service history, and other omitted information. Treat estimates as exploratory decision support and verify against current comparable listings and a vehicle inspection.
