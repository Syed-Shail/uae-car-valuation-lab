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

The first model run benchmarks three CatBoost configurations and may take a few minutes. It uses seed 42. It removes exact duplicate rows and enforces basic target/physical validity, but avoids arbitrary tail trimming. `vehicle_age_years` is excluded because it is redundant with model year in this snapshot. Address, location labels below city, and coordinates are excluded to reduce identifier-like signals and improve generalization.

## Evaluation design

- 60% fit, 20% calibration, and 20% final test rows, using fixed seed 42.
- Candidate model settings are ranked on a validation slice inside the fit set; the test set is not used for tuning.
- Target is `log1p(price)`; outputs and errors are shown in AED.
- Baseline predicts the fit-set median price for each make/model, falling back to the overall fit-set median.
- Reported metrics: MAE, median absolute error, RMSE, R², fraction within 10%/20% of asking price, and empirical held-out coverage of the nominal 90% split-conformal interval.
- Feature importance is predictive attribution, not causal explanation.

## Snapshot result

On the recorded snapshot, the untouched 1,541-row test split produced **AED 50,714 MAE** and **AED 16,188 median absolute error**. The training-set make/model median baseline had AED 86,520 MAE, so the model's MAE was **41.4% lower**; paired bootstrap 95% CI for the AED MAE improvement: AED 31,291–40,460. The model reached R² 0.576 and landed within 20% of listed price for 53.6% of test rows. The nominal 90% conformal interval covered 92.1% of this test set, but its global radius was AED 130,160 each way, which is too broad to imply precision for an individual estimate. Test-set MAE exceeded AED 293k for vehicles listed above AED 500k, so segment-level uncertainty matters.

These are held-out listing-price results from this source snapshot. They are not evidence of realized transaction-price accuracy or today's market performance. Exact metrics, split details, snapshot hash, candidate validation scores, and segment breakdowns are in `artifacts/results.json`.

## Reproducibility and data terms

Read [`data/README.md`](data/README.md) before downloading or republishing data. Upstream currently provides no explicit license in its repository README; the CSV is therefore kept out of version control. The source URL is on a mutable branch, so `train.py` saves the input SHA-256 in `artifacts/results.json` to identify the exact snapshot used. Re-running after an upstream file changes can produce different scores.

## Limitations and responsible use

The source gives no listing collection timestamps, outcomes, or detailed sampling methodology. A random split can contain similar vehicles across fit and test; it does not simulate a future-time deployment. Data is concentrated in Dubai and some makes/models are sparse. The global conformal radius provides marginal calibration only under exchangeability; coverage may vary by segment and is not an individual guarantee. Prices can reflect errors, negotiation strategy, condition, accident history, service history, and other omitted information. Treat estimates as exploratory decision support and verify against current comparable listings and a vehicle inspection.
