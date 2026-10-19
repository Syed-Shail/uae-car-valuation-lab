"""Leakage-aware training, benchmark, calibration, and artifact export."""
from __future__ import annotations
import hashlib, json, platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/raw/dubizzle_cars_dataset.csv"
OUT = ROOT / "artifacts"
SEED = 42
TARGET = "price"
PRICE_BAND_EDGES = [0, 50_000, 100_000, 250_000, 500_000, float("inf")]
PRICE_BAND_LABELS = ["<=50k", "50-100k", "100-250k", "250-500k", ">500k"]
CATEGORICAL = ["brand", "model", "trim", "regional_specs", "body_type", "fuel_type", "transmission_type", "engine_capacity_cc", "horsepower", "exterior_color", "interior_color", "warranty", "city", "seller_type"]
NUMERIC = ["year", "kilometers", "doors", "seating_capacity", "no_of_cylinders"]
FEATURES = CATEGORICAL + NUMERIC


def read_data() -> tuple[pd.DataFrame, dict]:
    if not DATA.exists():
        raise SystemExit(f"Dataset missing at {DATA}. Run: python fetch_data.py")
    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    df = pd.read_csv(DATA, low_memory=False)
    start = len(df)
    # Only enforce basic physical/target validity. Keep legitimate expensive/old/high-mileage vehicles.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["kilometers"] = pd.to_numeric(df["kilometers"], errors="coerce")
    duplicate_rows=int(df.duplicated().sum())
    df = df.drop_duplicates()
    df = df[df[TARGET].between(1_000, 2_000_000) & df["year"].between(1950, date.today().year) & df["kilometers"].ge(0)].copy()
    df["year"] = df["year"].astype(int)
    df["kilometers"] = df["kilometers"].astype(float)
    # Avoid leakage or identifiers unavailable/unstable at quote time; age is derivable from year and snapshot year.
    for col in CATEGORICAL:
        if col not in df: df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    for col in NUMERIC:
        if col not in df: df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET, "year", "kilometers"])
    info = {"source_rows": start, "exact_duplicate_rows_removed": duplicate_rows, "eligible_rows": len(df), "excluded_rows": start-len(df), "sha256": sha,
            "data_file": DATA.name, "target": "asking price in AED", "snapshot_date": "2026-10-19"}
    return df.reset_index(drop=True), info


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    pct = np.abs(pred-y) / np.maximum(y, 1)
    return {"mae_aed": float(mean_absolute_error(y,pred)), "median_ae_aed": float(median_absolute_error(y,pred)),
            "rmse_aed": float(np.sqrt(mean_squared_error(y,pred))), "r2": float(r2_score(y,pred)),
            "within_10pct": float(np.mean(pct <= .10)), "within_20pct": float(np.mean(pct <= .20))}


def main():
    OUT.mkdir(exist_ok=True)
    df, data_info = read_data()
    # Split rows once into 60% model fit, 20% interval calibration, 20% final untouched test.
    idx = np.arange(len(df))
    fit_idx, rest_idx = train_test_split(idx, test_size=.40, random_state=SEED)
    cal_idx, test_idx = train_test_split(rest_idx, test_size=.50, random_state=SEED)
    fit_df, cal_df, test_df = df.iloc[fit_idx], df.iloc[cal_idx], df.iloc[test_idx]
    subtrain_idx, valid_idx = train_test_split(np.arange(len(fit_df)), test_size=.18, random_state=SEED)
    xsub, xval = fit_df.iloc[subtrain_idx][FEATURES], fit_df.iloc[valid_idx][FEATURES]
    ysub = np.log1p(fit_df.iloc[subtrain_idx][TARGET]); yval = fit_df.iloc[valid_idx][TARGET].to_numpy()
    candidates = [
        {"depth": 7, "learning_rate": .05, "l2_leaf_reg": 5, "objective": "RMSE", "log_target": True},
        {"depth": 9, "learning_rate": .04, "l2_leaf_reg": 7, "objective": "RMSE", "log_target": True},
        {"depth": 6, "learning_rate": .06, "l2_leaf_reg": 8, "objective": "RMSE", "log_target": True},
        {"depth": 7, "learning_rate": .05, "l2_leaf_reg": 5, "objective": "MAE", "log_target": False},
        {"depth": 9, "learning_rate": .04, "l2_leaf_reg": 7, "objective": "MAE", "log_target": False},
        {"depth": 6, "learning_rate": .06, "l2_leaf_reg": 8, "objective": "MAE", "log_target": False},
    ]
    trials=[]
    for candidate in candidates:
        params={k:v for k,v in candidate.items() if k not in ("objective","log_target")}
        log_target=candidate["log_target"]
        train_y=np.log1p(fit_df.iloc[subtrain_idx][TARGET]) if log_target else fit_df.iloc[subtrain_idx][TARGET]
        valid_y=np.log1p(yval) if log_target else yval
        m=CatBoostRegressor(iterations=1400, loss_function=candidate["objective"], eval_metric="MAE", random_seed=SEED,
                            verbose=False, allow_writing_files=False, thread_count=-1, **params)
        m.fit(xsub, train_y, cat_features=CATEGORICAL, eval_set=(xval, valid_y), early_stopping_rounds=100, verbose=False)
        raw=m.predict(xval); pred=np.maximum(1, np.expm1(raw) if log_target else raw)
        trials.append((mean_absolute_error(yval,pred), candidate, max(100,m.best_iteration_)))
        print(f"Validation MAE AED {trials[-1][0]:,.0f}; params={candidate}; trees={trials[-1][2]}", flush=True)
    trials.sort(key=lambda z:z[0]); _, best_candidate, iterations=trials[0]
    best_params={k:v for k,v in best_candidate.items() if k not in ("objective","log_target")}
    log_target=best_candidate["log_target"]
    objective=best_candidate["objective"]
    model=CatBoostRegressor(iterations=iterations, loss_function=objective, random_seed=SEED, verbose=False,
                            allow_writing_files=False, thread_count=-1, **best_params)
    model_y=np.log1p(fit_df[TARGET]) if log_target else fit_df[TARGET]
    model.fit(fit_df[FEATURES], model_y, cat_features=CATEGORICAL, verbose=False)
    cal_raw=model.predict(cal_df[FEATURES])
    cal_pred=np.maximum(1,np.expm1(cal_raw) if log_target else cal_raw)
    cal_abs=np.abs(cal_df[TARGET].to_numpy()-cal_pred)
    # Split-conformal absolute residual radius; finite-sample 90% marginal calibration.
    q_idx=min(len(cal_abs)-1, int(np.ceil((len(cal_abs)+1)*.90))-1)
    radius=float(np.sort(cal_abs)[q_idx])
    test_raw=model.predict(test_df[FEATURES])
    pred=np.maximum(1,np.expm1(test_raw) if log_target else test_raw)
    y=test_df[TARGET].to_numpy()
    cal_bands=pd.cut(cal_pred, bins=PRICE_BAND_EDGES, labels=PRICE_BAND_LABELS, include_lowest=True)
    test_bands=pd.cut(pred, bins=PRICE_BAND_EDGES, labels=PRICE_BAND_LABELS, include_lowest=True)
    tier_intervals={}
    tier_radii={}
    for band in PRICE_BAND_LABELS:
        cal_mask=np.asarray(cal_bands==band)
        test_mask=np.asarray(test_bands==band)
        errors=cal_abs[cal_mask]
        if len(errors)>=30:
            band_q=min(len(errors)-1, int(np.ceil((len(errors)+1)*.90))-1)
            band_radius=float(np.sort(errors)[band_q])
            interval_source="price-band calibration set"
        else:
            band_radius=radius
            interval_source="global fallback (small calibration group)"
        tier_radii[band]=band_radius
        tier_intervals[band]={
            "radius_aed":band_radius,
            "calibration_n":int(len(errors)),
            "test_n":int(test_mask.sum()),
            "test_coverage":float(np.mean(np.abs(y[test_mask]-pred[test_mask])<=band_radius)) if test_mask.any() else None,
            "interval_source":interval_source,
        }
    test_metrics=metrics(y,pred)
    test_metrics.update({"n":int(len(y)), "conformal_90_coverage":float(np.mean((y>=pred-radius)&(y<=pred+radius))),
                         "conformal_90_radius_aed":radius, "conformal_90_median_width_aed":2*radius,
                         "price_band_conformal_90_coverage":float(np.mean([abs(actual-estimate)<=tier_radii[str(band)] for actual,estimate,band in zip(y,pred,test_bands)]))})
    # Strong transparent baseline: median asking price by make/model, with a training-set median fallback.
    base=fit_df.groupby(["brand","model"])[TARGET].median()
    global_med=float(fit_df[TARGET].median())
    baseline=np.array([base.get((r.brand,r.model),global_med) for r in test_df.itertuples()])
    baseline_metrics=metrics(y,baseline); baseline_metrics["n"]=int(len(y))
    improvement=1-test_metrics["mae_aed"]/baseline_metrics["mae_aed"]
    # Paired bootstrap CI quantifies uncertainty in the test-set MAE difference.
    rng=np.random.default_rng(SEED); diffs=[]
    for _ in range(2000):
        b=rng.integers(0,len(y),len(y))
        diffs.append(np.mean(np.abs(y[b]-baseline[b]))-np.mean(np.abs(y[b]-pred[b])))
    diff_ci=np.quantile(diffs,[.025,.975]).tolist()
    model.save_model(str(OUT/"price_model.cbm"))
    model_imp=sorted([{ "feature": f, "importance": float(v)} for f,v in zip(FEATURES,model.get_feature_importance())],key=lambda z:-z["importance"])
    test_output=test_df[["brand","model","year","kilometers","seller_type","city",TARGET]].copy().reset_index(drop=True)
    test_output["prediction_aed"]=pred; test_output["abs_error_aed"]=abs(y-pred)
    test_output["price_band"]=[str(band) for band in test_bands]
    test_output["interval_radius_aed"]=[tier_radii[str(band)] for band in test_bands]
    test_output["lower_90_aed"]=np.maximum(0,pred-test_output["interval_radius_aed"].to_numpy()); test_output["upper_90_aed"]=pred+test_output["interval_radius_aed"].to_numpy()
    test_output.to_csv(OUT/"test_predictions.csv",index=False)
    # Report segment metrics only where the held-out sample is not tiny.
    segment_metrics={}
    for col in ["city","seller_type"]:
        segment_metrics[col]={}
        for value,positions in test_df.groupby(col).indices.items():
            positions=np.asarray(positions)
            if len(positions)>=30:
                segment_metrics[col][str(value)]={"n":int(len(positions)),"model_mae_aed":float(mean_absolute_error(y[positions],pred[positions])),"baseline_mae_aed":float(mean_absolute_error(y[positions],baseline[positions]))}
    # Price-band error is useful because absolute error scales with vehicle price.
    bands=pd.cut(y,bins=PRICE_BAND_EDGES,labels=PRICE_BAND_LABELS)
    for band in bands.categories:
        positions=np.where(np.asarray(bands)==band)[0]
        if len(positions)>=30:
            segment_metrics.setdefault("price_band",{})[str(band)]={"n":int(len(positions)),"model_mae_aed":float(mean_absolute_error(y[positions],pred[positions])),"baseline_mae_aed":float(mean_absolute_error(y[positions],baseline[positions]))}
    result={"project":"UAE Used-Car Asking-Price Valuation", "data":data_info,
      "split":{"seed":SEED,"fit_rows":len(fit_df),"calibration_rows":len(cal_df),"test_rows":len(test_df),"method":"random row split; exact duplicate rows removed before split"},
      "model":{"name":"CatBoostRegressor","target_transform":"log1p(AED asking price)" if log_target else "AED asking price (raw scale)","objective":objective,"features":FEATURES,"categorical_features":CATEGORICAL,"best_params":best_params,"iterations":iterations,"excluded_columns":["vehicle_age_years (redundant with year in this snapshot)","address/location coordinates (identifier-like, privacy and generalization concerns)","latitude","longitude","location_name","area_name"],"validation_candidates":[{"validation_mae_aed":float(mae),"params":p,"iterations":n} for mae,p,n in trials]},
      "baseline":{"name":"training median by brand + model; global median fallback",**baseline_metrics},
      "test":{"model":test_metrics,"mae_improvement_over_baseline":float(improvement),"mae_improvement_aed":float(baseline_metrics["mae_aed"]-test_metrics["mae_aed"]),"paired_bootstrap_95pct_ci_mae_improvement_aed":diff_ci,"bootstrap_replicates":2000,"segment_metrics":segment_metrics},
      "calibration":{"method":"90% split-conformal absolute residual intervals calibrated within model-predicted price bands on a separate 20% split","global_radius_aed":radius,"predicted_price_band_intervals":tier_intervals,"warning":"Coverage is calibrated within predicted-price bands on exchangeable data. It is not a guarantee for an individual car or changed market conditions; sparse or expensive-car bands remain uncertain."},
      "feature_importance":model_imp,"limitations":["Predicts asking prices, not transaction prices.","One historical listing snapshot; random holdout does not establish future-market performance.","No listing timestamp or outcome field was provided.","Coverage and errors can vary substantially by model, price segment, and data quality."],
      "environment":{"python":platform.python_version(),"pandas":pd.__version__}}
    (OUT/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"rows":len(df),"test":test_metrics,"baseline":baseline_metrics,"mae_improvement":improvement,"model":best_candidate,"iterations":iterations},indent=2))

if __name__=="__main__": main()
