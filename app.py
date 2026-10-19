"""Interactive valuation demo backed by the saved, held-out-evaluated model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from catboost import CatBoostRegressor
from train import read_data

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
DATA=ROOT/"data/raw/dubizzle_cars_dataset.csv"
st.set_page_config(page_title="UAE Car Valuation Lab",page_icon="🚘",layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp{background:#f7f8ef;color:#000}.block-container{max-width:1220px;padding-top:2.2rem}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"], [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div {color:#000!important}
input,textarea,[data-baseweb="select"]>div{color:#000!important;background:#fff!important;border-color:#dbe7cf!important}
[data-baseweb="select"] [role="option"],[data-baseweb="popover"],[data-baseweb="menu"]{color:#000!important;background:#fff!important}
[data-baseweb="popover"] [role="option"]:hover{background:#eef7df!important}
.js-plotly-plot text,.js-plotly-plot .hoverlayer text{fill:#000!important;color:#000!important}
.js-plotly-plot .hoverlayer .hovertext rect{fill:#fff!important;stroke:#dbe7cf!important}
.js-plotly-plot .hoverlayer .hovertext path{fill:#fff!important;stroke:#dbe7cf!important}
h1,h2,h3{font-family:Manrope,sans-serif;letter-spacing:-.04em;color:#000!important}.eyebrow{font:11px 'DM Mono',monospace;letter-spacing:.12em;color:#000!important}
.hero{padding:30px 34px;background:#e8f2dc;border:1px solid #d5e4c6;color:#000;border-radius:8px;margin:8px 0 25px}.hero p{color:#000!important;max-width:720px}.hero h1{font-size:42px!important;color:#000!important}.hero .eyebrow{color:#000!important}
.stMetric{background:#fff;border:1px solid #dbe7cf;padding:14px;border-radius:6px}.stButton button{background:#dff2b8;color:#000!important;border:1px solid #cfe4a8;font-weight:700}
.note{padding:13px 16px;background:#eff5e8;border-left:3px solid #9cbd70;color:#000!important;font-size:13px}
</style>""",unsafe_allow_html=True)
@st.cache_data
def load_results(version=None): return json.loads((ART/"results.json").read_text())
@st.cache_data
def load_data():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    return read_data()[0]
@st.cache_resource
def load_model():
    m=CatBoostRegressor();m.load_model(str(ART/"price_model.cbm"));return m

def category_values(series):
    """Use the same missing-category labels as the training pipeline."""
    return series.fillna("Unknown").astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})

def options_for(rows, column):
    # Keep every value observed under the current parent selections; do not
    # suppress rare trims or body/fuel/spec combinations with a count cutoff.
    values=category_values(rows[column]).unique()
    return sorted(values) if len(values) else ["Unknown"]

def option_counts(rows, column):
    return category_values(rows[column]).value_counts().to_dict()

def listing_label(counts):
    return lambda value: f"{value} · {counts.get(value, 0):,} matching listings"

if not (ART/"results.json").exists() or not (ART/"price_model.cbm").exists():
    st.error("Model artifacts are missing. Follow the README setup steps to fetch data and train the model.");st.stop()
try: df=load_data()
except Exception:
    st.error("Raw source data is missing. Follow data/README.md and run `python fetch_data.py`.");st.stop()
res=load_results((ART/"results.json").stat().st_mtime_ns);model=load_model();test=res["test"]["model"]
st.markdown(f"<div class='hero'><div class='eyebrow'>FIELD NOTES 01 · UAE USED-CAR MARKET</div><h1>What might this car be listed for?</h1><p>Trained on {res['data']['eligible_rows']:,} eligible listings from a 9,953-row UAE snapshot. Explore the held-out evaluation, then try a configuration. Estimates describe historical asking prices, not completed sales or today's market.</p></div>",unsafe_allow_html=True)
st.markdown("<div class='eyebrow'>MODEL SCORE · HELD-OUT TEST SET</div>",unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4)
c1.metric("Median absolute error",f"AED {test['median_ae_aed']:,.0f}")
c2.metric("Mean absolute error",f"AED {test['mae_aed']:,.0f}",f"{res['test']['mae_improvement_over_baseline']:.1%} lower than baseline")
c3.metric("Median absolute percentage error",f"{test['median_ape']:.1%}")
c4.metric("Within 20% of asking price",f"{test['within_20pct']:.1%}")
st.caption(f"Evaluation uses a fixed 20% held-out split ({test['n']:,} listings). The dataset's published analysis identifies asking prices below AED 5,000 as likely incorrect/fake, so those rows are excluded before splitting. Price ranges are calibrated separately by predicted-price band; group coverage is not an individual guarantee.")
meaning_tab, calculation_tab, process_tab=st.tabs(["What the statistics mean", "How they were calculated", "Project: start to finish"])
with meaning_tab:
    st.markdown(f"""
### Median absolute error — AED {test['median_ae_aed']:,.0f}
For each held-out listing, we took the absolute difference between the model estimate and the listed asking price. The median of those differences is AED {test['median_ae_aed']:,.0f}: half the test listings had a smaller absolute error and half had a larger one. The median is less affected by a few very large misses.

### Mean absolute error — AED {test['mae_aed']:,.0f}
This is the average size of the model’s misses in AED, regardless of whether it estimated too high or too low. The make-and-model median baseline had MAE AED {res['baseline']['mae_aed']:,.0f}; the model’s MAE was {res['test']['mae_improvement_over_baseline']:.1%} lower on the same test listings.

### Median absolute percentage error — {test['median_ape']:.1%}
For each listing, divide its absolute AED error by its asking price, then take the median. Half the test listings had a percentage error below {test['median_ape']:.1%} and half above.

### Mean absolute percentage error — {test['mape']:.1%}
This averages those per-listing percentage errors. It is sensitive to low asking prices, which is why the source's below-AED-5,000 rows are excluded using its published data-quality rule.

### Within 20% of asking price — {test['within_20pct']:.1%}
This is the share of held-out listings where the model estimate was no more than 20% above or below that listing’s asking price. Here, that was {test['within_20pct']:.1%} of the test rows.

### Held-out listings — {test['n']:,}
This is the number of listings used to calculate the three scores. They were kept out of model fitting and model selection, so they provide a check on data the model did not train on.
""")
with calculation_tab:
    st.markdown("The calculations below use the model’s predictions and actual asking prices for the same held-out test set. `n` is the number of test listings; `yᵢ` is a listing’s asking price and `ŷᵢ` is the model’s estimate.")
    st.latex(r"AE_i = |\hat{y}_i - y_i|")
    st.latex(r"MedianAE = median(AE_1, AE_2, \ldots, AE_n)")
    st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i-y_i|")
    st.latex(r"Within20 = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left(\frac{|\hat{y}_i-y_i|}{y_i}\leq 0.20\right)")
    st.latex(r"MAPE = \frac{1}{n}\sum_{i=1}^{n}\frac{|\hat{y}_i-y_i|}{y_i}")
    st.latex(r"MdAPE = median\left(\frac{|\hat{y}_i-y_i|}{y_i}\right)")
    st.markdown(f"Here, `yᵢ` is positive for every eligible listing. The baseline predicts the fit-set median asking price for the same make and model, falling back to the overall fit-set median when that combination is absent. MAE improvement is `(baseline MAE − model MAE) / baseline MAE`, which is {res['test']['mae_improvement_over_baseline']:.1%} here.")
    st.caption(f"The source snapshot had {res['data']['source_rows']:,} rows. After exact duplicates, {res['data']['below_upstream_price_quality_threshold_rows_removed']:,} rows below the upstream AED {res['data']['upstream_price_quality_threshold_aed']:,} quality threshold, and invalid target/year/mileage rows were removed, {res['data']['eligible_rows']:,} remained. A fixed seed split assigned {res['split']['fit_rows']:,} to model fitting, {res['split']['calibration_rows']:,} to interval calibration, and {res['split']['test_rows']:,} to the final test set. These scores describe historical asking prices in this snapshot, not completed sale prices or today's market.")
with process_tab:
    model_info=res["model"]
    bp=model_info["best_params"]
    st.markdown("""
### 1. Define the prediction task
The target is the **advertised asking price in AED** for a used-car listing. Each row is one listing from a single historical UAE snapshot. This is a supervised regression task: the model learns a numeric price from listing attributes. It does not learn the final sale price, negotiation outcome, or current market value.

### 2. Inspect and prepare the source data
The project starts with **9,953 rows**. It removes exact duplicate rows and applies the source notebook's stated data-quality rule that prices below AED 5,000 are likely incorrect or fake. It also keeps records with valid price, model year (1950 through the snapshot year), and nonnegative mileage. Missing categorical attributes are represented as `Unknown`; numeric fields are parsed as numbers. After these rules, **7,296 listings** are eligible. The raw CSV is not redistributed with the code because its upstream repository does not give a clear data license.

### 3. Keep model fitting, interval calibration, and final scoring separate
Using random seed **42**, the cleaned rows are divided into **60% fit**, **20% calibration**, and **20% final test** partitions: {fit_rows:,}, {cal_rows:,}, and {test_rows:,} rows. Inside the fit portion, an **18% validation slice** is used for early stopping and choosing model settings. The calibration rows set the prediction-interval widths. The final test rows estimate performance after those choices. Because the split is random by row, this evaluates interpolation within the snapshot; it is not a future-time test.

### 4. Give the model vehicle and listing attributes
The model uses {feature_count} input fields: {features}. CatBoost is a gradient-boosted decision-tree model. It builds trees in sequence, with later trees correcting patterns left by earlier ones. CatBoost was selected because it can handle the categorical fields directly, including make, model, trim, body type, and seller type, while also using numeric mileage, year, and specifications.

### 5. Compare six candidate configurations
Each candidate used CatBoost with at most **1,400 boosting iterations**, validation-based early stopping after **100** rounds without improvement, random seed **42**, and validation **MAE in AED** as the selection criterion. The candidate with the lowest validation MAE was then refit on the full fit partition for the selected number of iterations. The table shows validation results saved by the training run.
""".format(fit_rows=res["split"]["fit_rows"],cal_rows=res["split"]["calibration_rows"],test_rows=res["split"]["test_rows"],feature_count=len(model_info["features"]),features=", ".join(f"`{f}`" for f in model_info["features"])))
    candidates=pd.DataFrame(model_info.get("validation_candidates", []))
    if not candidates.empty:
        params_df=pd.json_normalize(candidates["params"])
        candidates_view=pd.concat([candidates[["validation_mae_aed","iterations"]].rename(columns={"validation_mae_aed":"Validation MAE (AED)"}),params_df],axis=1)
        st.dataframe(candidates_view.style.format({"Validation MAE (AED)":"{:,.0f}","learning_rate":"{:.2f}"}),use_container_width=True,hide_index=True)
    st.markdown("""
**Selected configuration:** CatBoostRegressor, **{objective}** loss on **{target_transform}**, depth **{depth}**, learning rate **{learning_rate}**, L2 leaf regularization **{l2}**, and **{iterations:,}** boosting iterations.

- **Depth** controls how many successive feature decisions a tree can make; larger values can model more complex patterns but can overfit.
- **Learning rate** controls how much each new tree adjusts the current predictions. Lower rates generally require more trees.
- **L2 leaf regularization** penalizes overly large leaf adjustments to help control overfitting.
- **RMSE objective** trains on squared errors in the transformed target. The `log1p` transform makes relative price differences matter more than they would on raw AED scale. Predictions are transformed back to AED before evaluation.
- **Validation MAE** is average absolute error on the validation slice, in AED. The selected setup was the lowest-MAE candidate; it was not chosen by maximizing the within-20% statistic.

The numeric inputs were year, kilometers, doors, seating capacity, and cylinder count. The categorical inputs were make, model, trim, regional specifications, body type, fuel type, transmission, engine capacity, horsepower, exterior/interior colors, warranty, city, and seller type. Vehicle age was left out because it duplicates information in year for this snapshot. Address-level location labels and coordinates were left out because they act like unstable identifiers and may not be available consistently at prediction time.

### 6. Compare with a simple baseline
The baseline predicts the median fit-set price for the same make and model. If that pair was absent from fit data, it uses the overall fit-set median. This gives a transparent benchmark for whether the feature-rich model adds value. On the held-out test rows, model MAE was **{mae:,.0f} AED** versus **{base_mae:,.0f} AED** for the baseline, a **{improvement:.1%}** reduction.

### 7. Calibrate prediction ranges and evaluate once
After fitting, absolute calibration-set errors are used to calculate a split-conformal radius for a nominal **90%** range. The app uses separate radii for predicted-price groups (`<=50k`, `50–100k`, `100–250k`, `250–500k`, and `>500k`) when a group has at least 30 calibration examples; smaller groups use the global radius. This is group-level coverage evidence, not a promise for an individual car.

The final test set contains **{n:,}** listings. The deployed model achieved **{within20:.1%} within 20%**, **{mdape:.1%} median absolute percentage error**, **{mape:.1%} mean absolute percentage error**, **AED {mae:,.0f} mean absolute error**, and **R² {r2:.3f}**. The 20% figure means that proportion of held-out asking prices had an absolute percentage error no greater than 20%; it does not mean the model is correct for that percentage of future vehicles. The reported test metrics belong to this saved model and this snapshot.
""".format(objective=model_info["objective"],target_transform=model_info["target_transform"],depth=bp["depth"],learning_rate=bp["learning_rate"],l2=bp["l2_leaf_reg"],iterations=model_info["iterations"],mae=test["mae_aed"],base_mae=res["baseline"]["mae_aed"],improvement=res["test"]["mae_improvement_over_baseline"],n=test["n"],within20=test["within_20pct"],mdape=test["median_ape"],mape=test["mape"],r2=test["r2"]))
left,right=st.columns([1,1],gap="large")
with left:
    st.subheader("Try a vehicle")
    brands=sorted(category_values(df.brand).unique())
    brand_counts=option_counts(df,"brand")
    brand=st.selectbox("Make",brands,index=brands.index("Toyota") if "Toyota" in brands else 0,
                       format_func=listing_label(brand_counts),key="make")
    make_rows=df[category_values(df.brand)==brand]
    model_counts=option_counts(make_rows,"model")
    mdls=sorted(category_values(make_rows.model).unique())
    mdl=st.selectbox("Model",mdls,format_func=listing_label(model_counts),key=f"model::{brand}")
    example=make_rows[category_values(make_rows.model)==mdl]
    trim_options=options_for(example,"trim")
    trim_counts=option_counts(example,"trim")
    trim=st.selectbox("Trim",trim_options,format_func=listing_label(trim_counts),key=f"trim::{brand}::{mdl}")
    config=example[category_values(example.trim)==trim]
    body_options=options_for(config,"body_type")
    body_counts=option_counts(config,"body_type")
    body=st.selectbox("Body type",body_options,format_func=listing_label(body_counts),key=f"body::{brand}::{mdl}::{trim}")
    config=config[category_values(config.body_type)==body]
    fuel_options=options_for(config,"fuel_type")
    fuel_counts=option_counts(config,"fuel_type")
    fuel=st.selectbox("Fuel type",fuel_options,format_func=listing_label(fuel_counts),key=f"fuel::{brand}::{mdl}::{trim}::{body}")
    config=config[category_values(config.fuel_type)==fuel]
    trans_options=options_for(config,"transmission_type")
    trans_counts=option_counts(config,"transmission_type")
    trans=st.selectbox("Transmission",trans_options,format_func=listing_label(trans_counts),key=f"transmission::{brand}::{mdl}::{trim}::{body}::{fuel}")
    config=config[category_values(config.transmission_type)==trans]
    region_options=options_for(config,"regional_specs")
    region_counts=option_counts(config,"regional_specs")
    region=st.selectbox("Regional specs",region_options,format_func=listing_label(region_counts),key=f"region::{brand}::{mdl}::{trim}::{body}::{fuel}::{trans}")
    config=config[category_values(config.regional_specs)==region]
    # Location is independent of vehicle configuration; keep all source locations available.
    city_name_fixes={"Fujeirah":"Fujairah","Umm Al Qawain":"Umm Al Quwain","Al Ain":"Al Ain (Abu Dhabi Emirate)"}
    city_raw_options=options_for(df,"city")
    city_counts=option_counts(df,"city")
    city_by_label={city_name_fixes.get(value,value):value for value in city_raw_options}
    city_label_counts={city_name_fixes.get(value,value):count for value,count in city_counts.items()}
    city_label=st.selectbox("Listed city / emirate",sorted(city_by_label),format_func=listing_label(city_label_counts),key="listed-city")
    city=city_by_label[city_label]
    seller_options=options_for(config,"seller_type")
    seller_counts=option_counts(config,"seller_type")
    seller=st.selectbox("Seller type",seller_options,format_func=listing_label(seller_counts),key=f"seller::{brand}::{mdl}::{trim}::{body}::{fuel}::{trans}::{region}")
    config=config[category_values(config.seller_type)==seller]
    default_year=int(example.year.median()) if len(example) else 2019
    default_km=int(example.kilometers.median()) if len(example) else 60000
    year=st.slider("Year",min_value=1990,max_value=int(df.year.max()),value=min(int(df.year.max()),max(1990,default_year)))
    km=st.number_input("Mileage (km)",min_value=0,max_value=1_000_000,value=min(1_000_000,default_km),step=5000)
    if st.button("Estimate asking price",use_container_width=True):
        sample=config.iloc[0] if len(config) else example.iloc[0] if len(example) else df.iloc[0]
        row={c:category_values(pd.Series([sample.get(c,"Unknown")])).iloc[0] for c in res["model"]["categorical_features"]}
        row.update({"brand":brand,"model":mdl,"trim":trim,"city":city,"regional_specs":region,"body_type":body,"fuel_type":fuel,"transmission_type":trans,"seller_type":seller,
                    "year":year,"kilometers":km,"doors":sample.get("doors",4),"seating_capacity":sample.get("seating_capacity",5),"no_of_cylinders":sample.get("no_of_cylinders",4)})
        raw_prediction=float(model.predict(pd.DataFrame([row]))[0])
        if res["model"]["target_transform"].startswith("log1p"):
            raw_prediction=float(np.expm1(raw_prediction))
        if not np.isfinite(raw_prediction):
            st.error("The model returned a non-finite estimate for this configuration. Try another vehicle configuration.")
            st.stop()
        pred=max(1,raw_prediction)
        bands=res["calibration"].get("predicted_price_band_intervals",{})
        band="<=50k" if pred<=50_000 else "50-100k" if pred<=100_000 else "100-250k" if pred<=250_000 else "250-500k" if pred<=500_000 else ">500k"
        band_info=bands.get(band,{})
        radius=float(band_info.get("radius_aed",test["conformal_90_radius_aed"]))
        st.session_state["estimate"]=(pred,radius,band,band_info)
    if "estimate" in st.session_state and (len(st.session_state["estimate"])<4 or not np.isfinite(st.session_state["estimate"][0])):
        st.session_state.pop("estimate",None)
    if "estimate" in st.session_state:
        pred,radius,band,band_info=st.session_state["estimate"]
        st.success(f"Estimated historical asking price: AED {pred:,.0f}\n\n90% calibrated range for the {band} predicted-price group: AED {max(0,pred-radius):,.0f} – {pred+radius:,.0f}")
        st.caption(f"Price band: {band}. Held-out test coverage in this group: {band_info.get('test_coverage',0):.1%} across {band_info.get('test_n',0):,} listings. This is group-level evidence, not a guarantee for this individual car.")
with right:
    st.subheader("What the snapshot contains")
    st.caption("Asking-price distribution among eligible listings, clipped at AED 2,000,000 for readability. Each bar groups listings into a price interval; taller bars mean more listings in that interval. The long right tail is why the horizontal scale is dominated by ordinary-price cars.")
    d=df.copy();d["price"]=pd.to_numeric(d.price,errors="coerce")
    d=d[d.price.between(1000,2_000_000)]
    fig=px.histogram(d,x="price",nbins=55,color_discrete_sequence=["#527d64"],labels={"price":"Asking price (AED)","count":"Listings"})
    fig.update_layout(height=285,margin=dict(l=8,r=8,t=15,b=10),paper_bgcolor="#f7f8ef",plot_bgcolor="#ffffff",font_family="DM Sans",font_color="#000000",hoverlabel=dict(bgcolor="#ffffff",bordercolor="#dbe7cf",font_color="#000000"),showlegend=False)
    fig.update_traces(hovertemplate="Asking price: AED %{x:,.0f}<br>Listings: %{y}<extra></extra>")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Top model inputs by CatBoost's feature-importance score. The score summarizes how much the fitted trees used each feature to improve predictions; it is relative within this model, does not show whether a feature raises or lowers price, and does not establish causation. Correlated inputs can share or shift importance.")
    fi=pd.DataFrame(res["feature_importance"][:10]).sort_values("importance")
    fig2=px.bar(fi,x="importance",y="feature",orientation="h",color_discrete_sequence=["#bc7954"],labels={"importance":"Model importance","feature":"Input feature"})
    fig2.update_layout(height=310,margin=dict(l=8,r=8,t=10,b=10),paper_bgcolor="#f7f8ef",plot_bgcolor="#ffffff",font_family="DM Sans",font_color="#000000",hoverlabel=dict(bgcolor="#ffffff",bordercolor="#dbe7cf",font_color="#000000"))
    fig2.update_traces(hovertemplate="Feature: %{y}<br>Importance: %{x:.2f}<extra></extra>")
    st.plotly_chart(fig2,use_container_width=True)
st.markdown("<div class='note'><b>How to read this:</b> these are asking prices in a single historical listing snapshot. The dataset has no sale outcomes or collection dates. The random holdout tests interpolation within the snapshot; it does not prove current-market accuracy. Feature importance describes model behavior, not causal effects.</div>",unsafe_allow_html=True)
with st.expander("Evaluation details and limitations"):
    st.json({"data":res["data"],"split":res["split"],"baseline":res["baseline"],"test":res["test"],"interval":res["calibration"],"model":res["model"]["best_params"]})
    st.write("The interval radius is calibrated within price bands based on the model prediction, which makes ordinary-car ranges more useful than one global margin. Errors remain much larger for expensive and rare vehicles. Group-level coverage is not an individual guarantee; compare current listings and inspect the vehicle before relying on an estimate.")
