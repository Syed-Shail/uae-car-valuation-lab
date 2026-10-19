"""Interactive valuation demo backed by the saved, held-out-evaluated model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from catboost import CatBoostRegressor

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
def load_results(): return json.loads((ART/"results.json").read_text())
@st.cache_data
def load_data(): return pd.read_csv(DATA,low_memory=False)
@st.cache_resource
def load_model():
    m=CatBoostRegressor();m.load_model(str(ART/"price_model.cbm"));return m

def options_for(rows, column, min_count=1):
    counts=rows[column].fillna("Unknown").astype(str).value_counts()
    supported=counts[counts>=min_count]
    if len(supported):
        counts=supported
    elif len(counts):
        counts=counts.iloc[:1]
    return sorted(counts.index.astype(str)) if len(counts) else ["Unknown"]

if not (ART/"results.json").exists() or not (ART/"price_model.cbm").exists():
    st.error("Model artifacts are missing. Follow the README setup steps to fetch data and train the model.");st.stop()
try: df=load_data()
except Exception:
    st.error("Raw source data is missing. Follow data/README.md and run `python fetch_data.py`.");st.stop()
res=load_results();model=load_model();test=res["test"]["model"]
st.markdown("<div class='hero'><div class='eyebrow'>FIELD NOTES 01 · UAE USED-CAR MARKET</div><h1>What might this car be listed for?</h1><p>A data-backed estimate from 9,953 UAE vehicle listings. Explore the held-out evaluation, then try a configuration. Estimates describe historical asking prices, not completed sales or today's market.</p></div>",unsafe_allow_html=True)
st.markdown("<div class='eyebrow'>MODEL SCORE · UNTOUCHED TEST SET</div>",unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4)
c1.metric("Median absolute error",f"AED {test['median_ae_aed']:,.0f}")
c2.metric("Mean absolute error",f"AED {test['mae_aed']:,.0f}",f"{res['test']['mae_improvement_over_baseline']:.1%} lower than baseline")
c3.metric("Within 20% of asking price",f"{test['within_20pct']:.1%}")
c4.metric("Held-out listings",f"{test['n']:,}")
st.caption("Evaluation uses a fixed untouched 20% test split. Price ranges are calibrated separately by model-predicted price band; held-out group coverage is shown after each estimate. This does not guarantee coverage for an individual car.")
left,right=st.columns([1,1],gap="large")
with left:
    st.subheader("Try a vehicle")
    brands=sorted(df.brand.dropna().astype(str).unique())
    brand=st.selectbox("Make",brands,index=brands.index("Toyota") if "Toyota" in brands else 0)
    mdls=sorted(df.loc[df.brand.astype(str)==brand,"model"].dropna().astype(str).unique())
    mdl=st.selectbox("Model",mdls)
    example=df[(df.brand.astype(str)==brand)&(df.model.astype(str)==mdl)]
    trim_options=options_for(example,"trim")
    trim=st.selectbox("Trim",trim_options)
    config=example[example.trim.fillna("Unknown").astype(str)==trim]
    body_options=options_for(config,"body_type",min_count=3)
    body=st.selectbox("Body type",body_options)
    st.caption("Body styles with fewer than three matching listings are treated as likely label noise; if none reach that count, the most common recorded style is shown.")
    config=config[config.body_type.fillna("Unknown").astype(str)==body]
    fuel_options=options_for(config,"fuel_type")
    fuel=st.selectbox("Fuel type",fuel_options)
    config=config[config.fuel_type.fillna("Unknown").astype(str)==fuel]
    trans_options=options_for(config,"transmission_type")
    trans=st.selectbox("Transmission",trans_options)
    config=config[config.transmission_type.fillna("Unknown").astype(str)==trans]
    region_options=options_for(config,"regional_specs")
    region=st.selectbox("Regional specs",region_options)
    config=config[config.regional_specs.fillna("Unknown").astype(str)==region]
    # Emirate is an independent location choice; retain every location in the source dataset.
    city_options=options_for(df,"city")
    city=st.selectbox("Emirate",city_options)
    seller_options=options_for(config,"seller_type")
    seller=st.selectbox("Seller type",seller_options)
    config=config[config.seller_type.fillna("Unknown").astype(str)==seller]
    default_year=int(example.year.median()) if len(example) else 2019
    default_km=int(example.kilometers.median()) if len(example) else 60000
    year=st.slider("Year",min_value=1990,max_value=int(df.year.max()),value=min(int(df.year.max()),max(1990,default_year)))
    km=st.number_input("Mileage (km)",min_value=0,max_value=1_000_000,value=min(1_000_000,default_km),step=5000)
    if st.button("Estimate asking price",use_container_width=True):
        sample=config.iloc[0] if len(config) else example.iloc[0] if len(example) else df.iloc[0]
        row={c:str(sample.get(c,"Unknown")) if pd.notna(sample.get(c)) else "Unknown" for c in res["model"]["categorical_features"]}
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
    d=df.copy();d["price"]=pd.to_numeric(d.price,errors="coerce")
    d=d[d.price.between(1000,2_000_000)]
    fig=px.histogram(d,x="price",nbins=55,color_discrete_sequence=["#527d64"],labels={"price":"Asking price (AED)","count":"Listings"})
    fig.update_layout(height=285,margin=dict(l=8,r=8,t=15,b=10),paper_bgcolor="#f7f8ef",plot_bgcolor="#ffffff",font_family="DM Sans",font_color="#000000",hoverlabel=dict(bgcolor="#ffffff",bordercolor="#dbe7cf",font_color="#000000"),showlegend=False)
    st.plotly_chart(fig,use_container_width=True)
    fi=pd.DataFrame(res["feature_importance"][:10]).sort_values("importance")
    fig2=px.bar(fi,x="importance",y="feature",orientation="h",color_discrete_sequence=["#bc7954"],labels={"importance":"Model importance","feature":"Input feature"})
    fig2.update_layout(height=310,margin=dict(l=8,r=8,t=10,b=10),paper_bgcolor="#f7f8ef",plot_bgcolor="#ffffff",font_family="DM Sans",font_color="#000000",hoverlabel=dict(bgcolor="#ffffff",bordercolor="#dbe7cf",font_color="#000000"))
    st.plotly_chart(fig2,use_container_width=True)
st.markdown("<div class='note'><b>How to read this:</b> these are asking prices in a single historical listing snapshot. The dataset has no sale outcomes or collection dates. The random holdout tests interpolation within the snapshot; it does not prove current-market accuracy. Feature importance describes model behavior, not causal effects.</div>",unsafe_allow_html=True)
with st.expander("Evaluation details and limitations"):
    st.json({"data":res["data"],"split":res["split"],"baseline":res["baseline"],"test":res["test"],"interval":res["calibration"],"model":res["model"]["best_params"]})
    st.write("The interval radius is calibrated within price bands based on the model prediction, which makes ordinary-car ranges more useful than one global margin. Errors remain much larger for expensive and rare vehicles. Group-level coverage is not an individual guarantee; compare current listings and inspect the vehicle before relying on an estimate.")
