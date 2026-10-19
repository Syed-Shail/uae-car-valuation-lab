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
.stApp{background:#f5f3ec;color:#202720}.block-container{max-width:1220px;padding-top:2.2rem}
h1,h2,h3{font-family:Manrope,sans-serif;letter-spacing:-.04em}.eyebrow{font:11px 'DM Mono',monospace;letter-spacing:.12em;color:#61776b}
.hero{padding:30px 34px;background:#132820;color:#f4f1e8;border-radius:8px;margin:8px 0 25px}.hero p{color:#c7d4cb;max-width:720px}.hero h1{font-size:42px!important;color:#f4f1e8!important}.hero .eyebrow{color:#c1e477}
.stMetric{background:white;border:1px solid #e4e0d6;padding:14px;border-radius:6px}.stButton button{background:#c6ee76;color:#132820;border:0;font-weight:700}
.note{padding:13px 16px;background:#e9e5d8;border-left:3px solid #b87754;color:#414c44;font-size:13px}
</style>""",unsafe_allow_html=True)
@st.cache_data
def load_results(): return json.loads((ART/"results.json").read_text())
@st.cache_data
def load_data(): return pd.read_csv(DATA,low_memory=False)
@st.cache_resource
def load_model():
    m=CatBoostRegressor();m.load_model(str(ART/"price_model.cbm"));return m

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
st.caption("Evaluation uses a fixed 20% test split. Baseline: training-set median by make and model. Interval below is split-conformal with marginal 90% calibration; individual coverage is not guaranteed.")
left,right=st.columns([1,1],gap="large")
with left:
    st.subheader("Try a vehicle")
    brands=sorted(df.brand.dropna().astype(str).unique())
    brand=st.selectbox("Make",brands,index=brands.index("Toyota") if "Toyota" in brands else 0)
    mdls=sorted(df.loc[df.brand.astype(str)==brand,"model"].dropna().astype(str).unique())
    mdl=st.selectbox("Model",mdls)
    example=df[(df.brand.astype(str)==brand)&(df.model.astype(str)==mdl)]
    default_year=int(example.year.median()) if len(example) else 2019
    default_km=int(example.kilometers.median()) if len(example) else 60000
    year=st.slider("Year",min_value=1990,max_value=int(df.year.max()),value=min(int(df.year.max()),max(1990,default_year)))
    km=st.number_input("Mileage (km)",min_value=0,max_value=1_000_000,value=min(1_000_000,default_km),step=5000)
    trim=st.selectbox("Trim",sorted(example.trim.fillna("Unknown").astype(str).unique()) if len(example) else ["Unknown"])
    city=st.selectbox("Emirate",sorted(df.city.fillna("Unknown").astype(str).unique()))
    region=st.selectbox("Regional specs",sorted(df.regional_specs.fillna("Unknown").astype(str).unique()))
    body=st.selectbox("Body type",sorted(df.body_type.fillna("Unknown").astype(str).unique()))
    fuel=st.selectbox("Fuel type",sorted(df.fuel_type.fillna("Unknown").astype(str).unique()))
    trans=st.selectbox("Transmission",sorted(df.transmission_type.fillna("Unknown").astype(str).unique()))
    seller=st.selectbox("Seller type",sorted(df.seller_type.fillna("Unknown").astype(str).unique()))
    if st.button("Estimate asking price",use_container_width=True):
        sample=example.iloc[0] if len(example) else df.iloc[0]
        row={c:str(sample.get(c,"Unknown")) if pd.notna(sample.get(c)) else "Unknown" for c in res["model"]["categorical_features"]}
        row.update({"brand":brand,"model":mdl,"trim":trim,"city":city,"regional_specs":region,"body_type":body,"fuel_type":fuel,"transmission_type":trans,"seller_type":seller,
                    "year":year,"kilometers":km,"doors":sample.get("doors",4),"seating_capacity":sample.get("seating_capacity",5),"no_of_cylinders":sample.get("no_of_cylinders",4)})
        pred=max(1,float(np.expm1(model.predict(pd.DataFrame([row]))[0])))
        st.session_state["estimate"]=(pred,test["conformal_90_radius_aed"])
    if "estimate" in st.session_state:
        pred,radius=st.session_state["estimate"]
        st.success(f"Estimated historical asking price: AED {pred:,.0f}\n\nBroad 90% calibrated interval: AED {max(0,pred-radius):,.0f} – {pred+radius:,.0f}")
with right:
    st.subheader("What the snapshot contains")
    d=df.copy();d["price"]=pd.to_numeric(d.price,errors="coerce")
    d=d[d.price.between(1000,2_000_000)]
    fig=px.histogram(d,x="price",nbins=55,color_discrete_sequence=["#527d64"],labels={"price":"Asking price (AED)","count":"Listings"})
    fig.update_layout(height=285,margin=dict(l=8,r=8,t=15,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white",font_family="DM Sans",showlegend=False)
    st.plotly_chart(fig,use_container_width=True)
    fi=pd.DataFrame(res["feature_importance"][:10]).sort_values("importance")
    fig2=px.bar(fi,x="importance",y="feature",orientation="h",color_discrete_sequence=["#bc7954"],labels={"importance":"Model importance","feature":"Input feature"})
    fig2.update_layout(height=310,margin=dict(l=8,r=8,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white",font_family="DM Sans")
    st.plotly_chart(fig2,use_container_width=True)
st.markdown("<div class='note'><b>How to read this:</b> these are asking prices in a single historical listing snapshot. The dataset has no sale outcomes or collection dates. The random holdout tests interpolation within the snapshot; it does not prove current-market accuracy. Feature importance describes model behavior, not causal effects.</div>",unsafe_allow_html=True)
with st.expander("Evaluation details and limitations"):
    st.json({"data":res["data"],"split":res["split"],"baseline":res["baseline"],"test":res["test"],"interval":res["calibration"],"model":res["model"]["best_params"]})
    st.write("Prediction intervals use one global absolute-error radius. They can be wide, particularly for expensive or rare vehicles. Do not use the estimate as a substitute for inspection, transaction comparables, or a professional valuation.")
