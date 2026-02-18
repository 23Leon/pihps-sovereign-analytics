import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data
from utils.ml_engine import ensemble_forecast, scan_all_regions
from utils.risk_engine import classify_risk
from utils.macro_engine import (
    compute_national_volatility,
    compute_multi_commodity_pressure,
    compute_contagion,
    compute_macro_score,
    national_forecast_30
)

st.set_page_config(layout="wide")
st.title("🌍 Sovereign Macro Risk Intelligence Platform")

df = load_data("data/pihps_full_detail_async_clean.csv")

# ============================
# FILTER
# ============================
kategori = st.sidebar.selectbox(
    "Kategori",
    sorted(df["kategori"].dropna().unique())
)
df = df[df["kategori"] == kategori]

komoditas = st.sidebar.selectbox(
    "Komoditas",
    sorted(df["komoditas"].dropna().unique())
)
df = df[df["komoditas"] == komoditas]

# ============================
# REGIONAL FORECAST
# ============================
st.header("🧠 Regional Forecast & Risk")

analysis_level = st.radio(
    "Level Analisis",
    ["Provinsi","Kabupaten/Kota"]
)

result = None

if analysis_level == "Provinsi":

    prov_list = sorted(df["provinsi"].dropna().unique())
    selected = st.selectbox("Pilih Provinsi", prov_list)

    df_w = (
        df.groupby(["provinsi","tanggal"])["harga"]
        .mean()
        .reset_index()
    )
    df_w = df_w[df_w["provinsi"] == selected]

else:

    prov_list = sorted(df["provinsi"].dropna().unique())
    prov_selected = st.selectbox("Pilih Provinsi", prov_list)

    df_temp = df[df["provinsi"] == prov_selected]

    kab_list = sorted(df_temp["wilayah"].dropna().unique())
    selected = st.selectbox("Pilih Kabupaten/Kota", kab_list)

    df_w = df_temp[df_temp["wilayah"] == selected]

if not df_w.empty:

    result = ensemble_forecast(df_w)

    if result:

        st.metric("Forecast 7 Hari", f"Rp {int(result['forecast_price']):,}")
        st.metric("MAPE (%)", f"{result['mape']:.2f}")

        volatility = df_w["harga"].rolling(14).std().iloc[-1]
        score, level = classify_risk(
            result["forecast_change"],
            volatility
        )

        st.metric("Regional Risk Score", f"{score:.2f}")
        st.write("Risk Level:", level)

    else:
        st.warning("Data belum cukup untuk forecast.")

# ============================
# RISK RANKING NASIONAL
# ============================
st.header("🔥 Regional Risk Ranking")

risk_df = scan_all_regions(df, level="provinsi")

if not risk_df.empty:

    risk_df["risk_score"] = risk_df["forecast_change"] * 100
    risk_df = risk_df.sort_values("risk_score", ascending=False)

    st.dataframe(risk_df.head(10))

    high_regions = (risk_df["forecast_change"] > 0.07).sum()
    total_regions = len(risk_df)

    if total_regions > 0:

        high_ratio = high_regions / total_regions

        if high_ratio > 0.3:
            st.error("🚨 NATIONAL EARLY WARNING TRIGGERED")
        else:
            st.success("National risk under control")

# ============================
# NATIONAL MACRO RISK
# ============================
st.header("📈 National Macro Risk Engine")

vol = compute_national_volatility(df)
pressure = compute_multi_commodity_pressure(df)
contagion = compute_contagion(df)

forecast_risk = result["forecast_change"] if result else 0

macro_score = compute_macro_score(
    forecast_risk,
    vol,
    contagion,
    pressure
)

st.metric("Macro Risk Score", f"{macro_score:.4f}")

if macro_score > 0.08:
    st.error("🚨 SYSTEMIC RISK")
elif macro_score > 0.04:
    st.warning("⚠ ELEVATED RISK")
else:
    st.success("Stable Macro Condition")

# ============================
# NATIONAL 30-DAY PROJECTION
# ============================
st.header("📈 National 30-Day Inflation Projection")

future_price, preds = national_forecast_30(df)

if future_price is not None:

    st.metric("Projected National Price (30d)", f"Rp {int(future_price):,}")

    df_nat = df.groupby("tanggal")["harga"].mean().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_nat["tanggal"],
        y=df_nat["harga"],
        name="Actual"
    ))

    future_dates = pd.date_range(
        df_nat["tanggal"].iloc[-1],
        periods=30,
        freq="D"
    )

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=preds,
        name="Forecast"
    ))

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Data belum cukup untuk 30-day projection.")
