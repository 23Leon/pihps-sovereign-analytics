import numpy as np
from sklearn.linear_model import LinearRegression

def compute_national_volatility(df):
    df_nat = df.groupby("tanggal")["harga"].mean().reset_index()
    return df_nat["harga"].rolling(14).std().iloc[-1]

def compute_multi_commodity_pressure(df):
    df["rolling30"] = df.groupby("komoditas")["harga"].transform(
        lambda x: x.rolling(30).mean()
    )
    df["pressure"] = (df["harga"] - df["rolling30"]) / df["rolling30"]
    return df["pressure"].mean()

def compute_contagion(df):
    pivot = df.pivot_table(
        index="tanggal",
        columns="wilayah",
        values="harga"
    )
    corr = pivot.corr().mean().mean()
    return corr

def compute_macro_score(forecast_risk, volatility, contagion, pressure):

    score = (
        0.35 * forecast_risk +
        0.25 * volatility +
        0.20 * contagion +
        0.20 * pressure
    )

    return score

def national_forecast_30(df):

    df_nat = df.groupby("tanggal")["harga"].mean().reset_index()
    df_nat = df_nat.dropna()

    if len(df_nat) < 60:
        return None, None

    df_nat["t"] = np.arange(len(df_nat))

    X = df_nat[["t"]]
    y = df_nat["harga"]

    model = LinearRegression()
    model.fit(X, y)

    future_t = np.arange(len(df_nat), len(df_nat)+30).reshape(-1,1)
    preds = model.predict(future_t)

    return preds[-1], preds
