import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def prepare_features(df):
    df = df.sort_values("tanggal")
    df["month"] = df["tanggal"].dt.month
    df["day"] = df["tanggal"].dt.day
    df["lag1"] = df["harga"].shift(1)
    df["lag7"] = df["harga"].shift(7)
    df = df.dropna()
    return df

def ensemble_forecast(df):

    df = prepare_features(df)

    if len(df) < 60:
        return None

    X = df[["month","day","lag1","lag7"]]
    y = df["harga"]

    split = int(len(df)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    lr_pred = lr.predict(X_test)

    ensemble_pred = (rf_pred + lr_pred) / 2

    mae = mean_absolute_error(y_test, ensemble_pred)
    mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100

    last_row = df.iloc[-1]
    last_price = last_row["harga"]

    future_preds = []

    for i in range(7):

        future_X = X.iloc[[-1]]

        rf_future = rf.predict(future_X)[0]
        lr_future = lr.predict(future_X)[0]

        final_pred = (rf_future + lr_future) / 2

        future_preds.append(final_pred)
        last_price = final_pred

    forecast_change = (
        future_preds[-1] - df["harga"].iloc[-1]
    ) / df["harga"].iloc[-1]

    return {
        "forecast_price": future_preds[-1],
        "forecast_change": forecast_change,
        "mae": mae,
        "mape": mape
    }

def scan_all_regions(df, level="provinsi"):

    results = []

    if level == "provinsi":
        regions = df["provinsi"].unique()
        group_cols = ["provinsi","tanggal"]
        region_col = "provinsi"
    else:
        regions = df["wilayah"].unique()
        group_cols = ["wilayah","tanggal"]
        region_col = "wilayah"

    df_grouped = df.groupby(group_cols)["harga"].mean().reset_index()

    for r in regions:
        df_w = df_grouped[df_grouped[region_col] == r]
        res = ensemble_forecast(df_w)

        if res:
            results.append({
                "wilayah": r,
                "forecast_change": res["forecast_change"],
                "mape": res["mape"]
            })

    return pd.DataFrame(results)
