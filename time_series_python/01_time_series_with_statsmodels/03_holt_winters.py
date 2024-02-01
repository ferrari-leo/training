import os
import numpy as np
import pandas as pd
from pylab import rcParams
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"
df = pd.read_csv(os.path.join(dir_data, "airline_passengers.csv"), index_col="Month")
df.dropna(inplace=True)
df.index = pd.to_datetime(df.index)
display(df.head())

df.index.freq = "MS"

span = 12
alpha = 2 / (span + 1)
df["ewma12"] = df["Thousands of Passengers"].ewm(alpha=alpha, adjust=False).mean()
df["ses12"] = (
    SimpleExpSmoothing(df["Thousands of Passengers"])
    .fit(smoothing_level=alpha, optimized=False)
    .fittedvalues.shift(-1)
)
df["des_add12"] = (
    ExponentialSmoothing(df["Thousands of Passengers"], trend="add")
    .fit()
    .fittedvalues.shift(-1)
)
df["des_mul12"] = (
    ExponentialSmoothing(df["Thousands of Passengers"], trend="mul")
    .fit()
    .fittedvalues.shift(-1)
)
df["tes_mul12"] = (
    ExponentialSmoothing(
        df["Thousands of Passengers"], trend="mul", seasonal="mul", seasonal_periods=12
    )
    .fit()
    .fittedvalues
)
df[["Thousands of Passengers", "des_add12", "des_mul12", "tes_mul12"]].iloc[:24].plot()
df[["Thousands of Passengers", "des_add12", "des_mul12", "tes_mul12"]].iloc[-24:].plot()
