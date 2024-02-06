import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from pylab import rcParams
from statsmodels.tsa.arima_model import ARMAResults, ARIMAResults
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tools.eval_measures import rmse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils import adf_test

rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"

df1 = pd.read_csv(
    os.path.join(dir_data, "airline_passengers.csv"),
    index_col="Month",
    parse_dates=True,
)
df1.index.freq = "MS"

df2 = pd.read_csv(
    os.path.join(dir_data, "DailyTotalFemaleBirths.csv"),
    index_col="Date",
    parse_dates=True,
)
df2.index.freq = "D"
df2 = df2[:120]

df3 = pd.read_csv(
    os.path.join(dir_data, "TradeInventories.csv"),
    index_col="Date",
    parse_dates=True,
)
df3.index.freq = "MS"

# Pyramid ARIMA (grid search)
stepwise_fit = auto_arima(
    df2["Births"], start_p=0, start_q=0, max_p=6, max_q=3, seasonal=False, trace=True
)

display(stepwise_fit.summary())

stepwise_fit = auto_arima(
    df1["Thousands of Passengers"],
    start_p=0,
    start_q=0,
    max_p=4,
    max_q=4,
    m=12,
    seasonal=True,
    trace=True,
)

display(stepwise_fit.summary())

# ARMA
adf_test(df2["Births"])
auto_arima(df2["Births"], seasonal=False).summary()

train = df2.iloc[:90]
test = df2.iloc[90:]

model = ARIMA(train["Births"], order=(2, 0, 2))
results = model.fit()
results.summary()

start = len(train)
end = len(train) + len(test) - 1
preds = results.predict(start, end).rename("ARMA 2,2")

test["Births"].plot(legend=True)
preds.plot(legend=True)

df3.plot()
result = seasonal_decompose(df3["Inventories"], model="add")
result.plot()

auto_arima(df3["Inventories"], seasonal=False).summary()

df3["diff1"] = diff(df3["Inventories"], k_diff=1)
adf_test(df3["diff1"])

plot_acf(df3["Inventories"], lags=40)
plot_pacf(df3["Inventories"], lags=40)

stepwise_fit = auto_arima(
    df3["Inventories"],
    start_p=0,
    start_q=0,
    seasonal=False,
    max_p=2,
    max_q=2,
    trace=True,
)

stepwise_fit.summary()

train = df3.iloc[:252]
test = df3.iloc[252:]

model = ARIMA(train["Inventories"], order=(1, 1, 1))
results = model.fit()
results.summary()

start = len(train)
end = len(train) + len(test) - 1

preds = results.predict(start, end, typ="levels").rename("ARIMA 1,1,1")

test["Inventories"].plot(legend=True)
preds.plot(legend=True)

error = rmse(test["Inventories"], preds)
error

# forecast
model = ARIMA(df3["Inventories"], order=(1, 1, 1))
results = model.fit()
fcast = results.predict(start=len(df3), end=len(df3) + 11, typ="levels").rename(
    "ARIMA 1,1,1 forecast"
)

df3["Inventories"].plot(legend=True)
fcast.plot(legend=True)
