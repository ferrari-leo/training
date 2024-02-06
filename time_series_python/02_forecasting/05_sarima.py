import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils import adf_test

rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"

df = pd.read_csv(os.path.join(dir_data, "co2_mm_mlo.csv"))

df["date"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})

df.set_index(df["date"], inplace=True)

df.index.freq = "MS"

df["interpolated"].plot()

result = seasonal_decompose(df["interpolated"], model="add")
result.plot()

auto_arima(df["interpolated"], seasonal=True, m=12).summary()

train = df.iloc[:717]
test = df.iloc[717:]

model = SARIMAX(df["interpolated"], order=(0, 1, 3), seasonal_order=(1, 0, 1, 12))
results = model.fit()
results.summary()

start = len(train)
end = len(train) + len(test) - 1
pred = results.predict(start, end, typ="levels").rename("SARIMA")

test["interpolated"].plot(legend=True)
pred.plot(legend=True)

rmse(test["interpolated"], pred)

# Forecast
model = SARIMAX(df["interpolated"], order=(0, 1, 3), seasonal_order=(1, 0, 1, 12))
results = model.fit()
results.summary()
start = len(df)
end = start + 10 * 12 - 1
fcast = results.predict(start, end, typ="levels").rename("SARIMA forecast")
df["interpolated"].plot(legend=True)
fcast.plot(legend=True)
