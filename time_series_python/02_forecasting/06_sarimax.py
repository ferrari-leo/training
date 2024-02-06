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

df = pd.read_csv(
    os.path.join(dir_data, "RestaurantVisitors.csv"), index_col="date", parse_dates=True
)
df.index.freq = "D"

df1 = df.dropna()

cols = ["rest1", "rest2", "rest3", "rest4", "total"]
for c in cols:
    df1[c] = df1[c].astype(int)

ax = df1["total"].plot()
for day in df1.query("holiday==1").index:
    ax.axvline(x=day, color="k")

result = seasonal_decompose(df1["total"])
result.plot()
result.seasonal.plot()

train = df1.iloc[:436]
test = df1.iloc[436:]

# Assess order
auto_arima(df1["total"], seasonal=True, m=7).summary()

# Fit with orders
model = SARIMAX(
    train["total"],
    order=(1, 0, 0),
    seasonal_order=(2, 0, 0, 7),
    enforce_invertibility=False,  # allows weights for error to have modulus >=1
)

results = model.fit()
results.summary()

start = len(train)
end = start + len(test) - 1
pred = results.predict(start, end).rename("SARIMA")

ax = test["total"].plot(legend=True)
pred.plot(legend=True)

for day in test.query("holiday==1").index:
    ax.axvline(x=day, color="k")

print(rmse(test["total"], pred))

# Now add ex variable - check again orders
auto_arima(
    df1["total"],
    exogenous=df1[["holiday"]],  # Need dataframe format
    seasonal=True,
    m=7,
).summary()

# Train SARIMAX
model = SARIMAX(
    train["total"],
    exog=train[["holiday"]],
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 7),
    enforce_invertibility=False,
)

result = model.fit()

result.summary()

start = len(train)
end = start + len(test) - 1
pred = result.predict(start, end, exog=test[["holiday"]]).rename("SARIMAX + holiday")

ax = test["total"].plot(legend=True)
pred.plot(legend=True)
for day in test.query("holiday==1").index:
    ax.axvline(x=day, color="k")

print(rmse(test["total"], pred))

# Run full model
model = SARIMAX(
    df1["total"],
    exog=df1[["holiday"]],
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 7),
    enforce_invertibility=False,
)

result = model.fit()

exog_fcast = df[478:][["holiday"]]  # Get forcast exogenous variables

fcast = results.predict(start=len(df1), end=len(df1) + 38, exog=exog_fcast).rename(
    "SARIMAX final"
)

ax = df1["total"].plot(legend=True)
fcast.plot(legend=True)

for day in df.query("holiday==1").index:
    ax.axvline(x=day, color="k")
