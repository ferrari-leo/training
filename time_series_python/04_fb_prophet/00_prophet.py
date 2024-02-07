import os
import pandas as pd
import numpy as np
from pylab import rcParams
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric, add_changepoints_to_plot
from statsmodels.tools.eval_measures import rmse

rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"

df = pd.read_csv(os.path.join(dir_data, "BeerWineLiquor.csv"))
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])

df2 = pd.read_csv(os.path.join(dir_data, "Miles_Traveled.csv"))
df2.columns = ["ds", "y"]
df2["ds"] = pd.to_datetime(df2["ds"])

df3 = pd.read_csv(os.path.join(dir_data, "HospitalityEmployees.csv"))
df3.columns = ["ds", "y"]
df3["ds"] = pd.to_datetime(df3["ds"])

df4 = pd.read_csv(os.path.join(dir_data, "airline_passengers.csv"))
df4.columns = ["ds", "y"]
df4["ds"] = pd.to_datetime(df4["ds"])

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=24, freq="MS")
fcast = m.predict(future)
m.plot(fcast)

m.plot_components(fcast)

nobs = 12
train = df2.iloc[:-nobs]
test = df2.iloc[-nobs:]

m = Prophet()
m.fit(train)
m.make_future_dataframe(periods=12, freq="MS")
fcast = m.predict(future)

ax = test.plot(x="ds", y="y", label="test", legend=True)
fcast.plot(
    x="ds",
    y="yhat",
    label="pred",
    legend=True,
    ax=ax,
    xlim=("2018-01-01", "2018-12-01"),
)

pred = fcast.iloc[-nobs:]["yhat"]
print(rmse(test["y"], pred))

# Evaluation
initial = str(5 * 365) + " days"
period = str(5 * 365) + " days"
horizon = str(365) + " days"

df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)

performance_metrics(df_cv)
plot_cross_validation_metric(df_cv, metric="rmse")

# Trend Changes
df3.plot(x="ds", y="y")
m = Prophet()
m.fit(df3)
future = m.make_future_dataframe(periods=12, freq="MS")
fcast = m.predict(future)

fig = m.plot(fcast)
a = add_changepoints_to_plot(fig.gca(), m, fcast)

# Seasonality
df4.plot(x="ds", y="y")
m = Prophet(seasonality_mode="multiplicative")
m.fit(df4)
future = m.make_future_dataframe(periods=50, freq="MS")
fcast = m.predict(future)

fig = m.plot(fcast)
a = add_changepoints_to_plot(fig.gca(), m, fcast)

m.plot_components(fcast)
