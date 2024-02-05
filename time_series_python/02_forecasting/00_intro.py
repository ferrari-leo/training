import os
import numpy as np
import pandas as pd
from pylab import rcParams
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.tools import diff
from sklearn.metrics import mean_absolute_error, mean_squared_error

rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"
df = pd.read_csv(os.path.join(dir_data, "airline_passengers.csv"), index_col="Month")
df.dropna(inplace=True)
df.index = pd.to_datetime(df.index)
df.index.freq = "MS"
display(df.tail())

train_data = df.iloc[:109]
test_data = df.iloc[108:]

fitted_model = ExponentialSmoothing(
    train_data["Thousands of Passengers"],
    trend="mul",
    seasonal="mul",
    seasonal_periods=12,
).fit()

test_predictions = fitted_model.forecast(36)

train_data["Thousands of Passengers"].plot(legend=True, label="TRAIN")
test_data["Thousands of Passengers"].plot(legend=True, label="TEST")
test_predictions.plot(legend=True, label="PREDICTION")

print(mean_absolute_error(test_data, test_predictions))
print(np.sqrt(mean_squared_error(test_data, test_predictions)))

final_model = ExponentialSmoothing(
    df["Thousands of Passengers"], trend="mul", seasonal="mul", seasonal_periods=12
).fit()

forecast_predictions = final_model.forecast(36)

df["Thousands of Passengers"].plot(legend=True, label="DATA")
forecast_predictions.plot(legend=True, label="PREDICTION")

# Examples of stationarity
df2 = pd.read_csv(os.path.join(dir_data, "samples.csv"), index_col=0, parse_dates=True)
df2["a"].plot()
df2["b"].plot()

diff(df2["b"], k_diff=1).plot()
