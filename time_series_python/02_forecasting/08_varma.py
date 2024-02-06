import os
import pandas as pd
import numpy as np
from pylab import rcParams
from pmdarima import auto_arima
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse
from utils import adf_test

rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"

df = pd.read_csv(
    os.path.join(dir_data, "M2SLMoneyStock.csv"), index_col=0, parse_dates=True
)
df.index.freq = "MS"

sp = pd.read_csv(
    os.path.join(dir_data, "PCEPersonalSpending.csv"), index_col=0, parse_dates=True
)
sp.index.freq = "MS"

df = df.join(sp)

df.dropna(inplace=True)

df.plot()

# Run autoarima to get orders
display(auto_arima(df["Money"]).summary())  # gives (1,2,1)
display(auto_arima(df["Spending"]).summary())  # gives (1,1,2)
# Use VARMA(1,2), need to manually difference twice

# Check for stationarity
adf_test(df["Money"])
adf_test(df["Spending"])

df_diff1 = df.diff().dropna()
adf_test(df_diff1["Money"])
adf_test(df_diff1["Spending"])  # This is stationary
df_diff1.plot()

df_diff2 = df_diff1.diff().dropna()
adf_test(df_diff2["Money"])  # This is now stationary
adf_test(df_diff2["Spending"])  # Just a check
df_diff2.plot()

# Train test split
nobs = 12
train = df_diff2[:-nobs]
test = df_diff2[-nobs:]

# Fit model

model = VARMAX(train, order=(1, 2), trend="c")
results = model.fit(disp=False)
display(results.summary())


# Forecast
df_fcast = results.forecast(nobs).rename(
    columns={"Money": "Money_2d", "Spending": "Spending_2d"}
)

# Invert differencing
df_fcast["Money_1d"] = (
    df["Money"].iloc[-nobs - 1] - df["Money"].iloc[-nobs - 2]
) + df_fcast["Money_2d"].cumsum()
df_fcast["MoneyForecast"] = df["Money"].iloc[-nobs - 1] + df_fcast["Money_1d"].cumsum()

df_fcast["Spending_1d"] = (
    df["Spending"].iloc[-nobs - 1] - df["Spending"].iloc[-nobs - 2]
) + df_fcast["Spending_2d"].cumsum()
df_fcast["SpendingForecast"] = (
    df["Spending"].iloc[-nobs - 1] + df_fcast["Spending_1d"].cumsum()
)

# Plot values
test_range = df[-nobs:]
test_range.plot(legend=True)
df_fcast[["MoneyForecast", "SpendingForecast"]].plot(legend=True)

test_range["Money"].plot(legend=True)
df_fcast[-nobs:]["MoneyForecast"].plot(legend=True)

test_range["Spending"].plot(legend=True)
df_fcast[-nobs:]["SpendingForecast"].plot(legend=True)

# Evaluate
print(rmse(test_range["Money"], df_fcast["MoneyForecast"]))
print(rmse(test_range["Spending"], df_fcast["SpendingForecast"]))
