import os
import pandas as pd
import numpy as np
from pylab import rcParams
from statsmodels.tsa.api import VAR
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

# Grid search for order p of AR component

model = VAR(train)

aic_min = np.inf
p_min = 0
for p in range(8):
    results = model.fit(p)
    if results.aic < aic_min:
        aic_min = results.aic
        p_min = p
    print(f"Order {p}")
    print(f"AIC {results.aic}")
    print()

print(f"Best order is p = {p_min} with AIC {aic_min}")

results = model.fit(p_min)
results.summary()

# Give k lagged values prior to forecast set as nparray
lagged_vals = train.values[-p_min:]

z = results.forecast(y=lagged_vals, steps=12)
idx = pd.date_range("2015-01-01", periods=12, freq="MS")
df_fcast = pd.DataFrame(data=z, index=idx, columns=["Money_2d", "Spending_2d"])

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
df_fcast["MoneyForecast"].plot(legend=True)

test_range["Spending"].plot(legend=True)
df_fcast["SpendingForecast"].plot(legend=True)

# Evaluate
print(rmse(test_range["Money"], df_fcast["MoneyForecast"]))
print(rmse(test_range["Spending"], df_fcast["SpendingForecast"]))
