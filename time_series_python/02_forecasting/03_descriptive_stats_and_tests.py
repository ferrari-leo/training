import os
import pandas as pd
import numpy as np
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tools.eval_measures import mse, rmse, meanabs
from statsmodels.graphics.tsaplots import month_plot, quarter_plot

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

df3 = pd.read_csv(os.path.join(dir_data, "samples.csv"), index_col=0, parse_dates=True)
df3.index.freq = "MS"

# Run Dickey-Fuller test

dftest = adfuller(df1["Thousands of Passengers"])
dfout = pd.Series(dftest[:4], index=["ADF stat", "p val", "# Lags", "# obs"])

for key, val in dftest[4].items():
    dfout[f"critical value ({key})"] = val

display(dfout)


def adf_test(series, title="", sig=0.05):
    print(f"Augmented Dicket-Fuller Test: {title}")
    result = adfuller(series.dropna(), autolag="AIC")
    labels = ["ADF stat", "p val", "# Lags", "# obs"]
    out = pd.Series(result[:4], index=labels)
    for key, val in result[4].items():
        out[f"critical value ({key})"] = val
    print(out.to_string())

    if result[1] <= sig:
        print(f"Strong evidence against the null hypothesis at {100*sig}% significance")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print(f"Weak evidence against the null hypothesis at {100*sig}% significance")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


adf_test(df1["Thousands of Passengers"])
adf_test(df2["Births"])

# Granger Causality
df3[["a", "d"]].plot()
grangercausalitytests(df3[["a", "d"]], maxlag=3)
grangercausalitytests(df3[["b", "d"]], maxlag=3)

# evaluation measures
np.random.seed(42)

df = pd.DataFrame(np.random.randint(20, 30, (50, 2)), columns=["test", "pred"])

mse(df["test"], df["pred"])

month_plot(df1["Thousands of Passengers"])
quarter_plot(df1["Thousands of Passengers"].resample(rule="Q").mean())
