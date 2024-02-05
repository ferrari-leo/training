import os
import pandas as pd
from pandas.plotting import lag_plot
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pylab import rcParams

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

display(df1.head())
display(df2.head())

lag_plot(df1["Thousands of Passengers"])
lag_plot(df2["Births"])

plot_acf(df1, lags=40)
plot_acf(df2, lags=40)
plot_pacf(df2, lags=40)
