import os
import pandas as pd
from pandas.plotting import lag_plot
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import (
    AutoReg,
    AutoRegResults,
    ar_select_order,
    AROrderSelectionResults,
)
from sklearn.metrics import mean_squared_error
from pylab import rcParams

rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"

df = pd.read_csv(
    os.path.join(dir_data, "uspopulation.csv"), index_col="DATE", parse_dates=True
)
df.index.freq = "MS"
display(df.head())
df.plot()

train = df.iloc[:84]
test = df.iloc[84:]

model = AutoReg(train["PopEst"], lags=1)
AR1fit = model.fit()
print(AR1fit.ar_lags)
print(AR1fit.params)

start = len(train)
end = start + len(test) - 1

pred1 = AR1fit.predict(start=start, end=end).rename("AR(1) pred")

test.plot(legend=True)
pred1.plot(legend=True)

AR2fit = AutoReg(train["PopEst"], lags=2).fit()
print(AR2fit.params)
pred2 = AR2fit.predict(start=start, end=end).rename("AR(2) pred")

test.plot(legend=True)
pred1.plot(legend=True)
pred2.plot(legend=True)

ARfit = ar_select_order(train["PopEst"], maxlag=10).model.fit()
pred = ARfit.predict(start=start, end=end).rename("AR pred")
print(ARfit.params)

test.plot(legend=True)
pred1.plot(legend=True)
pred2.plot(legend=True)
pred.plot(legend=True)

labels = ["AR1", "AR2", "ARS"]
preds = [pred1, pred2, pred]

for i in range(3):
    error = mean_squared_error(test["PopEst"], preds[i])
    print(f"{labels[i]} MSE = {error}")

# retrain model on full dataset
ARfit = ar_select_order(df, maxlag=10).model.fit()
forecasted = ARfit.predict(start=len(df), end=len(df) + 12).rename("Forecast")

df["PopEst"].plot(legend=True)
forecasted.plot(legend=True)
