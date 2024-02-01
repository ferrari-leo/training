import os
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import dates

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"

# region Basic datetime index
my_year = 2020
my_month = 1
my_day = 2
my_hour = 13
my_min = 30
my_sec = 15

my_date = datetime(my_year, my_month, my_day)
print(my_date)

my_datetime = datetime(my_year, my_month, my_day, my_hour, my_min, my_sec)
print(my_datetime)
print(type(my_datetime))

np.array(["2020-03-15", "2020-03-16", "2020-03-17"], dtype="datetime64")

np.arange("2018-06-01", "2018-06-23", 7, dtype="datetime64[D]")
# endregion
# region Pandas datetime index
pd.date_range("2020-01-01", periods=7, freq="D")

data = np.random.randn(3, 2)
cols = ["A", "B"]
idx = pd.date_range("2020-01-01", periods=3, freq="D")
df = pd.DataFrame(data, index=idx, columns=cols)
display(df)
print(df.index.max())
print(df.index.argmax())
# endregion
# region Resampling
df = pd.read_csv(
    os.path.join(dir_data, "starbucks.csv"), index_col="Date", parse_dates=True
)
display(df.head())

# resample daily data to yearly data
df.resample(rule="A").mean()


# custom resampling function
def first_day(entry):
    if len(entry):
        return entry[0]


df.resample(rule="A").apply(first_day)

df["Close"].resample(rule="M").mean().plot.bar()
# endregion
# region Time Shifting
display(df.shift(1))
display(df.shift(-1))
display(df.shift(periods=1, freq="M"))
# endregion
# region Rolling and Expanding
df["Close"].plot(figsize=(12, 5))
df["Close 30 day mean"] = df["Close"].rolling(window=30).mean()
df[["Close", "Close 30 day mean"]].plot(figsize=(12, 5))

df["Close"].expanding().mean().plot()
# endregion
# region Visualisation
title = "TITLE"
ylabel = "Y LABEL"
xlabel = "X LABEL"

ax = df["Close"].plot(figsize=(12, 6), title=title)
ax.autoscale(axis="both", tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)

# set axis limits
df["Close"]["2017-01-01":"2017-12-31"].plot(figsize=(12, 5))
df["Close"].plot(
    figsize=(12, 5), xlim=["2017-01-01", "2017-12-31"], ylim=[0, 70], ls="--", c="green"
)

# set ticks and format
ax = df["Close"].plot(figsize=(12, 5), xlim=["2017-01-01", "2017-03-01"], ylim=[50, 60])
ax.set(xlabel="")
ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(dates.DateFormatter("%d"))
ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter("\n\n%b"))
ax.yaxis.grid(True)
ax.xaxis.grid(True)
#

# endregion
