import os
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"
df = pd.read_csv(os.path.join(dir_data, "macrodata.csv"), index_col=0, parse_dates=True)
display(df.head())

df["realgdp"].plot(figsize=(12, 5))

# HP filter
gdp_cycle, gdp_trend = hpfilter(df["realgdp"], lamb=1600)
df["trend"] = gdp_trend
df[["realgdp", "trend"]].plot(figsize=(12, 5))
