import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"
airline = pd.read_csv(
    os.path.join(dir_data, "airline_passengers.csv"),
    index_col="Month",
    parse_dates=True,
)
airline.dropna(inplace=True)

airline.plot()

result = seasonal_decompose(airline["Thousands of Passengers"], model="multiplicative")
result.plot()
