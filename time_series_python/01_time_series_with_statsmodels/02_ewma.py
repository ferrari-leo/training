import os
import numpy as np
import pandas as pd
from pylab import rcParams


rcParams["figure.figsize"] = 12, 5

dir_data = "/Users/IFLM/Desktop/vsc/training/time_series_python/data"
airline = pd.read_csv(
    os.path.join(dir_data, "airline_passengers.csv"), index_col="Month"
)
airline.dropna(inplace=True)
airline.index = pd.to_datetime(airline.index)
display(airline.head())
airline["6m sma"] = airline["Thousands of Passengers"].rolling(6).mean()
airline["12m sma"] = airline["Thousands of Passengers"].rolling(12).mean()
airline["ewma 12"] = airline["Thousands of Passengers"].ewm(span=12).mean()
airline[["Thousands of Passengers", "ewma 12"]].plot()
