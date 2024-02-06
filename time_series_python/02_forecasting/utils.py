import pandas as pd
from statsmodels.tsa.stattools import adfuller


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
