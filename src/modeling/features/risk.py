"""Volatility / drawdown features computed from a wide price table (rows = dates, columns = tickers).

Like the slopes, every feature uses only the rows passed in, so passing prices up to an as-of date
gives that date's features with no look-ahead. Volatility is the standard deviation of daily log
returns; drawdown and the distance from the 52-week high are log differences (e.g. -0.1 is about
10% below the peak).
"""
import numpy as np
import pandas as pd

YEAR = 252  # trading days
VOL_WINDOWS = {"1m": 21, "3m": 63}
MIN_COVERAGE = 0.8  # share of a window that must have prices, as for the slopes

RISK_FEATURES = [f"vol_{k}" for k in VOL_WINDOWS] + ["max_dd_1y", "off_52w_high"]


def compute_risk(prices):
    """DataFrame of risk features, one row per ticker, from the trailing year of `prices`."""
    recent = prices.iloc[-(YEAR + 1):]
    log_p = np.log(recent.where(recent > 0))
    ret = log_p.diff()
    out = {}
    for name, n in VOL_WINDOWS.items():
        window = ret.iloc[-n:]
        out[f"vol_{name}"] = window.std().where(window.count() >= MIN_COVERAGE * n)
    year = log_p.iloc[-YEAR:]
    enough = year.count() >= MIN_COVERAGE * YEAR
    filled = year.ffill()
    out["max_dd_1y"] = (filled - filled.cummax()).min().where(enough)
    out["off_52w_high"] = (filled.iloc[-1] - year.max()).where(enough)
    return pd.DataFrame(out, index=prices.columns)
