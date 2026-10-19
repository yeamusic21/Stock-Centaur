"""Trend / slope features computed from a wide price table (rows = dates, columns = tickers).

A slope is the least-squares fit of log(price) against time over a trailing window, expressed
as the fitted % change over that window (e.g. a 1y slope of 20 means the trend line rose 20%
over the year). Using log prices makes slopes comparable across stocks of any price level.
These same features will feed the 30-day price model.
"""
import numpy as np
import pandas as pd

# trailing windows in trading days
WINDOWS = {
    "5y": 1260,
    "1y": 252,
    "6m": 126,
    "3m": 63,
    "1m": 21,
    "2w": 10,
    "3d": 3,
}

# share of a window that must have prices for its slope to be trusted
MIN_COVERAGE = 0.8


def window_slopes(prices, window):
    """Fitted % change from a log-price regression over the last `window` rows of `prices`.

    Vectorized across all columns and tolerant of missing values. Returns a Series indexed by
    ticker, NaN where coverage is too thin.
    """
    recent = prices.iloc[-window:]
    log_p = np.log(recent.where(recent > 0)).to_numpy()
    mask = ~np.isnan(log_p)
    x = np.arange(len(recent), dtype=float)[:, None]
    y = np.where(mask, log_p, 0.0)

    n = mask.sum(axis=0)
    sx = (x * mask).sum(axis=0)
    sy = y.sum(axis=0)
    sxx = (x * x * mask).sum(axis=0)
    sxy = (x * y).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)

    enough = n >= max(2, MIN_COVERAGE * window)
    growth = np.where(enough, np.expm1(slope * (window - 1)) * 100, np.nan)
    return pd.Series(growth, index=prices.columns)


def compute_slopes(prices, windows=WINDOWS):
    """DataFrame of slopes, one column per window (e.g. 'slope_1y'), one row per ticker."""
    return pd.DataFrame({f"slope_{name}": window_slopes(prices, n) for name, n in windows.items()})
