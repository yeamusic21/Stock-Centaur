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


# moving-average windows in trading days, for where the price sits relative to its recent averages
MA_WINDOWS = {"20d": 20, "50d": 50, "200d": 200}
RETURN_WINDOW = 21  # trading days for the plain 1-month return

MA_FEATURES = [f"vs_sma_{k}" for k in MA_WINDOWS] + ["sma_50d_vs_200d", "ret_1m"]


def compute_moving_averages(prices):
    """Log distance of the last price from each simple moving average, the 50d/200d cross, and the 1m return.

    E.g. vs_sma_50d of 0.05 means the price is about 5% above its 50-day average. Unlike the slopes
    (the trend line's direction), these say how far the price has run ahead of or fallen behind its trend.
    """
    filled = prices.iloc[-max(MA_WINDOWS.values()):].where(lambda p: p > 0)
    last = filled.ffill().iloc[-1]
    sma = {}
    for name, n in MA_WINDOWS.items():
        window = filled.iloc[-n:]
        sma[name] = window.mean().where(window.count() >= MIN_COVERAGE * n)
    out = {f"vs_sma_{k}": np.log(last / v) for k, v in sma.items()}
    out["sma_50d_vs_200d"] = np.log(sma["50d"] / sma["200d"])
    then = filled.iloc[-(RETURN_WINDOW + 1):].ffill().iloc[0]
    out["ret_1m"] = np.log(last / then)
    return pd.DataFrame(out, index=prices.columns)
