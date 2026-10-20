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


BETA_FEATURES = ["beta_1y", "idio_vol_1y"]


def compute_beta(prices, market):
    """Beta and idiosyncratic volatility against `market` (daily market log returns) over the trailing year.

    Beta is how much the stock moves per unit of market move; idiosyncratic volatility is the daily
    standard deviation of what's left after removing that market part, i.e. stock-specific risk.
    """
    recent = prices.iloc[-(YEAR + 1):]
    r = np.log(recent.where(recent > 0)).diff().iloc[1:].to_numpy()
    m = market.reindex(recent.index[1:]).to_numpy()[:, None]
    mask = ~np.isnan(r) & ~np.isnan(m)
    n = mask.sum(axis=0)
    r0, m0 = np.where(mask, r, 0.0), np.where(mask, m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_r, mean_m = r0.sum(axis=0) / n, m0.sum(axis=0) / n
        cov = (r0 * m0).sum(axis=0) / n - mean_r * mean_m
        var_m = (m0 * m0).sum(axis=0) / n - mean_m ** 2
        var_r = (r0 * r0).sum(axis=0) / n - mean_r ** 2
        beta = cov / var_m
        idio = np.sqrt(np.maximum(var_r - beta * cov, 0))
    enough = n >= MIN_COVERAGE * YEAR
    return pd.DataFrame({"beta_1y": np.where(enough, beta, np.nan), "idio_vol_1y": np.where(enough, idio, np.nan)},
                        index=prices.columns)
