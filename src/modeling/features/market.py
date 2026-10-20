"""Market-wide ("regime") features: the same value for every stock on a given date.

The market's return is the median across all stocks, matching how the model's target removes it.
These say what kind of month it's been (rallying or falling, calm or volatile, stocks moving
together or apart) and mostly help the 10th-90th percentile range, which should be wider in
volatile months.
"""
import numpy as np
import pandas as pd

MONTH = 21  # trading days

REGIME_FEATURES = ["market_ret_1m", "market_vol_1m", "dispersion_1m", "vix", "vix_chg_1m"]


def market_returns(prices):
    """Daily market log return: the median daily log return across all stocks."""
    return np.log(prices.where(prices > 0)).diff().median(axis=1)


def compute_regime(prices, market, vix=None):
    """One row per ticker, every row identical: the market's state as of the last date in `prices`."""
    log_p = np.log(prices.iloc[-(MONTH + 1):].where(lambda p: p > 0)).ffill()
    month = (log_p.iloc[-1] - log_p.iloc[0]).dropna()
    q75, q25 = month.quantile([0.75, 0.25])
    out = {
        "market_ret_1m": month.median(),
        "market_vol_1m": market.reindex(prices.index[-MONTH:]).std(),
        "dispersion_1m": (q75 - q25) / 1.349,  # interquartile range scaled to a standard deviation, robust to bad ticks
        "vix": np.nan, "vix_chg_1m": np.nan,
    }
    if vix is not None:
        level = vix.loc[:prices.index[-1]].dropna()
        if len(level) > MONTH:
            out["vix"], out["vix_chg_1m"] = level.iloc[-1], np.log(level.iloc[-1] / level.iloc[-(MONTH + 1)])
    return pd.DataFrame([out] * prices.shape[1], index=prices.columns)
