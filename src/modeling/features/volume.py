"""Trading-volume features from wide price and volume tables (rows = dates, columns = tickers).

Dollar volume (price x shares traded) is a stand-in for company size and liquidity; the volume
surge compares the last month's trading with the past year's; illiquidity (Amihud) is the average
daily price move per dollar traded, high for thinly traded stocks. All are logged, so they're on a
readable scale; NaN where there isn't enough history.
"""
import numpy as np
import pandas as pd

MONTH, YEAR = 21, 252  # trading days
MIN_COVERAGE = 0.8

VOLUME_FEATURES = ["dollar_vol_1m", "volume_surge", "illiquidity_1m"]


def compute_volume(prices, volumes):
    """DataFrame of volume features, one row per ticker, from the trailing year of `prices` and `volumes`."""
    p = prices.iloc[-(YEAR + 1):].where(lambda x: x > 0)
    v = volumes.reindex_like(p).where(lambda x: x >= 0)
    dollars = (p * v).iloc[-MONTH:]
    month_ok = dollars.count() >= MIN_COVERAGE * MONTH
    year_vol = v.iloc[-YEAR:]
    with np.errstate(divide="ignore", invalid="ignore"):
        move = np.log(p).diff().abs().iloc[-MONTH:]
        per_dollar = (move / dollars.where(dollars > 0)).mean()
        out = {
            "dollar_vol_1m": np.log(dollars.mean()).where(month_ok),
            "volume_surge": np.log(v.iloc[-MONTH:].mean() / year_vol.mean())
            .where(month_ok & (year_vol.count() >= MIN_COVERAGE * YEAR)),
            "illiquidity_1m": np.log(per_dollar * 1e6).where(month_ok),  # log of % move per $1M traded, roughly
        }
    return pd.DataFrame(out, index=prices.columns).replace([np.inf, -np.inf], np.nan)
