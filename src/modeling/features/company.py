"""Company features: sector, industry momentum and earnings timing.

Sector and industry come from Yahoo's current classification, applied to every past date (a small
look-ahead, as companies rarely change industry). Industry momentum is the median slope of the
stock's industry, a well-documented return predictor; the vs_industry features say how the stock
compares with that median. Industries with fewer than MIN_INDUSTRY_SIZE stocks fall back to their sector.

The earnings feature is the calendar days since the stock last reported. It only uses past report
dates, so it has no look-ahead, and since companies report about every 91 days it tells the model
when the next report (and its price jump) is likely due.
"""
import numpy as np
import pandas as pd

# Yahoo's sector names, in a fixed order so the categorical codes match between training and prediction
SECTORS = ["Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive", "Energy",
           "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology", "Utilities"]
MIN_INDUSTRY_SIZE = 5
MOMENTUM_SLOPES = ["slope_3m", "slope_6m"]

INDUSTRY_FEATURES = (["sector"] + [f"industry_{s.removeprefix('slope_')}" for s in MOMENTUM_SLOPES]
                     + [f"vs_industry_{s.removeprefix('slope_')}" for s in MOMENTUM_SLOPES])
EARNINGS_FEATURES = ["days_since_earnings"]


def compute_industry(slopes, companies):
    """Sector (categorical) and industry-median slopes, given each ticker's `slopes` and the companies table."""
    info = companies.reindex(slopes.index) if companies is not None else pd.DataFrame(index=slopes.index)
    sector = info.get("sector", pd.Series(np.nan, index=slopes.index))
    industry = info.get("industry", pd.Series(np.nan, index=slopes.index))
    out = pd.DataFrame({"sector": pd.Categorical(sector, categories=SECTORS)}, index=slopes.index)
    for col in MOMENTUM_SLOPES:
        name = col.removeprefix("slope_")
        by_industry = slopes[col].groupby(industry)
        median = by_industry.transform("median").where(by_industry.transform("count") >= MIN_INDUSTRY_SIZE)
        median = median.fillna(slopes[col].groupby(sector).transform("median"))
        out[f"industry_{name}"] = median
        out[f"vs_industry_{name}"] = slopes[col] - median
    return out


def compute_earnings(tickers, as_of, earnings):
    """Calendar days since each ticker's last earnings report on or before `as_of` (NaN if unknown)."""
    if earnings is None:
        return pd.DataFrame({"days_since_earnings": np.nan}, index=tickers)
    last = earnings.loc[earnings["date"] <= as_of].groupby("ticker")["date"].max()
    return pd.DataFrame({"days_since_earnings": (as_of - last).dt.days.reindex(tickers).astype(float)}, index=tickers)
