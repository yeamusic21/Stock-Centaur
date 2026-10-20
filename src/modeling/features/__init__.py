"""Every price-model feature, computed from data up to an as-of date.

`compute_features` takes the wide price/volume tables cut off at the as-of date (their last row)
plus the slower-changing VIX, company and earnings tables, and returns one row per ticker. Training
calls it once per past as-of date and prediction calls it on today's data, so both see exactly the
same features. Inputs that are missing (e.g. the company data hasn't been pulled yet) leave their
features NaN, which the tree models treat as missing.
"""
from pathlib import Path

import pandas as pd

from .company import EARNINGS_FEATURES, INDUSTRY_FEATURES, compute_earnings, compute_industry
from .market import REGIME_FEATURES, compute_regime, market_returns
from .risk import BETA_FEATURES, RISK_FEATURES, compute_beta, compute_risk
from .trend import MA_FEATURES, WINDOWS, compute_moving_averages, compute_slopes
from .volume import VOLUME_FEATURES, compute_volume

SLOPE_WINDOWS = {k: WINDOWS[k] for k in ["1y", "6m", "3m", "1m", "2w", "3d"]}

FEATURE_GROUPS = {
    "trend": [f"slope_{k}" for k in SLOPE_WINDOWS],
    "risk": RISK_FEATURES,
    "moving_avg": MA_FEATURES,
    "beta": BETA_FEATURES,
    "volume": VOLUME_FEATURES,
    "industry": INDUSTRY_FEATURES,
    "regime": REGIME_FEATURES,
    "earnings": EARNINGS_FEATURES,
}
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
CATEGORICAL_FEATURES = ["sector"]
HISTORY = max(SLOPE_WINDOWS.values())  # trading days of prices needed before the first as-of date

INPUT_FILES = {"prices": "prices.parquet", "volumes": "volumes.parquet", "vix": "vix.parquet",
               "companies": "companies.parquet", "earnings": "earnings.parquet"}


def load_inputs(data_dir):
    """Every input table that exists in `data_dir` (missing ones are None), with volumes aligned to prices."""
    inputs = {k: pd.read_parquet(Path(data_dir) / f) if (Path(data_dir) / f).exists() else None
              for k, f in INPUT_FILES.items()}
    if inputs["vix"] is not None:
        inputs["vix"] = inputs["vix"]["VIX"]
    if inputs["volumes"] is not None:
        inputs["volumes"] = inputs["volumes"].reindex_like(inputs["prices"])
    return inputs


def compute_features(prices, volumes=None, vix=None, companies=None, earnings=None, market=None):
    """Every feature in ALL_FEATURES for each ticker in `prices`, as of its last row.

    `market` is the daily market return series (market_returns(prices)); pass it in when calling this
    for many as-of dates so it isn't recomputed, otherwise it's computed from `prices`.
    """
    as_of = prices.index[-1]
    if market is None:
        market = market_returns(prices.iloc[-(HISTORY + 1):])
    market = market.loc[:as_of]
    slopes = compute_slopes(prices, SLOPE_WINDOWS)
    parts = [
        slopes,
        compute_risk(prices),
        compute_moving_averages(prices),
        compute_beta(prices, market),
        compute_volume(prices, volumes) if volumes is not None
        else pd.DataFrame(index=prices.columns, columns=VOLUME_FEATURES, dtype=float),
        compute_industry(slopes, companies),
        compute_regime(prices, market, vix),
        compute_earnings(prices.columns, as_of, earnings),
    ]
    return pd.concat(parts, axis=1)[ALL_FEATURES]
