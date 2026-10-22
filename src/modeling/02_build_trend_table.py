"""Build the trend rankings table the web app reads.

Reads data/prices.parquet (from data_gathering/01_pull_stock_data.py), computes slope features,
and writes data/trend_table.json. If data/sp500.parquet exists, the S&P 500 (SPY) gets the same stats as a
benchmark row, and each stock's 1-year return is compared against it.

The trend score blends the 5yr, 1yr and 1mon slopes. Each slope is first converted to a
percentile rank across all stocks so that the noisy, large-magnitude short-term slopes don't
drown out the long-term ones; the score is the weighted average of those percentiles (0-100).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.trend import compute_slopes

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"
SP500_PATH = DATA_DIR / "sp500.parquet"
TREND_TABLE_PATH = DATA_DIR / "trend_table.json"

TREND_WEIGHTS = {"5y": 0.3, "1y": 0.4, "1m": 0.3}
STALE_AFTER_DAYS = 10  # drop tickers whose last price is this far behind the newest data
MAX_DAILY_JUMP = 5  # a 5x move in one day is almost always an unadjusted split or bad data
SPARKLINE_POINTS = 52  # weekly closes over the last year


def trend_scores(pct_ranks, weights):
    """Weighted average of percentile ranks, re-weighting over whichever slopes are available."""
    w = pd.Series(weights)
    cols = [f"pct_{k}" for k in w.index]
    values = pct_ranks[cols].to_numpy()
    w_arr = np.where(np.isnan(values), 0.0, w.to_numpy())
    with np.errstate(invalid="ignore"):
        return np.nansum(values * w_arr, axis=1) / w_arr.sum(axis=1)


def price_stats(prices):
    """Last price, 1-day change, 1-year return and a weekly sparkline for each column of `prices`."""
    filled = prices.ffill()
    year = filled.iloc[-252:]
    weekly = year.iloc[::-5][::-1].iloc[-SPARKLINE_POINTS:]
    return pd.DataFrame({
        "price": filled.iloc[-1],
        "change_1d": (filled.iloc[-1] / filled.iloc[-2] - 1) * 100,
        "return_1y": (year.iloc[-1] / year.iloc[0] - 1) * 100,
        "sparkline": [weekly[t].dropna().round(2).tolist() for t in prices.columns],
    }, index=prices.columns)


def benchmark_row(sp500, dates):
    """The S&P 500's slopes and price stats over the same trading days as the stocks."""
    spy = sp500.reindex(dates)
    row = compute_slopes(spy).join(price_stats(spy)).iloc[0]
    return {"ticker": sp500.columns[0], "name": "S&P 500", **row.to_dict()}


def build_trend_table(prices):
    last_date = prices.index.max()
    last_valid = prices.apply(pd.Series.last_valid_index)
    active = last_valid >= last_date - pd.Timedelta(days=STALE_AFTER_DAYS)
    daily = prices.ffill().pct_change(fill_method=None) + 1
    clean = ((daily < MAX_DAILY_JUMP) & (daily > 1 / MAX_DAILY_JUMP) | daily.isna()).all()
    prices = prices.loc[:, active & clean]

    table = compute_slopes(prices)
    # a stock must have a full year of history to be ranked
    table = table[table["slope_1y"].notna()]
    prices = prices[table.index]

    for k in TREND_WEIGHTS:
        table[f"pct_{k}"] = table[f"slope_{k}"].rank(pct=True) * 100
    table["trend"] = trend_scores(table, TREND_WEIGHTS)

    table = table.join(price_stats(prices))
    table = table.sort_values("trend", ascending=False)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.index.name = "ticker"
    return table.reset_index(), last_date


def to_records(table):
    numeric = table.select_dtypes("number").columns
    table[numeric] = table[numeric].round(2)
    records = table.to_dict(orient="records")
    for r in records:  # NaN isn't valid JSON
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
    return records


if __name__ == "__main__":
    prices = pd.read_parquet(PRICES_PATH)
    table, as_of = build_trend_table(prices)
    benchmark = None
    if SP500_PATH.exists():
        benchmark = benchmark_row(pd.read_parquet(SP500_PATH), prices.index)
        # percentage points the stock's 1-year return beat (or trailed) the S&P 500's by
        table["vs_sp_1y"] = table["return_1y"] - benchmark["return_1y"]
        benchmark = to_records(pd.DataFrame([benchmark]))[0]
    output = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "weights": TREND_WEIGHTS,
        "benchmark": benchmark,
        "stocks": to_records(table),
    }
    TREND_TABLE_PATH.write_text(json.dumps(output, allow_nan=False))
    print(f"Ranked {len(table)} stocks as of {output['as_of']} -> {TREND_TABLE_PATH}")
    print(table[["rank", "ticker", "trend", "slope_5y", "slope_1y", "slope_1m", "price"]].head(10).to_string(index=False))
