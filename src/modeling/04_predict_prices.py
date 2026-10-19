"""Predict every ranked stock's price ~30 days out with the trained price model.

Reads data/prices.parquet, data/trend_table.json (for the list of ranked tickers) and
data/price_model.joblib (from 03_train_price_model.py), and writes data/predictions.json with the
predicted price and a calibrated 10th-90th percentile range for each ticker. It also stores the log-return
mean (mu) and spread (sigma) the Portfolio Sandbox uses for its Monte Carlo simulation.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from features.risk import compute_risk
from features.trend import WINDOWS, compute_slopes

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"
TREND_TABLE_PATH = DATA_DIR / "trend_table.json"
MODEL_PATH = DATA_DIR / "price_model.joblib"
PREDICTIONS_PATH = DATA_DIR / "predictions.json"

HORIZON_CALENDAR_DAYS = 30


def predict(prices, tickers, bundle):
    models, quantiles = bundle["models"], bundle["quantiles"]
    windows = {f.removeprefix("slope_"): WINDOWS[f.removeprefix("slope_")]
               for f in bundle["features"] if f.startswith("slope_")}
    history = prices[tickers]
    X = compute_slopes(history, windows).join(compute_risk(history))[bundle["features"]]
    # the models predict the excess return over the market; add back the average market return
    drift, widen = bundle["market_drift"], bundle["range_adjustment"]
    mu = models["point"].predict(X) + drift
    # quantile models are fit separately and can cross; order them, widen by the conformal calibration
    # from training, and keep the range around the point estimate
    a, b = models["low"].predict(X) + drift, models["high"].predict(X) + drift
    low = np.minimum(np.minimum(a, b) - widen, mu)
    high = np.maximum(np.maximum(a, b) + widen, mu)
    # spread of a normal distribution with the same low/high percentiles
    sigma = (high - low) / (norm.ppf(quantiles["high"]) - norm.ppf(quantiles["low"]))

    price = prices[tickers].ffill().iloc[-1].to_numpy()
    return pd.DataFrame({
        "ticker": tickers,
        "price": price,
        "pred_price": price * np.exp(mu),
        "pred_return": np.expm1(mu) * 100,
        "low_price": price * np.exp(low),
        "high_price": price * np.exp(high),
        "mu": mu,
        "sigma": sigma,
    })


if __name__ == "__main__":
    prices = pd.read_parquet(PRICES_PATH)
    trend = json.loads(TREND_TABLE_PATH.read_text())
    bundle = joblib.load(MODEL_PATH)
    tickers = [s["ticker"] for s in trend["stocks"]]

    table = predict(prices, tickers, bundle)
    as_of = prices.index.max()
    output = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "target_date": (as_of + pd.Timedelta(days=HORIZON_CALENDAR_DAYS)).strftime("%Y-%m-%d"),
        "horizon_trading_days": bundle["horizon_trading_days"],
        "model": {k: bundle[k] for k in ["trained_through", "features", "train_rows", "validation"]},
        "stocks": table.round({"price": 2, "pred_price": 2, "pred_return": 2, "low_price": 2,
                               "high_price": 2, "mu": 5, "sigma": 5}).to_dict(orient="records"),
    }
    PREDICTIONS_PATH.write_text(json.dumps(output, allow_nan=False))
    print(f"Predicted {len(table)} stocks as of {output['as_of']} -> {PREDICTIONS_PATH}")
    print(table.sort_values("pred_return", ascending=False)[["ticker", "price", "pred_price", "pred_return", "low_price", "high_price"]]
          .head(10).round(2).to_string(index=False))
