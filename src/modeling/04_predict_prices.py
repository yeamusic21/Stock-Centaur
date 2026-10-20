"""Predict every ranked stock's price ~30 days out with the trained price model.

Reads the input tables in data/ (prices, volumes, VIX, companies, earnings dates), data/trend_table.json
(for the list of ranked tickers) and data/price_model.joblib (from 03_train_price_model.py), and
writes data/predictions.json with the
predicted price and a calibrated 10th-90th percentile range for each ticker. It also stores the log-return
mean (mu) and spread (sigma) the Portfolio Sandbox uses for its Monte Carlo simulation.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from features import compute_features, load_inputs

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TREND_TABLE_PATH = DATA_DIR / "trend_table.json"
MODEL_PATH = DATA_DIR / "price_model.joblib"
PREDICTIONS_PATH = DATA_DIR / "predictions.json"

HORIZON_CALENDAR_DAYS = 30


def predict(inputs, tickers, bundle):
    models, quantiles = bundle["models"], bundle["quantiles"]
    prices = inputs["prices"]
    # market-wide and industry features need every stock, not just the ranked ones
    X = compute_features(prices, inputs["volumes"], inputs["vix"], inputs["companies"], inputs["earnings"])
    X = X.loc[tickers, bundle["features"]]
    # the point model predicts the excess return over the market; add back the average market return
    drift, widen = bundle["market_drift"], bundle["range_adjustment"]
    mu = models["point"].predict(X) + drift
    # quantile models are fit separately and can cross; order them, widen by the conformal calibration
    # from training, and keep the range around the point estimate. They predict either the raw return
    # or (older models) the excess return, which needs the drift added back too
    shift = drift if bundle.get("range_target_column", "excess") == "excess" else 0.0
    a, b = models["low"].predict(X) + shift, models["high"].predict(X) + shift
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
    inputs = load_inputs(DATA_DIR)
    trend = json.loads(TREND_TABLE_PATH.read_text())
    bundle = joblib.load(MODEL_PATH)
    tickers = [s["ticker"] for s in trend["stocks"]]

    table = predict(inputs, tickers, bundle)
    as_of = inputs["prices"].index.max()
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
