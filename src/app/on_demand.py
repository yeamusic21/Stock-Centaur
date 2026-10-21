"""Price-model predictions for tickers outside the ranked set, such as ETFs or funds held at a brokerage.

data/predictions.json only covers the ranked stocks. For any other ticker this downloads its 5-year
price history (and sector/industry), adds it to the price table so it gets the same market-relative
features, and runs the saved model on it. Results are cached in data/extra_predictions.json (see
sandbox.load_extra) until
the main predictions are refreshed, so the Monte Carlo can treat these tickers like any other.
"""
import copy
import importlib.util
import json
import sys
import threading
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

import sandbox as sandbox_sim

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELING_DIR = REPO_ROOT / "src" / "modeling"
MODEL_PATH = DATA_DIR / "price_model.joblib"
MIN_DAYS = 20  # need some recent history for the short-window features

sys.path.insert(0, str(MODELING_DIR))
from features import load_inputs  # noqa: E402

_spec = importlib.util.spec_from_file_location("predict_prices", MODELING_DIR / "04_predict_prices.py")
predict_prices = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_prices)

_lock = threading.Lock()


def yahoo_symbol(ticker):
    """Brokers write share classes as BRK.B; Yahoo uses BRK-B."""
    return ticker.replace(".", "-")


def _download(tickers, index):
    symbols = {yahoo_symbol(t): t for t in tickers}
    data = yf.download(list(symbols), period="5y", auto_adjust=True, progress=False, threads=True)
    if data.empty:
        return pd.DataFrame(index=index), pd.DataFrame(index=index)
    closes, volumes = data["Close"], data["Volume"]
    if isinstance(closes, pd.Series):
        closes, volumes = closes.to_frame(list(symbols)[0]), volumes.to_frame(list(symbols)[0])
    closes, volumes = closes.rename(columns=symbols), volumes.rename(columns=symbols)
    # line up with the trading days the rest of the model sees
    closes.index, volumes.index = closes.index.tz_localize(None), volumes.index.tz_localize(None)
    return closes.reindex(index).ffill(limit=3), volumes.reindex(index)


def _save_returns_history(closes):
    """Keep the last year of the new tickers' prices so the simulation can correlate them with everything else."""
    recent = closes.iloc[-sandbox_sim.CORR_DAYS - 1:].dropna(axis=1, how="all")
    path = sandbox_sim.EXTRA_PRICES_PATH
    if path.exists():
        old = pd.read_parquet(path)
        if old.index.equals(recent.index):
            recent = pd.concat([old.drop(columns=recent.columns, errors="ignore"), recent], axis=1)
    recent.to_parquet(path)


def _company(ticker):
    try:
        info = yf.Ticker(yahoo_symbol(ticker)).info or {}
    except Exception:
        info = {}
    return {"sector": info.get("sector"), "industry": info.get("industry")}


def predict_missing(tickers):
    """Make sure every ticker has a prediction. Returns {"stocks": {ticker: row}, "failed": {ticker: reason}}
    for the tickers that aren't in data/predictions.json."""
    base = sandbox_sim.load_predictions()
    if base is None:
        raise ValueError("No predictions yet. Run the modeling pipeline first.")
    known = {s["ticker"] for s in base["stocks"]}
    wanted = sorted({str(t).upper() for t in tickers} - known)
    with _lock:
        cache = copy.deepcopy(sandbox_sim.load_extra())
        todo = [t for t in wanted if t not in cache["stocks"] and t not in cache["failed"]]
        if todo:
            _predict_into(cache, todo)
            sandbox_sim.EXTRA_PATH.write_text(json.dumps(cache, allow_nan=False))
    return {"stocks": {t: cache["stocks"][t] for t in wanted if t in cache["stocks"]},
            "failed": {t: cache["failed"][t] for t in wanted if t in cache["failed"]}}


def _predict_into(cache, todo):
    inputs = load_inputs(DATA_DIR)
    prices, volumes, companies = inputs["prices"], inputs["volumes"], inputs["companies"]
    new = [t for t in todo if t not in prices.columns]
    if new:
        closes, vols = _download(new, prices.index)
        prices = pd.concat([prices, closes], axis=1)
        volumes = pd.concat([volumes, vols.reindex(columns=closes.columns)], axis=1) if volumes is not None else None
        _save_returns_history(closes)
        if companies is not None:
            extra = pd.DataFrame({t: _company(t) for t in closes.columns}).T
            companies = pd.concat([companies, extra[~extra.index.isin(companies.index)]])

    ok = []
    for t in todo:
        days = prices[t].notna().sum() if t in prices.columns else 0
        if t not in prices.columns or days == 0:
            cache["failed"][t] = "No price data found (cash, crypto and options can't be modeled)."
        elif days < MIN_DAYS or pd.isna(prices[t].iloc[-1]):
            cache["failed"][t] = "Not enough recent price history to model."
        else:
            ok.append(t)
    if not ok:
        return
    inputs = {**inputs, "prices": prices, "volumes": volumes, "companies": companies}
    table = predict_prices.predict(inputs, ok, joblib.load(MODEL_PATH))
    table = table.round({"price": 2, "pred_price": 2, "pred_return": 2, "low_price": 2, "high_price": 2,
                         "mu": 5, "sigma": 5})
    for row in table.to_dict(orient="records"):
        if pd.isna(row["mu"]) or pd.isna(row["sigma"]) or row["sigma"] <= 0:
            cache["failed"][row["ticker"]] = "The model couldn't score this ticker."
        else:
            cache["stocks"][row["ticker"]] = {**row, "extra": True}
