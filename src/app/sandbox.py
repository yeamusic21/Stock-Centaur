"""Monte Carlo simulation of a hypothetical portfolio's value ~30 days out.

Each stock's log return is drawn from a normal distribution with the mean (mu) and spread (sigma)
the price model predicted for it: data/predictions.json for the ranked stocks, plus
data/extra_predictions.json for other tickers scored on demand (see on_demand.py). Stocks don't move independently, so the draws are correlated using the
correlation of each pair's daily returns over the last year.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = REPO_ROOT / "data" / "predictions.json"
EXTRA_PATH = REPO_ROOT / "data" / "extra_predictions.json"
PRICES_PATH = REPO_ROOT / "data" / "prices.parquet"
EXTRA_PRICES_PATH = REPO_ROOT / "data" / "extra_prices.parquet"  # last year of on-demand tickers' prices

SIMULATIONS = 10_000
CORR_DAYS = 252
CORR_MIN_DAYS = 60  # pairs with less shared history are treated as uncorrelated
PERCENTILES = [5, 10, 25, 50, 75, 90, 95]
HIST_BINS = 40
MAX_PICKS = 20
MIN_PICK_PRICE = 5  # skip penny stocks, whose forecasts are the noisiest
PICK_POOL = 60  # candidates the lowest-chance-of-loss search chooses from

_cache = {}


def _load(path, reader):
    """Read a data file, re-reading it only when it changes."""
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    entry = _cache.get(path)
    if not entry or entry[0] != mtime:
        _cache[path] = entry = (mtime, reader(path))
    return entry[1]


def load_predictions():
    return _load(PREDICTIONS_PATH, lambda p: json.loads(p.read_text()))


def load_extra():
    """Predictions made on demand for unranked tickers, dropped once the main predictions move to a newer date."""
    base = load_predictions()
    cache = _load(EXTRA_PATH, lambda p: json.loads(p.read_text()))
    if base is None or cache is None or cache.get("as_of") != base["as_of"]:
        return {"as_of": base and base["as_of"], "stocks": {}, "failed": {}}
    return cache


def predictions_by_ticker():
    """Every prediction the simulation can use, ranked and on-demand, keyed by ticker."""
    data = load_predictions()
    return {**load_extra()["stocks"], **{s["ticker"]: s for s in data["stocks"]}}


def _daily_returns():
    prices = _load(PRICES_PATH, pd.read_parquet)
    if prices is None:
        return None
    prices = prices.iloc[-CORR_DAYS - 1:]
    extra = _load(EXTRA_PRICES_PATH, pd.read_parquet)
    if extra is not None:
        prices = pd.concat([prices, extra.reindex(prices.index).drop(columns=prices.columns, errors="ignore")], axis=1)
    return np.log(prices).diff().iloc[1:]


def correlation(tickers):
    """Correlation matrix of daily log returns, nudged to be positive semi-definite."""
    returns = _daily_returns()
    if returns is None:
        return np.eye(len(tickers))
    cols = [t if t in returns.columns else None for t in tickers]
    frame = pd.DataFrame({i: returns[c] if c else np.nan for i, c in enumerate(cols)})
    corr = frame.corr(min_periods=CORR_MIN_DAYS).fillna(0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1)
    # pairwise correlations with gaps aren't always a valid matrix; clip negative eigenvalues
    vals, vecs = np.linalg.eigh(corr)
    corr = vecs @ np.diag(np.clip(vals, 1e-8, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def simulate(holdings):
    """holdings: list of {"ticker", "amount"} where amount is dollars invested today."""
    by_ticker = predictions_by_ticker()
    rows = [(h["ticker"], float(h["amount"])) for h in holdings if h["ticker"] in by_ticker and float(h["amount"]) > 0]
    if not rows:
        raise ValueError("Add at least one stock with an amount above $0.")

    tickers = [t for t, _ in rows]
    amounts = np.array([a for _, a in rows])
    mu = np.array([by_ticker[t]["mu"] for t in tickers])
    sigma = np.array([by_ticker[t]["sigma"] for t in tickers])

    chol = np.linalg.cholesky(correlation(tickers))
    z = np.random.default_rng(0).standard_normal((SIMULATIONS, len(tickers))) @ chol.T
    values = np.exp(mu + sigma * z) * amounts
    totals = values.sum(axis=1)

    invested = float(amounts.sum())
    counts, edges = np.histogram(totals, bins=HIST_BINS, range=tuple(np.percentile(totals, [0.5, 99.5])))
    return {
        "invested": invested,
        "simulations": SIMULATIONS,
        "expected": float(totals.mean()),
        "percentiles": {str(p): float(v) for p, v in zip(PERCENTILES, np.percentile(totals, PERCENTILES))},
        "prob_loss": float((totals < invested).mean()),
        "histogram": {"counts": counts.tolist(), "edges": edges.round(2).tolist()},
        "holdings": [
            {"ticker": t, "amount": float(a), "median": float(np.median(v)),
             "low": float(np.percentile(v, 10)), "high": float(np.percentile(v, 90))}
            for t, a, v in zip(tickers, amounts, values.T)
        ],
    }


def _loss_chance(mu, cov, weights):
    """Normal approximation of the chance a weighted portfolio's log return is below zero."""
    m, s = weights @ mu, np.sqrt(weights @ cov @ weights)
    return norm.cdf(-m / s) if s > 0 else float(m < 0)


def autopick(objective, budget, count, min_price=MIN_PICK_PRICE):
    """Pick `count` stocks and split `budget` evenly between them.

    "return": the stocks with the highest predicted 30-day return.
    "safety": greedily builds the equal-weight portfolio with the lowest chance of a loss, so it
    favors stocks that are both unlikely to fall and don't tend to fall together.
    """
    budget, count = float(budget), int(count)
    if objective not in ("return", "safety"):
        raise ValueError("Choose a goal: highest projected return or lowest chance of loss.")
    if budget <= 0 or not 1 <= count <= MAX_PICKS:
        raise ValueError(f"Use a budget above $0 and between 1 and {MAX_PICKS} stocks.")

    stocks = pd.DataFrame(load_predictions()["stocks"])
    stocks = stocks[(stocks["price"] >= float(min_price)) & (stocks["sigma"] > 0)]
    if len(stocks) < count:
        raise ValueError("Not enough stocks match. Try a lower minimum price.")

    if objective == "return":
        picks = stocks.nlargest(count, "mu")["ticker"].tolist()
    else:
        stocks = stocks.assign(loss=norm.cdf(-stocks["mu"] / stocks["sigma"]))
        pool = stocks.nsmallest(max(PICK_POOL, count), "loss").reset_index(drop=True)
        mu, sigma = pool["mu"].to_numpy(), pool["sigma"].to_numpy()
        cov = correlation(pool["ticker"].tolist()) * np.outer(sigma, sigma)
        chosen = []
        for _ in range(count):
            best = min((i for i in range(len(pool)) if i not in chosen),
                       key=lambda i: _loss_chance(mu[chosen + [i]], cov[np.ix_(chosen + [i], chosen + [i])],
                                                  np.full(len(chosen) + 1, 1 / (len(chosen) + 1))))
            chosen.append(best)
        picks = pool.loc[chosen, "ticker"].tolist()

    return {"objective": objective, "holdings": [{"ticker": t, "amount": round(budget / count, 2)} for t in picks]}
