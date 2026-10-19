"""Train the 30-day price model the Portfolio Sandbox uses.

Reads data/prices.parquet and builds a training set by sampling many historical as-of dates. For
each as-of date, the features are every stock's slopes over the 1y, 6m, 3m, 1m, 2w and 3d windows
plus its 1m/3m volatility, 1y max drawdown and distance from its 52-week high, all computed only
from prices up to that date.

The target is the stock's *excess* log return over the next HORIZON trading days (~30 calendar
days): its return minus that date's median return across all stocks. Market-wide swings are most
of the noise in a single month and can't be predicted from these features, so removing them lets
the trees focus on which stocks beat the rest. A predicted return is the predicted excess return
plus the average market return over the training dates (MARKET_DRIFT in the saved model).

Gradient-boosted trees predict the expected excess return, and two quantile models predict its
10th and 90th percentiles. The quantile range is then conformally calibrated: it is widened by the
amount that makes it contain 80% of validation outcomes, since raw quantile trees run too narrow.

Validation is time-based: the model is fit on earlier as-of dates and scored on the most recent
ones, with a HORIZON-day gap so no training target overlaps the validation period. The final
models are then refit on every date and saved to data/price_model.joblib, with the validation
metrics in data/price_model_metrics.json.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features.risk import RISK_FEATURES, compute_risk
from features.trend import WINDOWS, compute_slopes

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"
MODEL_PATH = DATA_DIR / "price_model.joblib"
METRICS_PATH = DATA_DIR / "price_model_metrics.json"

FEATURE_WINDOWS = {k: WINDOWS[k] for k in ["1y", "6m", "3m", "1m", "2w", "3d"]}
FEATURES = [f"slope_{k}" for k in FEATURE_WINDOWS] + RISK_FEATURES
HORIZON = 21  # trading days, about 30 calendar days
SAMPLE_EVERY = 2  # trading days between training as-of dates
VALID_EVERY = 5  # validation keeps the original weekly as-of dates so its numbers stay comparable
VALID_FRACTION = 0.2  # most recent share of as-of dates held out for validation
QUANTILES = {"low": 0.1, "high": 0.9}
MAX_ABS_TARGET = 1.0  # clip training targets to +/-1 log return (~-63% / +172%) so bad ticks don't dominate
MAX_DAILY_JUMP = 5  # same bad-data filter as the trend table

# picked by a small sweep over learning_rate / min_samples_leaf / max_iter on the time split
TREE_PARAMS = dict(max_iter=100, learning_rate=0.02, max_leaf_nodes=31, min_samples_leaf=8000,
                   l2_regularization=1.0, early_stopping=False, random_state=0)

# validation numbers of the previous model (slopes only, raw target, every 5 days), for comparison
PREVIOUS_MODEL = {"mae_pct": 8.782, "direction_accuracy": 0.528, "rank_ic": 0.019, "range_coverage": 0.772}


def clean_prices(prices):
    """Drop tickers with implausible one-day moves (usually unadjusted splits or bad data)."""
    daily = prices.ffill().pct_change(fill_method=None) + 1
    clean = ((daily < MAX_DAILY_JUMP) & (daily > 1 / MAX_DAILY_JUMP) | daily.isna()).all()
    return prices.loc[:, clean]


def compute_features(prices):
    """Every model feature for each ticker, from `prices` up to (and including) the as-of date."""
    return compute_slopes(prices, FEATURE_WINDOWS).join(compute_risk(prices))[FEATURES]


def build_dataset(prices):
    """One row per (as-of date, ticker): features, the forward log return and its excess over the market."""
    log_p = np.log(prices.where(prices > 0))
    forward = log_p.shift(-HORIZON) - log_p
    first = max(FEATURE_WINDOWS.values()) - 1
    frames = []
    for i in range(first, len(prices) - HORIZON):
        on_train_grid, on_valid_grid = (i - first) % SAMPLE_EVERY == 0, (i - first) % VALID_EVERY == 0
        if not (on_train_grid or on_valid_grid):
            continue
        feats = compute_features(prices.iloc[: i + 1])
        feats["target"] = forward.iloc[i]
        feats = feats[feats["slope_1y"].notna() & feats["target"].notna()]
        feats["market"] = feats["target"].median()
        feats["as_of"] = prices.index[i]
        feats["train_grid"], feats["valid_grid"] = on_train_grid, on_valid_grid
        frames.append(feats)
    data = pd.concat(frames)
    data.index.name = "ticker"
    data["excess"] = data["target"] - data["market"]
    return data.reset_index()


def fit_models(train):
    X, y = train[FEATURES], train["excess"].clip(-MAX_ABS_TARGET, MAX_ABS_TARGET)
    models = {"point": HistGradientBoostingRegressor(loss="squared_error", **TREE_PARAMS).fit(X, y)}
    for name, q in QUANTILES.items():
        models[name] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **TREE_PARAMS).fit(X, y)
    # average market log return per as-of date, added back to turn excess predictions into returns
    drift = float(train.groupby("as_of")["market"].first().mean())
    return models, drift


def predict_returns(models, drift, X):
    """Predicted log return plus the raw (uncalibrated) 10th/90th percentile, all including the market drift."""
    pred = models["point"].predict(X) + drift
    a, b = models["low"].predict(X) + drift, models["high"].predict(X) + drift
    return pred, np.minimum(a, b), np.maximum(a, b)


def conformal_adjustment(y, low, high, target=QUANTILES["high"] - QUANTILES["low"]):
    """How much to widen [low, high] on each side so it contains `target` of the outcomes y."""
    scores = np.maximum(low - y, y - high)
    return float(np.quantile(scores, min(1.0, target * (1 + 1 / len(y)))))


def evaluate(models, drift, valid):
    X, y = valid[FEATURES], valid["target"].to_numpy()
    pred, low, high = predict_returns(models, drift, X)
    moved = y != 0
    up_share = float((y[moved] > 0).mean())
    # information coefficient: per-date rank correlation between predicted and actual returns
    ranks = valid.assign(pred=pred).groupby("as_of")[["pred", "target"]].rank()
    ic = ranks.groupby(valid["as_of"]).apply(lambda g: g["pred"].corr(g["target"])).mean()

    # calibrate the range on the validation set; to report honest coverage, calibrate on one half of
    # the validation dates and score on the other (both ways round), then use all of it for the final model
    adjustment = conformal_adjustment(y, low, high)
    first_half = (valid["as_of"] < np.sort(valid["as_of"].unique())[valid["as_of"].nunique() // 2]).to_numpy()
    held_out = []
    for cal, test in [(first_half, ~first_half), (~first_half, first_half)]:
        q = conformal_adjustment(y[cal], low[cal], high[cal])
        held_out.append(((y[test] >= low[test] - q) & (y[test] <= high[test] + q)).mean())

    pct = lambda r: np.expm1(r) * 100  # log return -> % price change
    return {
        "rows": int(len(valid)),
        "mae_pct": float(np.mean(np.abs(pct(pred) - pct(y)))),
        "baseline_mae_pct": float(np.mean(np.abs(pct(y)))),
        "direction_accuracy": float((np.sign(pred[moved]) == np.sign(y[moved])).mean()),
        "baseline_direction_accuracy": max(up_share, 1 - up_share),
        "rank_ic": float(ic),
        "range_coverage": float(np.mean(held_out)),
        "raw_range_coverage": float(((y >= low) & (y <= high)).mean()),
        "range_adjustment": adjustment,
    }


def print_metrics(m):
    prev = PREVIOUS_MODEL
    rows = [
        ("MAE (% of price)", f"{m['mae_pct']:.2f}%", f"{prev['mae_pct']:.2f}%", f"{m['baseline_mae_pct']:.2f}% (no change)"),
        ("Direction accuracy", f"{m['direction_accuracy']:.1%}", f"{prev['direction_accuracy']:.1%}",
         f"{m['baseline_direction_accuracy']:.1%} (majority)"),
        ("Rank IC", f"{m['rank_ic']:.3f}", f"{prev['rank_ic']:.3f}", "0.000 (random)"),
        ("10th-90th coverage", f"{m['range_coverage']:.1%}", f"{prev['range_coverage']:.1%}", "80.0% (target)"),
    ]
    widths = [max(len(r[i]) for r in rows + [("Metric", "This model", "Previous", "Baseline")]) for i in range(4)]
    for r in [("Metric", "This model", "Previous", "Baseline")] + rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print(f"  (raw quantile range covered {m['raw_range_coverage']:.1%}; widened by "
          f"{m['range_adjustment']:+.3f} log return on each side; coverage above is on held-out halves)")


if __name__ == "__main__":
    start = time.time()
    prices = clean_prices(pd.read_parquet(PRICES_PATH))
    print(f"Building training set from {prices.shape[1]} tickers x {len(prices)} days…")
    data = build_dataset(prices)
    print(f"{len(data):,} rows over {data['as_of'].nunique()} as-of dates ({time.time() - start:.0f}s)")

    # time-based split with a HORIZON-day embargo so training targets end before validation starts
    valid_dates = np.sort(data.loc[data["valid_grid"], "as_of"].unique())
    valid_start = valid_dates[int(len(valid_dates) * (1 - VALID_FRACTION))]
    embargo = prices.index[prices.index.get_loc(valid_start) - HORIZON]
    train = data[data["train_grid"] & (data["as_of"] < embargo)]
    valid = data[data["valid_grid"] & (data["as_of"] >= valid_start)]
    data = data[data["train_grid"]]
    print(f"Train: {len(train):,} rows through {pd.Timestamp(embargo).date()}; "
          f"validate: {len(valid):,} rows from {pd.Timestamp(valid_start).date()}")

    metrics = evaluate(*fit_models(train), valid)
    print_metrics(metrics)

    print("Refitting on all dates…")
    models, drift = fit_models(data)
    trained_through = pd.Timestamp(data["as_of"].max()).strftime("%Y-%m-%d")
    summary = {
        "trained_through": trained_through,
        "horizon_trading_days": HORIZON,
        "features": FEATURES,
        "target": "excess log return over the cross-sectional median, plus market_drift",
        "market_drift": drift,
        "quantiles": QUANTILES,
        "range_adjustment": metrics["range_adjustment"],
        "tree_params": TREE_PARAMS,
        "sample_every": SAMPLE_EVERY,
        "train_rows": int(len(data)),
        "validation": {**metrics, "start": pd.Timestamp(valid_start).strftime("%Y-%m-%d")},
        "previous_model_validation": PREVIOUS_MODEL,
    }
    joblib.dump({"models": models, **summary}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Saved model -> {MODEL_PATH} ({time.time() - start:.0f}s)")
