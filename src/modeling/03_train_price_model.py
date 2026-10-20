"""Train the 30-day price model the Portfolio Sandbox uses.

Reads the tables in data/ (prices, volumes, VIX, companies, earnings dates) and builds a training
set by sampling many historical as-of dates. For each as-of date, every stock gets the FEATURE_SET
groups from features/ (trend slopes, risk, beta, industry momentum, market regime), all computed only
from data up to that date.

The target is the stock's *excess* log return over the next HORIZON trading days (~30 calendar
days): its return minus that date's median return across all stocks. Market-wide swings are most
of the noise in a single month and can't be predicted from stock features, so removing them lets
the trees focus on which stocks beat the rest. A predicted return is the predicted excess return
plus the average market return over the training dates (MARKET_DRIFT in the saved model).

Gradient-boosted trees predict the expected excess return, and two quantile models predict the
10th and 90th percentile of the stock's raw return (RANGE_TARGET), so the range also reflects how
volatile the market is right now. The range is then conformally calibrated: widened by the amount
that makes it contain 80% of out-of-sample outcomes, since raw quantile trees run too narrow.

Validation is walk-forward: the last N_FOLDS blocks of FOLD_DAYS trading days are each scored by a
model fit only on as-of dates before that block, with a HORIZON-day gap so no training target
overlaps it. Metrics are pooled over all blocks, with per-block numbers alongside. The latest block
also checks that the model isn't memorising individual stocks: a model fit without a random
TICKER_HOLDOUT share of tickers should score as well on those tickers as on the ones it saw.
The final models are then refit on every date and saved to data/price_model.joblib, with the
validation metrics in data/price_model_metrics.json.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import CATEGORICAL_FEATURES, FEATURE_GROUPS, HISTORY, compute_features, load_inputs, market_returns

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_PATH = DATA_DIR / "price_model.joblib"
METRICS_PATH = DATA_DIR / "price_model_metrics.json"

# kept because each improved rank IC in most walk-forward blocks; moving averages, volume and earnings
# timing were tried and left out (no gain, or no gain on top of these)
FEATURE_SET = ["trend", "risk", "beta", "industry", "regime"]
FEATURES = [f for group in FEATURE_SET for f in FEATURE_GROUPS[group]]
RANGE_TARGET = "target"  # quantile models predict the raw return ("target") or the excess return ("excess")
HORIZON = 21  # trading days, about 30 calendar days
SAMPLE_EVERY = 2  # trading days between training as-of dates
VALID_EVERY = 5  # validation keeps the original weekly as-of dates so its numbers stay comparable
N_FOLDS = 4  # walk-forward validation blocks, most recent last
FOLD_DAYS = 63  # trading days per block, about 3 months
TICKER_HOLDOUT = 0.2  # share of tickers left out of training in the memorisation check
QUANTILES = {"low": 0.1, "high": 0.9}
MAX_ABS_TARGET = 1.0  # clip training targets to +/-1 log return (~-63% / +172%) so bad ticks don't dominate
MAX_DAILY_JUMP = 5  # same bad-data filter as the trend table

# picked by a small sweep over learning_rate / min_samples_leaf / max_iter on the walk-forward folds
TREE_PARAMS = dict(max_iter=100, learning_rate=0.02, max_leaf_nodes=31, min_samples_leaf=8000,
                   l2_regularization=1.0, early_stopping=False, random_state=0)

# walk-forward validation of the previous model (slopes + risk features only, range on excess returns)
PREVIOUS_MODEL = {"mae_pct": 8.783, "direction_accuracy": 0.539, "rank_ic": 0.058, "range_coverage": 0.798,
                  "range_width_pct": 26.3}


def clean_prices(prices):
    """Drop tickers with implausible one-day moves (usually unadjusted splits or bad data)."""
    daily = prices.ffill().pct_change(fill_method=None) + 1
    clean = ((daily < MAX_DAILY_JUMP) & (daily > 1 / MAX_DAILY_JUMP) | daily.isna()).all()
    return prices.loc[:, clean]


def build_dataset(inputs):
    """One row per (as-of date, ticker): features, the forward log return and its excess over the market."""
    prices = inputs["prices"]
    volumes = inputs["volumes"].reindex_like(prices) if inputs["volumes"] is not None else None
    market = market_returns(prices)  # each day's median uses only that day, so this has no look-ahead
    log_p = np.log(prices.where(prices > 0))
    forward = log_p.shift(-HORIZON) - log_p
    first = HISTORY - 1
    frames = []
    for i in range(first, len(prices) - HORIZON):
        on_train_grid, on_valid_grid = (i - first) % SAMPLE_EVERY == 0, (i - first) % VALID_EVERY == 0
        if not (on_train_grid or on_valid_grid):
            continue
        feats = compute_features(prices.iloc[: i + 1], None if volumes is None else volumes.iloc[: i + 1],
                                 inputs["vix"], inputs["companies"], inputs["earnings"], market=market)
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


def make_model(loss, features, **kw):
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    return HistGradientBoostingRegressor(loss=loss, categorical_features=categorical or None, **kw, **TREE_PARAMS)


def fit_models(train, features=FEATURES, range_target=RANGE_TARGET, quantiles=True):
    # a feature whose input table is missing is entirely NaN, which the trees can't bin; leave it out
    features = [f for f in features if train[f].notna().any()]
    X = train[features]
    clip = lambda col: train[col].clip(-MAX_ABS_TARGET, MAX_ABS_TARGET)
    models = {"point": make_model("squared_error", features).fit(X, clip("excess"))}
    if quantiles:
        for name, q in QUANTILES.items():
            models[name] = make_model("quantile", features, quantile=q).fit(X, clip(range_target))
    # average market log return per as-of date, added back to turn excess predictions into returns
    drift = float(train.groupby("as_of")["market"].first().mean())
    return {"models": models, "drift": drift, "features": features, "range_target": range_target}


def predict_returns(fit, X):
    """Predicted log return plus the raw (uncalibrated) 10th/90th percentile, all including the market drift."""
    models, drift = fit["models"], fit["drift"]
    pred = models["point"].predict(X[fit["features"]]) + drift
    if "low" not in models:
        return pred, None, None
    shift = drift if fit["range_target"] == "excess" else 0.0
    a, b = (models[k].predict(X[fit["features"]]) + shift for k in ("low", "high"))
    return pred, np.minimum(a, b), np.maximum(a, b)


def conformal_adjustment(y, low, high, target=QUANTILES["high"] - QUANTILES["low"]):
    """How much to widen [low, high] on each side so it contains `target` of the outcomes y."""
    scores = np.maximum(low - y, y - high)
    return float(np.quantile(scores, min(1.0, target * (1 + 1 / len(y)))))


def walk_forward_folds(data, dates):
    """(train, valid) row selections for each block, oldest first; `dates` is the full trading-day index."""
    valid_dates = np.sort(data.loc[data["valid_grid"], "as_of"].unique())
    last = dates.get_loc(valid_dates[-1])
    for k in reversed(range(N_FOLDS)):
        start, end = dates[last - (k + 1) * FOLD_DAYS + 1], dates[min(last - k * FOLD_DAYS + 1, len(dates) - 1)]
        embargo = dates[dates.get_loc(start) - HORIZON]
        train = data["train_grid"] & (data["as_of"] < embargo)
        valid = data["valid_grid"] & (data["as_of"] >= start) & ((data["as_of"] < end) if k else True)
        yield train.to_numpy(), valid.to_numpy()


def score(valid, pred):
    """Point-prediction metrics on the rows in `valid`."""
    y = valid["target"].to_numpy()
    moved = y != 0
    up_share = float((y[moved] > 0).mean())
    # information coefficient: per-date rank correlation between predicted and actual returns
    ranks = valid.assign(pred=pred)[["as_of", "pred", "target"]].groupby("as_of").rank()
    ic = ranks.groupby(valid["as_of"].to_numpy()).apply(lambda g: g["pred"].corr(g["target"]))
    pct = lambda r: np.expm1(r) * 100  # log return -> % price change
    return {
        "rows": int(len(valid)),
        "start": pd.Timestamp(valid["as_of"].min()).strftime("%Y-%m-%d"),
        "mae_pct": float(np.mean(np.abs(pct(pred) - pct(y)))),
        "baseline_mae_pct": float(np.mean(np.abs(pct(y)))),
        "direction_accuracy": float((np.sign(pred[moved]) == np.sign(y[moved])).mean()),
        "baseline_direction_accuracy": max(up_share, 1 - up_share),
        "rank_ic": float(ic.mean()),
    }


def walk_forward(data, dates, features=FEATURES, range_target=RANGE_TARGET, quantiles=True, ticker_check=True):
    """Fit and score a model on each walk-forward block; returns pooled metrics with a 'folds' list."""
    folds, oof = [], []
    for k, (train_rows, valid_rows) in enumerate(walk_forward_folds(data, dates)):
        train, valid = data[train_rows], data[valid_rows]
        fit = fit_models(train, features, range_target, quantiles)
        pred, low, high = predict_returns(fit, valid)
        fold = score(valid, pred)
        part = valid[["as_of", "ticker", "target"]].assign(pred=pred, fold=k)
        if quantiles:
            y = valid["target"].to_numpy()
            fold["raw_range_coverage"] = float(((y >= low) & (y <= high)).mean())
            part = part.assign(low=low, high=high)
        if ticker_check and k == N_FOLDS - 1:
            fold["ticker_check"] = memorisation_check(train, valid, features)
        folds.append(fold)
        oof.append(part)
    oof = pd.concat(oof)
    metrics = {**score(oof, oof["pred"].to_numpy()), "folds": folds}
    if quantiles:
        metrics.update(range_metrics(oof))
    if ticker_check:
        metrics["ticker_check"] = folds[-1].pop("ticker_check")
    return metrics


def range_metrics(oof):
    """Calibrate the range on the pooled out-of-sample predictions and report honest coverage and width.

    Each block's coverage is measured with the adjustment calibrated on the *other* blocks, so the
    reported coverage is out of sample; the final adjustment uses every block.
    """
    y, low, high, fold = (oof[c].to_numpy() for c in ["target", "low", "high", "fold"])
    adjustment = conformal_adjustment(y, low, high)
    covered, width = np.zeros(len(y), bool), np.zeros(len(y))
    per_fold = []
    for k in np.unique(fold):
        test = fold == k
        q = conformal_adjustment(y[~test], low[~test], high[~test])
        covered[test] = (y[test] >= low[test] - q) & (y[test] <= high[test] + q)
        width[test] = (np.exp(high[test] + q) - np.exp(low[test] - q)) * 100
        per_fold.append(float(covered[test].mean()))
    return {
        "range_coverage": float(covered.mean()),
        "range_coverage_by_fold": per_fold,
        "range_width_pct": float(np.mean(width)),
        "raw_range_coverage": float(((y >= low) & (y <= high)).mean()),
        "range_adjustment": adjustment,
    }


def memorisation_check(train, valid, features):
    """Rank IC on tickers the model trained on vs tickers it never saw (TICKER_HOLDOUT of them, at random)."""
    rng = np.random.default_rng(0)
    tickers = train["ticker"].unique()
    held_out = set(rng.choice(tickers, int(len(tickers) * TICKER_HOLDOUT), replace=False))
    fit = fit_models(train[~train["ticker"].isin(held_out)], features, quantiles=False)
    pred = predict_returns(fit, valid)[0]
    unseen = valid["ticker"].isin(held_out).to_numpy()
    return {"seen_rank_ic": score(valid[~unseen], pred[~unseen])["rank_ic"],
            "unseen_rank_ic": score(valid[unseen], pred[unseen])["rank_ic"]}


def print_metrics(m):
    prev = PREVIOUS_MODEL
    fmt = lambda v, f: "–" if v is None else format(v, f)
    rows = [
        ("MAE (% of price)", f"{m['mae_pct']:.2f}%", fmt(prev["mae_pct"], ".2f") + "%",
         f"{m['baseline_mae_pct']:.2f}% (no change)"),
        ("Direction accuracy", f"{m['direction_accuracy']:.1%}", fmt(prev["direction_accuracy"], ".1%"),
         f"{m['baseline_direction_accuracy']:.1%} (majority)"),
        ("Rank IC", f"{m['rank_ic']:.3f}", fmt(prev["rank_ic"], ".3f"), "0.000 (random)"),
        ("10th-90th coverage", f"{m['range_coverage']:.1%}", fmt(prev["range_coverage"], ".1%"), "80.0% (target)"),
        ("10th-90th width", f"{m['range_width_pct']:.1f} pts", fmt(prev["range_width_pct"], ".1f") + " pts",
         "narrower is better"),
    ]
    header = ("Metric", "This model", "Previous", "Baseline")
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(4)]
    for r in [header] + rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print(f"  (pooled over {len(m['folds'])} walk-forward blocks from {m['start']}; raw quantile range covered "
          f"{m['raw_range_coverage']:.1%}, widened by {m['range_adjustment']:+.3f} log return on each side)")
    print("  Per block:  " + "   ".join(
        f"{f['start']}: IC {f['rank_ic']:.3f}, cover {c:.0%}" for f, c in zip(m["folds"], m["range_coverage_by_fold"])))
    t = m["ticker_check"]
    print(f"  Memorisation check (latest block): rank IC {t['seen_rank_ic']:.3f} on tickers seen in training, "
          f"{t['unseen_rank_ic']:.3f} on held-out tickers")


def load_training_inputs():
    inputs = load_inputs(DATA_DIR)
    inputs["prices"] = clean_prices(inputs["prices"])
    missing = [k for k, v in inputs.items() if v is None]
    if missing:
        print(f"Missing inputs ({', '.join(missing)}); their features will be empty. Run the data_gathering scripts.")
    return inputs


if __name__ == "__main__":
    start = time.time()
    inputs = load_training_inputs()
    prices = inputs["prices"]
    print(f"Building training set from {prices.shape[1]} tickers x {len(prices)} days…")
    data = build_dataset(inputs)
    print(f"{len(data):,} rows over {data['as_of'].nunique()} as-of dates ({time.time() - start:.0f}s)")

    print(f"Walk-forward validation over {N_FOLDS} blocks of {FOLD_DAYS} trading days…")
    metrics = walk_forward(data, prices.index)
    print_metrics(metrics)

    print("Refitting on all dates…")
    data = data[data["train_grid"]]
    fit = fit_models(data)
    trained_through = pd.Timestamp(data["as_of"].max()).strftime("%Y-%m-%d")
    summary = {
        "trained_through": trained_through,
        "horizon_trading_days": HORIZON,
        "features": fit["features"],
        "feature_groups": FEATURE_SET,
        "target": "excess log return over the cross-sectional median, plus market_drift",
        "range_target": "raw log return" if RANGE_TARGET == "target" else "excess log return, plus market_drift",
        "market_drift": fit["drift"],
        "quantiles": QUANTILES,
        "range_adjustment": metrics["range_adjustment"],
        "tree_params": TREE_PARAMS,
        "sample_every": SAMPLE_EVERY,
        "train_rows": int(len(data)),
        "validation": metrics,
        "previous_model_validation": PREVIOUS_MODEL,
    }
    joblib.dump({"models": fit["models"], "range_target_column": RANGE_TARGET, **summary}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Saved model -> {MODEL_PATH} ({time.time() - start:.0f}s)")
