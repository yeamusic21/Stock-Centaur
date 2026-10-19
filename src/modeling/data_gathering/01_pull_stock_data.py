"""Pull 5 years of daily closing prices for every ticker in the ticker list.

Results are saved as a wide parquet file (rows = dates, columns = tickers) at
data/prices.parquet in the repo root.

Usage:
    python 01_pull_stock_data.py              # pull all tickers (skips if data < 7 days old)
    python 01_pull_stock_data.py --force      # pull even if data is fresh
    python 01_pull_stock_data.py --limit 200  # only pull the first 200 tickers (handy for testing)
    python 01_pull_stock_data.py --tickers AAPL MSFT NVDA
"""
import argparse
import ast
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"
TICKERS_PATH = Path(__file__).resolve().parent / "full_stock_tickers_list_061817_6k.csv"

MAX_DATA_AGE = timedelta(days=7)
CHUNK_SIZE = 100  # yfinance handles batches of tickers much faster than one at a time
CHUNK_PAUSE = 2  # seconds between batches, to stay under Yahoo's rate limit
RETRY_WAITS = [60, 120, 300, 600]  # back-off schedule (seconds) when Yahoo rate limits us


def load_tickers(limit=None):
    tickers = pd.read_csv(TICKERS_PATH)["Symbol"].dropna().astype(str).str.strip()
    # yahoo uses '-' for share classes (e.g. BRK-B), the list uses '.' or '^' in places
    tickers = tickers.str.replace(".", "-", regex=False).str.replace("^", "-P", regex=False)
    tickers = list(dict.fromkeys(tickers))  # de-dupe, keep order
    return tickers[:limit] if limit else tickers


def data_is_fresh():
    if not PRICES_PATH.exists():
        return False
    modified = datetime.fromtimestamp(PRICES_PATH.stat().st_mtime)
    return datetime.now() - modified < MAX_DATA_AGE


class RateLimitCollector(logging.Handler):
    """yfinance only logs failed tickers, so collect the ones that failed due to rate limiting."""
    pattern = re.compile(r"(\[.*?\]): YFRateLimitError")

    def __init__(self):
        super().__init__()
        self.tickers = set()

    def emit(self, record):
        for match in self.pattern.finditer(record.getMessage()):
            self.tickers.update(ast.literal_eval(match.group(1)))


def download_closes(tickers):
    """Download 5 years of adjusted closes for a batch of tickers.

    Returns a wide DataFrame and the list of tickers that were rate limited (to retry later).
    """
    collector = RateLimitCollector()
    yf_logger = logging.getLogger("yfinance")
    yf_logger.addHandler(collector)
    try:
        data = yf.download(tickers, period="5y", auto_adjust=True, progress=False, threads=True)
    finally:
        yf_logger.removeHandler(collector)
    rate_limited = [t for t in tickers if t in collector.tickers]
    if data.empty:
        return pd.DataFrame(), rate_limited
    closes = data["Close"]
    if isinstance(closes, pd.Series):  # single ticker
        closes = closes.to_frame(tickers[0])
    return closes.dropna(axis=1, how="all"), rate_limited


def download_with_retry(tickers):
    """Download a batch, backing off and retrying any tickers Yahoo rate limited."""
    frames = []
    for attempt, wait in enumerate([0] + RETRY_WAITS):
        if wait:
            print(f"  rate limited on {len(tickers)} tickers, waiting {wait}s before retry {attempt}")
            time.sleep(wait)
        closes, tickers = download_closes(tickers)
        if not closes.empty:
            frames.append(closes)
        if not tickers:
            break
    else:
        print(f"  giving up on {len(tickers)} rate-limited tickers: {' '.join(tickers)}")
    return frames


def pull_stock_data(tickers):
    frames = []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        print(f"Pulling tickers {i + 1}-{i + len(chunk)} of {len(tickers)}")
        frames.extend(download_with_retry(chunk))
        time.sleep(CHUNK_PAUSE)
    if not frames:
        raise RuntimeError("No price data was downloaded.")
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    if PRICES_PATH.exists():
        # keep tickers from earlier pulls that failed this time (e.g. rate limited); new data wins
        previous = pd.read_parquet(PRICES_PATH)
        prices = prices.combine_first(previous)
        prices = prices[prices.index >= prices.index.max() - pd.DateOffset(years=5)]
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_parquet(PRICES_PATH)
    print(f"Saved {prices.shape[1]} tickers x {prices.shape[0]} days to {PRICES_PATH}")
    return prices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="pull even if existing data is fresh")
    parser.add_argument("--limit", type=int, help="only pull the first N tickers from the list")
    parser.add_argument("--tickers", nargs="+", help="pull these tickers instead of the ticker list")
    args = parser.parse_args()

    if data_is_fresh() and not (args.force or args.tickers or args.limit):
        print(f"{PRICES_PATH} is less than {MAX_DATA_AGE.days} days old, skipping (use --force to re-pull).")
    else:
        pull_stock_data(args.tickers or load_tickers(args.limit))
