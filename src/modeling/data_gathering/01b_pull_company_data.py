"""Pull each stock's sector, industry and earnings report dates from Yahoo Finance.

Covers every ticker in data/prices.parquet, one Yahoo request pair per ticker, so a full pull takes
an hour or more. Results are cached and only stale tickers are re-pulled:
  data/companies.parquet  one row per ticker: sector, industry, when each part was last pulled
  data/earnings.parquet   one row per (ticker, report date)

Sector and industry rarely change, so they're re-pulled every INFO_MAX_AGE. Earnings dates are
re-pulled every EARNINGS_MAX_AGE, so newly reported quarters show up within about a month. Tickers
that have never been pulled are always pulled; of the stale ones, only the MAX_STALE_PER_RUN oldest
are re-pulled per run, which keeps a routine refresh to about 20 minutes. Progress is saved every
SAVE_EVERY tickers, so an interrupted pull picks up where it left off.

Usage:
    python 01b_pull_company_data.py                    # pull stale tickers
    python 01b_pull_company_data.py --force            # re-pull everything
    python 01b_pull_company_data.py --tickers AAPL MSFT
    python 01b_pull_company_data.py --skip-earnings    # sector/industry only (the price model doesn't use earnings yet)
"""
import argparse
import logging
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"
COMPANIES_PATH = DATA_DIR / "companies.parquet"
EARNINGS_PATH = DATA_DIR / "earnings.parquet"

INFO_MAX_AGE = timedelta(days=180)
EARNINGS_MAX_AGE = timedelta(days=30)
EARNINGS_LIMIT = 40  # quarters of history requested (Yahoo also returns a few upcoming ones)
MAX_STALE_PER_RUN = 800
SAVE_EVERY = 100
PAUSE = 0.2  # seconds between tickers
RETRY_WAITS = [60, 120, 300, 600]  # back-off schedule (seconds) when Yahoo rate limits us


def with_retry(fn):
    for attempt, wait in enumerate([0] + RETRY_WAITS):
        if wait:
            print(f"  rate limited, waiting {wait}s before retry {attempt}")
            time.sleep(wait)
        try:
            return fn()
        except YFRateLimitError:
            continue
    raise YFRateLimitError()


def pull_info(ticker):
    info = with_retry(lambda: yf.Ticker(ticker).info) or {}
    return {"sector": info.get("sector"), "industry": info.get("industry")}


def pull_earnings(ticker):
    """Report dates (dates only, no time zone), oldest first. Empty if Yahoo has none."""
    try:
        dates = with_retry(lambda: yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LIMIT))
    except YFRateLimitError:
        raise
    except Exception:  # yfinance raises assorted errors for tickers without earnings (funds, ETFs, SPACs)
        return []
    if dates is None or dates.empty:
        return []
    return sorted({d.date() for d in dates.index})


def load_cache():
    companies = (pd.read_parquet(COMPANIES_PATH) if COMPANIES_PATH.exists() else pd.DataFrame({
        "sector": pd.Series(dtype=object), "industry": pd.Series(dtype=object),
        "info_pulled": pd.Series(dtype="datetime64[ns]"), "earnings_pulled": pd.Series(dtype="datetime64[ns]")}))
    earnings = (pd.read_parquet(EARNINGS_PATH) if EARNINGS_PATH.exists() else
                pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")}))
    return companies, earnings


def save(companies, earnings):
    companies.index.name = "ticker"
    for col in ["info_pulled", "earnings_pulled"]:
        companies[col] = pd.to_datetime(companies[col])
    companies.to_parquet(COMPANIES_PATH)
    earnings.sort_values(["ticker", "date"]).reset_index(drop=True).to_parquet(EARNINGS_PATH)


def pull_company_data(tickers, force=False, earnings_dates=True):
    companies, earnings = load_cache()
    now = pd.Timestamp.now()
    stale = lambda col, age: (companies[col].reindex(tickers).isna() | (now - companies[col].reindex(tickers) > age))
    need_info = set(pd.Index(tickers)[force | stale("info_pulled", INFO_MAX_AGE)])
    need_earnings = set(pd.Index(tickers)[force | stale("earnings_pulled", EARNINGS_MAX_AGE)]) if earnings_dates else set()
    todo = [t for t in tickers if t in need_info or t in need_earnings]
    if not force:
        # never-pulled tickers first, then the stalest, capped so a routine refresh stays short
        pulled = companies[["info_pulled", "earnings_pulled"] if earnings_dates else ["info_pulled"]].min(axis=1).reindex(todo)
        new, old = pulled[pulled.isna()].index, pulled.dropna().sort_values().index[:MAX_STALE_PER_RUN]
        todo = list(new) + list(old)
    print(f"{len(todo)} of {len(tickers)} tickers need pulling "
          f"({len(need_info)} sector/industry, {len(need_earnings)} earnings dates)")

    new_earnings = {}
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # it logs every ticker without earnings data
    for i, ticker in enumerate(todo):
        if i % SAVE_EVERY == 0:
            print(f"Pulling tickers {i + 1}-{min(i + SAVE_EVERY, len(todo))} of {len(todo)}")
        try:
            if ticker in need_info:
                for col, value in pull_info(ticker).items():
                    companies.loc[ticker, col] = value
                companies.loc[ticker, "info_pulled"] = now
            if ticker in need_earnings:
                new_earnings[ticker] = pull_earnings(ticker)
                companies.loc[ticker, "earnings_pulled"] = now
        except YFRateLimitError:
            print(f"  giving up on {ticker} after repeated rate limits")
        except Exception as err:
            print(f"  {ticker}: {err}")
        time.sleep(PAUSE)
        if (i + 1) % SAVE_EVERY == 0 or i + 1 == len(todo):
            earnings = merge_earnings(earnings, new_earnings)
            new_earnings = {}
            save(companies, earnings)
    if not todo:
        save(companies, earnings)
    print(f"Saved {companies['sector'].notna().sum()} tickers with a sector to {COMPANIES_PATH} and "
          f"{len(earnings):,} earnings dates for {earnings['ticker'].nunique()} tickers to {EARNINGS_PATH}")


def merge_earnings(earnings, new):
    """Add re-pulled tickers' dates to the saved ones (old quarters Yahoo stops returning are kept)."""
    if not new:
        return earnings
    fresh = pd.DataFrame([(t, pd.Timestamp(d)) for t, dates in new.items() for d in dates], columns=["ticker", "date"])
    return pd.concat([earnings, fresh]).drop_duplicates()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-pull every ticker, even ones pulled recently")
    parser.add_argument("--tickers", nargs="+", help="pull these tickers instead of everything in prices.parquet")
    parser.add_argument("--skip-earnings", action="store_true", help="only pull sector and industry")
    args = parser.parse_args()
    tickers = args.tickers or list(pd.read_parquet(PRICES_PATH).columns)
    pull_company_data(tickers, force=args.force or bool(args.tickers), earnings_dates=not args.skip_earnings)
