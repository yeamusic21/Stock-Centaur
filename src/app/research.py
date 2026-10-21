"""Free data the agent and Stock Insights pull on demand: fresh prices from Yahoo, company profiles, and
news headlines scraped from public RSS feeds (Yahoo Finance and Google News) and article pages."""
import json
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from on_demand import yahoo_symbol

REPO_ROOT = Path(__file__).resolve().parents[2]
TREND_TABLE_PATH = REPO_ROOT / "data" / "trend_table.json"

HEADERS = {"User-Agent": "Mozilla/5.0"}  # Yahoo rate-limits full browser user agents from scripts
TIMEOUT = 10
FEEDS = {
    "Yahoo Finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
    "Google News": "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
}
ARTICLE_CHARS = 1500
PROFILE_FIELDS = ["longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "dividendYield",
                  "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "recommendationKey", "numberOfAnalystOpinions",
                  "targetMeanPrice", "quoteType", "website"]

_trend_cache = {"mtime": None, "by_ticker": {}}


def ranked(ticker):
    """The ticker's row in the trend rankings, if it's ranked."""
    if not TREND_TABLE_PATH.exists():
        return None
    mtime = TREND_TABLE_PATH.stat().st_mtime
    if _trend_cache["mtime"] != mtime:
        data = json.loads(TREND_TABLE_PATH.read_text())
        _trend_cache.update(mtime=mtime, count=len(data["stocks"]),
                            by_ticker={s["ticker"]: {k: v for k, v in s.items() if k != "sparkline"} for s in data["stocks"]})
    row = _trend_cache["by_ticker"].get(ticker)
    return row and {**row, "of": _trend_cache["count"]}


def history(tickers, period="1y"):
    """Daily adjusted closes downloaded fresh from Yahoo, one column per ticker."""
    symbols = {yahoo_symbol(t): t for t in tickers}
    data = yf.download(list(symbols), period=period, auto_adjust=True, progress=False, threads=True)
    if data.empty:
        return pd.DataFrame()
    closes = data["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(list(symbols)[0])
    closes.index = closes.index.tz_localize(None)
    return closes.rename(columns=symbols).dropna(axis=1, how="all")


def trend_stats(closes):
    """Recent returns, moving averages, volatility and drawdown for one price series."""
    s = closes.dropna()
    if len(s) < 2:
        return None
    last = float(s.iloc[-1])
    ret = lambda days: float(s.iloc[-1] / s.iloc[-days - 1] - 1) * 100 if len(s) > days else None
    ma = lambda days: float(s.iloc[-days:].mean()) if len(s) >= days else None
    daily = np.log(s).diff().dropna()
    recent = s.iloc[-63:]
    ma50, ma200 = ma(50), ma(200)
    return {
        "price": round(last, 2),
        "as_of": s.index[-1].strftime("%Y-%m-%d"),
        "change_1d": ret(1), "return_1w": ret(5), "return_1m": ret(21), "return_3m": ret(63),
        "return_1y": float(s.iloc[-1] / s.iloc[0] - 1) * 100,
        "vs_ma50": (last / ma50 - 1) * 100 if ma50 else None,
        "vs_ma200": (last / ma200 - 1) * 100 if ma200 else None,
        "volatility_1m": float(daily.iloc[-21:].std() * np.sqrt(252) * 100) if len(daily) >= 21 else None,
        "max_drawdown_3m": float((recent / recent.cummax() - 1).min() * 100),
        "off_high_1y": float((last / s.max() - 1) * 100),
    }


def profile(ticker):
    try:
        info = yf.Ticker(yahoo_symbol(ticker)).info or {}
    except Exception:
        return {}
    out = {k: info.get(k) for k in PROFILE_FIELDS if info.get(k) is not None}
    summary = info.get("longBusinessSummary")
    if summary:
        out["summary"] = summary if len(summary) < 900 else summary[:900].rsplit(". ", 1)[0] + "."
    return out


def _text(el, tag):
    found = el.find(tag)
    return (found.text or "").strip() if found is not None and found.text else ""


def news(ticker, limit=8):
    """Recent headlines for a ticker from free RSS feeds, newest first, without duplicates."""
    items, seen = [], set()
    for source, url in FEEDS.items():
        try:
            res = requests.get(url.format(symbol=yahoo_symbol(ticker)), headers=HEADERS, timeout=TIMEOUT)
            res.raise_for_status()
            root = ET.fromstring(res.content)
        except (requests.RequestException, ET.ParseError):
            continue
        for item in root.iter("item"):
            title = _text(item, "title")
            key = re.sub(r"\W+", " ", title.lower()).strip()[:80]
            if not title or key in seen:
                continue
            seen.add(key)
            try:
                published = parsedate_to_datetime(_text(item, "pubDate")).isoformat()
            except (TypeError, ValueError):
                published = ""
            summary = BeautifulSoup(_text(item, "description"), "html.parser").get_text(" ", strip=True)
            items.append({"title": title, "link": _text(item, "link"), "published": published,
                          "publisher": _text(item, "source") or source, "via": source,
                          "summary": "" if summary.startswith(title[:40]) else summary[:400]})
    # Yahoo's items link straight to the article and carry a summary, so prefer them over Google's redirects
    newest = lambda rows: sorted(rows, key=lambda i: i["published"], reverse=True)
    picked = newest([i for i in items if i["via"] == "Yahoo Finance"])[:limit]
    picked += newest([i for i in items if i["via"] != "Yahoo Finance"])[:limit - len(picked)]
    return newest(picked)


def article_text(url):
    """The readable paragraphs of a news article, trimmed. Empty if the page can't be fetched or parsed."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        res.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in (soup.find("article") or soup).find_all("p")]
    text = " ".join(p for p in paragraphs if len(p) > 60)
    return text[:ARTICLE_CHARS]
