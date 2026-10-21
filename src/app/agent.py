"""The research agent: pulls fresh data for a portfolio or a single stock, runs the price model and the
Monte Carlo, scrapes recent news, and has the local LM Studio model write a report from it.

Runs one job at a time in a background thread (like refresh.py) and saves each report to the store.
Everything is free: Yahoo prices, public RSS feeds and article pages, and a model running locally.
"""
import threading
import time
from datetime import datetime

import llm
import on_demand
import research
import sandbox as sandbox_sim
import store

DETAILED_HOLDINGS = 8  # in detailed mode, the largest holdings get their own LLM brief
NEWS_PER_STOCK = 6
ARTICLES_PER_STOCK = 2  # article pages scraped for the largest holdings / a single stock
ARTICLE_HOLDINGS = 3
ARTICLE_PROMPT_CHARS = 600  # keep prompts short: local models read and reason slowly
REPLY_TOKENS = 3500  # room left in the model's context for a reasoning model to think and then write
FULL_FACTS = 10  # holdings described in full; smaller ones get one short line
NOT_ADVICE = ("You are a careful equity research assistant. Describe what the data and news show, including risks "
              "and uncertainty. Do not tell the reader to buy, sell or hold anything and do not give personalized "
              "financial advice. Only use the facts provided; say so when information is missing. "
              "Write in Markdown.")

_lock = threading.Lock()
_status = {"state": "idle"}
_stop = threading.Event()


def status():
    with _lock:
        s = dict(_status)
    if s.get("state") == "running":
        s["elapsed"] = round(time.time() - s["started_ts"])
    return s


def _update(**fields):
    with _lock:
        _status.update(fields)


def _step(index, label, message=""):
    _update(step=label, step_index=index, message=message)


def _ask(prompt, model, label="the report"):
    """One LLM call, reporting its progress in the job status and honoring Stop."""
    started, total = time.time(), [0]

    def progress(phase, tokens):
        total[0] += 1
        rate = total[0] / max(time.time() - started, 1)
        verb = "Thinking" if phase == "thinking" else "Writing"
        _update(message=f"{model}: {verb} {label}… {tokens:,} tokens ({rate:.1f}/s)")

    _update(message=f"{model}: reading the data for {label}…")
    return llm.chat([{"role": "system", "content": NOT_ADVICE}, {"role": "user", "content": prompt}], model=model,
                    on_progress=progress, should_stop=_stop.is_set)


def _prompt_budget(model):
    """Tokens a prompt can use so the reply still fits in the loaded model's context."""
    ctx = llm.context_length(model)
    budget = ctx - REPLY_TOKENS - llm.estimate_tokens(NOT_ADVICE)
    if budget < 800:
        raise ValueError(f"The model in LM Studio is loaded with only {ctx:,} tokens of context, too little for a "
                         "report. Raise its context length to 8,192 or more in LM Studio's model load settings.")
    return budget


def _fit(parts, budget):
    """Join as many of the optional text parts as fit in the token budget, in order."""
    kept = []
    for part in parts:
        if llm.estimate_tokens(part) > budget:
            break
        kept.append(part)
        budget -= llm.estimate_tokens(part)
    return kept


def _short_facts(r):
    stats, about = r["stats"] or {}, f" ({r['about']})" if r.get("about") else ""
    return (f"{r['ticker']}{about}: 1m {_pct(stats.get('return_1m'))}, 1y {_pct(stats.get('return_1y'))}, "
            f"model 30-day forecast {r['prediction']['pred_return']:+.1f}%")


def _pct(v):
    return "n/a" if v is None else f"{v:+.1f}%"


# ---------- data gathering ----------
def _gather_news(tickers, article_for):
    out = {}
    for i, t in enumerate(tickers):
        _update(message=f"News for {t} ({i + 1} of {len(tickers)})")
        items = research.news(t, NEWS_PER_STOCK)
        if t in article_for:
            for item in items[:ARTICLES_PER_STOCK]:
                item["text"] = research.article_text(item["link"])
        out[t] = items
    return out


def _news_block(items):
    lines = []
    for n in items:
        line = f"- {n['published'][:10]} {n['publisher']}: {n['title']}"
        if n.get("summary"):
            line += f" — {n['summary']}"
        if n.get("text"):
            line += f"\n  Article excerpt: {n['text'][:ARTICLE_PROMPT_CHARS]}"
        lines.append(line)
    return "\n".join(lines) or "- No recent headlines found."


def _describe(prof):
    """e.g. "Schwab U.S. Dividend Equity ETF, ETF" or "Apple Inc., Technology / Consumer Electronics"."""
    kind = prof.get("quoteType") if prof.get("quoteType") not in (None, "EQUITY") else None
    sector = " / ".join(x for x in (prof.get("sector"), prof.get("industry")) if x)
    return ", ".join(x for x in (prof.get("longName"), kind, sector) if x)


def _stock_facts(t, pred, stats, rank, prof=None):
    # the model predicted from its last data refresh; carry its forecast returns over to today's price
    price = stats["price"] if stats else pred["price"]
    scale = price / pred["price"]
    name = _describe(prof or {})
    parts = [f"{t}{f' ({name})' if name else ''}: price ${price:,.2f}"]
    if stats:
        parts.append(f"1w {_pct(stats['return_1w'])}, 1m {_pct(stats['return_1m'])}, 3m {_pct(stats['return_3m'])}, "
                     f"1y {_pct(stats['return_1y'])}; {_pct(stats['vs_ma50'])} vs 50-day avg, "
                     f"{_pct(stats['vs_ma200'])} vs 200-day avg; {_pct(stats['off_high_1y'])} from 1-year high; "
                     f"1-month volatility {stats['volatility_1m'] or 0:.0f}% annualized")
    if rank:
        parts.append(f"trend score {rank['trend']:.0f}/100 (rank {rank['rank']} of {rank['of']})")
    parts.append(f"model 30-day forecast {pred['pred_return']:+.1f}% "
                 f"(80% range ${pred['low_price'] * scale:,.2f}–${pred['high_price'] * scale:,.2f})")
    return "; ".join(parts)


# ---------- portfolio report ----------
PORTFOLIO_STEPS = ["Scoring holdings with the price model", "Pulling fresh prices", "Running 10,000 simulations",
                   "Scraping news", "Writing the report"]


def _portfolio_report(pid, detailed):
    portfolio = store.get_portfolio(pid)
    if not portfolio:
        raise ValueError("That portfolio no longer exists.")
    model = llm.active_model()

    _step(0, PORTFOLIO_STEPS[0])
    tickers = [h["ticker"] for h in portfolio["holdings"]]
    missing = on_demand.predict_missing(tickers)
    preds = sandbox_sim.predictions_by_ticker()
    holdings, skipped = [], []
    for h in portfolio["holdings"]:
        p = preds.get(h["ticker"])
        if not p:
            skipped.append({"ticker": h["ticker"], "reason": missing["failed"].get(h["ticker"], "No prediction.")})
            continue
        amount = h["value"] * p["price"] if h["mode"] == "shares" else h["value"]
        holdings.append({**h, "amount": amount})
    if not holdings:
        raise ValueError("None of this portfolio's holdings could be modeled.")
    total = sum(h["amount"] for h in holdings)
    holdings.sort(key=lambda h: h["amount"], reverse=True)

    _step(1, PORTFOLIO_STEPS[1])
    closes = research.history([h["ticker"] for h in holdings])
    stats = {h["ticker"]: research.trend_stats(closes[h["ticker"]]) if h["ticker"] in closes else None for h in holdings}
    _update(message="Company profiles")
    profiles = {h["ticker"]: research.profile(h["ticker"]) for h in holdings}

    _step(2, PORTFOLIO_STEPS[2])
    sim = sandbox_sim.simulate([{"ticker": h["ticker"], "amount": h["amount"]} for h in holdings])

    _step(3, PORTFOLIO_STEPS[3])
    ordered = [h["ticker"] for h in holdings]
    news = _gather_news(ordered, set(ordered[:ARTICLE_HOLDINGS]))

    rows = []
    for h in holdings:
        t = h["ticker"]
        rows.append({"ticker": t, "name": profiles[t].get("longName"), "about": _describe(profiles[t]),
                     "amount": round(h["amount"], 2), "weight": h["amount"] / total * 100,
                     "prediction": {k: preds[t][k] for k in ("price", "pred_price", "pred_return", "low_price", "high_price")},
                     "stats": stats[t], "rank": research.ranked(t), "extra": bool(preds[t].get("extra"))})

    _step(4, PORTFOLIO_STEPS[4])
    briefs = {}
    if detailed:
        brief_budget = _prompt_budget(model) - 300
        for i, r in enumerate(rows[:DETAILED_HOLDINGS]):
            briefs[r["ticker"]] = _ask(
                f"Facts: {_stock_facts(r['ticker'], r['prediction'], r['stats'], r['rank'], profiles[r['ticker']])}\n\n"
                f"Recent news:\n{''.join(_fit([_news_block([n]) + chr(10) for n in news[r['ticker']]], brief_budget))}\n"
                "In 3-4 short bullet points: the news tone (positive, negative or mixed) and "
                "main themes, how the price trend looks, and the key risk to watch.",
                model, f"the {r['ticker']} brief ({i + 1} of {min(len(rows), DETAILED_HOLDINGS)})")

    p = sim["percentiles"]
    head = (
        f"Portfolio \"{portfolio['title']}\", ${total:,.0f} across {len(rows)} holdings.\n\n"
        f"Monte Carlo (10,000 simulations of the price model's 30-day forecasts, correlated by past co-movement): "
        f"median value ${p['50']:,.0f} ({(p['50'] / total - 1) * 100:+.1f}%), 10th-90th percentile "
        f"${p['10']:,.0f}–${p['90']:,.0f}, {sim['prob_loss'] * 100:.0f}% chance of a loss.\n\n"
        "Holdings, largest first:\n")
    ask = (
        "Write a portfolio review with these sections: ## Summary (3-4 sentences), ## Trends (what the recent price "
        "trends and the model show), ## Holdings (one short paragraph or bullet per notable holding, citing its news), "
        "## Risks (concentration, volatility, correlated holdings, upcoming events), ## What to watch (bullets). "
        "Keep it under 500 words."
    )
    # fit the prompt to the model's context: every holding gets a line (the largest in full), then news and
    # briefs are added largest holding first for as long as they fit
    budget = _prompt_budget(model) - llm.estimate_tokens(head + ask)
    full = lambda r: _stock_facts(r["ticker"], r["prediction"], r["stats"], r["rank"], profiles[r["ticker"]])
    lines = [f"- {r['weight']:.1f}%: {full(r) if i < FULL_FACTS else _short_facts(r)}" for i, r in enumerate(rows)]
    if llm.estimate_tokens("\n".join(lines)) > budget:
        lines = [f"- {r['weight']:.1f}%: {_short_facts(r)}" for r in rows]
    lines = _fit([line + "\n" for line in lines], budget)
    budget -= llm.estimate_tokens("".join(lines))
    extras = _fit([f"\n{t} brief:\n{briefs[t]}\n" if t in briefs else f"\n{t} headlines:\n{_news_block(news[t][:4])}\n"
                   for t in ordered], budget)
    left_out = len(rows) - len(lines)
    note = f"(The {left_out} smallest holdings are left out for space.)\n" if left_out else ""
    narrative = _ask(head + "".join(lines) + note + "".join(extras) + "\n" + ask, model)

    return {
        "kind": "portfolio", "portfolio_id": int(pid), "title": f"{portfolio['title']} review",
        "model": model, "markdown": narrative, "briefs": briefs,
        "data": {"total": total, "simulation": {k: sim[k] for k in ("invested", "expected", "percentiles", "prob_loss",
                                                                   "simulations", "histogram")},
                 "holdings": rows, "skipped": skipped,
                 "news": {t: [{k: n[k] for k in ("title", "link", "publisher", "published")} for n in items]
                          for t, items in news.items()}},
    }


# ---------- single-stock research ----------
STOCK_STEPS = ["Scoring with the price model", "Pulling prices and company profile", "Scraping news",
               "Writing the report"]


def _stock_report(ticker):
    ticker = ticker.strip().upper()
    model = llm.active_model()
    _step(0, STOCK_STEPS[0])
    missing = on_demand.predict_missing([ticker])
    pred = sandbox_sim.predictions_by_ticker().get(ticker)
    if not pred:
        raise ValueError(missing["failed"].get(ticker, f"Couldn't model {ticker}."))

    _step(1, STOCK_STEPS[1])
    closes = research.history([ticker])
    stats = research.trend_stats(closes[ticker]) if ticker in closes else None
    prof, rank = research.profile(ticker), research.ranked(ticker)

    _step(2, STOCK_STEPS[2])
    news = _gather_news([ticker], {ticker})[ticker]

    _step(3, STOCK_STEPS[3])
    about = "; ".join(f"{k}: {v}" for k, v in prof.items() if k != "summary")
    head = (f"Company: {about}\nBusiness: {prof.get('summary', 'n/a')}\n\n"
            f"Facts: {_stock_facts(ticker, pred, stats, rank, prof)}\n\nRecent news:\n")
    ask = ("\nWrite a research note with these sections: ## Snapshot (2-3 sentences), ## Recent news (themes and tone, "
           "citing headlines), ## Trend and model (what the price trend and the 30-day forecast range show), "
           "## Risks, ## What to watch (bullets). Keep it under 400 words.")
    budget = _prompt_budget(model) - llm.estimate_tokens(head + ask)
    prompt = head + ("".join(_fit([_news_block([n]) + "\n" for n in news], budget)) or "- No recent headlines found.\n") + ask
    narrative = _ask(prompt, model)
    return {
        "kind": "stock", "ticker": ticker, "title": f"{ticker} research", "model": model, "markdown": narrative,
        "data": {"profile": prof, "stats": stats, "rank": rank,
                 "prediction": {k: pred[k] for k in ("price", "pred_price", "pred_return", "low_price", "high_price")},
                 "news": [{k: n[k] for k in ("title", "link", "publisher", "published")} for n in news]},
    }


# ---------- job control ----------
def _run(kind, fn, *args):
    try:
        report = fn(*args)
        rid = store.save_report(report)
        _update(state="done", report_id=rid, message="Report saved.",
                finished_at=datetime.now().isoformat(timespec="seconds"))
    except llm.Stopped:
        _update(state="stopped", message="Stopped.", finished_at=datetime.now().isoformat(timespec="seconds"))
    except Exception as err:
        _update(state="error", message=str(err) or err.__class__.__name__,
                finished_at=datetime.now().isoformat(timespec="seconds"))


def stop():
    """Ask the running job to stop; it does so between model tokens."""
    with _lock:
        running = _status.get("state") == "running"
    if running:
        _stop.set()
    return running


def start(kind, target, detailed=False):
    """kind "portfolio" (target: portfolio id) or "stock" (target: ticker). Returns False if a job is running."""
    if kind == "portfolio":
        fn, steps, args = _portfolio_report, PORTFOLIO_STEPS, (int(target), bool(detailed))
        if not store.get_portfolio(target):
            raise ValueError("Pick a saved portfolio.")
    elif kind == "stock":
        if not str(target or "").strip():
            raise ValueError("Enter a ticker.")
        target = str(target).strip().upper()
        fn, steps, args = _stock_report, STOCK_STEPS, (target,)
    else:
        raise ValueError(f"Unknown report type {kind!r}.")
    with _lock:
        if _status.get("state") == "running":
            return False
        _stop.clear()
        _status.clear()
        _status.update(state="running", kind=kind, target=target, steps=len(steps),
                       step=steps[0], step_index=0, message="", started_ts=time.time(),
                       started_at=datetime.now().isoformat(timespec="seconds"))
    threading.Thread(target=_run, args=(kind, fn, *args), daemon=True).start()
    return True
