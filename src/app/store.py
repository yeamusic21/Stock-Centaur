"""Saved portfolios, agent reports and app settings in a TinyDB file (data/stock_centaur.json).

Portfolios have a source: "sandbox" for ones saved from the Portfolio Sandbox, or "robinhood" /
"etrade" for ones pulled from a brokerage (re-pulling the same account replaces its holdings).
Holdings are stored as {"ticker", "mode": "dollars" | "shares", "value"}, the sandbox's format.
Broker logins are never stored here.
"""
import threading
from datetime import datetime
from pathlib import Path

from tinydb import Query, TinyDB

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "stock_centaur.json"
SOURCES = ("sandbox", "robinhood", "etrade")
MAX_TITLE = 80
DEFAULT_SETTINGS = {
    "lmstudio_url": "http://localhost:1234/v1",
    "lmstudio_model": "",
    "lmstudio_command": "~/bin/lmstudio",
}

_lock = threading.Lock()  # TinyDB isn't thread-safe and the agent writes from background threads
_db = None


def _table(name):
    global _db
    if _db is None:
        DB_PATH.parent.mkdir(exist_ok=True)
        _db = TinyDB(DB_PATH, indent=1)
    return _db.table(name)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _with_id(doc):
    return None if doc is None else {**doc, "id": doc.doc_id}


def clean_holdings(holdings):
    rows = []
    for h in holdings or []:
        ticker = str(h.get("ticker", "")).strip().upper()
        mode = h.get("mode", "dollars")
        value = float(h.get("value", 0) or 0)
        if ticker and mode in ("dollars", "shares") and value > 0:
            rows.append({"ticker": ticker, "mode": mode, "value": round(value, 6)})
    return rows


# ---------- portfolios ----------
def list_portfolios():
    with _lock:
        docs = _table("portfolios").all()
    docs = sorted(docs, key=lambda d: d.get("updated_at", ""), reverse=True)
    return [{**{k: d[k] for k in ("title", "source", "updated_at")}, "id": d.doc_id, "count": len(d["holdings"]),
             "account": d.get("account")} for d in docs]


def get_portfolio(pid):
    with _lock:
        return _with_id(_table("portfolios").get(doc_id=int(pid)))


def save_portfolio(title, holdings, source="sandbox", pid=None, account=None, extra=None):
    """Create or update a portfolio. Brokerage portfolios are matched on (source, account)."""
    title = str(title or "").strip()[:MAX_TITLE]
    if not title:
        raise ValueError("Give the portfolio a title.")
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}.")
    holdings = clean_holdings(holdings)
    if not holdings:
        raise ValueError("A portfolio needs at least one holding with an amount above 0.")
    doc = {"title": title, "source": source, "holdings": holdings, "updated_at": _now(), **(extra or {})}
    if account:
        doc["account"] = account
    with _lock:
        table = _table("portfolios")
        if pid is None and account:
            q = Query()
            match = table.get((q.source == source) & (q.account == account))
            pid = match.doc_id if match else None
        if pid is not None:
            if not table.contains(doc_id=int(pid)):
                raise KeyError("That portfolio no longer exists.")
            table.update(doc, doc_ids=[int(pid)])
            return int(pid)
        doc["created_at"] = doc["updated_at"]
        return table.insert(doc)


def delete_portfolio(pid):
    with _lock:
        return bool(_table("portfolios").remove(doc_ids=[int(pid)]))


# ---------- reports ----------
def list_reports(portfolio_id=None):
    with _lock:
        docs = _table("reports").all()
    if portfolio_id is not None:
        docs = [d for d in docs if d.get("portfolio_id") == int(portfolio_id)]
    docs = sorted(docs, key=lambda d: d["created_at"], reverse=True)
    return [{"id": d.doc_id, **{k: d.get(k) for k in ("title", "kind", "created_at", "portfolio_id", "ticker", "model")}}
            for d in docs]


def get_report(rid):
    with _lock:
        return _with_id(_table("reports").get(doc_id=int(rid)))


def save_report(report):
    with _lock:
        return _table("reports").insert({**report, "created_at": _now()})


def delete_report(rid):
    with _lock:
        return bool(_table("reports").remove(doc_ids=[int(rid)]))


# ---------- settings ----------
def get_settings():
    with _lock:
        doc = _table("settings").get(doc_id=1) or {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in doc.items() if k in DEFAULT_SETTINGS}}


def update_settings(**fields):
    fields = {k: str(v).strip() for k, v in fields.items() if k in DEFAULT_SETTINGS and v is not None}
    with _lock:
        table = _table("settings")
        if table.contains(doc_id=1):
            table.update(fields, doc_ids=[1])
        else:
            table.insert({**DEFAULT_SETTINGS, **fields})
    return get_settings()
