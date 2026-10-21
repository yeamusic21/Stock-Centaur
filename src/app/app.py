import html
import json
import re
from pathlib import Path

import markdown
from flask import Flask, abort, jsonify, render_template, request
from markupsafe import Markup

import agent
import brokers
import llm
import on_demand
import refresh
import research
import sandbox as sandbox_sim
import store

REPO_ROOT = Path(__file__).resolve().parents[2]
TREND_TABLE_PATH = REPO_ROOT / "data" / "trend_table.json"

app = Flask(__name__)

_cache = {"mtime": None, "data": None}


def load_trend_table():
    """Load data/trend_table.json, re-reading it only when the file changes."""
    if not TREND_TABLE_PATH.exists():
        return None
    mtime = TREND_TABLE_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["data"] = json.loads(TREND_TABLE_PATH.read_text())
        _cache["mtime"] = mtime
    return _cache["data"]


@app.route("/")
def rankings():
    data = load_trend_table()
    meta = None if data is None else {"as_of": data["as_of"], "weights": data["weights"], "count": len(data["stocks"])}
    return render_template("rankings.html", active="rankings", meta=meta)


@app.route("/api/stocks")
def api_stocks():
    data = load_trend_table()
    if data is None:
        return jsonify({"error": "No trend data yet. Run the modeling pipeline first."}), 404
    return jsonify(data)


@app.route("/api/refresh", methods=["GET"])
def api_refresh_status():
    return jsonify(refresh.status())


@app.route("/api/refresh", methods=["POST"])
def api_refresh_start():
    started = refresh.start()
    return jsonify(refresh.status()), 202 if started else 409


@app.route("/sandbox")
def sandbox():
    data = sandbox_sim.load_predictions()
    meta = None if data is None else {k: data[k] for k in ["as_of", "target_date", "model"]}
    return render_template("sandbox.html", active="sandbox", meta=meta)


@app.route("/api/predictions")
def api_predictions():
    data = sandbox_sim.load_predictions()
    if data is None:
        return jsonify({"error": "No predictions yet. Run the modeling pipeline first."}), 404
    # tickers scored on demand (e.g. brokerage ETFs) come after the ranked ones
    return jsonify({**data, "stocks": data["stocks"] + list(sandbox_sim.load_extra()["stocks"].values())})


@app.route("/api/predictions/extra", methods=["POST"])
def api_predictions_extra():
    """Score tickers outside the ranked set with the price model so the sandbox can simulate them."""
    if sandbox_sim.load_predictions() is None:
        return no_predictions()
    tickers = (request.get_json(silent=True) or {}).get("tickers") or []
    if not isinstance(tickers, list) or not tickers or len(tickers) > 50:
        return bad_request("Send between 1 and 50 tickers.")
    return jsonify(on_demand.predict_missing([str(t).strip().upper() for t in tickers if str(t).strip()]))


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    if sandbox_sim.load_predictions() is None:
        return jsonify({"error": "No predictions yet. Run the modeling pipeline first."}), 404
    try:
        return jsonify(sandbox_sim.simulate((request.get_json(silent=True) or {}).get("holdings", [])))
    except (ValueError, TypeError, KeyError) as err:
        return jsonify({"error": str(err)}), 400


@app.route("/api/autopick", methods=["POST"])
def api_autopick():
    if sandbox_sim.load_predictions() is None:
        return jsonify({"error": "No predictions yet. Run the modeling pipeline first."}), 404
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(sandbox_sim.autopick(body.get("objective"), body.get("budget", 0), body.get("count", 0),
                                            body.get("min_price", sandbox_sim.MIN_PICK_PRICE)))
    except (ValueError, TypeError) as err:
        return jsonify({"error": str(err)}), 400


def no_predictions():
    return jsonify({"error": "No predictions yet. Run the modeling pipeline first."}), 404


def bad_request(err):
    return jsonify({"error": str(err)}), 400


# ---------- saved portfolios ----------
@app.route("/api/portfolios")
def api_portfolios():
    return jsonify({"portfolios": store.list_portfolios()})


@app.route("/api/portfolios", methods=["POST"])
def api_portfolio_save():
    body = request.get_json(silent=True) or {}
    try:
        pid = store.save_portfolio(body.get("title"), body.get("holdings"), pid=body.get("id"))
    except (ValueError, TypeError, KeyError) as err:
        return bad_request(err)
    return jsonify({"id": pid, "portfolios": store.list_portfolios()})


@app.route("/api/portfolios/<int:pid>")
def api_portfolio(pid):
    portfolio = store.get_portfolio(pid)
    if not portfolio:
        return jsonify({"error": "That portfolio no longer exists."}), 404
    if sandbox_sim.load_predictions() is None:
        return no_predictions()
    # brokerage holdings can include tickers outside the ranked set; score those with the model first
    extra = on_demand.predict_missing([h["ticker"] for h in portfolio["holdings"]])
    return jsonify({**portfolio, "predictions": extra["stocks"], "failed": extra["failed"]})


@app.route("/api/portfolios/<int:pid>", methods=["DELETE"])
def api_portfolio_delete(pid):
    store.delete_portfolio(pid)
    return jsonify({"portfolios": store.list_portfolios()})


# ---------- connections ----------
@app.route("/connections")
def connections():
    return render_template("connections.html", active="connections")


@app.route("/api/connections")
def api_connections():
    return jsonify({"brokers": brokers.status(), "lmstudio": {**llm.status(), **store.get_settings()}})


@app.route("/api/connections/lmstudio", methods=["POST"])
def api_lmstudio_save():
    body = request.get_json(silent=True) or {}
    store.update_settings(lmstudio_url=body.get("url"), lmstudio_model=body.get("model"),
                          lmstudio_command=body.get("command"))
    return jsonify({**llm.status(), **store.get_settings()})


@app.route("/api/connections/lmstudio/launch", methods=["POST"])
def api_lmstudio_launch():
    try:
        llm.launch()
    except llm.LLMError as err:
        return bad_request(err)
    return jsonify({"ok": True})


@app.route("/api/connections/robinhood", methods=["POST"])
def api_robinhood_pull():
    body = request.get_json(silent=True) or {}
    try:
        started = brokers.start_robinhood(body.get("username", "").strip(), body.get("password", ""),
                                          body.get("mfa_code", "").strip())
    except ValueError as err:
        return bad_request(err)
    return jsonify(brokers.status()["robinhood"]), 202 if started else 409


@app.route("/api/connections/robinhood/code", methods=["POST"])
def api_robinhood_code():
    if not brokers.submit_robinhood_code((request.get_json(silent=True) or {}).get("code", "")):
        return bad_request("Robinhood isn't waiting for a code right now.")
    return jsonify(brokers.status()["robinhood"])


@app.route("/api/connections/etrade/authorize", methods=["POST"])
def api_etrade_authorize():
    body = request.get_json(silent=True) or {}
    try:
        url = brokers.etrade_authorize_url(body.get("consumer_key", ""), body.get("consumer_secret", ""),
                                           sandbox=bool(body.get("sandbox")))
    except ValueError as err:
        return bad_request(err)
    return jsonify({"url": url})


@app.route("/api/connections/etrade/verify", methods=["POST"])
def api_etrade_verify():
    try:
        return jsonify(brokers.etrade_pull((request.get_json(silent=True) or {}).get("verifier", "")))
    except ValueError as err:
        return bad_request(err)


# ---------- research agent ----------
FENCE_RE = re.compile(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", re.S)


def render_markdown(text):
    # models sometimes wrap the whole answer in a ```markdown fence, which would render as one code block
    match = FENCE_RE.match(text.strip())
    text = match.group(1) if match else text
    # escape first: the text comes from a language model and shouldn't be able to inject HTML
    return Markup(markdown.markdown(html.escape(text, quote=False), extensions=["tables", "sane_lists"]))


@app.route("/agent")
def agent_page():
    return render_template("agent.html", active="agent", has_predictions=sandbox_sim.load_predictions() is not None)


@app.route("/api/agent")
def api_agent_status():
    return jsonify(agent.status())


@app.route("/api/agent", methods=["POST"])
def api_agent_start():
    body = request.get_json(silent=True) or {}
    if sandbox_sim.load_predictions() is None:
        return no_predictions()
    try:
        started = agent.start(body.get("kind"), body.get("target"), detailed=body.get("detailed", False))
    except (ValueError, TypeError) as err:
        return bad_request(err)
    if not started:
        return jsonify({**agent.status(), "error": "The agent is already working on a report."}), 409
    return jsonify(agent.status()), 202


@app.route("/api/agent/stop", methods=["POST"])
def api_agent_stop():
    agent.stop()
    return jsonify(agent.status())


@app.route("/api/reports")
def api_reports():
    pid, ticker = request.args.get("portfolio_id", type=int), request.args.get("ticker")
    reports = store.list_reports(pid)
    if ticker:
        reports = [r for r in reports if r.get("ticker") == ticker.upper()]
    return jsonify({"reports": reports})


@app.route("/api/reports/<int:rid>")
def api_report(rid):
    report = store.get_report(rid)
    if not report:
        return jsonify({"error": "That report no longer exists."}), 404
    briefs = {t: str(render_markdown(b)) for t, b in (report.get("briefs") or {}).items()}
    return jsonify({**report, "html": str(render_markdown(report["markdown"])), "briefs_html": briefs})


@app.route("/api/reports/<int:rid>", methods=["DELETE"])
def api_report_delete(rid):
    store.delete_report(rid)
    return jsonify({"ok": True})


# ---------- stock insights ----------
@app.route("/insights")
@app.route("/insights/<ticker>")
def insights(ticker=None):
    ticker = ticker.strip().upper() if ticker else None
    if ticker is not None and not ticker.replace(".", "").replace("-", "").isalnum():
        abort(404)
    return render_template("insights.html", active="insights", ticker=ticker,
                           has_predictions=sandbox_sim.load_predictions() is not None)


@app.route("/api/insights/<ticker>")
def api_insights(ticker):
    ticker = ticker.strip().upper()
    if sandbox_sim.load_predictions() is None:
        return no_predictions()
    missing = on_demand.predict_missing([ticker])
    pred = sandbox_sim.predictions_by_ticker().get(ticker)
    closes = research.history([ticker])
    if ticker not in closes:
        return jsonify({"error": f"No price data found for {ticker}."}), 404
    series = closes[ticker].dropna()
    return jsonify({
        "ticker": ticker,
        "profile": research.profile(ticker),
        "stats": research.trend_stats(series),
        "rank": research.ranked(ticker),
        "prediction": pred, "prediction_error": missing["failed"].get(ticker),
        "history": {"dates": series.index.strftime("%Y-%m-%d").tolist(), "closes": series.round(2).tolist()},
        "news": research.news(ticker),
    })


@app.route("/methodology")
def methodology():
    data = load_trend_table()
    weights = data["weights"] if data else None
    return render_template("methodology.html", active="methodology", weights=weights)


if __name__ == "__main__":
    app.run(debug=True)
