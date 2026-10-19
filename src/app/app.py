import json
from pathlib import Path

from flask import Flask, jsonify, render_template

import refresh

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
    return render_template(
        "coming_soon.html",
        active="sandbox",
        title="Portfolio Sandbox",
        blurb="Pick stocks, build a hypothetical portfolio, and simulate projected earnings with a "
              "tree-based model that forecasts each stock's price 30 days out from its trend features.",
    )


@app.route("/insights")
def insights():
    return render_template(
        "coming_soon.html",
        active="insights",
        title="Stock Insights",
        blurb="A per-stock overview focused on ML and AI insights. LLM-powered research will be an "
              "optional feature you can turn on with your own OpenAI API key.",
    )


@app.route("/methodology")
def methodology():
    data = load_trend_table()
    weights = data["weights"] if data else None
    return render_template("methodology.html", active="methodology", weights=weights)


if __name__ == "__main__":
    app.run(debug=True)
