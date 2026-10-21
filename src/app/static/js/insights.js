(() => {
  const { api, escapeHtml, errorText, fmtMoney, fmtPrice, fmtPct, fmtCompact, fmtWhen, tone, watchAgent, renderReport, newsList } = window.Centaur;
  const $ = (id) => document.getElementById(id);
  const root = $("insight");
  const ticker = root?.dataset.ticker;
  const HORIZON_DAYS = 30;
  let data = null;

  $("insight-search").addEventListener("submit", (e) => {
    e.preventDefault();
    const t = $("insight-ticker").value.trim().toUpperCase();
    if (t) location.href = `/insights/${encodeURIComponent(t)}`;
  });

  // ---------- no ticker: suggest a few top-ranked ones ----------
  async function quickPicks() {
    try {
      const { stocks } = await api("/api/predictions");
      $("quick-picks").innerHTML = stocks.slice(0, 8).map((s) =>
        `<a class="btn" href="/insights/${encodeURIComponent(s.ticker)}">${escapeHtml(s.ticker)}</a>`).join("");
    } catch (err) { /* the search box still works */ }
  }

  // ---------- chart: a year of prices, then the forecast range fanning out 30 days ----------
  function renderChart() {
    const { dates, closes } = data.history;
    const pred = data.prediction;
    const W = Math.max(300, $("ins-chart").clientWidth), H = 260, padL = 8, padR = 56, padT = 12, padB = 26;
    const t0 = new Date(dates[0]).getTime(), tLast = new Date(dates[dates.length - 1]).getTime();
    const tEnd = pred ? tLast + HORIZON_DAYS * 864e5 : tLast;
    const vals = [...closes, ...(pred ? [pred.low_price, pred.high_price] : [])];
    const lo = Math.min(...vals) * 0.97, hi = Math.max(...vals) * 1.03;
    const x = (t) => padL + ((t - t0) / (tEnd - t0 || 1)) * (W - padL - padR);
    const y = (v) => padT + (1 - (v - lo) / (hi - lo || 1)) * (H - padT - padB);
    const pts = closes.map((c, i) => `${x(new Date(dates[i]).getTime()).toFixed(1)},${y(c).toFixed(1)}`);
    const last = closes[closes.length - 1];

    let fan = "";
    if (pred) {
      fan = `<path class="fan" d="M${x(tLast)},${y(last)}L${x(tEnd)},${y(pred.high_price)}L${x(tEnd)},${y(pred.low_price)}Z"/>
        <line class="fan-mid" x1="${x(tLast)}" y1="${y(last)}" x2="${x(tEnd)}" y2="${y(pred.pred_price)}"/>
        <text class="end-label" x="${x(tEnd) + 6}" y="${y(pred.high_price) + 4}">${fmtPrice(pred.high_price)}</text>
        <text class="end-label strong" x="${x(tEnd) + 6}" y="${y(pred.pred_price) + 4}">${fmtPrice(pred.pred_price)}</text>
        <text class="end-label" x="${x(tEnd) + 6}" y="${y(pred.low_price) + 4}">${fmtPrice(pred.low_price)}</text>`;
    }
    const ticks = [];
    for (let k = 0; k <= 4; k++) {
      const t = t0 + ((tLast - t0) * k) / 4;
      ticks.push(`<text x="${x(t)}" y="${H - 6}" text-anchor="${k === 0 ? "start" : "middle"}">${new Date(t).toLocaleDateString(undefined, { month: "short", year: "2-digit" })}</text>`);
    }
    $("ins-chart").innerHTML = `
      <svg class="hist line-chart" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img"
           aria-label="${escapeHtml(ticker)} price over the last year${pred ? `, with a 30-day forecast of ${fmtPrice(pred.pred_price)}` : ""}">
        <line class="baseline" x1="${padL}" x2="${W - padR}" y1="${H - padB}" y2="${H - padB}"/>
        <line class="today" x1="${x(tLast)}" x2="${x(tLast)}" y1="${padT}" y2="${H - padB}"/>
        ${fan}
        <polyline class="price-line ${tone(last - closes[0])}" points="${pts.join(" ")}"/>
        <g class="axis">${ticks.join("")}</g>
      </svg>`;
    $("ins-chart-note").textContent = pred ? `Forecast to ${new Date(tEnd).toLocaleDateString(undefined, { month: "short", day: "numeric" })}` : "";
  }

  const fact = (label, value, cls = "") => value == null || value === "" ? "" : `<dt>${label}</dt><dd class="${cls}">${value}</dd>`;

  function render() {
    const { profile: p, stats: s, rank: r, prediction: pred } = data;
    $("ins-name").textContent = p.longName ? `${p.longName} (${ticker})` : ticker;
    $("ins-sub").textContent = [p.sector, p.industry, p.quoteType && p.quoteType !== "EQUITY" ? p.quoteType : null].filter(Boolean).join(" · ") || "No company profile";
    $("ins-price").textContent = fmtPrice(s.price);
    $("ins-change").innerHTML = `<span class="${tone(s.change_1d)}">${fmtPct(s.change_1d, 2)}</span> <span class="muted">on ${escapeHtml(s.as_of)}</span>`;

    if (pred) {
      $("ins-forecast").innerHTML = `${fmtPrice(pred.pred_price)} <small class="${tone(pred.pred_return)}">${fmtPct(pred.pred_return)}</small>`;
      $("ins-range").textContent = `${fmtPrice(pred.low_price)} – ${fmtPrice(pred.high_price)}`;
    } else {
      $("ins-forecast").innerHTML = `<small class="muted">${escapeHtml(data.prediction_error || "Not modeled")}</small>`;
    }
    $("ins-trend").innerHTML = r ? `${Math.round(r.trend)} <small>rank ${r.rank.toLocaleString()} of ${r.of.toLocaleString()}</small>` : `<small class="muted">Not ranked</small>`;
    $("ins-1y").innerHTML = `<span class="${tone(s.return_1y)}">${fmtPct(s.return_1y)}</span>`;

    $("ins-trend-facts").innerHTML = [
      fact("1 week", fmtPct(s.return_1w), tone(s.return_1w)),
      fact("1 month", fmtPct(s.return_1m), tone(s.return_1m)),
      fact("3 months", fmtPct(s.return_3m), tone(s.return_3m)),
      fact("vs 50-day average", fmtPct(s.vs_ma50), tone(s.vs_ma50)),
      fact("vs 200-day average", fmtPct(s.vs_ma200), tone(s.vs_ma200)),
      fact("From 1-year high", fmtPct(s.off_high_1y)),
      fact("Worst drop, 3 months", fmtPct(s.max_drawdown_3m)),
      fact("Volatility (1 month, annualized)", s.volatility_1m == null ? null : s.volatility_1m.toFixed(0) + "%"),
    ].join("");
    const rec = p.recommendationKey ? p.recommendationKey.replace(/_/g, " ") : null;
    $("ins-profile").innerHTML = [
      fact("Market cap", p.marketCap ? fmtCompact(p.marketCap) : null),
      fact("P/E (trailing)", p.trailingPE ? p.trailingPE.toFixed(1) : null),
      fact("P/E (forward)", p.forwardPE ? p.forwardPE.toFixed(1) : null),
      fact("Dividend yield", p.dividendYield ? p.dividendYield.toFixed(2) + "%" : null),
      fact("52-week range", p.fiftyTwoWeekLow ? `${fmtPrice(p.fiftyTwoWeekLow)} – ${fmtPrice(p.fiftyTwoWeekHigh)}` : null),
      fact("Analyst view", rec ? `${escapeHtml(rec)}${p.numberOfAnalystOpinions ? ` <span class="muted">(${p.numberOfAnalystOpinions})</span>` : ""}` : null),
      fact("Analyst mean target", p.targetMeanPrice ? fmtPrice(p.targetMeanPrice) : null),
    ].join("") + (p.summary ? `<p class="muted company-summary">${escapeHtml(p.summary)}</p>` : "");

    $("ins-news").innerHTML = newsList(data.news);
    $("insight-loading").hidden = true;
    $("insight-body").hidden = false;
    renderChart();
  }

  // ---------- AI research ----------
  async function loadReports(showId) {
    const { reports } = await api(`/api/reports?ticker=${encodeURIComponent(ticker)}`);
    const id = showId || reports[0]?.id;
    $("ins-report").innerHTML = id ? renderReport(await api(`/api/reports/${id}`)) : `<p class="muted">No research yet. The agent reads the latest news and writes a note with your local model.</p>`;
    const older = reports.filter((r) => r.id !== id);
    $("ins-older").innerHTML = older.length ? `<details class="fold"><summary>Earlier research <span class="muted">${older.length}</span></summary>
      <ul class="report-list">${older.map((r) => `<li><button type="button" class="report-item" data-report="${r.id}"><strong>${escapeHtml(r.title)}</strong><span class="muted">${escapeHtml(fmtWhen(r.created_at))}</span></button></li>`).join("")}</ul></details>` : "";
  }

  async function loadLlm() {
    try {
      const s = (await api("/api/connections")).lmstudio;
      $("ins-llm").innerHTML = s.online && s.model
        ? `Uses <strong>${escapeHtml(s.model)}</strong> in LM Studio. Local models can take several minutes.`
        : `<span class="down">${escapeHtml(s.message)}</span> <a href="/connections">Set up LM Studio</a>`;
      $("ins-research").disabled = !(s.online && s.model);
    } catch (err) { $("ins-llm").textContent = errorText(err); }
  }

  function watch() {
    watchAgent($("ins-job"), {
      onUpdate: (s) => {
        const mine = s.kind === "stock" && s.target === ticker;
        $("ins-job").hidden = $("ins-job").hidden || !mine;
        if (s.state === "running") $("ins-research").disabled = true;
        if (s.state === "running" && !mine) $("ins-llm").innerHTML = `The agent is busy with another report. <a href="/agent">See progress</a>`;
      },
      onDone: (s) => { loadLlm(); if (s.kind === "stock" && s.target === ticker) loadReports(s.report_id); },
    });
  }

  async function research() {
    $("ins-research").disabled = true;
    try {
      await api("/api/agent", { kind: "stock", target: ticker });
    } catch (err) {
      if (err.status !== 409) { $("ins-research").disabled = false; alert(errorText(err)); return; }
      alert("The agent is already writing another report. This one can start when it finishes.");
    }
    watch();
  }

  async function init() {
    if (!ticker) { quickPicks(); return; }
    try {
      data = await api(`/api/insights/${encodeURIComponent(ticker)}`);
      // the model predicted from its last data refresh; carry its forecast returns over to today's price
      const pred = data.prediction, scale = pred ? data.stats.price / pred.price : 1;
      if (pred) data.prediction = { ...pred, price: data.stats.price, pred_price: pred.pred_price * scale, low_price: pred.low_price * scale, high_price: pred.high_price * scale };
    } catch (err) {
      $("insight-loading").textContent = errorText(err);
      return;
    }
    render();
    $("ins-research").addEventListener("click", research);
    $("ins-older").addEventListener("click", (e) => { const b = e.target.closest("[data-report]"); if (b) loadReports(+b.dataset.report); });
    $("ins-report").addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-delete-report]");
      if (!btn || !confirm("Delete this research note?")) return;
      await api(`/api/reports/${btn.dataset.deleteReport}`, null, "DELETE");
      loadReports();
    });
    let resizeTimer = null;
    addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderChart, 150); });
    await Promise.all([loadLlm(), loadReports()]);
    watch();
  }

  init();
})();
