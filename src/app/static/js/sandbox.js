(() => {
  const state = {
    stocks: {},      // ticker -> prediction
    ranked: [],      // tickers in trend-rank order, for search
    holdings: [],    // [{ticker, mode: "dollars" | "shares", value}]
    sim: null,
    active: -1,      // highlighted search suggestion
  };

  const $ = (id) => document.getElementById(id);
  const STORAGE_KEY = "sandbox-holdings";
  const DEFAULT_AMOUNT = 1000;
  const MAX_SUGGESTIONS = 8;

  const fmtMoney = (v, digits = 0) => v == null ? "–" : "$" + v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const fmtPrice = (v) => fmtMoney(v, 2);
  const fmtPct = (v) => v == null ? "–" : (v > 0 ? "+" : "") + v.toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 }) + "%";
  const fmtCompact = (v) => "$" + v.toLocaleString(undefined, { notation: "compact", maximumFractionDigits: 1 });
  const tone = (v) => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // fetch() rejects with a TypeError only when the server can't be reached
  const errorText = (err) => err instanceof TypeError ? "Couldn't reach the server. Is the app still running?" : err.message;

  const dollars = (h) => {
    const v = Math.max(0, +h.value || 0);
    return h.mode === "shares" ? v * state.stocks[h.ticker].price : v;
  };

  function save() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.holdings)); } catch (e) {}
  }

  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      state.holdings = saved.filter((h) => state.stocks[h.ticker] && (h.mode === "dollars" || h.mode === "shares"));
    } catch (e) { state.holdings = []; }
  }

  // ---------- holdings table ----------
  function renderRows() {
    const tbody = $("holding-rows");
    if (!state.holdings.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="loading">Search for a ticker above, or let Auto-pick build a portfolio.</td></tr>`;
      return;
    }
    tbody.innerHTML = state.holdings.map((h, i) => {
      const s = state.stocks[h.ticker];
      return `
      <tr data-i="${i}">
        <td class="ticker">${escapeHtml(h.ticker)}</td>
        <td>
          <div class="amount">
            <input type="number" min="0" step="any" inputmode="decimal" value="${h.value}" aria-label="${escapeHtml(h.ticker)} amount" data-field="value">
            <select aria-label="${escapeHtml(h.ticker)} unit" data-field="mode">
              <option value="dollars"${h.mode === "dollars" ? " selected" : ""}>$</option>
              <option value="shares"${h.mode === "shares" ? " selected" : ""}>shares</option>
            </select>
          </div>
          <div class="amount-alt muted" data-cell="alt"></div>
        </td>
        <td class="num hide-sm">${fmtPrice(s.price)}</td>
        <td class="num">${fmtPrice(s.pred_price)} <span class="forecast-pct ${tone(s.pred_return)}">${fmtPct(s.pred_return)}</span></td>
        <td class="num hide-md muted">${fmtPrice(s.low_price)} – ${fmtPrice(s.high_price)}</td>
        <td class="num hide-sm" data-cell="projected"></td>
        <td><button class="icon-btn remove" type="button" data-action="remove" aria-label="Remove ${escapeHtml(h.ticker)}">
          <svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg></button></td>
      </tr>`;
    }).join("");
    updateCells();
  }

  // update computed cells in place so typing in an input doesn't lose focus
  function updateCells() {
    state.holdings.forEach((h, i) => {
      const row = document.querySelector(`#holding-rows tr[data-i="${i}"]`);
      if (!row) return;
      const s = state.stocks[h.ticker];
      const d = dollars(h);
      row.querySelector('[data-cell="alt"]').textContent = h.mode === "shares"
        ? fmtMoney(d, 2)
        : (d / s.price).toLocaleString(undefined, { maximumFractionDigits: 2 }) + " shares";
      row.querySelector('[data-cell="projected"]').textContent = d ? fmtMoney(d * s.pred_price / s.price, 2) : "–";
    });
  }

  // ---------- summary + chart ----------
  function renderSummary() {
    const sim = state.sim;
    const invested = state.holdings.reduce((sum, h) => sum + dollars(h), 0);
    $("stat-invested").textContent = invested ? fmtMoney(invested) : "–";
    if (!sim) {
      $("stat-projected").textContent = $("stat-range").textContent = $("stat-loss").textContent = "–";
      $("chart").innerHTML = `<p class="muted chart-empty">Add stocks to see the range of outcomes.</p>`;
      $("chart-key").hidden = true;
      $("chart-note").textContent = "";
      return;
    }
    const median = sim.percentiles["50"];
    const change = (median / sim.invested - 1) * 100;
    $("stat-projected").innerHTML = `${fmtMoney(median)} <small class="${tone(change)}">${fmtPct(change)}</small>`;
    $("stat-range").textContent = `${fmtMoney(sim.percentiles["10"])} – ${fmtMoney(sim.percentiles["90"])}`;
    $("stat-loss").textContent = Math.round(sim.prob_loss * 100) + "%";
    $("chart-note").textContent = `${sim.simulations.toLocaleString()} simulations`;
    $("chart-key").hidden = false;
    renderChart(sim);
  }

  function renderChart(sim) {
    const { counts, edges } = sim.histogram;
    const W = Math.max(280, $("chart").clientWidth), H = 220, padL = 8, padR = 8, padT = 12, padB = 28;
    const lo = edges[0], hi = edges[edges.length - 1];
    const x = (v) => padL + ((v - lo) / (hi - lo || 1)) * (W - padL - padR);
    const maxCount = Math.max(...counts) || 1;
    const y = (c) => H - padB - (c / maxCount) * (H - padT - padB);
    const p10 = sim.percentiles["10"], p90 = sim.percentiles["90"];
    const gap = 1;

    const bars = counts.map((c, i) => {
      const x0 = x(edges[i]) + gap / 2, x1 = x(edges[i + 1]) - gap / 2;
      const mid = (edges[i] + edges[i + 1]) / 2;
      const inBand = mid >= p10 && mid <= p90;
      const top = y(c), h = H - padB - top;
      const r = Math.min(3, (x1 - x0) / 2, h);
      // bar with rounded top corners, square at the baseline
      const d = h <= 0 ? "" : `M${x0},${H - padB}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x1 - r}Q${x1},${top} ${x1},${top + r}V${H - padB}Z`;
      return `<g class="bar${inBand ? "" : " faint"}" data-i="${i}">
        <rect class="hit" x="${x(edges[i])}" y="${padT}" width="${x(edges[i + 1]) - x(edges[i])}" height="${H - padT - padB}"/>
        <path d="${d}"/></g>`;
    }).join("");

    const ticks = [];
    for (let k = 0; k <= 4; k++) ticks.push(lo + ((hi - lo) * k) / 4);
    const axis = ticks.map((v, k) => `<text x="${x(v)}" y="${H - 8}" text-anchor="${k === 0 ? "start" : k === 4 ? "end" : "middle"}">${fmtCompact(v)}</text>`).join("");

    const inv = sim.invested;
    const invLine = inv >= lo && inv <= hi
      ? `<line class="invested" x1="${x(inv)}" x2="${x(inv)}" y1="${padT - 4}" y2="${H - padB}"/>`
      : "";

    $("chart").innerHTML = `
      <svg class="hist" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img"
           aria-label="Histogram of simulated portfolio values; median ${fmtMoney(sim.percentiles["50"])}, 10th to 90th percentile ${fmtMoney(p10)} to ${fmtMoney(p90)}">
        <rect class="band" x="${x(p10)}" y="${padT}" width="${Math.max(0, x(p90) - x(p10))}" height="${H - padT - padB}"/>
        <line class="baseline" x1="${padL}" x2="${W - padR}" y1="${H - padB}" y2="${H - padB}"/>
        ${bars}${invLine}
        <g class="axis">${axis}</g>
      </svg>
      <div class="chart-tip" id="chart-tip" hidden></div>`;

    const tip = $("chart-tip");
    const svg = $("chart").querySelector("svg");
    svg.querySelectorAll(".bar").forEach((g) => {
      g.addEventListener("pointerenter", () => {
        const i = +g.dataset.i;
        const share = (counts[i] / sim.simulations) * 100;
        tip.innerHTML = `<strong>${fmtMoney(edges[i])} – ${fmtMoney(edges[i + 1])}</strong>${share.toFixed(1)}% of simulations`;
        tip.hidden = false;
        const mid = (x(edges[i]) + x(edges[i + 1])) / 2;
        tip.style.left = Math.min(Math.max(mid, 80), W - 80) + "px";
        g.classList.add("hover");
      });
      g.addEventListener("pointerleave", () => { tip.hidden = true; g.classList.remove("hover"); });
    });
  }

  let simTimer = null, simSeq = 0;
  function scheduleSimulation() {
    clearTimeout(simTimer);
    simTimer = setTimeout(runSimulation, 250);
  }

  async function runSimulation() {
    const holdings = state.holdings.map((h) => ({ ticker: h.ticker, amount: dollars(h) })).filter((h) => h.amount > 0);
    const seq = ++simSeq;
    if (!holdings.length) { state.sim = null; renderSummary(); return; }
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ holdings }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      if (seq === simSeq) { state.sim = data; renderSummary(); }
    } catch (err) {
      if (seq === simSeq) $("chart").innerHTML = `<p class="muted chart-empty">Couldn't run the simulation: ${escapeHtml(errorText(err))}</p>`;
    }
  }

  function changed({ rows = false } = {}) {
    save();
    if (rows) renderRows(); else updateCells();
    renderSummary();
    scheduleSimulation();
  }

  // ---------- adding tickers ----------
  function addTicker(ticker, value = DEFAULT_AMOUNT) {
    if (!state.stocks[ticker]) return;
    const existing = state.holdings.findIndex((h) => h.ticker === ticker);
    if (existing === -1) state.holdings.push({ ticker, mode: "dollars", value });
    changed({ rows: true });
    const i = existing === -1 ? state.holdings.length - 1 : existing;
    document.querySelector(`#holding-rows tr[data-i="${i}"] input`)?.focus();
  }

  function suggestions(q) {
    q = q.trim().toUpperCase();
    if (!q) return [];
    const starts = state.ranked.filter((t) => t.startsWith(q));
    const contains = state.ranked.filter((t) => !t.startsWith(q) && t.includes(q));
    return [...starts, ...contains].slice(0, MAX_SUGGESTIONS);
  }

  function renderSuggestions() {
    const input = $("add-search"), list = $("add-list");
    const items = suggestions(input.value);
    state.active = Math.min(state.active, items.length - 1);
    list.hidden = !input.value.trim();
    input.setAttribute("aria-expanded", String(!list.hidden));
    list.innerHTML = items.length ? items.map((t, i) => {
      const s = state.stocks[t];
      const held = state.holdings.some((h) => h.ticker === t);
      return `<li role="option" id="opt-${i}" data-ticker="${t}" aria-selected="${i === state.active}" class="${i === state.active ? "active" : ""}">
        <span class="ticker">${escapeHtml(t)}</span>
        <span class="muted">${fmtPrice(s.price)}</span>
        <span class="${tone(s.pred_return)}">${fmtPct(s.pred_return)}</span>
        ${held ? '<span class="pill">Added</span>' : ""}
      </li>`;
    }).join("") : `<li class="combo-empty muted">No ranked ticker matches.</li>`;
    if (state.active >= 0) input.setAttribute("aria-activedescendant", `opt-${state.active}`);
    else input.removeAttribute("aria-activedescendant");
  }

  function closeSuggestions() {
    $("add-list").hidden = true;
    $("add-search").setAttribute("aria-expanded", "false");
    state.active = -1;
  }

  async function autopick() {
    const objective = new FormData($("autopick")).get("objective");
    const go = $("ap-go"), msg = $("ap-msg");
    go.disabled = true;
    msg.classList.remove("down");
    msg.textContent = "Picking stocks…";
    try {
      const res = await fetch("/api/autopick", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objective, budget: +$("ap-budget").value, count: +$("ap-count").value, min_price: +$("ap-min-price").value }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      state.holdings = data.holdings.map((h) => ({ ticker: h.ticker, mode: "dollars", value: h.amount }));
      changed({ rows: true });
      const goal = objective === "return" ? "highest projected return" : "lowest chance of loss";
      msg.textContent = `Picked ${data.holdings.length} stocks for the ${goal}. Edit or remove any of them below.`;
    } catch (err) {
      msg.classList.add("down");
      msg.textContent = errorText(err);
    } finally {
      go.disabled = false;
    }
  }

  function bind() {
    const input = $("add-search");
    input.addEventListener("input", () => { state.active = 0; renderSuggestions(); });
    input.addEventListener("focus", () => { if (input.value.trim()) renderSuggestions(); });
    input.addEventListener("keydown", (e) => {
      const items = suggestions(input.value);
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!items.length) return;
        state.active = (state.active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
        renderSuggestions();
      } else if (e.key === "Enter") {
        e.preventDefault();
        const pick = items[Math.max(state.active, 0)];
        if (pick) { input.value = ""; closeSuggestions(); addTicker(pick); }
      } else if (e.key === "Escape") {
        closeSuggestions();
      }
    });
    $("add-list").addEventListener("pointerdown", (e) => {
      const li = e.target.closest("li[data-ticker]");
      if (!li) return;
      e.preventDefault();  // keep focus off the input so the list closes cleanly
      input.value = "";
      closeSuggestions();
      addTicker(li.dataset.ticker);
    });
    document.addEventListener("pointerdown", (e) => { if (!e.target.closest("#combo")) closeSuggestions(); });

    const tbody = $("holding-rows");
    tbody.addEventListener("input", (e) => {
      const row = e.target.closest("tr[data-i]");
      if (!row || e.target.dataset.field !== "value") return;
      state.holdings[+row.dataset.i].value = e.target.value;
      changed();
    });
    tbody.addEventListener("change", (e) => {
      const row = e.target.closest("tr[data-i]");
      if (!row || e.target.dataset.field !== "mode") return;
      const h = state.holdings[+row.dataset.i];
      // keep the same dollar amount when switching units
      const d = dollars(h), price = state.stocks[h.ticker].price;
      h.mode = e.target.value;
      h.value = +(h.mode === "shares" ? d / price : d).toFixed(h.mode === "shares" ? 4 : 2);
      changed({ rows: true });
    });
    tbody.addEventListener("click", (e) => {
      const btn = e.target.closest('[data-action="remove"]');
      if (!btn) return;
      state.holdings.splice(+btn.closest("tr").dataset.i, 1);
      changed({ rows: true });
    });

    const panel = $("autopick"), apBtn = $("autopick-btn");
    apBtn.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      apBtn.setAttribute("aria-expanded", String(!panel.hidden));
    });
    panel.addEventListener("submit", (e) => { e.preventDefault(); autopick(); });
    $("clear-btn").addEventListener("click", () => { state.holdings = []; changed({ rows: true }); });
  }

  async function init() {
    try {
      const res = await fetch("/api/predictions");
      if (!res.ok) throw new Error((await res.json()).error || res.statusText);
      const data = await res.json();
      for (const s of data.stocks) state.stocks[s.ticker] = s;
      state.ranked = data.stocks.map((s) => s.ticker);
    } catch (err) {
      $("holding-rows").innerHTML = `<tr><td colspan="7" class="loading">Couldn't load predictions: ${escapeHtml(err.message)}</td></tr>`;
      return;
    }
    restore();
    bind();
    let resizeTimer = null;
    addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { if (state.sim) renderChart(state.sim); }, 150);
    });
    changed({ rows: true });
  }

  init();
})();
