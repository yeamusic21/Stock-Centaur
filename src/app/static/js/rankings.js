(() => {
  const state = {
    stocks: [],
    defaults: {},
    weights: {},
    sortKey: "trend",
    sortDir: -1,
    query: "",
    minPrice: 5,
    pageSize: 50,
    page: 0,
  };

  const $ = (id) => document.getElementById(id);
  const tbody = $("stock-rows");
  const WEIGHT_KEYS = ["5y", "1y", "1m"];

  const fmtPrice = (v) => v == null ? "–" : "$" + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtPct = (v) => v == null ? "–" : (v > 0 ? "+" : "") + v.toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 }) + "%";
  const tone = (v) => v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "";
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function sparkline(points) {
    if (!points || points.length < 2) return "";
    const w = 96, h = 28, pad = 2;
    const min = Math.min(...points), max = Math.max(...points);
    const span = max - min || 1;
    const step = (w - pad * 2) / (points.length - 1);
    const d = points.map((p, i) => `${i ? "L" : "M"}${(pad + i * step).toFixed(1)},${(h - pad - ((p - min) / span) * (h - pad * 2)).toFixed(1)}`).join("");
    const cls = points[points.length - 1] >= points[0] ? "up" : "down";
    return `<svg class="spark ${cls}" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true"><path d="${d}"/></svg>`;
  }

  // Recompute trend scores and ranks from the per-slope percentiles using the current weights.
  function rescore() {
    const total = WEIGHT_KEYS.reduce((s, k) => s + state.weights[k], 0);
    for (const s of state.stocks) {
      let num = 0, den = 0;
      for (const k of WEIGHT_KEYS) {
        const p = s["pct_" + k];
        if (p != null) { num += p * state.weights[k]; den += state.weights[k]; }
      }
      s.trend = den ? num / den : null;
    }
    [...state.stocks].sort((a, b) => (b.trend ?? -1) - (a.trend ?? -1)).forEach((s, i) => { s.rank = i + 1; });
    const pct = (k) => total ? Math.round((state.weights[k] / total) * 100) + "%" : "0%";
    for (const k of WEIGHT_KEYS) $(`w-${k}-out`).textContent = pct(k);
    $("trend-tip-weights").textContent = `5-year slope ${pct("5y")}, 1-year ${pct("1y")}, 1-month ${pct("1m")}`;
    $("learn-more-weights").textContent = `5-year ${pct("5y")}, 1-year ${pct("1y")}, 1-month ${pct("1m")}`;
  }

  function filtered() {
    const q = state.query.trim().toUpperCase();
    const rows = state.stocks.filter((s) => (s.price ?? 0) >= state.minPrice && (!q || s.ticker.includes(q)));
    const k = state.sortKey, dir = state.sortDir;
    rows.sort((a, b) => {
      const av = a[k], bv = b[k];
      if (av == null) return 1;
      if (bv == null) return -1;
      return (typeof av === "string" ? av.localeCompare(bv) : av - bv) * dir;
    });
    return rows;
  }

  function renderStats(rows) {
    const top = rows.reduce((best, s) => (!best || (s.trend ?? -1) > (best.trend ?? -1) ? s : best), null);
    $("stat-count").textContent = rows.length.toLocaleString();
    $("stat-top").innerHTML = top ? `${escapeHtml(top.ticker)} <small>${top.trend.toFixed(1)}</small>` : "–";
    const rets = rows.map((s) => s.return_1y).filter((v) => v != null).sort((a, b) => a - b);
    const median = rets.length ? rets[Math.floor(rets.length / 2)] : null;
    $("stat-median").innerHTML = `<span class="${tone(median)}">${fmtPct(median)}</span>`;
    const up = rows.filter((s) => s.change_1d > 0).length;
    $("stat-up").textContent = rows.length ? Math.round((up / rows.length) * 100) + "%" : "–";
  }

  function render() {
    const rows = filtered();
    const pages = Math.max(1, Math.ceil(rows.length / state.pageSize));
    state.page = Math.min(state.page, pages - 1);
    const start = state.page * state.pageSize;
    const pageRows = rows.slice(start, start + state.pageSize);

    tbody.innerHTML = pageRows.length ? pageRows.map((s) => `
      <tr>
        <td class="num muted">${s.rank}</td>
        <td class="ticker">${escapeHtml(s.ticker)}</td>
        <td><div class="score"><div class="score-bar"><span style="width:${(s.trend ?? 0).toFixed(1)}%"></span></div><b>${s.trend == null ? "–" : s.trend.toFixed(1)}</b></div></td>
        <td class="num">${fmtPrice(s.price)}</td>
        <td class="num ${tone(s.change_1d)}">${fmtPct(s.change_1d)}</td>
        <td class="num hide-sm ${tone(s.return_1y)}">${fmtPct(s.return_1y)}</td>
        <td class="num hide-md ${tone(s.slope_5y)}">${fmtPct(s.slope_5y)}</td>
        <td class="num hide-md ${tone(s.slope_1y)}">${fmtPct(s.slope_1y)}</td>
        <td class="num hide-md ${tone(s.slope_1m)}">${fmtPct(s.slope_1m)}</td>
        <td class="hide-sm">${sparkline(s.sparkline)}</td>
      </tr>`).join("") : `<tr><td colspan="10" class="loading">No stocks match.</td></tr>`;

    $("pager-info").textContent = rows.length
      ? `${(start + 1).toLocaleString()}–${Math.min(start + state.pageSize, rows.length).toLocaleString()} of ${rows.length.toLocaleString()}`
      : "";
    $("prev").disabled = state.page === 0;
    $("next").disabled = state.page >= pages - 1;

    document.querySelectorAll("th[data-sort]").forEach((th) => {
      const active = th.dataset.sort === state.sortKey;
      th.classList.toggle("sorted", active);
      th.setAttribute("aria-sort", active ? (state.sortDir > 0 ? "ascending" : "descending") : "none");
    });
    renderStats(rows);
  }

  function bind() {
    $("search").addEventListener("input", (e) => { state.query = e.target.value; state.page = 0; render(); });
    $("min-price").addEventListener("change", (e) => { state.minPrice = +e.target.value; state.page = 0; render(); });
    $("page-size").addEventListener("change", (e) => { state.pageSize = +e.target.value; state.page = 0; render(); });
    $("prev").addEventListener("click", () => { state.page--; render(); });
    $("next").addEventListener("click", () => { state.page++; render(); });

    // tapping an info icon shows its tooltip rather than re-sorting the column
    document.querySelectorAll(".tip").forEach((tip) => tip.addEventListener("click", (e) => e.stopPropagation()));
    document.querySelectorAll("th[data-sort]").forEach((th) => th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (state.sortKey === key) state.sortDir *= -1;
      else { state.sortKey = key; state.sortDir = key === "ticker" || key === "rank" ? 1 : -1; }
      render();
    }));

    const panel = $("weights"), btn = $("weights-btn");
    btn.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      btn.setAttribute("aria-expanded", String(!panel.hidden));
    });
    for (const k of WEIGHT_KEYS) {
      $(`w-${k}`).addEventListener("input", (e) => { state.weights[k] = +e.target.value; rescore(); render(); });
    }
    $("weights-reset").addEventListener("click", () => {
      for (const k of WEIGHT_KEYS) { state.weights[k] = state.defaults[k]; $(`w-${k}`).value = state.defaults[k]; }
      rescore(); render();
    });
  }

  async function init() {
    state.minPrice = +$("min-price").value;
    state.pageSize = +$("page-size").value;
    try {
      const res = await fetch("/api/stocks");
      if (!res.ok) throw new Error((await res.json()).error || res.statusText);
      const data = await res.json();
      state.stocks = data.stocks;
      for (const k of WEIGHT_KEYS) state.defaults[k] = state.weights[k] = Math.round(data.weights[k] * 100);
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="10" class="loading">Couldn't load stocks: ${escapeHtml(err.message)}</td></tr>`;
      return;
    }
    bind();
    rescore();
    render();
  }

  init();
})();
