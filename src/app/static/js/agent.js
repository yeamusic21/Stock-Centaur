(() => {
  const { api, escapeHtml, errorText, fmtWhen, watchAgent, renderReport } = window.Centaur;
  const $ = (id) => document.getElementById(id);
  const SOURCES = { sandbox: "Sandbox", robinhood: "Robinhood", etrade: "E*Trade" };
  const params = new URLSearchParams(location.search);
  let currentReport = null;

  async function loadPortfolios() {
    const { portfolios } = await api("/api/portfolios");
    const select = $("agent-portfolio");
    if (!portfolios.length) {
      select.innerHTML = `<option value="">No saved portfolios</option>`;
      $("agent-go").disabled = true;
      return;
    }
    const wanted = params.get("portfolio");
    select.innerHTML = Object.entries(SOURCES).map(([src, label]) => {
      const group = portfolios.filter((p) => p.source === src);
      return group.length ? `<optgroup label="${label}">${group.map((p) =>
        `<option value="${p.id}"${String(p.id) === wanted ? " selected" : ""}>${escapeHtml(p.title)} (${p.count})</option>`).join("")}</optgroup>` : "";
    }).join("");
  }

  async function loadLlm() {
    const el = $("agent-llm");
    try {
      const s = (await api("/api/connections")).lmstudio;
      el.innerHTML = s.online && s.model
        ? `Model: <strong>${escapeHtml(s.model)}</strong> · <a href="/connections">change</a>`
        : `<span class="down">${escapeHtml(s.message)}</span> <a href="/connections">Set up LM Studio</a>`;
    } catch (err) { el.textContent = errorText(err); }
  }

  async function loadReports() {
    const { reports } = await api("/api/reports");
    const list = $("report-list");
    list.innerHTML = reports.length ? reports.map((r) => `
      <li><button type="button" class="report-item${r.id === currentReport ? " active" : ""}" data-report="${r.id}">
        <span class="pill pill-kind">${r.kind === "stock" ? "Stock" : "Portfolio"}</span>
        <strong>${escapeHtml(r.title)}</strong>
        <span class="muted">${escapeHtml(fmtWhen(r.created_at))}</span>
      </button></li>`).join("") : `<li class="muted">No reports yet.</li>`;
  }

  async function showReport(id) {
    currentReport = +id;
    const el = $("report");
    el.innerHTML = `<div class="chart-empty muted">Loading report…</div>`;
    try {
      el.innerHTML = renderReport(await api(`/api/reports/${id}`));
      history.replaceState(null, "", `?report=${id}`);
    } catch (err) {
      el.innerHTML = `<div class="chart-empty muted">${escapeHtml(errorText(err))}</div>`;
    }
    document.querySelectorAll(".report-item").forEach((b) => b.classList.toggle("active", +b.dataset.report === currentReport));
  }

  function watch() {
    watchAgent($("agent-job"), {
      onUpdate: (s) => { $("agent-go").disabled = s.state === "running"; },
      onDone: async (s) => { await loadReports(); if (s.report_id) showReport(s.report_id); },
    });
  }

  async function run(e) {
    e.preventDefault();
    const id = $("agent-portfolio").value;
    if (!id) return;
    $("agent-go").disabled = true;
    try {
      await api("/api/agent", { kind: "portfolio", target: +id, detailed: $("agent-detailed").checked });
    } catch (err) {
      if (err.status !== 409) { $("agent-go").disabled = false; alert(errorText(err)); return; }
    }
    watch();
  }

  function bind() {
    $("agent-form").addEventListener("submit", run);
    $("report-list").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-report]");
      if (btn) showReport(btn.dataset.report);
    });
    $("report").addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-delete-report]");
      if (!btn || !confirm("Delete this report?")) return;
      await api(`/api/reports/${btn.dataset.deleteReport}`, null, "DELETE");
      currentReport = null;
      history.replaceState(null, "", location.pathname);
      $("report").innerHTML = `<div class="chart-empty muted">Report deleted.</div>`;
      loadReports();
    });
  }

  async function init() {
    bind();
    await Promise.all([loadPortfolios(), loadLlm(), loadReports()]).catch((err) => alert(errorText(err)));
    if (params.get("report")) showReport(params.get("report"));
    watch();  // pick up a report that's already being written
  }

  init();
})();
