(() => {
  const $ = (id) => document.getElementById(id);
  const btn = $("refresh-btn");
  const panel = $("refresh-status");
  const POLL_MS = 2000;
  // remembers (per tab) that this page started a refresh, so it reloads when the refresh finishes
  const flag = (value) => {
    try {
      if (value === undefined) return sessionStorage.getItem("refreshing");
      if (value) sessionStorage.setItem("refreshing", value); else sessionStorage.removeItem("refreshing");
    } catch (e) { return null; }
  };

  function show(s) {
    const running = s.state === "running";
    btn.disabled = running;
    btn.classList.toggle("spinning", running);
    btn.querySelector("span").textContent = running ? "Refreshing…" : "Refresh data";
    // a finished refresh is only worth showing once, right after this tab reloads with the new data
    const justFinished = flag() === "done";
    if (justFinished) flag(false);
    if (s.state === "idle" || (s.state === "done" && !justFinished)) { panel.hidden = true; return; }

    panel.hidden = false;
    panel.classList.toggle("error", s.state === "error");
    const step = s.step ? `Step ${s.step_index + 1} of ${s.steps}: ${s.step}` : "Starting…";
    $("refresh-step").textContent = s.state === "error" ? "Refresh failed" : s.state === "done" ? "Refresh complete" : step;
    $("refresh-count").textContent = running && s.progress != null ? Math.round(s.progress * 100) + "%" : "";
    $("refresh-msg").textContent = s.message || "";
    // indeterminate bar while a step has no measurable progress
    const bar = $("refresh-bar");
    bar.parentElement.classList.toggle("indeterminate", running && s.progress == null);
    bar.style.width = s.state === "done" ? "100%" : s.progress != null ? s.progress * 100 + "%" : "";
  }

  async function poll() {
    try {
      const s = await (await fetch("/api/refresh")).json();
      show(s);
      if (s.state === "running") setTimeout(poll, POLL_MS);
      else if (s.state === "done" && flag() === "running") {
        flag("done");
        location.reload();  // pick up the new data
      }
    } catch (err) {
      setTimeout(poll, POLL_MS * 2);
    }
  }

  btn.addEventListener("click", async () => {
    if (!confirm("Download fresh prices for every ticker, rebuild the rankings and retrain the price model? This takes around 15 minutes, longer if Yahoo rate-limits the download.")) return;
    flag("running");
    const res = await fetch("/api/refresh", { method: "POST" });
    show(await res.json());
    poll();
  });

  poll();  // pick up a refresh that's already running (e.g. after a page reload)
})();
