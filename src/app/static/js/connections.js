(() => {
  const $ = (id) => document.getElementById(id);
  const POLL_MS = 2000;
  const errorText = (err) => err instanceof TypeError ? "Couldn't reach the server. Is the app still running?" : err.message;
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function api(url, body, method = body ? "POST" : "GET") {
    const res = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await res.json();
    if (!res.ok && res.status !== 409) throw new Error(data.error || data.message || res.statusText);
    return data;
  }

  function setBadge(el, text, kind = "") {
    el.textContent = text;
    el.className = "badge" + (kind ? " " + kind : "");
  }

  function setMsg(el, text, kind = "") {
    el.textContent = text || "";
    el.classList.toggle("down", kind === "error");
    el.classList.toggle("up", kind === "ok");
  }

  // ---------- LM Studio ----------
  function showLm(s, { fill = false } = {}) {
    if (fill) {
      $("lm-url").value = s.lmstudio_url;
      $("lm-command").value = s.lmstudio_command;
    }
    const select = $("lm-model");
    select.innerHTML = s.models.length
      ? s.models.map((m) => `<option value="${escapeHtml(m)}"${m === s.model ? " selected" : ""}>${escapeHtml(m)}</option>`).join("")
      : `<option value="">No models found</option>`;
    setBadge($("lm-badge"), s.online ? "Connected" : "Offline", s.online ? "ok" : "off");
    setMsg($("lm-msg"), s.message, s.online && s.models.length ? "ok" : "error");
    $("lm-launch").hidden = s.online;
  }

  async function saveLm(e) {
    e?.preventDefault();
    setMsg($("lm-msg"), "Checking…");
    try {
      showLm(await api("/api/connections/lmstudio", { url: $("lm-url").value, model: $("lm-model").value, command: $("lm-command").value }));
    } catch (err) { setMsg($("lm-msg"), errorText(err), "error"); }
  }

  async function launchLm() {
    await saveLm();
    try {
      await api("/api/connections/lmstudio/launch", {});
      setMsg($("lm-msg"), "Starting LM Studio… start its local server, then press Test.");
      // check back a few times while it boots
      let tries = 0;
      const timer = setInterval(async () => {
        const s = (await api("/api/connections")).lmstudio;
        if (s.online || ++tries > 15) { clearInterval(timer); showLm(s); }
      }, 3000);
    } catch (err) { setMsg($("lm-msg"), errorText(err), "error"); }
  }

  // ---------- brokers ----------
  const fmtWhen = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  function showLast(id, last) {
    $(id).innerHTML = last ? `Last pulled ${escapeHtml(fmtWhen(last.updated_at))} · <a href="/sandbox?portfolio=${last.id}">Open in sandbox</a>` : "";
  }

  function showRobinhood(s) {
    const busy = s.state === "running" || s.state === "needs_code";
    $("rh-go").disabled = busy;
    $("rh-code-row").hidden = s.state !== "needs_code";
    const badge = { running: ["Signing in…", "busy"], needs_code: ["Needs code", "busy"], done: ["Pulled", "ok"], error: ["Failed", "off"] }[s.state];
    setBadge($("rh-badge"), badge ? badge[0] : s.last ? "Saved" : "Not pulled", badge ? badge[1] : "");
    setMsg($("rh-msg"), s.state === "idle" ? "" : s.message, s.state === "error" ? "error" : s.state === "done" ? "ok" : "");
    showLast("rh-last", s.last);
    if (s.state === "done" || s.state === "error") $("rh-pass").value = $("rh-mfa").value = "";
    if (s.state === "needs_code" && document.activeElement !== $("rh-code")) $("rh-code").focus();
    return busy;
  }

  function showEtrade(s) {
    const badge = { running: ["Pulling…", "busy"], awaiting_code: ["Waiting for code", "busy"], done: ["Pulled", "ok"], error: ["Failed", "off"] }[s.state];
    setBadge($("et-badge"), badge ? badge[0] : s.last ? "Saved" : "Not pulled", badge ? badge[1] : "");
    showLast("et-last", s.last);
  }

  let rhTimer = null;
  async function pollRobinhood() {
    clearTimeout(rhTimer);
    try {
      const s = (await api("/api/connections")).brokers.robinhood;
      if (showRobinhood(s)) rhTimer = setTimeout(pollRobinhood, POLL_MS);
    } catch (err) { rhTimer = setTimeout(pollRobinhood, POLL_MS * 2); }
  }

  async function pullRobinhood(e) {
    e.preventDefault();
    try {
      const s = await api("/api/connections/robinhood", { username: $("rh-user").value, password: $("rh-pass").value, mfa_code: $("rh-mfa").value });
      showRobinhood({ ...s, last: null });
      pollRobinhood();
    } catch (err) { setMsg($("rh-msg"), errorText(err), "error"); }
  }

  async function sendRobinhoodCode() {
    try {
      await api("/api/connections/robinhood/code", { code: $("rh-code").value });
      $("rh-code").value = "";
      pollRobinhood();
    } catch (err) { setMsg($("rh-msg"), errorText(err), "error"); }
  }

  async function etradeLink(e) {
    e.preventDefault();
    $("et-go").disabled = true;
    setMsg($("et-msg"), "Asking E*Trade for a sign-in link…");
    try {
      const { url } = await api("/api/connections/etrade/authorize", { consumer_key: $("et-key").value, consumer_secret: $("et-secret").value, sandbox: $("et-sandbox").checked });
      $("et-link").href = url;
      $("et-verify-row").hidden = false;
      setMsg($("et-msg"), "");
      showEtrade({ state: "awaiting_code" });
      window.open(url, "_blank", "noopener");
      $("et-verifier").focus();
    } catch (err) { setMsg($("et-msg"), errorText(err), "error"); }
    finally { $("et-go").disabled = false; }
  }

  async function etradeVerify() {
    const btn = $("et-verify-btn");
    btn.disabled = true;
    setMsg($("et-msg"), "Downloading positions…");
    showEtrade({ state: "running" });
    try {
      const s = await api("/api/connections/etrade/verify", { verifier: $("et-verifier").value });
      setMsg($("et-msg"), s.message, "ok");
      showEtrade(s);
      $("et-verify-row").hidden = true;
      $("et-secret").value = $("et-verifier").value = "";
    } catch (err) {
      setMsg($("et-msg"), errorText(err) + " Get a new sign-in link to try again.", "error");
      showEtrade({ state: "error" });
      $("et-verify-row").hidden = true;
    } finally { btn.disabled = false; }
  }

  async function init() {
    $("lm-form").addEventListener("submit", saveLm);
    $("lm-test").addEventListener("click", saveLm);
    $("lm-launch").addEventListener("click", launchLm);
    $("rh-form").addEventListener("submit", pullRobinhood);
    $("rh-code-btn").addEventListener("click", sendRobinhoodCode);
    $("rh-code").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); sendRobinhoodCode(); } });
    $("et-form").addEventListener("submit", etradeLink);
    $("et-verify-btn").addEventListener("click", etradeVerify);
    $("et-verifier").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); etradeVerify(); } });
    try {
      const data = await api("/api/connections");
      showLm(data.lmstudio, { fill: true });
      if (showRobinhood(data.brokers.robinhood)) pollRobinhood();
      showEtrade(data.brokers.etrade);
    } catch (err) {
      setMsg($("lm-msg"), errorText(err), "error");
    }
  }

  init();
})();
