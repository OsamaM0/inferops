/* Shared connection, preferences, notifications and refresh lifecycle. */
(() => {
  "use strict";
  const defaults = {theme: "dark", accent: "mint", density: "comfortable", refresh: 5, sourceLayout: "grid", telemetry: true, landing: "overview"};
  const storage = {
    get(key, fallback = null) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Private browsing may disable storage. */ } },
  };
  const preferences = {...defaults, ...storage.get("inferops.console.preferences", {})};
  const allowed = {theme: ["dark", "light", "system"], accent: ["mint", "blue", "violet"], density: ["comfortable", "compact"], sourceLayout: ["grid", "list"], landing: ["overview", "sources", "playground", "api"]};
  for (const [key, values] of Object.entries(allowed)) if (!values.includes(preferences[key])) preferences[key] = defaults[key];
  if (![0, 2, 5, 10, 30].includes(Number(preferences.refresh))) preferences.refresh = defaults.refresh;
  preferences.telemetry = preferences.telemetry !== false;
  const media = matchMedia("(prefers-color-scheme: light)");
  function applyPreferences() {
    document.documentElement.dataset.theme = preferences.theme === "system" ? (media.matches ? "light" : "dark") : preferences.theme;
    document.documentElement.dataset.accent = preferences.accent;
    document.documentElement.dataset.density = preferences.density;
    document.documentElement.dataset.sourceLayout = preferences.sourceLayout;
    document.documentElement.dataset.telemetry = String(preferences.telemetry);
  }
  applyPreferences();
  media.addEventListener("change", applyPreferences);
  const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  function url(path) {
    const result = new URL(path, location.origin);
    if (result.origin !== location.origin || !path.startsWith("/") || path.startsWith("//")) throw new Error("Use an API path on this server, starting with /.");
    return result.href;
  }
  function detail(body, status) {
    if (Array.isArray(body?.detail)) return body.detail.map(item => `${item.loc?.slice(1).join(".") || "Request"}: ${item.msg}`).join("; ");
    return typeof body?.detail === "string" ? body.detail : `Request failed (${status})`;
  }
  async function raw(path, options = {}) {
    const {timeoutMs = 12000, ...init} = options;
    const controller = new AbortController();
    const abort = () => controller.abort(options.signal?.reason);
    if (options.signal?.aborted) abort();
    else options.signal?.addEventListener("abort", abort, {once: true});
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url(path), {...init, signal: controller.signal});
      // Read inside the timeout, including streaming response bodies.
      const contentType = response.headers.get("content-type") || "";
      const isBinary = /^(image|video|audio)\//.test(contentType) || contentType.includes("application/octet-stream");
      const blob = isBinary ? await response.blob() : null;
      const text = blob ? `${contentType} response · ${blob.size.toLocaleString()} bytes. Use Download response to save the file.` : await response.text();
      let body = text;
      try { body = text ? JSON.parse(text) : null; } catch { /* Metrics and logs are text. */ }
      return {response, body, text, blob};
    } catch (error) {
      if (controller.signal.aborted) throw new Error(options.signal?.aborted ? "Request cancelled." : `Request timed out after ${Math.round(timeoutMs / 1000)}s.`);
      throw error;
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", abort);
    }
  }
  async function request(path, options = {}) {
    const {response, body} = await raw(path, options);
    if (!response.ok) throw new Error(detail(body, response.status));
    return body;
  }
  let notificationTimer;
  function notify(message, error = false) {
    const target = document.getElementById("notice");
    if (!target) return;
    clearTimeout(notificationTimer);
    const content = document.createElement("span"), dismiss = document.createElement("button");
    content.textContent = message;
    dismiss.textContent = "×";
    dismiss.type = "button";
    dismiss.className = "notice-dismiss";
    dismiss.setAttribute("aria-label", "Dismiss notification");
    dismiss.addEventListener("click", () => target.classList.add("hidden"));
    target.replaceChildren(content, dismiss);
    target.className = `notice${error ? " error" : ""}`;
    target.setAttribute("role", error ? "alert" : "status");
    if (!error) notificationTimer = setTimeout(() => target.classList.add("hidden"), 8000);
  }
  const refreshers = new Set();
  let lastRefresh = 0;
  setInterval(() => {
    const interval = Number(preferences.refresh) * 1000;
    if (!interval || document.hidden || Date.now() - lastRefresh < interval) return;
    lastRefresh = Date.now();
    for (const refresh of refreshers) Promise.resolve().then(refresh).catch(error => console.warn("Refresh failed", error));
  }, 1000);
  window.InferOps = {
    request, raw, url, escape, storage, preferences, defaults, notify,
    poll(fn) { refreshers.add(fn); return () => refreshers.delete(fn); },
    savePreferences(next) {
      Object.assign(preferences, next);
      storage.set("inferops.console.preferences", preferences);
      applyPreferences();
      document.dispatchEvent(new CustomEvent("inferops:preferences"));
    },
  };
})();
