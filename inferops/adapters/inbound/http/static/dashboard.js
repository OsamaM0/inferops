(() => {
    "use strict";

    const STORAGE_KEYS = {
      apiBase: "inferops.workspace.apiBase",
      domainId: "inferops.dashboard.domainId",
    };

    const state = {
      apiBase: "",
      connected: false,
      configsIndex: null,
      models: [],
      streams: [],
      alerts: [],
      streamCapabilities: {},
      events: [],
      system: null,
      interfaces: [],
      selectedDomainId: "",
      selectedMode: "camera",
      selectedConfig: null,
      configDirty: false,
      activeStreamId: "",
      cameraMediaStream: null,
      cameraLoopEnabled: false,
      cameraLoopBusy: false,
      cameraLoopTimer: null,
      videoLoopEnabled: false,
      videoLoopRvfcHandle: 0,
      liveSocket: null,
      liveSocketSeq: 0,
      liveSocketLatency: 0,
      liveSocketFps: 0,
      liveSocketLastFrameAt: 0,
      liveSocketFrameCount: 0,
      liveSocketRvfcHandle: 0,
      alertSocket: null,
      previewUrls: {
        image: "",
        video: "",
      },
      previewAnnotations: [],
      previewAnnotationTarget: "",
      previewAnnotationModel: "",
      lastResult: null,
      activityTimer: null,
    };

    const $ = (id) => document.getElementById(id);
    let configForm = null, configLoadId = 0;

    const DEFAULT_REQUEST_TIMEOUT_MS = 12000;
    const LONG_REQUEST_TIMEOUT_MS = 90000;

    function inferDefaultApiBase() {
      return window.location.origin;
    }

    function normalizeBase(input) {
      let value = (input || "").trim();
      if (!value) {
        return "";
      }
      if (!/^https?:\/\//i.test(value)) {
        value = `http://${value}`;
      }
      return value.replace(/\/+$/, "");
    }

    function apiUrl(path) {
      return `${state.apiBase}${path}`;
    }

    function requestTimeoutMs(path, options = {}) {
      if (Number.isFinite(options.timeoutMs) && options.timeoutMs > 0) {
        return options.timeoutMs;
      }
      if (path.includes("/v1/infer/") || path.includes("/v1/run/")) {
        return LONG_REQUEST_TIMEOUT_MS;
      }
      return DEFAULT_REQUEST_TIMEOUT_MS;
    }

    function buildWsUrl(channel) {
      const url = new URL(state.apiBase);
      url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
      url.pathname = `/v1/ws/${encodeURIComponent(channel)}`;
      url.search = "";
      return url.toString();
    }

    // Every InferOps socket fans out binary orjson frames, but a socket without
    // `binaryType = "arraybuffer"` yields a Blob instead. Both live and alert
    // sockets decode through here so the two paths cannot drift apart again.
    async function decodeSocketPayload(data) {
      if (data === null || data === undefined) return null;
      if (typeof data === "string") return JSON.parse(data);
      if (data instanceof ArrayBuffer) {
        return JSON.parse(new TextDecoder().decode(new Uint8Array(data)));
      }
      if (typeof Blob !== "undefined" && data instanceof Blob) {
        return JSON.parse(await data.text());
      }
      if (ArrayBuffer.isView(data)) {
        return JSON.parse(new TextDecoder().decode(data));
      }
      throw new TypeError(`unsupported websocket payload type: ${typeof data}`);
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function prettyName(value) {
      return String(value || "-")
        .replace(/[_-]+/g, " ")
        .replace(/\b\w/g, (match) => match.toUpperCase());
    }

    function formatTime(ms) {
      if (!ms) {
        return "-";
      }
      const value = Number(ms);
      if (!Number.isFinite(value)) {
        return "-";
      }
      return new Date(value).toLocaleString();
    }

    function formatCount(value) {
      if (value === null || value === undefined || value === "") {
        return "-";
      }
      return Number(value).toLocaleString();
    }

    function formatDetail(detail) {
      if (Array.isArray(detail)) {
        return detail.map((item) => {
          const path = Array.isArray(item.loc) ? item.loc.join(".") : "config";
          return `${path}: ${item.msg}`;
        }).join(" | ");
      }
      if (typeof detail === "object" && detail) {
        return JSON.stringify(detail);
      }
      return String(detail || "Request failed");
    }

    function setPill(id, tone, text) {
      const element = $(id);
      element.className = `status-pill ${tone}`;
      element.textContent = text;
    }

    function setNotice(text, tone = "status-info") {
      setPill("apiConnectionBadge", tone, tone === "status-ok" ? "Connected" : tone === "status-err" ? "Error" : tone === "status-warn" ? "Attention" : "Working");
      $("dashboardNotice").textContent = text;
      if (!text.startsWith("Connected to") && !text.startsWith("Refreshing dashboard")) window.InferOps.notify(text, tone === "status-err");
    }

    async function request(path, options = {}) {
      return window.InferOps.request(path, {...options, timeoutMs: requestTimeoutMs(path, options)});
    }

    function selectedDomain() {
      if (!state.configsIndex || !Array.isArray(state.configsIndex.domains)) {
        return null;
      }
      return state.configsIndex.domains.find((domain) => domain.id === state.selectedDomainId) || state.configsIndex.domains[0] || null;
    }

    function ensureSelectedDomain() {
      const domains = state.configsIndex && Array.isArray(state.configsIndex.domains) ? state.configsIndex.domains : [];
      if (!domains.length) {
        state.selectedDomainId = "";
        return null;
      }

      const saved = window.InferOps.storage.get(STORAGE_KEYS.domainId);
      const candidate = state.selectedDomainId || saved;
      const found = domains.find((domain) => domain.id === candidate) || domains[0];
      state.selectedDomainId = found.id;
      window.InferOps.storage.set(STORAGE_KEYS.domainId, found.id);
      return found;
    }

    function setSelectedMode(mode) {
      state.selectedMode = mode;
      for (const button of document.querySelectorAll("#modeTabs .mode-tab")) {
        button.classList.toggle("is-active", button.dataset.mode === mode);
        button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
      }
      for (const panel of document.querySelectorAll(".mode-panel")) {
        panel.classList.toggle("is-active", panel.id === `mode-${mode}`);
      }
      setPill("modeBadge", "status-info", prettyName(mode));
    }

    function resetPreviewUrls() {
      for (const key of Object.keys(state.previewUrls)) {
        if (state.previewUrls[key]) {
          URL.revokeObjectURL(state.previewUrls[key]);
          state.previewUrls[key] = "";
        }
      }
    }

    function clearPreviewAnnotations(options = {}) {
      const preserveState = Boolean(options.preserveState);
      const overlay = $("previewOverlay");
      overlay.innerHTML = "";
      overlay.classList.add("hidden");
      if (!preserveState) {
        state.previewAnnotations = [];
        state.previewAnnotationTarget = "";
        state.previewAnnotationModel = "";
      }
    }

    function getPreviewMediaElement(kind) {
      if (kind === "camera") {
        return $("cameraPreview");
      }
      if (kind === "image") {
        return $("imagePreview");
      }
      if (kind === "video") {
        return $("videoPreview");
      }
      return null;
    }

    function getMediaIntrinsicSize(element) {
      if (!element) {
        return null;
      }
      if (element.tagName === "VIDEO") {
        if (!element.videoWidth || !element.videoHeight) {
          return null;
        }
        return { width: element.videoWidth, height: element.videoHeight };
      }
      if (element.tagName === "IMG") {
        if (!element.naturalWidth || !element.naturalHeight) {
          return null;
        }
        return { width: element.naturalWidth, height: element.naturalHeight };
      }
      return null;
    }

    function clampUnit(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) {
        return 0;
      }
      return Math.max(0, Math.min(1, number));
    }

    // Generic per-frame signals: a flat { name: number | string } object.
    function normalizeSignals(signals) {
      if (!signals || typeof signals !== "object" || Array.isArray(signals)) {
        return {};
      }
      const out = {};
      for (const [key, value] of Object.entries(signals)) {
        if (typeof value === "number" || typeof value === "string") {
          out[key] = value;
        }
      }
      return out;
    }

    // Generic overlay boxes: [{ label, box: [x1,y1,x2,y2] (0..1), tone: ok|warn|danger }].
    function normalizeOverlayBoxes(boxes) {
      if (!Array.isArray(boxes)) {
        return [];
      }
      const out = [];
      for (const entry of boxes) {
        if (!entry || !Array.isArray(entry.box) || entry.box.length !== 4) {
          continue;
        }
        const tone = entry.tone === "warn" || entry.tone === "danger" ? entry.tone : "ok";
        out.push({
          label: entry.label ? String(entry.label) : "",
          box: entry.box.map((value) => clampUnit(value)),
          tone,
        });
      }
      return out;
    }

    function formatSignalLabel(key) {
      return String(key)
        .split("_")
        .filter(Boolean)
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
    }

    function formatSignalValue(value) {
      if (typeof value === "number") {
        if (!Number.isFinite(value)) {
          return String(value);
        }
        return String(Number(value.toFixed(3)));
      }
      return String(value);
    }

    function signalTone(key, value) {
      const text = typeof value === "string" ? value.toLowerCase() : "";
      if (/_alert$/.test(String(key)) || /drowsy|danger|critical/.test(text)) {
        return "danger";
      }
      if (/warn|looking/.test(text)) {
        return "warn";
      }
      return "";
    }

    function renderPreviewAnnotations() {
      const overlay = $("previewOverlay");
      const targetKind = state.previewAnnotationTarget;
      const media = getPreviewMediaElement(targetKind);
      const overlayBoxes = state.lastResult && Array.isArray(state.lastResult.overlayBoxes)
        ? state.lastResult.overlayBoxes
        : [];
      const hasBoxes = state.previewAnnotations.length > 0;
      const hasOverlayBoxes = overlayBoxes.length > 0;

      if (!media || media.classList.contains("hidden") || (!hasBoxes && !hasOverlayBoxes)) {
        clearPreviewAnnotations({ preserveState: true });
        return;
      }

      const intrinsic = getMediaIntrinsicSize(media);
      if (!intrinsic) {
        clearPreviewAnnotations({ preserveState: true });
        return;
      }

      overlay.classList.remove("hidden");
      const containerWidth = overlay.clientWidth;
      const containerHeight = overlay.clientHeight;
      if (!containerWidth || !containerHeight) {
        clearPreviewAnnotations({ preserveState: true });
        return;
      }

      const mediaRatio = intrinsic.width / intrinsic.height;
      const containerRatio = containerWidth / containerHeight;
      let renderWidth = containerWidth;
      let renderHeight = containerHeight;
      let offsetX = 0;
      let offsetY = 0;

      if (mediaRatio > containerRatio) {
        renderHeight = containerWidth / mediaRatio;
        offsetY = (containerHeight - renderHeight) / 2;
      } else {
        renderWidth = containerHeight * mediaRatio;
        offsetX = (containerWidth - renderWidth) / 2;
      }

      const boxHtml = state.previewAnnotations.map((detection) => {
        const [x1, y1, x2, y2] = detection.box;
        const left = offsetX + (x1 * renderWidth);
        const top = offsetY + (y1 * renderHeight);
        const width = Math.max(2, (x2 - x1) * renderWidth);
        const height = Math.max(2, (y2 - y1) * renderHeight);
        const label = `${detection.label} ${(detection.score * 100).toFixed(1)}%`;
        return `
          <div class="preview-box" style="left:${left}px;top:${top}px;width:${width}px;height:${height}px;">
            <span class="preview-box-label">${escapeHtml(label)}</span>
          </div>
        `;
      }).join("");

      const overlayBoxHtml = overlayBoxes.map((entry) => {
        const [x1, y1, x2, y2] = entry.box;
        const left = offsetX + (x1 * renderWidth);
        const top = offsetY + (y1 * renderHeight);
        const width = Math.max(2, (x2 - x1) * renderWidth);
        const height = Math.max(2, (y2 - y1) * renderHeight);
        const toneClass = entry.tone === "danger" ? " is-danger" : entry.tone === "warn" ? " is-warn" : "";
        const labelHtml = entry.label
          ? `<span class="preview-box-label${toneClass}">${escapeHtml(entry.label)}</span>`
          : "";
        return `
          <div class="preview-box is-overlay${toneClass}" style="left:${left}px;top:${top}px;width:${width}px;height:${height}px;">
            ${labelHtml}
          </div>
        `;
      }).join("");

      overlay.innerHTML = boxHtml + overlayBoxHtml;
    }

    function extractPreviewDetections(outputs) {
      const detections = [];
      const modelsUsed = [];
      const seenModels = new Set();

      for (const item of outputs || []) {
        for (const result of item.results || []) {
          if (result.model && !seenModels.has(result.model)) {
            seenModels.add(result.model);
            modelsUsed.push(result.model);
          }
          for (const detection of result.detections || []) {
            if (!Array.isArray(detection.box) || detection.box.length !== 4) {
              continue;
            }
            const box = detection.box.map((value) => {
              const number = Number(value);
              if (!Number.isFinite(number)) {
                return 0;
              }
              return Math.max(0, Math.min(1, number));
            });
            detections.push({
              label: detection.label || "object",
              score: Number.isFinite(Number(detection.score)) ? Number(detection.score) : 0,
              box,
            });
          }
        }
      }

      return { detections, modelsUsed };
    }

    function setPreview(kind, detail) {
      const camera = $("cameraPreview");
      const video = $("videoPreview");
      const image = $("imagePreview");
      const placeholder = $("previewPlaceholder");
      const meta = $("previewMeta");

      camera.classList.add("hidden");
      video.classList.add("hidden");
      image.classList.add("hidden");
      placeholder.classList.add("hidden");
      meta.innerHTML = "";

      if (kind === "camera") {
        camera.classList.remove("hidden");
      } else if (kind === "video") {
        video.classList.remove("hidden");
      } else if (kind === "image") {
        image.classList.remove("hidden");
      } else {
        placeholder.classList.remove("hidden");
        placeholder.innerHTML = `<strong>${escapeHtml(detail && detail.title ? detail.title : "Choose a source.")}</strong><span>${escapeHtml(detail && detail.body ? detail.body : "Start the camera, select an image, upload a video, or launch an RTSP stream.")}</span>`;
      }

      const bits = [];
      if (detail && detail.source) {
        bits.push(`<span class="chip">Source <span class="mono">${escapeHtml(detail.source)}</span></span>`);
      }
      if (detail && detail.pipeline) {
        bits.push(`<span class="chip">Pipeline <span class="mono">${escapeHtml(detail.pipeline)}</span></span>`);
      }
      if (detail && detail.streamId) {
        bits.push(`<span class="chip">Stream <span class="mono">${escapeHtml(detail.streamId)}</span></span>`);
      }
      if ((kind === "camera" || kind === "image" || kind === "video") && state.previewAnnotationTarget === kind && state.previewAnnotations.length) {
        bits.push(`<span class="chip">Boxes <span class="mono">${escapeHtml(String(state.previewAnnotations.length))}</span></span>`);
      }
      if ((kind === "camera" || kind === "image" || kind === "video") && state.previewAnnotationTarget === kind && state.previewAnnotationModel) {
        bits.push(`<span class="chip">Model <span class="mono">${escapeHtml(state.previewAnnotationModel)}</span></span>`);
      }
      meta.innerHTML = bits.join("");
      renderPreviewAnnotations();
    }

    function summarizeInferOutputs(outputs) {
      const detectionsByClass = {};
      const alerts = [];
      const fusionEvents = [];
      const preview = extractPreviewDetections(outputs);

      for (const item of outputs || []) {
        for (const result of item.results || []) {
          for (const detection of result.detections || []) {
            detectionsByClass[detection.label] = (detectionsByClass[detection.label] || 0) + 1;
          }
        }
        if (Array.isArray(item.payload && item.payload.alerts)) {
          alerts.push(...item.payload.alerts);
        }
        if (Array.isArray(item.payload && item.payload.fusion_events)) {
          fusionEvents.push(...item.payload.fusion_events);
        }
      }

      // Take signals / overlay boxes from the first item whose payload carries them,
      // falling back to the last item.
      let signalsSource = null;
      let overlaySource = null;
      let lastPayload = null;
      for (const item of outputs || []) {
        const payload = item && item.payload;
        if (!payload) {
          continue;
        }
        lastPayload = payload;
        if (!signalsSource && payload.signals && typeof payload.signals === "object") {
          signalsSource = payload.signals;
        }
        if (!overlaySource && Array.isArray(payload.overlay_boxes)) {
          overlaySource = payload.overlay_boxes;
        }
      }
      if (!signalsSource && lastPayload) {
        signalsSource = lastPayload.signals;
      }
      if (!overlaySource && lastPayload) {
        overlaySource = lastPayload.overlay_boxes;
      }

      return {
        pipeline: selectedDomain() ? selectedDomain().pipeline : "-",
        frames: Array.isArray(outputs) ? outputs.length : 0,
        detectionsByClass,
        alerts,
        fusionEvents,
        previewDetections: preview.detections,
        modelsUsed: preview.modelsUsed,
        signals: normalizeSignals(signalsSource),
        overlayBoxes: normalizeOverlayBoxes(overlaySource),
        raw: outputs,
      };
    }

    function normalizeResult(payload) {
      if (Array.isArray(payload)) {
        return summarizeInferOutputs(payload);
      }
      const detectionsByClass = payload && typeof payload === "object" && payload.detections_by_class ? payload.detections_by_class : {};
      return {
        pipeline: payload && payload.pipeline ? payload.pipeline : selectedDomain() ? selectedDomain().pipeline : "-",
        frames: payload && payload.frames ? payload.frames : 1,
        detectionsByClass,
        alerts: Array.isArray(payload && payload.alerts) ? payload.alerts : [],
        fusionEvents: Array.isArray(payload && payload.fusion_events) ? payload.fusion_events : [],
        previewDetections: [],
        modelsUsed: Array.isArray(payload && payload.models_used) ? payload.models_used : [],
        signals: normalizeSignals(payload && payload.signals),
        overlayBoxes: normalizeOverlayBoxes(payload && payload.overlay_boxes),
        raw: payload,
      };
    }

    function renderInterfaces() {
      const items = state.interfaces.filter((item) => ["minio", "grafana", "prometheus", "swagger"].includes(item.id));
      if (!items.length) return;
      const serviceKeys = {minio: "minio_console", grafana: "grafana", prometheus: "prometheus", swagger: "api"};
      const icons = {minio: "M", grafana: "G", prometheus: "P", swagger: "{ }"};
      $("interfaceGrid").innerHTML = items.map((item) => {
        const health = state.system?.services?.[serviceKeys[item.id]];
        const online = health && Boolean(health.ok || health.open || health.status === 200);
        const disabled = health?.enabled === false;
        const tone = disabled || !health ? "status-info" : online ? "status-ok" : "status-warn";
        const label = disabled ? "Optional / off" : !health ? "Not checked" : online ? "Online" : "Unavailable";
        return `<a class="interface-card" href="${item.id === "swagger" ? "#api" : escapeHtml(item.url)}" ${item.id === "swagger" ? "" : 'target="_blank" rel="noopener noreferrer"'}>
          <div class="interface-top"><span class="app-icon ${item.id}-icon">${icons[item.id]}</span>
          <span class="status-pill ${tone}">${label}</span></div>
          <strong>${item.id === "swagger" ? "API explorer →" : `${escapeHtml(item.title)} ↗`}</strong><span>${escapeHtml(item.description)}</span></a>`;
      }).join("");
    }

    function renderServiceGrid() {
      const container = $("serviceGrid");
      const system = state.system;
      if (!system || !system.services) {
        container.innerHTML = '<div class="empty-state">System status is not available yet.</div>';
        setPill("serviceBadge", "status-warn", "No data");
        return;
      }

      const services = [
        ["API", system.services.api],
        ["Triton HTTP", system.services.triton_http],
        ["Triton metrics", system.services.triton_metrics],
        ["MinIO S3", system.services.minio_s3 || system.services.s3],
        ["MinIO console", system.services.minio_console],
        ["Grafana", system.services.grafana],
        ["Prometheus", system.services.prometheus],
        ["MediaMTX", system.services.mediamtx],
      ].filter(([, value]) => value && value.enabled !== false);

      let okCount = 0;
      container.innerHTML = services.map(([label, value]) => {
        const isOk = Boolean(value && (value.ok || value.open || value.status === 200));
        if (isOk) {
          okCount += 1;
        }
        const tone = isOk ? "status-ok" : "status-err";
        const meta = value && value.port ? `Port ${value.port}` : "No port";
        const suffix = value && value.status ? `HTTP ${value.status}` : value && value.open ? "Open" : "Unavailable";
        const interfaceId = {"MinIO console": "minio", "Grafana": "grafana", "Prometheus": "prometheus"}[label];
        const name = interfaceId
          ? `<a href="${escapeHtml(apiUrl(`/v1/interfaces/${interfaceId}`))}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)} ↗</a>`
          : escapeHtml(label);
        return `
          <div class="service-item">
            <div>
              <div class="service-name">${name}</div>
              <div class="service-meta">${escapeHtml(meta)}</div>
            </div>
            <span class="status-pill ${tone}">${escapeHtml(suffix)}</span>
          </div>
        `;
      }).join("");

      const dbEvents = system.database && system.database.events !== null ? formatCount(system.database.events) : "n/a";
      const dbAlerts = system.database && system.database.alerts !== null ? formatCount(system.database.alerts) : "n/a";
      const jsonRecords = system.evidence ? formatCount(system.evidence.json_records) : "-";
      const clipCount = system.evidence ? formatCount(system.evidence.clips) : "-";

      container.insertAdjacentHTML("beforeend", `
        <div class="service-item">
          <div>
            <div class="service-name">Database</div>
            <div class="service-meta">Events ${escapeHtml(dbEvents)} / alerts ${escapeHtml(dbAlerts)}</div>
          </div>
          <span class="status-pill status-info">Counts</span>
        </div>
        <div class="service-item">
          <div>
            <div class="service-name">Evidence</div>
            <div class="service-meta">JSON ${escapeHtml(jsonRecords)} / clips ${escapeHtml(clipCount)}</div>
          </div>
          <span class="status-pill status-info">Storage</span>
        </div>
      `);

      setPill("serviceBadge", okCount === services.length ? "status-ok" : okCount > 0 ? "status-warn" : "status-err", `${okCount}/${services.length} online`);
    }

    function renderDomainSummary() {
      const select = $("domainSelect");
      const domains = state.configsIndex && Array.isArray(state.configsIndex.domains) ? state.configsIndex.domains : [];
      select.innerHTML = "";

      if (!domains.length) {
        select.innerHTML = '<option value="">No domains available</option>';
        $("domainPipeline").textContent = "-";
        $("domainRefs").textContent = "-";
        $("domainFlow").innerHTML = '<span class="flow-pill">No domain metadata</span>';
        $("domainConfigLinks").innerHTML = '<div class="empty-state">Add a pipeline to connect models and policies.</div>';
        $("configPipelineDescription").textContent = "";
        return;
      }

      const domain = ensureSelectedDomain();
      for (const item of domains) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.label} (${item.pipeline})`;
        option.selected = item.id === domain.id;
        select.appendChild(option);
      }

      $("domainPipeline").textContent = domain.pipeline;
      const refs = domain.config_names || {};
      $("configPipelineDescription").textContent = domain.description || "";
      $("domainRefs").textContent = [refs.model, refs.fusion, refs.alerts].filter(Boolean).join(" / ") || "Linked config missing";

      const flowBits = [];
      if (domain.source_stage) {
        flowBits.push(`<span class="flow-pill">source: ${escapeHtml(domain.source_stage)}</span>`);
      }
      for (const stage of domain.stages || []) {
        flowBits.push(`<span class="flow-pill">${escapeHtml(stage.name)}: ${escapeHtml(stage.stage)}</span>`);
      }
      $("domainFlow").innerHTML = flowBits.join("") || '<span class="flow-pill">No stage metadata</span>';

      const quickLinks = domain.config_links || [];

      $("domainConfigLinks").innerHTML = quickLinks.map((item) => {
        return `<button class="link-btn" data-config-section="${escapeHtml(item.section)}" data-config-name="${escapeHtml(item.name)}">${escapeHtml(item.title)}</button>`;
      }).join("");
    }

    function renderRuntimeSummary() {
      const alerts = Array.isArray(state.alerts) ? state.alerts : [];
      const events = Array.isArray(state.events) ? state.events : [];
      const activeStreams = (state.streams || []).filter((stream) => !stream.done);

      const loaded = (state.models || []).filter((model) => (model.loaded_variants || []).length > 0).length;
      $("summaryModels").textContent = formatCount(loaded);
      const logicalCount = (state.configsIndex?.files || []).filter(file => file.section === "models").length;
      $("summaryModelsLabel").textContent = `Loaded implementations (${formatCount(logicalCount)} models / ${formatCount(state.models.length)} runtime variants)`;
      $("summaryModelStates").textContent = state.models.length
        ? state.models.map((model) => `${model.name}: ${(model.loaded_variants || []).length ? "loaded" : "configured; loads when used"}`).join(" · ")
        : "This project has no models configured.";
      $("summaryStreams").textContent = formatCount(activeStreams.length);
      $("summaryActivity").textContent = `${formatCount(alerts.length)} alerts / ${formatCount(events.length)} events`;
      $("summaryEvidence").textContent = state.system && state.system.evidence ? formatCount(state.system.evidence.clips) : "-";
      setPill("runtimeBadge", activeStreams.length ? "status-ok" : "status-info", activeStreams.length ? `${activeStreams.length} live` : "No live streams");
    }

    function renderStreams() {
      const body = $("streamsBody");
      const rows = Array.isArray(state.streams) ? state.streams : [];

      if (!rows.length) {
        body.innerHTML = '<tr><td colspan="6"><div class="empty-state">No registered streams right now.</div></td></tr>';
        return;
      }

      body.innerHTML = rows.map((stream) => {
        const status = stream.done ? "Stopped" : "Running";
        const tone = stream.done ? "status-warn" : "status-ok";
        return `
          <tr>
            <td class="mono">${escapeHtml(stream.stream_id || "-")}</td>
            <td class="mono">${escapeHtml(stream.pipeline || "-")}</td>
            <td>${escapeHtml(stream.input || "-")}</td>
            <td>${escapeHtml(String(stream.fps || "-"))}</td>
            <td><span class="status-pill ${tone}">${escapeHtml(status)}</span></td>
            <td>
              <button class="ghost-btn is-danger" data-stop-stream="${escapeHtml(stream.stream_id || "")}">Stop</button>
            </td>
          </tr>
        `;
      }).join("");
    }

    function renderAlertsAndEvents() {
      const alertsList = $("alertsList");
      const eventsList = $("eventsList");

      const alerts = (Array.isArray(state.alerts) ? state.alerts : []).slice(0, 12);
      const domain = selectedDomain();
      const events = (Array.isArray(state.events) ? state.events : [])
        .filter((row) => !domain || !row.pipeline || row.pipeline === domain.pipeline)
        .slice(0, 12);

      if (!alerts.length) {
        alertsList.innerHTML = '<div class="empty-state">No recent alerts in storage yet.</div>';
      } else {
        alertsList.innerHTML = alerts.map((row) => `
          <article class="activity-item">
            <div class="activity-title">${escapeHtml(prettyName(row.type || "alert"))}</div>
            <div class="activity-meta">Severity ${escapeHtml(row.severity || "-")} | Entity ${escapeHtml(row.entity_id || "-")}</div>
            <div class="activity-meta">${escapeHtml(formatTime(row.ts_ms))}</div>
            <div class="activity-meta">Evidence ${escapeHtml(row.payload && row.payload.evidence_state || "Legacy")} ${row.payload && row.payload.evidence_id ? `<a href="${escapeHtml(apiUrl(`/v1/evidence/records/${row.payload.evidence_id}`))}" target="_blank" rel="noreferrer">Details / failure reason</a>` : ""}</div>
            <div class="activity-meta">${(row.payload && row.payload.evidence_uris || []).filter((uri) => /\.(mp4|jpg)$/.test(uri)).map((uri) => {
              const path = uri.replace(/^s3:\/\//, "").split("/").map(encodeURIComponent).join("/");
              return `<a href="${escapeHtml(apiUrl(`/v1/evidence/file/${path}`))}" target="_blank" rel="noreferrer">${uri.endsWith(".mp4") ? "Play clip" : "Snapshot"}</a>`;
            }).join(" | ")}</div>
          </article>
        `).join("");
      }

      if (!events.length) {
        eventsList.innerHTML = '<div class="empty-state">No recent events in storage for the active domain.</div>';
      } else {
        eventsList.innerHTML = events.map((row) => `
          <article class="activity-item">
            <div class="activity-title">${escapeHtml(prettyName(row.type || "event"))}</div>
            <div class="activity-meta">Pipeline ${escapeHtml(row.pipeline || "-")} | Entity ${escapeHtml(row.entity_id || "-")}</div>
            <div class="activity-meta">${escapeHtml(formatTime(row.ts_ms))}</div>
            <div class="activity-meta">Evidence ${escapeHtml(row.payload && row.payload.evidence_state || "Legacy")} ${row.payload && row.payload.evidence_id ? `<a href="${escapeHtml(apiUrl(`/v1/evidence/records/${row.payload.evidence_id}`))}" target="_blank" rel="noreferrer">Details / failure reason</a>` : ""}</div>
          </article>
        `).join("");
      }
    }

    function renderConfigList() {
      const container = $("configList");
      const index = state.configsIndex;
      if (!index || !Array.isArray(index.files) || !index.files.length) {
        container.innerHTML = '<div class="empty-state">No settings yet. Add your first model or pipeline.</div>';
        return;
      }
      $("configPrecedence").textContent = `Settings precedence: ${(index.precedence || []).join(" → ")}. Environment overrides apply at startup.`;

      const search = $("configSearch").value.toLowerCase();
      const order = Object.keys(index.groups || {});
      const files = [...index.files].filter(file => `${file.title} ${file.section} ${file.description}`.toLowerCase().includes(search)).sort((left, right) => {
        if (left.section === right.section) {
          return left.name.localeCompare(right.name);
        }
        return order.indexOf(left.section) - order.indexOf(right.section);
      });
      let previousSection = "";
      container.innerHTML = files.map((file) => {
        const isActive = state.selectedConfig && state.selectedConfig.section === file.section && state.selectedConfig.name === file.name;
        const heading = previousSection === file.section ? "" : `<h5 class="config-section">${escapeHtml(file.group)}</h5>`;
        previousSection = file.section;
        return `
          ${heading}<button type="button" class="config-entry ${isActive ? "is-active" : ""}" data-config-section="${escapeHtml(file.section)}" data-config-name="${escapeHtml(file.name)}" aria-pressed="${Boolean(isActive)}">
            <strong>${escapeHtml(file.title)}</strong>
            <small>${escapeHtml(file.description || (file.variants.length ? `Runtime variants: ${file.variants.join(", ")}` : file.group))}</small>
            <small>${file.reload_behavior === "live" ? "Reloads live" : file.reload_behavior === "conditional" ? "Artifact changes need preparation and restart" : "Restart to apply"}</small>
          </button>
        `;
      }).join("");
    }

    function renderResult() {
      const result = state.lastResult;
      if (!result) {
        $("resultFrames").textContent = "-";
        $("resultAlerts").textContent = "-";
        $("resultEvents").textContent = "-";
        $("resultDetections").textContent = "-";
        $("resultAlertList").innerHTML = '<div class="empty-state">Run the pipeline to populate alert output.</div>';
        $("resultDetectionList").innerHTML = '<div class="empty-state">Detection counts appear here after inference.</div>';
        $("signalsSection").style.display = "none";
        $("resultSignalsList").innerHTML = '<div class="empty-state">No signals in last run.</div>';
        $("resultJson").textContent = "Waiting for the first pipeline result.";
        setPill("resultBadge", "status-info", "No run yet");
        clearPreviewAnnotations();
        return;
      }

      const detections = Object.entries(result.detectionsByClass || {});
      const alerts = Array.isArray(result.alerts) ? result.alerts : [];
      const fusionEvents = Array.isArray(result.fusionEvents) ? result.fusionEvents : [];

      $("resultFrames").textContent = formatCount(result.frames || 0);
      $("resultAlerts").textContent = formatCount(alerts.length);
      $("resultEvents").textContent = formatCount(fusionEvents.length);
      $("resultDetections").textContent = formatCount(detections.length);

      $("resultAlertList").innerHTML = alerts.length
        ? alerts.slice(0, 8).map((alert) => `<span class="chip">${escapeHtml(prettyName(alert.type || "alert"))} <span class="mono">${escapeHtml(alert.severity || "-")}</span></span>`).join("")
        : '<div class="empty-state">No alerts were emitted in the last run.</div>';

      $("resultDetectionList").innerHTML = detections.length
        ? detections.map(([label, count]) => `<span class="chip">${escapeHtml(label)} <span class="mono">${escapeHtml(String(count))}</span></span>`).join("")
        : '<div class="empty-state">No detections were reported in the last run.</div>';

      // Generic signals panel: one chip per signal key.
      const signalEntries = Object.entries(result.signals || {});
      const signalsSection = $("signalsSection");
      if (signalEntries.length) {
        signalsSection.style.display = "";
        $("resultSignalsList").innerHTML = signalEntries.map(([key, value]) => {
          const tone = signalTone(key, value);
          const toneClass = tone === "danger" ? " chip-danger" : tone === "warn" ? " chip-warn" : "";
          return `<span class="chip${toneClass}">${escapeHtml(formatSignalLabel(key))} <span class="mono">${escapeHtml(formatSignalValue(value))}</span></span>`;
        }).join("");
      } else {
        signalsSection.style.display = "none";
        $("resultSignalsList").innerHTML = '<div class="empty-state">No signals in last run.</div>';
      }

      $("resultJson").textContent = JSON.stringify(result.raw, null, 2);
      setPill("resultBadge", alerts.length || fusionEvents.length || detections.length ? "status-ok" : "status-warn", result.pipeline || "Pipeline result");
      renderPreviewAnnotations();
    }

    async function refreshSnapshot(options = {}) {
      const keepNotice = Boolean(options.keepNotice);
      if (!state.apiBase) {
        return false;
      }

      $("docsLink").href = "#api";
      $("evidenceLink").href = apiUrl("/v1/interfaces/minio");
      $("guideLink").href = "#guide";
      $("healthLink").href = "#api";

      if (!keepNotice) {
        setNotice("Refreshing dashboard snapshot...", "status-info");
      }

      try {
        const root = await request("/");
        const [system, configs, models, streams, alerts, events, interfaces] = await Promise.allSettled([
          request("/v1/system/status"),
          request("/v1/configs"),
          request("/v1/models?view=logical"),
          request("/v1/streams"),
          request("/v1/alerts?limit=12"),
          request("/v1/events?limit=12"),
          request("/v1/interfaces"),
        ]);

        if (configs.status !== "fulfilled") {
          throw configs.reason;
        }
        if (models.status !== "fulfilled") {
          throw models.reason;
        }
        if (streams.status !== "fulfilled") {
          throw streams.reason;
        }
        if (alerts.status !== "fulfilled") {
          throw alerts.reason;
        }
        if (events.status !== "fulfilled") {
          throw events.reason;
        }

        state.system = system.status === "fulfilled" ? system.value : null;
        state.interfaces = interfaces.status === "fulfilled" ? interfaces.value.interfaces : [];
        renderInterfaces();
        const runtime = state.system && state.system.runtime;
        if (runtime) {
          const models = (runtime.models || []).map((m) => `${m.model}: ${m.device} (${(m.active_providers || []).join(", ") || m.backend})`).join("; ");
          $("executionModeInfo").textContent = `Server mode: ${(runtime.execution_mode || "auto").toUpperCase()}. ${models || "No models loaded yet"}.`;
        }
        state.configsIndex = configs.value;
        state.models = Array.isArray(models.value) ? models.value : [];
        state.streams = Array.isArray(streams.value) ? streams.value : [];
        state.alerts = Array.isArray(alerts.value) ? alerts.value : Array.isArray(alerts.value && alerts.value.alerts) ? alerts.value.alerts : [];
        state.events = Array.isArray(events.value) ? events.value : Array.isArray(events.value && events.value.events) ? events.value.events : [];

        ensureSelectedDomain();
        renderServiceGrid();
        renderDomainSummary();
        renderRuntimeSummary();
        renderStreams();
        renderAlertsAndEvents();
        renderConfigList();
        renderResult();

        state.connected = true;
        setNotice(
          system.status === "fulfilled"
            ? `Connected to ${root.name || "InferOps"} ${root.version || ""}.`
            : `Connected to ${root.name || "InferOps"} ${root.version || ""}, but system status is unavailable.`,
          system.status === "fulfilled" ? "status-ok" : "status-warn",
        );

        if (!state.selectedConfig) {
          const domain = selectedDomain();
          if (domain && domain.config_names && domain.config_names.pipeline) await loadConfig("pipelines", domain.config_names.pipeline, { quiet: true });
          else if (state.configsIndex.files.length) await loadConfig(state.configsIndex.files[0].section, state.configsIndex.files[0].name, {quiet:true});
        }
        return true;
      } catch (error) {
        state.connected = false;
        if (state.activityTimer) {
          state.activityTimer();
          state.activityTimer = null;
        }
        setNotice(`Connection failed: ${error.message}`, "status-err");
        return false;
      }
    }

    async function refreshActivityOnly() {
      if (!state.apiBase || !state.connected) {
        return;
      }
      try {
        const [alerts, events, streams, system] = await Promise.all([
          request("/v1/alerts?limit=12"),
          request("/v1/events?limit=12"),
          request("/v1/streams"),
          request("/v1/system/status"),
        ]);
        state.alerts = Array.isArray(alerts) ? alerts : Array.isArray(alerts && alerts.alerts) ? alerts.alerts : [];
        state.events = Array.isArray(events) ? events : Array.isArray(events && events.events) ? events.events : [];
        state.streams = Array.isArray(streams) ? streams : [];
        state.system = system;
        renderServiceGrid();
        renderInterfaces();
        renderRuntimeSummary();
        renderStreams();
        renderAlertsAndEvents();
      } catch (error) {
        state.connected = false;
        if (state.activityTimer) {
          state.activityTimer();
          state.activityTimer = null;
        }
        console.warn("activity refresh failed", error);
      }
    }

    function connectAlertSocket(streamId) {
      if (state.alertSocket) {
        state.alertSocket.close();
        state.alertSocket = null;
      }
      if (!streamId) {
        return;
      }

      try {
        const socket = new WebSocket(buildWsUrl(`alerts.${streamId}`));
        // The server fans out with WebSocket.send_bytes (see sink_ws.py), so
        // frames arrive as binary. Without this the browser hands us a Blob and
        // JSON.parse sees the string "[object Blob]" — every alert was dropped.
        socket.binaryType = "arraybuffer";
        state.alertSocket = socket;
        socket.onmessage = async (event) => {
          try {
            const payload = await decodeSocketPayload(event.data);
            if (payload === null) return;
            state.alerts = [payload, ...(state.alerts || [])].slice(0, 12);
            renderAlertsAndEvents();
            renderRuntimeSummary();
          } catch (error) {
            console.warn("alert socket payload parse failed", error);
          }
        };
      } catch (error) {
        console.warn("alert socket setup failed", error);
      }
    }

    async function connectDashboard() {
      const base = normalizeBase($("apiBaseInput").value);
      if (!base) {
        setNotice("Provide a valid API base before connecting.", "status-warn");
        return;
      }
      state.apiBase = base;

      const connected = await refreshSnapshot();

      if (connected && !state.activityTimer) {
        state.activityTimer = window.InferOps.poll(() => ["activity-view", "playground", "services"].includes(window.InferOps.view) && refreshActivityOnly());
      }
    }

    async function loadConfig(section, name, options = {}) {
      const quiet = Boolean(options.quiet);
      if (!state.apiBase) {
        return;
      }
      if (state.configDirty && !confirm("Discard unsaved changes and open these settings?")) return;
      const loadId = ++configLoadId;
      try {
        const payload = await request(`/v1/configs/${encodeURIComponent(section)}/${encodeURIComponent(name)}`);
        if (loadId !== configLoadId) return;
        openConfig(payload);
        $("configStatus").textContent = payload.reload_behavior === "live"
          ? "Live sections are validated and reloaded after save when the checkbox is enabled."
          : "Project and platform changes require a restart. New model sources require preparation before restart.";
        if (!quiet) {
          setNotice(`Loaded config ${payload.section}/${payload.name}.`, "status-ok");
        }
      } catch (error) {
        console.error(error);
        setNotice(`Could not load config: ${error.message}`, "status-err");
      }
    }

    function openConfig(payload) {
      state.selectedConfig = payload;
      state.configDirty = false;
      $("configTitle").textContent = payload.title || payload.name;
      $("configMeta").textContent = payload.reload_behavior === "live" ? "Applies to new sessions. Running sessions keep their current settings."
        : payload.reload_behavior === "conditional" ? "Shared model settings and runtime overrides. Artifact changes need preparation and restart." : "Restart required to apply these settings.";
      const overrides = ["platform", "project"].includes(payload.section) ? (state.configsIndex?.environment_overrides || []) : [];
      $("configDescription").textContent = (payload.description || "") + (overrides.length ? `\nStartup environment overrides: ${overrides.join(", ")}. Values are not copied into this document.` : "");
      $("configMode").value = "form";
      $("configMode").disabled = Boolean(payload.isNew);
      $("configEditor").value = payload.content || "";
      $("configEditor").hidden = true; $("configForm").hidden = false;
      configForm = new window.InferOpsConfigForm($("configForm"), payload, state.configsIndex, () => {
        state.configDirty = true;
        $("configStatus").textContent = "Unsaved changes. Validate to check the project before saving.";
      });
      setPill("configBadge", payload.isNew ? "status-info" : payload.reload_behavior === "live" ? "status-ok" : "status-warn", payload.isNew ? "New draft" : payload.group || payload.section);
      renderConfigList();
    }

    async function saveCurrentConfig(validateOnly = false) {
      validateOnly = validateOnly === true;
      if (!state.selectedConfig) {
        setNotice("Choose settings before saving.", "status-warn");
        return;
      }
      if ($("configSaveReloadBtn").disabled) return;
      $("configSaveReloadBtn").disabled = true;
      try {
        const payload = await request(
          `/v1/configs/${encodeURIComponent(state.selectedConfig.section)}/${encodeURIComponent(state.selectedConfig.name)}`,
          {
            method: state.selectedConfig.isNew ? "POST" : "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...($("configMode").value === "form" ? {data:configForm.read()} : {content:$("configEditor").value}),
              reload: $("configReloadToggle").checked,
              version: state.selectedConfig.version,
              validate_only: validateOnly,
            }),
          }
        );
        if (validateOnly) {
          $("configStatus").textContent = `Configuration is valid. Nothing saved.${payload.restart_required ? " These changes require a restart." : " Ready to apply to new sessions."}`;
          return;
        }
        openConfig(payload);
        $("configStatus").textContent = payload.restart_required
          ? payload.preparation_required ? "Saved. Run project preparation and restart to apply model artifact changes." : "Saved. Restart to apply startup settings."
          : payload.reload
            ? `Saved and reloaded ${payload.reload.section}.`
            : "Saved without reload.";
        if (payload.enabled === false) $("configStatus").textContent += " Enable this pipeline in Project settings before using it.";
        await refreshSnapshot({ keepNotice: true });
        setNotice(`Saved ${payload.title || payload.name}.`, payload.restart_required ? "status-warn" : "status-ok");
      } catch (error) {
        console.error(error);
        $("configStatus").textContent = `${validateOnly ? "Validation" : "Save"} failed: ${error.message}`;
        setNotice($("configStatus").textContent, "status-err");
      } finally {
        $("configSaveReloadBtn").disabled = false;
      }
    }

    async function reloadSelectedSection() {
      if (!state.selectedConfig) {
        setNotice("Select a config file first.", "status-warn");
        return;
      }
      try {
        const result = await request(`/v1/configs/reload?section=${encodeURIComponent(state.selectedConfig.section)}`, { method: "POST" });
        await refreshSnapshot({ keepNotice: true });
        setNotice(result.restart_required ? "Saved startup changes require a restart before settings can be applied." : `Applied saved configuration to new sessions (${result.reloaded} pipelines).`, result.restart_required ? "status-warn" : "status-ok");
      } catch (error) {
        console.error(error);
        setNotice(`Reload failed: ${error.message}`, "status-err");
      }
    }

    async function reloadAllLiveConfigs() {
      try {
        const result = await request("/v1/configs/reload?section=all", { method: "POST" });
        await refreshSnapshot({ keepNotice: true });
        setNotice(result.restart_required ? "Saved startup changes require a restart before settings can be applied." : `Applied saved configuration to new sessions (${result.reloaded} pipelines).`, result.restart_required ? "status-warn" : "status-ok");
      } catch (error) {
        console.error(error);
        setNotice(`Global reload failed: ${error.message}`, "status-err");
      }
    }

    function currentPipeline() {
      const domain = selectedDomain();
      if (!domain) {
        throw new Error("Select a domain first.");
      }
      return domain.pipeline;
    }

    async function startCamera() {
      if (state.cameraMediaStream) {
        setPreview("camera", { source: "camera", pipeline: currentPipeline(), streamId: $("cameraStreamIdInput").value.trim() });
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      state.cameraMediaStream = stream;
      const video = $("cameraPreview");
      video.srcObject = stream;
      await video.play();
      state.activeStreamId = $("cameraStreamIdInput").value.trim() || "dashboard-camera";
      connectAlertSocket(state.activeStreamId);
      setPreview("camera", { source: "camera", pipeline: currentPipeline(), streamId: state.activeStreamId });
      setNotice("Camera preview started.", "status-ok");
    }

    function stopCameraLoop() {
      state.cameraLoopEnabled = false;
      if (state.cameraLoopTimer) {
        window.clearTimeout(state.cameraLoopTimer);
        state.cameraLoopTimer = null;
      }
      const video = $("cameraPreview");
      if (state.liveSocketRvfcHandle && video && typeof video.cancelVideoFrameCallback === "function") {
        try { video.cancelVideoFrameCallback(state.liveSocketRvfcHandle); } catch (_) { /* ignore */ }
      }
      state.liveSocketRvfcHandle = 0;
      if (state.liveSocket) {
        try { state.liveSocket.close(); } catch (_) { /* ignore */ }
        state.liveSocket = null;
      }
      $("cameraLoopBtn").textContent = "Start analysis loop";
    }

    function stopCamera() {
      stopCameraLoop();
      if (state.cameraMediaStream) {
        for (const track of state.cameraMediaStream.getTracks()) {
          track.stop();
        }
      }
      state.cameraMediaStream = null;
      $("cameraPreview").srcObject = null;
      setPreview("empty", { title: "Camera stopped.", body: "Start the camera again or switch to another source mode." });
      setNotice("Camera stopped.", "status-info");
    }

    function captureCameraBlob() {
      return new Promise((resolve, reject) => {
        const video = $("cameraPreview");
        if (!video.videoWidth || !video.videoHeight) {
          reject(new Error("Camera frame is not ready yet."));
          return;
        }
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error("Could not capture frame from camera."));
            return;
          }
          resolve(blob);
        }, "image/jpeg", 0.92);
      });
    }

    async function runInferBlob(blob, streamId, filename) {
      const form = new FormData();
      form.append("image", blob, filename);
      form.append("stream_id", streamId);
      connectAlertSocket(streamId);
      state.activeStreamId = streamId;
      const payload = await request(`/v1/infer/${encodeURIComponent(currentPipeline())}`, {
        method: "POST",
        body: form,
      });
      state.lastResult = normalizeResult(payload);
      state.previewAnnotationTarget = state.selectedMode === "camera" ? "camera" : "image";
      state.previewAnnotations = Array.isArray(state.lastResult.previewDetections) ? state.lastResult.previewDetections : [];
      state.previewAnnotationModel = Array.isArray(state.lastResult.modelsUsed) && state.lastResult.modelsUsed.length ? state.lastResult.modelsUsed[0] : "";
      renderResult();
      await refreshActivityOnly();
      return state.lastResult;
    }

    async function analyzeCameraFrame(options = {}) {
      const quiet = Boolean(options.quiet);
      try {
        await startCamera();
        const streamId = $("cameraStreamIdInput").value.trim() || "dashboard-camera";
        const blob = await captureCameraBlob();
        await runInferBlob(blob, streamId, "camera-frame.jpg");
        setPreview("camera", { source: "camera", pipeline: currentPipeline(), streamId });
        if (!quiet) {
          setNotice(`Processed one camera frame through ${currentPipeline()}.`, "status-ok");
        }
      } catch (error) {
        console.error(error);
        stopCameraLoop();
        setNotice(`Camera inference failed: ${error.message}`, "status-err");
      }
    }

    function stopVideoLoop() {
      state.videoLoopEnabled = false;
      const video = $("videoPreview");
      if (state.videoLoopRvfcHandle && video && typeof video.cancelVideoFrameCallback === "function") {
        try { video.cancelVideoFrameCallback(state.videoLoopRvfcHandle); } catch (_) { /* ignore */ }
      }
      state.videoLoopRvfcHandle = 0;
      if (!state.cameraLoopEnabled && state.liveSocket) {
        try { state.liveSocket.close(); } catch (_) { /* ignore */ }
        state.liveSocket = null;
      }
      const btn = $("videoRunBtn");
      btn.disabled = false;
      btn.textContent = "Stream and analyze video";
    }

    function buildLiveWsUrl(pipeline, streamId) {
      const url = new URL(state.apiBase);
      url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
      url.pathname = `/v1/live/${encodeURIComponent(pipeline)}`;
      url.search = `?stream_id=${encodeURIComponent(streamId)}`;
      return url.toString();
    }

    function applyLivePayload(msg, pipeline, annotTarget) {
      const dets = Array.isArray(msg.detections) ? msg.detections : [];
      state.previewAnnotations = dets
        .map((d) => {
          const box = Array.isArray(d.box) && d.box.length === 4
            ? d.box.map((v) => {
                const n = Number(v);
                return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0;
              })
            : null;
          if (!box) return null;
          return {
            label: d.label || "object",
            score: Number.isFinite(Number(d.score)) ? Number(d.score) : 0,
            box,
            track_id: d.track_id,
          };
        })
        .filter(Boolean);
      state.previewAnnotationTarget = annotTarget || "camera";
      state.previewAnnotationModel = msg.model || (Array.isArray(msg.results) && msg.results[0] && msg.results[0].model) || "";
      const detectionsByClass = {};
      for (const d of state.previewAnnotations) {
        detectionsByClass[d.label] = (detectionsByClass[d.label] || 0) + 1;
      }
      // Generic per-frame state forwarded by any domain stage.
      state.lastResult = {
        pipeline,
        frames: 1,
        detectionsByClass,
        alerts: Array.isArray(msg.alerts) ? msg.alerts : [],
        fusionEvents: Array.isArray(msg.fusion_events) ? msg.fusion_events : [],
        previewDetections: state.previewAnnotations,
        modelsUsed: state.previewAnnotationModel ? [state.previewAnnotationModel] : [],
        signals: normalizeSignals(msg.signals),
        overlayBoxes: normalizeOverlayBoxes(msg.overlay_boxes),
        raw: msg,
      };
      state.liveSocketLatency = Number(msg.latency_ms) || 0;
      const nowMs = performance.now();
      if (state.liveSocketLastFrameAt) {
        const dt = nowMs - state.liveSocketLastFrameAt;
        if (dt > 0) {
          // EMA smoothing
          const inst = 1000 / dt;
          state.liveSocketFps = state.liveSocketFps
            ? state.liveSocketFps * 0.8 + inst * 0.2
            : inst;
        }
      }
      state.liveSocketLastFrameAt = nowMs;
      renderResult();
    }

    async function toggleCameraLoop() {
      if (state.cameraLoopEnabled) {
        stopCameraLoop();
        setNotice("Live analysis stopped.", "status-info");
        return;
      }

      let pipeline;
      try {
        pipeline = currentPipeline();
      } catch (error) {
        setNotice(error.message, "status-err");
        return;
      }
      try {
        await startCamera();
      } catch (error) {
        setNotice(`Camera start failed: ${error.message}`, "status-err");
        return;
      }

      const streamId = $("cameraStreamIdInput").value.trim() || "dashboard-camera";
      const targetFps = Math.max(1, Math.min(30, Number($("cameraIntervalInput").value) || 15));
      const minPeriodMs = 1000 / targetFps;
      const video = $("cameraPreview");

      state.cameraLoopEnabled = true;
      state.liveSocketSeq = 0;
      state.liveSocketFps = 0;
      state.liveSocketLastFrameAt = 0;
      $("cameraLoopBtn").textContent = "Stop analysis loop";

      const ws = new WebSocket(buildLiveWsUrl(pipeline, streamId));
      ws.binaryType = "arraybuffer";
      state.liveSocket = ws;

      ws.addEventListener("open", () => {
        setNotice(`Live socket open → ${pipeline} @ ${targetFps} fps target.`, "status-ok");
      });
      ws.addEventListener("close", () => {
        if (state.cameraLoopEnabled) {
          setNotice("Live socket closed.", "status-warn");
          stopCameraLoop();
        }
      });
      ws.addEventListener("error", () => {
        setNotice("Live socket error.", "status-err");
      });
      ws.addEventListener("message", async (event) => {
        let msg;
        try {
          msg = await decodeSocketPayload(event.data);
          if (msg === null) return;
        } catch (err) {
          console.error("live ws parse failed", err);
          return;
        }
        if (msg.error) {
          setNotice(`Live infer error: ${msg.error}`, "status-err");
          return;
        }
        applyLivePayload(msg, pipeline, "camera");
      });

      // Reusable canvas + JPEG capture; drop oldest, single in-flight sender.
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      let lastSentAt = 0;
      let sending = false;

      const captureAndSend = async () => {
        if (!state.cameraLoopEnabled || ws.readyState !== WebSocket.OPEN) return;
        if (sending) return;
        const now = performance.now();
        if (now - lastSentAt < minPeriodMs) return;
        if (!video.videoWidth || !video.videoHeight) return;
        sending = true;
        try {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.6));
          if (!blob || !state.cameraLoopEnabled || ws.readyState !== WebSocket.OPEN) return;
          const buffer = await blob.arrayBuffer();
          // optional control packet with client timestamp
          try {
            ws.send(JSON.stringify({ client_ts_ms: Date.now(), media_time_ns: Math.round(performance.now() * 1000000), seq: ++state.liveSocketSeq }));
          } catch (_) { /* ignore */ }
          ws.send(buffer);
          lastSentAt = now;
        } catch (err) {
          console.error("live capture failed", err);
        } finally {
          sending = false;
        }
      };

      if (typeof video.requestVideoFrameCallback === "function") {
        const onFrame = () => {
          if (!state.cameraLoopEnabled) return;
          captureAndSend();
          state.liveSocketRvfcHandle = video.requestVideoFrameCallback(onFrame);
        };
        state.liveSocketRvfcHandle = video.requestVideoFrameCallback(onFrame);
      } else {
        const tick = () => {
          if (!state.cameraLoopEnabled) return;
          captureAndSend();
          state.cameraLoopTimer = window.setTimeout(tick, minPeriodMs);
        };
        tick();
      }
    }

    function previewImageFile(file) {
      resetPreviewUrls();
      clearPreviewAnnotations();
      state.previewUrls.image = URL.createObjectURL(file);
      const image = $("imagePreview");
      image.src = state.previewUrls.image;
      setPreview("image", { source: "image", pipeline: selectedDomain() ? selectedDomain().pipeline : "-", streamId: $("imageStreamIdInput").value.trim() || "dashboard-image" });
    }

    function previewVideoFile(file) {
      resetPreviewUrls();
      clearPreviewAnnotations();
      state.previewUrls.video = URL.createObjectURL(file);
      const video = $("videoPreview");
      video.src = state.previewUrls.video;
      setPreview("video", { source: "video", pipeline: selectedDomain() ? selectedDomain().pipeline : "-", streamId: $("videoStreamIdInput").value.trim() || "dashboard-video" });
    }

    async function runImage() {
      const file = $("imageFileInput").files && $("imageFileInput").files[0];
      if (!file) {
        setNotice("Choose an image file first.", "status-warn");
        return;
      }
      try {
        previewImageFile(file);
        const streamId = $("imageStreamIdInput").value.trim() || "dashboard-image";
        await runInferBlob(file, streamId, file.name || "image.jpg");
        setPreview("image", { source: "image", pipeline: currentPipeline(), streamId });
        setNotice(`Processed image through ${currentPipeline()}.`, "status-ok");
      } catch (error) {
        console.error(error);
        setNotice(`Image inference failed: ${error.message}`, "status-err");
      }
    }

    async function uploadVideo() {
      const file = $("videoFileInput").files && $("videoFileInput").files[0];
      if (!file) { setNotice("Choose a video file first.", "status-warn"); return; }
      const button = $("videoUploadBtn");
      button.disabled = true;
      try {
        stopVideoLoop();
        const pipeline = currentPipeline();
        const streamId = $("videoStreamIdInput").value.trim() || "dashboard-upload";
        connectAlertSocket(streamId);
        const form = new FormData();
        form.append("video", file, file.name);
        form.append("stream_id", streamId);
        setNotice("Analyzing every frame and saving evidence to MinIO...", "status-info");
        const result = await request(`/v1/run/${encodeURIComponent(pipeline)}/upload`, {
          method: "POST", body: form, timeoutMs: 600000,
        });
        state.lastResult = normalizeResult(result);
        renderResult();
        await refreshSnapshot({ keepNotice: true });
        const evidence = result.evidence_uris || [];
        setNotice(`Completed ${result.frames} frames on ${result.execution_mode}: ${result.alerts.length} alerts, ${evidence.length} saved evidence objects.`, "status-ok");
      } catch (error) {
        setNotice(`Video analysis failed: ${error.message}`, "status-err");
      } finally { button.disabled = false; }
    }

    async function runVideo() {
      if (state.videoLoopEnabled) {
        stopVideoLoop();
        setNotice("Video analysis stopped.", "status-info");
        return;
      }

      const file = $("videoFileInput").files && $("videoFileInput").files[0];
      if (!file) {
        setNotice("Choose a video file first.", "status-warn");
        return;
      }

      let pipeline;
      try {
        pipeline = currentPipeline();
      } catch (error) {
        setNotice(error.message, "status-err");
        return;
      }

      const streamId = $("videoStreamIdInput").value.trim() || "dashboard-video";
      previewVideoFile(file);
      const video = $("videoPreview");

      try { await video.play(); } catch (_) { /* requires user gesture */ }

      const btn = $("videoRunBtn");
      btn.textContent = "Stop analysis";
      btn.disabled = false;

      state.videoLoopEnabled = true;
      state.liveSocketSeq = 0;
      state.liveSocketFps = 0;
      state.liveSocketLastFrameAt = 0;

      const ws = new WebSocket(buildLiveWsUrl(pipeline, streamId));
      ws.binaryType = "arraybuffer";
      state.liveSocket = ws;

      ws.addEventListener("open", () => {
        setNotice(`Video stream socket open → ${pipeline}.`, "status-ok");
      });
      ws.addEventListener("close", () => {
        if (state.videoLoopEnabled) stopVideoLoop();
      });
      ws.addEventListener("error", () => {
        setNotice("Video WS error.", "status-err");
        stopVideoLoop();
      });
      ws.addEventListener("message", async (event) => {
        let msg;
        try {
          msg = await decodeSocketPayload(event.data);
          if (msg === null) return;
        } catch (err) {
          console.error("video ws parse failed", err);
          return;
        }
        if (msg && !msg.error) applyLivePayload(msg, pipeline, "video");
      });

      const canvas = document.createElement("canvas");
      const ctx2d = canvas.getContext("2d");
      let sending = false;

      const captureVideoFrame = async () => {
        if (!state.videoLoopEnabled || ws.readyState !== WebSocket.OPEN) return;
        if (sending || !video.videoWidth || !video.videoHeight || video.paused || video.ended) return;
        sending = true;
        try {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          ctx2d.drawImage(video, 0, 0, canvas.width, canvas.height);
          const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.65));
          if (!blob || !state.videoLoopEnabled || ws.readyState !== WebSocket.OPEN) return;
          try { ws.send(JSON.stringify({ client_ts_ms: Date.now(), media_time_ns: Math.round(video.currentTime * 1000000000), seq: ++state.liveSocketSeq })); } catch (_) {}
          ws.send(await blob.arrayBuffer());
        } catch (err) {
          console.error("video capture failed", err);
        } finally {
          sending = false;
        }
      };

      if (typeof video.requestVideoFrameCallback === "function") {
        const onVfc = () => {
          if (!state.videoLoopEnabled) return;
          captureVideoFrame();
          state.videoLoopRvfcHandle = video.requestVideoFrameCallback(onVfc);
        };
        state.videoLoopRvfcHandle = video.requestVideoFrameCallback(onVfc);
      } else {
        const tick = () => {
          if (!state.videoLoopEnabled) return;
          captureVideoFrame();
          window.setTimeout(tick, 67);
        };
        tick();
      }

      video.addEventListener("ended", () => { if (state.videoLoopEnabled) stopVideoLoop(); }, { once: true });
      setNotice(`Streaming ${file.name} through ${pipeline} — annotations update in real time.`, "status-info");
      setPreview("video", { source: "video", pipeline, streamId });
      connectAlertSocket(streamId);
      state.activeStreamId = streamId;
    }

    async function startRtspStream() {
      const url = $("rtspUrlInput").value.trim();
      if (!url) {
        setNotice("Enter an RTSP URL before starting a stream.", "status-warn");
        return;
      }

      try {
        const streamId = $("rtspStreamIdInput").value.trim() || `rtsp-${Date.now()}`;
        const fps = String(Math.max(1, Number($("rtspFpsInput").value) || 5));
        const params = new URLSearchParams({
          pipeline: currentPipeline(),
          rtsp_url: url,
          stream_id: streamId,
          fps,
        });
        const payload = await request(`/v1/streams?${params.toString()}`, { method: "POST" });
        state.activeStreamId = payload.stream_id;
        if (payload.capability) state.streamCapabilities[payload.stream_id] = payload.capability;
        connectAlertSocket(payload.stream_id);
        setPreview("empty", {
          title: "RTSP stream started.",
          body: `Backend pipeline ${payload.pipeline} is now consuming ${payload.input}. Browser playback is separate from the server-side stream run.`,
          source: "rtsp",
          pipeline: payload.pipeline,
          streamId: payload.stream_id,
        });
        await refreshActivityOnly();
        setNotice(`Started live RTSP stream ${payload.stream_id}.`, "status-ok");
      } catch (error) {
        console.error(error);
        setNotice(`RTSP stream start failed: ${error.message}`, "status-err");
      }
    }

    async function stopStream(streamId) {
      if (!streamId) {
        setNotice("There is no active stream to stop.", "status-warn");
        return;
      }
      try {
        await request(`/v1/streams/${encodeURIComponent(streamId)}`, { method: "DELETE",
          headers: state.streamCapabilities[streamId] ? {"X-Session-Capability": state.streamCapabilities[streamId]} : {} });
        delete state.streamCapabilities[streamId];
        if (state.activeStreamId === streamId) {
          state.activeStreamId = "";
          connectAlertSocket("");
        }
        await refreshActivityOnly();
        setNotice(`Stopped stream ${streamId}.`, "status-ok");
      } catch (error) {
        console.error(error);
        setNotice(`Stop stream failed: ${error.message}`, "status-err");
      }
    }

    function bindEvents() {
      $("apiBaseInput").value = inferDefaultApiBase();

      $("connectBtn").addEventListener("click", connectDashboard);
      $("refreshBtn").addEventListener("click", () => refreshSnapshot());
      $("activityRefreshBtn").addEventListener("click", refreshActivityOnly);
      $("reloadLiveConfigsBtn").addEventListener("click", reloadAllLiveConfigs);
      $("configSaveReloadBtn").addEventListener("click", saveCurrentConfig);
      $("configReloadSectionBtn").addEventListener("click", reloadSelectedSection);
      $("configReloadAllBtn").addEventListener("click", reloadAllLiveConfigs);
      $("configValidateBtn").addEventListener("click", () => saveCurrentConfig(true));
      $("configSearch").addEventListener("input", renderConfigList);
      $("configMode").addEventListener("change", () => {
        if (!state.selectedConfig) { $("configMode").value = "form"; return; }
        const selected = $("configMode").value;
        if (state.configDirty && !confirm("Switching editors discards unsaved changes. Continue?")) {
          $("configMode").value = selected === "form" ? "yaml" : "form"; return;
        }
        openConfig(state.selectedConfig);
        $("configMode").value = selected;
        $("configEditor").hidden = selected !== "yaml"; $("configForm").hidden = selected !== "form";
      });
      $("configNewBtn").addEventListener("click", () => $("configNewDialog").showModal());
      $("configNewForm").addEventListener("submit", async event => {
        event.preventDefault();
        if (state.configDirty && !confirm("Discard unsaved changes and start a new configuration?")) return;
        const section = $("configNewSection").value, name = $("configNewName").value;
        const drafts = {
          models:{name, defaults:{task:"custom", inputs:[{name:"input", shape:[], layout:"ND", color:"NONE"}], artifact:{path:`model_repository/${name}/1/model.onnx`}}, variants:{onnx:{backend:"onnxruntime"}}},
          pipelines:{name, tags:{label:prettyName(name), description:""}, source:{name:"input", stage:"source_rest"}, stages:[]},
          fusion:{name, signals:[], extract:{}, windows:{}, rules:[]},
          alerts:{name, severity:{levels:[{name:"low", rank:1}], default:"low", mapping:{}}, sinks:[]},
        };
        try {
          const schema = await request(`/v1/configs/schema/${section}?model_port=${section === "models"}`);
          ++configLoadId;
          openConfig({section, name, title:prettyName(name), data:drafts[section], schema, content:"", isNew:true,
            reload_behavior:section === "models" ? "conditional" : "live"});
          state.configDirty = true;
          $("configStatus").textContent = "New draft. Fill in the settings, validate, then save.";
          $("configNewDialog").close();
        } catch(error) { setNotice(error.message, "status-err"); }
      });
      $("configEditor").addEventListener("input", () => {
        state.configDirty = $("configEditor").value !== (state.selectedConfig?.content || "");
        $("configStatus").textContent = state.configDirty ? "Unsaved changes. Save to validate and apply this configuration." : "No unsaved changes.";
      });
      $("configEditor").addEventListener("keydown", event => {
        if (event.key === "Tab") {
          event.preventDefault();
          const editor = event.currentTarget;
          editor.setRangeText("  ", editor.selectionStart, editor.selectionEnd, "end");
          editor.dispatchEvent(new Event("input"));
        }
      });
      document.addEventListener("keydown", event => {
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s" && window.InferOps.view === "settings") {
          event.preventDefault(); saveCurrentConfig();
        }
      });

      $("domainSelect").addEventListener("change", async (event) => {
        state.selectedDomainId = event.target.value;
        window.InferOps.storage.set(STORAGE_KEYS.domainId, state.selectedDomainId);
        renderDomainSummary();
        renderAlertsAndEvents();
        const domain = selectedDomain();
        if (domain && domain.config_names && domain.config_names.pipeline) {
          await loadConfig("pipelines", domain.config_names.pipeline, { quiet: true });
        }
      });

      document.querySelectorAll("#modeTabs .mode-tab").forEach((button) => {
        button.addEventListener("click", () => setSelectedMode(button.dataset.mode));
      });

      $("cameraStartBtn").addEventListener("click", () => startCamera().catch((error) => {
        console.error(error);
        setNotice(`Camera start failed: ${error.message}`, "status-err");
      }));
      $("cameraStopBtn").addEventListener("click", stopCamera);
      $("cameraCaptureBtn").addEventListener("click", () => analyzeCameraFrame());
      $("cameraLoopBtn").addEventListener("click", toggleCameraLoop);

      $("imageFileInput").addEventListener("change", () => {
        const file = $("imageFileInput").files && $("imageFileInput").files[0];
        if (file) {
          previewImageFile(file);
        }
      });
      $("imageRunBtn").addEventListener("click", runImage);

      $("videoFileInput").addEventListener("change", () => {
        const file = $("videoFileInput").files && $("videoFileInput").files[0];
        if (file) {
          previewVideoFile(file);
        }
      });
      $("videoRunBtn").addEventListener("click", runVideo);
      $("videoUploadBtn").addEventListener("click", uploadVideo);

      $("rtspStartBtn").addEventListener("click", startRtspStream);
      $("rtspStopBtn").addEventListener("click", () => stopStream(state.activeStreamId));

      document.addEventListener("click", (event) => {
        const configButton = event.target.closest("[data-config-section][data-config-name]");
        if (configButton) {
          loadConfig(configButton.dataset.configSection, configButton.dataset.configName);
          return;
        }
        const stopButton = event.target.closest("[data-stop-stream]");
        if (stopButton) {
          stopStream(stopButton.dataset.stopStream);
        }
      });

      window.addEventListener("beforeunload", event => {
        if (state.configDirty) { event.preventDefault(); event.returnValue = ""; }
        stopCameraLoop();
        if (state.cameraMediaStream) {
          for (const track of state.cameraMediaStream.getTracks()) {
            track.stop();
          }
        }
        if (state.alertSocket) {
          state.alertSocket.close();
        }
        resetPreviewUrls();
      });

      $("imagePreview").addEventListener("load", renderPreviewAnnotations);
      $("cameraPreview").addEventListener("loadedmetadata", renderPreviewAnnotations);
      window.addEventListener("resize", renderPreviewAnnotations);
    }

    document.addEventListener("inferops:refresh", () => refreshSnapshot());
    document.addEventListener("inferops:navigate", event => {
      if (event.detail !== "playground") {
        if (state.cameraMediaStream) stopCamera();
        if (state.videoLoopEnabled) { stopVideoLoop(); $("videoPreview").pause(); }
      }
      if (["activity-view", "playground", "services"].includes(event.detail)) refreshActivityOnly();
    });
    bindEvents();
    setSelectedMode("camera");
    renderResult();
    connectDashboard();

})();
