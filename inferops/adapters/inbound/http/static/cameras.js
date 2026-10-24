/* Camera/source controls. Previews reuse the server's latest annotated JPEG. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const state = { sources: [], pipelines: [], filter: "all", editing: null, cards: new Map(), refreshing: false, capacityRequest: 0, live: null, liveRequest: 0 };
  const esc = value => String(value ?? "").replace(/[&<>"']/g, x => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
  const number = (value, digits = 1) => value == null ? "—" : Number(value).toFixed(digits);
  const gib = bytes => number(bytes / 1024 ** 3);
  const notice = window.InferOps.notify;
  const api = (path, options = {}) => window.InferOps.request(`/v1${path}`, {
    ...options, headers: { ...(typeof options.body === "string" ? {"Content-Type": "application/json"} : {}), ...options.headers },
  });
  function inputKind() {
    const kind = $("sourceKind").value;
    $("fileLabel").classList.toggle("hidden", kind === "rtsp");
    $("sourceFile").accept = kind === "image" ? "image/*" : "video/*";
    $("sourceUri").placeholder = state.editing ? "Leave blank to keep the current source" : kind === "rtsp" ? "rtsp://camera/stream" : "Upload below, or enter a server file path";
    $("sourceForm").elements.loop.disabled = kind !== "video";
    if (kind !== "video") $("sourceForm").elements.loop.checked = false;
  }
  function previewOptions() {
    for (const name of ["preview_fps", "jpeg_quality"]) $("sourceForm").elements[name].disabled = !$("sourceForm").elements.preview_enabled.checked;
  }
  function openForm(source = null) {
    state.editing = source?.camera_id || null;
    const form = $("sourceForm"); form.reset();
    $("formError").textContent = "";
    $("sourceDialogTitle").textContent = source ? "Edit source" : "Add a source";
    $("saveSource").textContent = source ? "Save changes" : "Register source";
    form.elements.camera_id.readOnly = Boolean(source);
    $("sourceWorker").value = source?.worker_id || "";
    $("sourceWorker").disabled = Boolean(source);
    if (source) {
      for (const key of ["camera_id", "name", "group", "location", "kind", "pipeline", "inference_fps", "evidence_fps", "max_frame_age_ms", "preview_fps", "jpeg_quality", "rtsp_transport", "timeout_s"]) form.elements[key].value = source[key] ?? "";
      for (const key of ["loop", "autostart", "preview_enabled"]) form.elements[key].checked = source[key] ?? key === "preview_enabled";
      const resolution = `${source.max_width}x${source.max_height}`;
      if (![...$("sourceResolution").options].some(x => x.value === resolution)) $("sourceResolution").add(new Option(resolution, resolution));
      $("sourceResolution").value = resolution;
    }
    inputKind(); previewOptions(); $("sourceDialog").showModal();
  }
  async function saveSource(event) {
    event.preventDefault(); const form = event.currentTarget; $("saveSource").disabled = true;
    $("formError").textContent = "";
    try {
      let uri = form.elements.uri.value.trim();
      const file = $("sourceFile").files[0];
      if (file && form.elements.kind.value !== "rtsp") {
        const body = new FormData(); body.append("file", file);
        const upload = await api("/cameras/media", {method:"POST", body});
        if (upload.kind !== form.elements.kind.value) throw new Error("The uploaded file does not match the selected input type.");
        uri = upload.uri;
      }
      if (!uri && !state.editing) throw new Error("Provide a source URL or upload a file.");
      const [max_width, max_height] = $("sourceResolution").value.split("x").map(Number);
      const payload = {camera_id:form.elements.camera_id.value.trim(),name:form.elements.name.value.trim(),
        group:form.elements.group.value.trim(),location:form.elements.location.value.trim(),
        kind:form.elements.kind.value, uri, pipeline:form.elements.pipeline.value,
        inference_fps:Number(form.elements.inference_fps.value),preview_fps:Number(form.elements.preview_fps.value),
        preview_enabled:form.elements.preview_enabled.checked,
        evidence_fps:form.elements.evidence_fps.value ? Number(form.elements.evidence_fps.value) : null,
        max_frame_age_ms:Number(form.elements.max_frame_age_ms.value),
        rtsp_transport:form.elements.rtsp_transport.value,timeout_s:Number(form.elements.timeout_s.value),
        jpeg_quality:Number(form.elements.jpeg_quality.value),max_width,max_height,
        loop:form.elements.loop.checked,autostart:form.elements.autostart.checked};
      const owner = $("sourceWorker").value;
      const saved = await api(state.editing ? `/cameras/${encodeURIComponent(state.editing)}` : `/cameras${owner ? `?worker_id=${encodeURIComponent(owner)}` : ""}`, {method:state.editing ? "PUT":"POST",body:JSON.stringify(payload)});
      $("sourceDialog").close(); notice(`${payload.name} ${state.editing ? "updated":"registered"}.${saved.active ? " Running stream restarted with the saved settings." : " Use Start to open a processing session."}`);
      await refresh();
    } catch (error) { $("formError").textContent = error.message; }
    finally { $("saveSource").disabled = false; }
  }
  async function action(source, kind, button) {
    if (kind === "edit") { openForm(source); return; }
    if (kind === "history") { await showHistory(source); return; }
    if (kind === "recordings") { await showRecordings(source); return; }
    if (kind === "live") { button.disabled = true; try { await showLive(source); } finally { button.disabled = false; } return; }
    if (kind === "remove" && !confirm(`Remove ${source.name}? Active processing will stop. Historical results and evidence are preserved.`)) return;
    button.disabled = true;
    try {
      await api(`/cameras/${encodeURIComponent(source.camera_id)}${kind === "remove" ? "":`/${kind}`}`, {method:kind === "remove" ? "DELETE":"POST"});
      notice(`${source.name}: ${kind === "remove" ? "removed; history preserved":kind === "start" ? "session started":"stopped"}.`);
      await refresh();
    } catch (error) { notice(error.message, true); }
    finally { button.disabled = false; }
  }
  function createCard(source) {
    const article = document.createElement("article"); article.className = "source-card";
    article.innerHTML = `<div class="card-preview"><div class="preview-placeholder"><span>▣</span><div>Waiting for a processed frame</div></div><img alt="" hidden><div class="card-tags"><span class="tag state-tag"></span><span class="tag kind-tag"></span></div><span class="frame-time"></span></div><div class="card-body"><div class="card-title"><h2></h2></div><p class="card-id"></p><div class="card-stats"></div><div class="card-controls"><button class="button toggle"></button><button class="button live" hidden>Live video</button><button class="button recordings" hidden>Recordings</button><button class="button history">History</button><button class="button edit">Edit</button><button class="button remove" aria-label="Remove source">✕</button></div><p class="state-note"></p></div>`;
    const card = {element:article,version:null, visible:true};
    for (const kind of ["toggle","live","recordings","history","edit","remove"]) article.querySelector(`.${kind}`).addEventListener("click", event => {
      const current = state.sources.find(s => s.camera_id === source.camera_id);
      action(current, kind === "toggle" ? current.active ? "stop":"start":kind,event.currentTarget).catch(e => notice(e.message,true));
    });
    const image = article.querySelector("img");
    image.addEventListener("load", () => { image.hidden = false; article.querySelector(".preview-placeholder").hidden = true; });
    image.addEventListener("error", () => {image.hidden = true;article.querySelector(".preview-placeholder").hidden = false;});
    observer.observe(article);
    return card;
  }
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) for (const card of state.cards.values()) if (card.element === entry.target) card.visible = entry.isIntersecting;
  }, {rootMargin:"150px"});
  function renderSources() {
    const search = $("search").value.toLowerCase();
    const visible = state.sources.filter(s => (state.filter === "all" || (state.filter === "active") === s.active) && `${s.name} ${s.camera_id} ${s.group || ""} ${s.location || ""}`.toLowerCase().includes(search));
    const ids = new Set(state.sources.map(s=>s.camera_id));
    for (const [id, card] of state.cards) if (!ids.has(id)) {observer.unobserve(card.element);card.element.remove();state.cards.delete(id);}
    for (const card of state.cards.values()) card.element.hidden = true;
    for (const source of visible) {
      let card = state.cards.get(source.camera_id);
      if (!card) {card = createCard(source);state.cards.set(source.camera_id,card);$("sourceGrid").append(card.element);}
      const el = card.element; el.hidden = false;
      el.querySelector("h2").textContent = source.name;
      el.querySelector(".card-id").textContent = [source.camera_id, source.pipeline, source.group && `Group: ${source.group}`, source.location && `Location: ${source.location}`, source.worker_id && `worker ${source.worker_id}`].filter(Boolean).join(" · ");
      const status = source.active && source.connection === "reconnecting" ? "reconnecting":source.status;
      const tag = el.querySelector(".state-tag"); tag.textContent = status === "running" ? "● ANALYZING":status.toUpperCase();
      tag.className = `tag state-tag${source.active ? " live":""}${status === "error" ? " error":""}`;
      el.querySelector(".kind-tag").textContent = source.kind.toUpperCase();
      const w = source.processed_width || source.max_width, h = source.processed_height || source.max_height;
      el.querySelector(".card-stats").innerHTML = `<span>${source.kind === "image" ? "Single image":`<b>${number(source.fps || 0)}</b> analysis / ${number(source.inference_fps,1)} target FPS`}</span><span>${esc(w)} × ${esc(h)}</span><span>${number(source.latency_ms,0)} ms</span>`;
      const toggle = el.querySelector(".toggle");toggle.textContent = source.active ? "Ⅱ Stop":"▷ Start";toggle.classList.toggle("primary",!source.active);
      el.querySelector(".edit").disabled = false;
      el.querySelector("button.live").hidden = !source.live_video_available;
      el.querySelector("button.recordings").hidden = !source.live_video_available;
      el.querySelector(".preview-placeholder div").textContent = source.preview_enabled === false ? "Annotated preview disabled · analysis continues when started":"Waiting for a processed frame";
      const age = source.last_result_at ? Math.max(0,Math.floor(Date.now()/1000-source.last_result_at)) : null;
      el.querySelector(".frame-time").textContent = age == null ? "No results":age <= 2 && source.active ? "Latest processed frame":`${age}s since last result`;
      el.querySelector(".state-note").textContent = source.error || (source.session_id ? `Session ${source.session_id.slice(0,10)} · ${source.processed || 0} processed · ${source.dropped || 0} skipped`:"Registered · ready to start");
      const diagnostics = [];
      if (source.below_target && source.active) diagnostics.push(`Below target. Slowest step: ${source.slowest_stage || "processing"} (${number(source.stage_timings_ms?.[source.slowest_stage],0)} ms).`);
      for (const model of source.model_runtime || []) diagnostics.push(`${model.model}: ${model.device} (${model.concurrency || 1} concurrent slots)${model.downgraded ? " (CPU fallback)" : ""}`);
      if (source.preview_error) diagnostics.push(source.preview_error);
      el.querySelector(".card-stats").title = diagnostics.join(" ");
      if (source.below_target && source.active) el.querySelector(".state-note").textContent += ` · ${diagnostics.join(" ")}`;
      const version = `${source.session_id}-${source.preview_version}`;
      if (source.preview_enabled !== false && source.preview_version && card.visible && card.version !== version && !document.hidden) {
        el.querySelector("img").alt = `${source.name}: latest processed frame with model annotations`;
        el.querySelector("img").src = `/v1/cameras/${encodeURIComponent(source.camera_id)}/preview.jpg?v=${encodeURIComponent(version)}`;card.version=version;
      } else if (source.preview_enabled === false || !source.preview_version) {
        el.querySelector("img").hidden = true;el.querySelector(".preview-placeholder").hidden = false;card.version=null;
      }
    }
    $("emptyState").hidden = visible.length > 0;
    $("emptyState").querySelector("h2").textContent = state.sources.length ? "No matching sources":"Your sources start here";
    $("navCount").textContent = $("allCount").textContent = state.sources.length;
    $("runningCount").textContent = state.sources.filter(s=>s.active).length;
  }
  function renderHardware(info) {
    $("serverName").textContent = info.hostname || "Server";
    const gpu = info.gpus?.[0];
    $("gpuMemory").textContent = gpu ? `${number(gpu.used_mib/1024)} / ${number(gpu.total_mib/1024)} GB`:"Unavailable";
    $("gpuName").textContent = gpu ? `${gpu.name} · ${gpu.utilization_percent}% utilization`:"GPU telemetry unavailable to this server";
    $("gpuMeter").style.width = gpu ? `${100 * gpu.used_mib/gpu.total_mib}%`:"0%";
    $("ramMemory").textContent = info.ram_bytes ? `${gib(info.ram_bytes-info.ram_available_bytes)} / ${gib(info.ram_bytes)} GB`:"Unavailable";
    $("ramMeter").style.width = info.ram_bytes ? `${100 * (1-info.ram_available_bytes/info.ram_bytes)}%`:"0%";
    $("cpuName").textContent = `${info.cpu_model || "CPU"} · ${info.cpu_count || "—"} logical cores`;
  }
  async function capacity(event) {
    event?.preventDefault(); if (!$("capacityPipeline").value) return;
    const requestId = ++state.capacityRequest;
    try {
      const [width,height] = $("capacityResolution").value.split("x");
      const params = new URLSearchParams({pipeline:$("capacityPipeline").value,width,height,fps:$("capacityFps").value});
      if ($("capacityEvidenceFps").value) params.set("evidence_fps",$("capacityEvidenceFps").value);
      const result = await api(`/system/capacity?${params}`);
      if (requestId !== state.capacityRequest) return;
      $("capacityNumber").textContent = result.tested_passing_cameras == null ? result.cases.length ? "Target not met":"Unmeasured":`${result.tested_passing_cameras} source${result.tested_passing_cameras === 1 ? "" : "s"} tested`;
      $("capacityCaption").textContent = `${width} × ${height} · ${$("capacityFps").value} FPS · evidence ${result.evidence_fps} FPS · ${result.runtime}`;
      const rows = result.cases.map(c => `<tr><td>${c.cameras}</td><td>${esc(c.per_camera_fps.join(" / "))}</td><td>${number(c.p95_ms,0)} ms</td><td>${number(c.gpu_used_peak_mib/1024)} GB</td><td>${c.passed ? "✓ Pass":"Below target"}</td></tr>`).join("");
      $("capacityDetails").innerHTML = `<p><strong>${result.tested_passing_cameras == null ? "No matching passing measurement":`${result.tested_passing_cameras} simultaneous source${result.tested_passing_cameras === 1 ? "" : "s"} passed this local profile`}</strong></p><p>${esc(result.explanation)}</p>${result.tested_passing_cameras === 1 ? `<p>One source passed. No additional camera-count headroom was demonstrated.</p>` : result.suggested_cameras_with_20_percent_headroom != null ? `<p>Starting recommendation with 20% count headroom: <strong>${result.suggested_cameras_with_20_percent_headroom} sources</strong>. Verify under your real workload.</p>`:""}<p>Runtime: ${esc(result.runtime)} · Evidence ${result.evidence_enabled ? "enabled":"disabled"} · Data filesystem ${esc(result.hardware.data_filesystem || "unknown")} · Configured active limit ${result.configured_active_limit}</p>${rows ? `<table><thead><tr><th>Sources</th><th>FPS per source</th><th>p95</th><th>Peak VRAM</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table>`:""}<p>Measured preview profile: 2 FPS, width 640, JPEG quality 75. Processing resolution is a size limit, not a model accuracy score or a change to the camera's native stream quality. Source FPS, inference FPS and preview FPS are separate.</p>`;
    } catch(error) { notice(error.message,true); }
  }
  async function showHistory(source) {
    $("historyTitle").textContent = `${source.name} · history`;$("historyBody").textContent="Loading…";$("historyDialog").showModal();
    try {
      const id = encodeURIComponent(source.camera_id);
      const [runs,results] = await Promise.all([api(`/cameras/${id}/runs?limit=20`),api(`/cameras/${id}/results?limit=20`)]);
      $("historyBody").innerHTML = `<p class="dialog-intro">Stable source ID: <code>${esc(source.camera_id)}</code>. Each start creates a separate session. Recent result history is bounded; saved incident evidence follows the configured pipeline storage.</p>` + runs.map(r => `<div class="history-run"><strong>${esc(r.status)}</strong> · ${esc(new Date(r.started_at*1000).toLocaleString())}<br><code>${esc(r.session_id)}</code> · ${r.processed || 0} frames<br><a href="#activity-view" data-evidence-session="${esc(r.session_id)}">Review evidence in dashboard →</a></div>`).join("") + `<h3>Latest results</h3><a href="#api" data-api-path="/v1/cameras/${id}/results?limit=100">Inspect result JSON →</a><pre>${esc(JSON.stringify(results.map(row=>({at:row.at,sequence:row.sequence,session_id:row.session_id,outputs:row.results.map(out=>({detections:out.results.flatMap(r=>r.detections || []),alerts:out.payload?.alerts || [],face_status:out.payload?.face_status_label}))})),null,2))}</pre>`;
    } catch(error) {$("historyBody").textContent=error.message;}
  }
  async function showRecordings(source) {
    $("recordingsTitle").textContent = `${source.name} · original recordings`;
    $("recordingsList").textContent = "Loading recordings…";
    $("recordingsDialog").showModal();
    try {
      const data = await api(`/cameras/${encodeURIComponent(source.camera_id)}/recordings`);
      $("recordingsList").replaceChildren();
      if (!data.recordings.length) $("recordingsList").textContent = "No finalized recordings yet. Recordings are created while inference or a live viewer reads the camera.";
      for (const row of data.recordings.slice().reverse()) {
        const button = document.createElement("button"); button.className = "button";
        button.textContent = `${new Date(row.start).toLocaleString()} · ${Math.round(row.duration)} seconds`;
        button.addEventListener("click", () => { $("recordingPlayer").src = row.url; });
        $("recordingsList").append(button);
      }
    } catch (error) { $("recordingsList").textContent = error.message; }
  }
  $("recordingsDialog").addEventListener("close", () => {
    $("recordingPlayer").pause(); $("recordingPlayer").removeAttribute("src"); $("recordingPlayer").load();
  });
  async function playLive() {
    const requestId = ++state.liveRequest;
    const frame = $("liveFrame");
    if (!state.live || document.hidden || !$("liveDialog").open) { frame.removeAttribute("src"); return; }
    const url = state.live[`${$("liveProtocol").value}_proxy_url`] || state.live[`${$("liveProtocol").value}_url`];
    $("liveExternal").href = state.live[`${$("liveProtocol").value}_url`];
    if (frame.getAttribute("src") === url) return;
    const status = $("liveStatus"); status.hidden = false; status.className = "dialog-intro"; status.textContent = "Connecting to live video…";
    frame.removeAttribute("src");
    try {
      const response = await fetch(url, {headers:{Accept:"text/html"}, credentials:"same-origin"});
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Live video service returned HTTP ${response.status}.`);
      }
      if (requestId !== state.liveRequest || document.hidden || !$("liveDialog").open) return;
      frame.src = url; status.hidden = true;
    } catch (error) {
      if (requestId === state.liveRequest) { status.className = "form-error"; status.textContent = `${error.message} Check the camera and MediaMTX connection, or try the other playback mode.`; }
    }
  }
  async function showLive(source) {
    state.live = await api(`/cameras/${encodeURIComponent(source.camera_id)}/live`);
    $("liveTitle").textContent = `${source.name} · live video`;
    $("liveDialog").showModal(); await playLive();
  }
  $("liveProtocol").addEventListener("change", playLive);
  $("liveDialog").addEventListener("close", () => { state.live = null; playLive(); });
  document.addEventListener("visibilitychange", playLive);
  window.addEventListener("pagehide", () => $("liveFrame").removeAttribute("src"));
  async function refresh() {
    if (state.refreshing) return; state.refreshing=true;
    try {
      const [catalog,hardware] = await Promise.all([api("/cameras"),api("/system/hardware")]);
      state.sources=catalog.cameras;
      $("activeSources").innerHTML = `${catalog.active} <em>/ ${catalog.active_limit} limit</em>`;
      $("registeredSources").textContent = `${catalog.cameras.length} registered · ${catalog.cameras.length-catalog.active} inactive`;
      renderSources();renderHardware(hardware);
      document.dispatchEvent(new CustomEvent("inferops:sources", {detail: {catalog, hardware}}));
    } catch(error) {notice(`Server connection: ${error.message}`,true);}
    finally {state.refreshing=false;}
  }
  $("addSource").addEventListener("click",()=>openForm());$("emptyAdd").addEventListener("click",()=>openForm());
  $("sourceKind").addEventListener("change",inputKind);$("sourceForm").addEventListener("submit",saveSource);
  $("sourceForm").elements.preview_enabled.addEventListener("change", previewOptions);
  document.querySelectorAll("[data-close]").forEach(b=>b.addEventListener("click",()=>$(b.dataset.close).close()));
  document.querySelectorAll("[data-filter]").forEach(b=>b.addEventListener("click",()=>{state.filter=b.dataset.filter;document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t===b));renderSources();}));
  $("search").addEventListener("input",renderSources);$("refresh").addEventListener("click",refresh);
  $("openCapacity").addEventListener("click",()=>{$("capacityDialog").showModal();capacity();});$("capacityForm").addEventListener("submit",capacity);
  async function init() {
    try {
      const data = await api("/workers");
      for (const worker of data.workers) $("sourceWorker").add(new Option(`${worker.worker_id} · ${worker.source_limit} source limit`,worker.worker_id));
      $("workerLabel").hidden = !data.workers.length;
    } catch(error) { notice(error.message,true); }
    try { state.pipelines=await api("/pipelines");for(const p of state.pipelines){$("pipelineSelect").add(new Option(p.tags?.label || p.name,p.name));$("capacityPipeline").add(new Option(p.name,p.name));} }
    catch(error){notice(error.message,true);}
    await refresh();await capacity();
    window.InferOps.poll(() => ["overview", "sources"].includes(window.InferOps.view) && refresh());
  }
  document.addEventListener("click", event => { if (event.target.closest("[data-api-path]") && $("historyDialog").open) $("historyDialog").close(); });
  document.addEventListener("inferops:refresh", refresh);
  document.addEventListener("inferops:navigate", event => { if (["sources", "overview"].includes(event.detail)) refresh(); });
  document.addEventListener("inferops:add-source", () => openForm());
  init();
})();
