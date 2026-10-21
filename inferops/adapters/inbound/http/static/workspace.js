/* One server, one navigation shell. Pipeline order comes from the executor. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const esc = x => String(x ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  let selected = "overview", busy = false;
  const api = (path, method = "GET") => window.InferOps.request(`/v1/${path}`, {method});
  function navigate(event) {
    selected = event.detail;
    if (selected === "pipeline" || selected === "services") refreshWorkspace();
    if (selected === "pipeline") checkpoints();
    if (selected === "activity-view") evidence();
  }
  async function refreshWorkspace() {
    if (busy) return;
    busy = true;
    try {
      const data = await api("system/workspace");
      document.dispatchEvent(new CustomEvent("inferops:workspace", {detail:data}));
      $("pipelineMap").innerHTML = data.pipelines.map(p => `<article class="pipeline-spec"><h3>${esc(p.label || p.name)}</h3><p class="footnote">${esc(p.description || "")}</p><ol class="pipeline-steps">${p.steps.map(s => `<li><span>${esc(s.kind)}</span><strong>${esc(s.name)}</strong><small>${esc(s.detail || Object.values(s.references || {}).flat().join(" · "))}</small></li>`).join("")}</ol></article>`).join("");
      const groups = new Map();
      for (const model of data.models) {
        const name = model.logical_name || model.name;
        if (!groups.has(name)) groups.set(name, []);
        groups.get(name).push(model);
      }
      $("modelRuntime").innerHTML = [...groups].map(([name, variants]) => {
        const current = variants.find(model => model.selected) || variants[0];
        return `<article class="model-card"><div><strong>${esc(name)}</strong><span class="status-pill ${current.loaded ? "status-ok":"status-info"}">${current.loaded ? "Loaded":current.selected ? "On demand":"Unavailable in current runtime"}</span></div><p>${esc(data.runtime)} → ${esc(current.selected ? current.name : "No matching variant")}</p><code>${esc(current.input_shapes.map(s => `[${s.join(", ")}]`).join(" / "))}</code><p>${current.batching.dynamic ? `Server dynamic batch ≤ ${esc(current.batching.server_max_batch_size)}` : "Dynamic batching not enabled for this loaded model"}</p><details><summary>${variants.length} runtime ${variants.length === 1 ? "implementation" : "variants"}</summary>${variants.map(model => `<p><strong>${esc(model.backend)}</strong> · ${esc(model.name)}${model.selected ? " · selected" : ""}${model.loaded ? " · loaded" : ""}</p>`).join("")}</details><small>${esc(current.placement.reason || "Placement is verified when the model loads.")}</small></article>`;
      }).join("");
      const triton = await api("system/triton");
      $("tritonStatistics").innerHTML = !triton.enabled ? "<p>Optional. Select Triton runtime and start the serving profile to enable shared inference.</p>" : triton.models.map(m => `<div class="model-card"><strong>${esc(m.model)}</strong>${m.available ? m.statistics.versions.map(v => `<p>Inferences ${v.inference_count} · executions ${v.execution_count} · average executed batch ${v.average_execution_batch_size?.toFixed(2) || "—"}</p>`).join("") : "<p>Statistics unavailable</p>"}</div>`).join("") || "<p>Models load on demand. Start a source to collect execution statistics.</p>";
      $("componentList").innerHTML = `<p>Project <strong>${esc(data.project)}</strong> · runtime <strong>${esc(data.runtime)}</strong> · local GPU slots ${data.admission.local_gpu_concurrency} · Triton outstanding requests ${data.admission.triton_outstanding_requests}</p><div class="component-grid">${data.components.map(c => `<article><strong>${esc(c.name)}</strong><span>${esc(c.state)}</span><small>${esc(c.purpose)}</small></article>`).join("")}</div>`;
    } catch (e) {$("componentList").textContent = e.message;$("pipelineMap").textContent=e.message;}
    finally {busy = false;}
  }
  let evidenceBusy = false;
  let retentionCutoff = null;
  $("retentionForm").addEventListener("submit", async event => {
    event.preventDefault();
    try {
      retentionCutoff = new Date($("retentionBefore").value).toISOString();
      const owner = $("retentionWorker").value;
      const data = await api(`retention/evidence?before=${encodeURIComponent(retentionCutoff)}${owner ? `&worker_id=${encodeURIComponent(owner)}` : ""}`);
      $("retentionPlan").innerHTML = data.candidates.length ? data.candidates.map(row => `<article class="model-card"><strong>${esc(row.camera_id || row.session_id)} · ${esc(row.state)}</strong><p><code>${esc(row.id)}</code></p><button class="ghost-btn" data-retention-id="${esc(row.id)}">Delete artifacts</button></article>`).join("") : "<p>No eligible evidence for this owner and cutoff.</p>";
    } catch(error) { $("retentionPlan").textContent = error.message; }
  });
  $("retentionPlan").addEventListener("click", async event => {
    const button = event.target.closest("[data-retention-id]");
    if (!button || !confirm("Permanently delete these evidence artifacts? The evidence record will remain.")) return;
    button.disabled = true;
    try {
      const result = await api(`retention/evidence/${encodeURIComponent(button.dataset.retentionId)}?before=${encodeURIComponent(retentionCutoff)}`, "POST");
      button.textContent = result[0]?.status || "Updated";
      await evidence();
    } catch(error) {button.textContent = error.message;button.disabled = false;}
  });
  api("workers").then(data => {for (const worker of data.workers) $("retentionWorker").add(new Option(worker.worker_id,worker.worker_id));}).catch(()=>{});
  async function checkpoints() {
    try {
      const data = await api("model-preparation");
      $("checkpointSubmit").disabled = !data.configured;
      if (!data.configured) $("checkpointNotice").textContent = data.prerequisite;
      $("checkpointJobs").innerHTML = data.jobs.map(j => `<article class="model-card"><strong>${esc(j.checkpoint)} · ${esc(j.status)}</strong><p>${esc(j.error || (j.report ? `${j.report.architecture} · ${j.report.classes} classes · batches ${j.report.checks.map(c=>c.batch).join(", ")}` : "CPU conversion and verification"))}</p>${j.artifact_path ? `<code>${esc(j.artifact_path)}</code>` : ""}<p><a href="#api" data-api-path="/v1/model-preparation/${encodeURIComponent(j.id)}/export.log">Export log</a>${j.status === "validated" ? ` · <a href="/v1/model-preparation/${encodeURIComponent(j.id)}/model.onnx">Download ONNX</a> · <a href="#api" data-api-path="/v1/model-preparation/${encodeURIComponent(j.id)}/report.json">Validation report</a>`:""}</p></article>`).join("");
    } catch(e) {$("checkpointNotice").textContent=e.message;}
  }
  $("checkpointForm").addEventListener("submit", async event => {
    event.preventDefault();$("checkpointSubmit").disabled=true;
    try {
      const form = new FormData(), file = $("checkpointFile").files[0];
      if (file) form.append("file",file);
      form.append("server_path",$("checkpointPath").value.trim());
      form.append("variant",$("checkpointVariant").value);form.append("resolution",$("checkpointResolution").value);form.append("max_batch",$("checkpointBatch").value);
      await window.InferOps.request("/v1/model-preparation", {method:"POST", body:form, timeoutMs:90000});
      $("checkpointNotice").textContent="Conversion queued. Results will appear here.";$("checkpointForm").reset();await checkpoints();
    } catch(e) {$("checkpointNotice").textContent=e.message;}
    finally {$("checkpointSubmit").disabled=false;}
  });
  async function evidence() {
    if (evidenceBusy) return;
    evidenceBusy = true;
    try {
      const id = $("evidenceSession").value.trim();
      const rows = await api(`evidence/records?limit=100${id ? `&session_id=${encodeURIComponent(id)}`:""}`);
      $("evidenceRecords").innerHTML = rows.length ? `<div class="evidence-grid">${rows.map(r => {
        const base = `/v1/evidence/records/${encodeURIComponent(r.id)}`;
        const snapshot = !r.deleted && r.artifacts?.snapshot;
        const clip = !r.deleted && r.artifacts?.clip;
        return `<article class="evidence-card">${snapshot ? `<a href="${base}/assets/snapshot" target="_blank" rel="noreferrer"><img loading="lazy" src="${base}/assets/snapshot" alt="Saved incident snapshot"></a>`:""}<div><strong>${esc(r.state)}${r.deleted ? " · Deleted":""}</strong><p>Source: ${esc(r.camera_id || r.stream_name || "Legacy / unregistered input")}</p><code>${esc(r.session_id)}</code><p>${esc((r.errors || []).join(" · ") || "No capture errors reported")}</p><div class="stack-row">${clip ? `<button class="ghost-btn" data-play="${base}/assets/clip">Play clip</button>`:""}<a class="ghost-btn" href="#api" data-api-path="${base}">Inspect record →</a>${r.retry_available && !r.deleted ? `<button class="btn" data-retry="${esc(r.id)}">Retry save</button>`:""}</div></div></article>`;
      }).join("")}</div>` : '<p class="empty-state">No evidence records for this selection. Accepted incidents create captures; incomplete saves show their reason here.</p>';
    } catch(e) {$("evidenceRecords").textContent=e.message;}
    finally {evidenceBusy=false;}
  }
  $("refreshEvidence").addEventListener("click",evidence);
  $("evidenceSession").addEventListener("change",evidence);
  document.addEventListener("click", event => {
    const link = event.target.closest("[data-evidence-session]");
    if (!link) return;
    $("evidenceSession").value = link.dataset.evidenceSession;
    $("historyDialog").close();
    if (selected === "activity-view") evidence();
  });
  $("evidencePlayer").addEventListener("close",()=>{$("evidenceVideo").pause();$("evidenceVideo").removeAttribute("src");$("evidenceVideo").load();});
  $("evidenceRecords").addEventListener("click", async event => {
    const play = event.target.closest("[data-play]");
    if (play) {$("evidenceVideo").src=play.dataset.play;$("evidencePlayer").showModal();return;}
    const button = event.target.closest("[data-retry]");if (!button) return;
    button.disabled = true;
    try {await api(`evidence/records/${encodeURIComponent(button.dataset.retry)}/retry`, "POST");await evidence();}
    catch(e) {button.textContent=e.message;button.disabled=false;}
  });
  document.addEventListener("inferops:navigate", navigate);
  function refreshSelected() {
    if (selected === "activity-view") return evidence();
    if (selected === "pipeline") { refreshWorkspace(); return checkpoints(); }
    if (selected === "services") return refreshWorkspace();
  }
  document.addEventListener("inferops:refresh", refreshSelected);
  window.InferOps.poll(refreshSelected);
})();
