/* Unified navigation, command palette, jobs and schema-driven API tools. */
(() => {
  "use strict";
  const C = window.InferOps, $ = id => document.getElementById(id), esc = C.escape;
  const paths = {
    overview:'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
    sources:'M4 7h12v12H4zM16 11l5-3v10l-5-3M7 4h6',
    pipeline:'M4 4h5v5H4zM15 15h5v5h-5zM15 4h5v5h-5zM6.5 9v8.5H15M9 6.5h6',
    play:'M8 4l13 8-13 8z', jobs:'M8 5V3h8v2M4 5h16v16H4zM8 10h8M8 14h8M8 18h4',
    activity:'M3 12h4l3-8 4 16 3-8h4', settings:'M4 6h16M4 12h16M4 18h16M8 3v6M16 9v6M10 15v6',
    server:'M3 3h18v7H3zM3 14h18v7H3zM7 6.5h.01M7 17.5h.01M11 6.5h6M11 17.5h6',
    code:'M8 5l-6 7 6 7M16 5l6 7-6 7M14 3l-4 18',
    palette:'M12 3a9 9 0 100 18h1a2 2 0 001.5-3.3 1.5 1.5 0 011.2-2.5H18a4 4 0 004-4c0-5-4-8.2-10-8.2zM7.5 10h.01M10 6.5h.01M15 6.5h.01M18 10h.01',
    book:'M12 5c-3-2-6-2-10-1v15c4-1 7-1 10 1 3-2 6-2 10-1V4c-4-1-7-1-10 1zM12 5v15',
    search:'M21 21l-5-5M18 10a8 8 0 11-16 0 8 8 0 0116 0z',
    sun:'M12 8a4 4 0 110 8 4 4 0 010-8zM12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5',
    refresh:'M20 7a9 9 0 10.5 9M20 2v6h-6', plus:'M12 5v14M5 12h14',
  };
  const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[name] || paths.overview}"/></svg>`;
  document.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = icon(el.dataset.icon); });
  const views = {
    overview: ['Overview', 'YOUR WORKSPACE, CONNECTED', 'Everything in view.', 'Your sources, pipelines and runtime. One place to take control.', 'overview'],
    sources: ['Sources', 'SOURCE CONTROL', 'Every source. One view.', 'Connect, tune and manage your cameras, videos and images.', 'sources'],
    pipeline: ['Pipeline & models', 'THE INTELLIGENCE LAYER', 'Shape the processing flow.', 'Inspect execution stages, model placement and checkpoint preparation.', 'pipeline'],
    playground: ['Inference playground', 'TEST & EXPLORE', 'Put your pipeline to work.', 'Run live camera, image, video and RTSP inference with real results.', 'play'],
    jobs: ['Background jobs', 'EXECUTION CONTROL', 'Keep the work moving.', 'Submit, inspect, cancel and retry background pipeline runs.', 'jobs'],
    'activity-view': ['Activity & evidence', 'RESULTS & HISTORY', 'A record of what matters.', 'Review events, incidents and saved evidence from your processing sessions.', 'activity'],
    settings: ['Configuration', 'ENGINE CONFIGURATION', 'Fine-tune every detail.', 'Edit the project configuration that drives models, pipelines and policies.', 'settings'],
    services: ['Services & runtime', 'SYSTEM OPERATIONS', 'Know what is running.', 'Inspect runtime placement, connected services and inference scheduling.', 'server'],
    api: ['API explorer', 'BUILD WITH INFEROPS', 'Your API, at your fingertips.', 'Discover endpoints, compose requests and inspect real server responses.', 'code'],
    preferences: ['Customize workspace', 'MAKE IT YOURS', 'Your workspace. Your way.', 'Choose your appearance, layout and refresh behavior.', 'palette'],
    guide: ['Interface guide', 'WORKSPACE REFERENCE', 'Find your way around.', 'A connected guide to workflows, interfaces and deployment ports.', 'book'],
  };
  C.view = 'overview';
  function navigate(initial = false) {
    const aliases = {execution:'playground', activity:'activity-view', configuration:'settings', interfaces:'services', mainContent:C.view};
    const hash = location.hash.slice(1);
    const legacy = location.pathname.replace(/\/$/, '') === '/ui/cameras' ? 'sources' : location.pathname === '/ui/guide' ? 'guide' : C.preferences.landing;
    C.view = aliases[hash] || (views[hash] ? hash : legacy);
    const meta = views[C.view];
    document.querySelectorAll('[data-pane]').forEach(el => { el.hidden = el.dataset.pane !== C.view; });
    document.querySelectorAll('[data-view]').forEach(el => {
      const current = el.dataset.view === C.view;
      el.classList.toggle('selected', current);
      if (current) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
    });
    $('viewTitle').textContent = meta[0]; $('pageEyebrow').textContent = meta[1];
    $('pageTitle').textContent = meta[2]; $('pageSubtitle').textContent = meta[3];
    document.title = `${meta[0]} · InferOps`;
    $('sidebar').classList.remove('open'); $('menuToggle').setAttribute('aria-expanded', 'false');
    document.dispatchEvent(new CustomEvent('inferops:navigate', {detail:C.view}));
    if (!initial) { window.scrollTo({top:0}); $('mainContent').focus({preventScroll:true}); }
  }
  window.addEventListener('hashchange', () => navigate());
  $('menuToggle').addEventListener('click', () => {
    const open = $('sidebar').classList.toggle('open'); $('menuToggle').setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('click', event => {
    if (event.target.closest('[data-view]') || (matchMedia('(max-width:760px)').matches && !$('sidebar').contains(event.target) && !$('menuToggle').contains(event.target))) {
      $('sidebar').classList.remove('open'); $('menuToggle').setAttribute('aria-expanded', 'false');
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') { $('sidebar').classList.remove('open'); $('menuToggle').setAttribute('aria-expanded','false'); }
  });
  $('globalRefresh').addEventListener('click', () => document.dispatchEvent(new CustomEvent('inferops:refresh')));
  const themeToggle = () => C.savePreferences({theme:document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'});
  $('themeToggle').addEventListener('click', themeToggle);
  function syncPreferences() {
    const form = $('preferencesForm');
    for (const [key, value] of Object.entries(C.preferences)) {
      const input = form.elements.namedItem(key);
      if (input) { if (input.type === 'checkbox') input.checked = Boolean(value); else input.value = value; }
    }
    $('refreshMode').textContent = Number(C.preferences.refresh) ? `Live · ${C.preferences.refresh}s` : 'Manual refresh';
  }
  $('preferencesForm').addEventListener('change', event => {
    const input = event.target;
    if (!input.name) return;
    C.savePreferences({[input.name]: input.type === 'checkbox' ? input.checked : input.name === 'refresh' ? Number(input.value) : input.value});
    $('preferencesStatus').textContent = 'Preferences saved';
  });
  $('resetPreferences').addEventListener('click', () => { C.savePreferences({...C.defaults}); $('preferencesStatus').textContent = 'Defaults restored'; });
  document.addEventListener('inferops:preferences', syncPreferences); syncPreferences();

  // Keyboard-first navigation. No API writes are triggered by palette selection.
  let commandIndex = 0, commands = [];
  function renderCommands() {
    const query = $('commandInput').value.toLowerCase().trim();
    const all = Object.entries(views).map(([id, meta]) => ({title:meta[0], search:meta.join(' '), glyph:meta[4], run:() => {location.hash=id;}}));
    all.push({title:'Add source', search:'register camera video image', glyph:'plus', run:() => document.dispatchEvent(new CustomEvent('inferops:add-source'))},
      {title:'Refresh workspace', search:'reload snapshot', glyph:'refresh', run:() => document.dispatchEvent(new CustomEvent('inferops:refresh'))},
      {title:'Toggle color theme', search:'light dark appearance', glyph:'sun', run:themeToggle});
    commands = all.filter(item => `${item.title} ${item.search}`.toLowerCase().includes(query));
    commandIndex = Math.min(commandIndex, Math.max(0, commands.length-1));
    $('commandResults').innerHTML = commands.length ? commands.map((item,i) => `<button class="command-result${i===commandIndex?' selected':''}" data-command="${i}">${icon(item.glyph)}${esc(item.title)}<span>↵</span></button>`).join('') : '<p class="empty-state">No matching views or actions.</p>';
  }
  function openCommands() { $('commandInput').value=''; commandIndex=0; renderCommands(); $('commandDialog').showModal(); $('commandInput').focus(); }
  $('commandOpen').addEventListener('click', openCommands); $('commandSidebar').addEventListener('click', openCommands);
  $('commandInput').addEventListener('input', () => {commandIndex=0;renderCommands();});
  function executeCommand(index) { const command=commands[index];if(command){$('commandDialog').close();command.run();} }
  $('commandResults').addEventListener('click', event => {const el=event.target.closest('[data-command]');if(el)executeCommand(Number(el.dataset.command));});
  $('commandInput').addEventListener('keydown', event => {
    if (['ArrowDown','ArrowUp'].includes(event.key)) {
      event.preventDefault();if(!commands.length)return;commandIndex=(commandIndex+(event.key==='ArrowDown'?1:-1)+commands.length)%commands.length;renderCommands();$('commandResults').children[commandIndex]?.scrollIntoView({block:'nearest'});
    } else if (event.key==='Enter') {event.preventDefault();executeCommand(commandIndex);}
  });
  document.addEventListener('keydown', event => {if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='k'){event.preventDefault();if($('commandDialog').open)$('commandDialog').close();else if(!document.querySelector('dialog[open]'))openCommands();}});

  let overviewBusy=false;
  document.addEventListener('inferops:sources', event => { $('overviewSources').textContent=event.detail.catalog.cameras.length; });
  function workspaceSummary(data) {
    $('workspaceProject').textContent=data.project;
    $('overviewPipelines').textContent=data.pipelines.length; $('overviewModels').textContent=new Set(data.models.map(model=>model.logical_name||model.name)).size;
    $('overviewRuntime').textContent=String(data.runtime).toUpperCase();
  }
  document.addEventListener('inferops:workspace', event => workspaceSummary(event.detail));
  async function refreshOverview() {
    if(overviewBusy)return;overviewBusy=true;
    try {
      const results=await Promise.allSettled([C.request('/v1/system/workspace'), C.request('/v1/jobs?limit=100'), C.request('/readyz')]);
      if(results[0].status==='fulfilled') workspaceSummary(results[0].value);
      if(results[1].status==='fulfilled') $('overviewJobs').textContent=results[1].value.length;
      const ready=results[2].status==='fulfilled';
      $('connectionDot').classList.toggle('disconnected',!ready); $('connectionLabel').textContent=ready?'Engine ready':'Engine unavailable';
      $('overviewUpdated').textContent=results[0].status==='fulfilled' ? `Workspace snapshot · ${new Date().toLocaleTimeString()}${ready?'':' · readiness check failed'}` : `Workspace unavailable: ${results[0].reason.message}`;
      if(results[0].status==='rejected') { $('overviewPipelines').textContent='—'; $('overviewModels').textContent='—'; }
    } finally {overviewBusy=false;}
  }

  let jobsBusy=false, selectedJob='';
  const jsonOptions = body => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const jobTone = status => ['succeeded','validated'].includes(status)?'status-ok':['failed','cancelled'].includes(status)?'status-err':'status-info';
  function dateLabel(value) {if(!value)return '—';const date=new Date(typeof value==='number'?value*1000:value);return Number.isNaN(date.valueOf())?'—':date.toLocaleString();}
  async function loadJobs() {
    if(jobsBusy)return;jobsBusy=true;
    try {
      const rows=await C.request('/v1/jobs?limit=100');
      $('overviewJobs').textContent=rows.length;
      $('jobsBody').innerHTML=rows.length?rows.map(row=>`<tr><td><strong>${esc(row.pipeline)}</strong><small>${esc(row.id)}</small></td><td><span class="status-pill ${jobTone(row.status)}">${esc(row.status)}</span></td><td>${esc(dateLabel(row.created_at))}</td><td><div class="job-actions"><button class="button" data-job="${esc(row.id)}" data-job-action="inspect">Inspect</button>${['queued','running'].includes(row.status)?`<button class="button is-danger" data-job="${esc(row.id)}" data-job-action="cancel">Cancel</button>`:''}${['failed','cancelled'].includes(row.status)?`<button class="button" data-job="${esc(row.id)}" data-job-action="retry">Retry</button>`:''}</div></td></tr>`).join(''):'<tr><td colspan="4"><div class="empty-state">No background jobs yet. Submit a pipeline run above.</div></td></tr>';
      if(selectedJob)await inspectJob(selectedJob);
    } catch(error){$('jobsBody').innerHTML=`<tr><td colspan="4" class="form-error">${esc(error.message)}</td></tr>`;}
    finally{jobsBusy=false;}
  }
  async function inspectJob(id) {
    const row=await C.request(`/v1/jobs/${encodeURIComponent(id)}`);selectedJob=id;
    $('jobDetail').textContent=JSON.stringify(row,null,2);$('jobDetailStatus').textContent=row.status;$('jobDetailStatus').className=`status-pill ${jobTone(row.status)}`;
  }
  $('jobForm').addEventListener('submit',async event=>{
    event.preventDefault();$('jobSubmit').disabled=true;
    try {
      const payload=JSON.parse($('jobPayload').value);
      if(!payload||Array.isArray(payload)||typeof payload!=='object')throw new Error('Job payload must be a JSON object.');
      const row=await C.request(`/v1/jobs/${encodeURIComponent($('jobPipeline').value)}`,jsonOptions(payload));
      selectedJob=row.job_id;$('jobStatus').textContent=`Queued job ${row.job_id}`;C.notify('Background job submitted.');await loadJobs();
    } catch(error){$('jobStatus').textContent=error.message;C.notify(error.message,true);}
    finally{$('jobSubmit').disabled=false;}
  });
  $('jobsBody').addEventListener('click',async event=>{
    const button=event.target.closest('[data-job-action]');if(!button)return;button.disabled=true;
    try{const id=button.dataset.job;if(button.dataset.jobAction==='inspect')await inspectJob(id);else{const row=await C.request(`/v1/jobs/${encodeURIComponent(id)}/${button.dataset.jobAction}`,{method:'POST'});if(row.job_id)selectedJob=row.job_id;await loadJobs();C.notify(button.dataset.jobAction==='cancel'?'Job cancelled.':'Retry queued.');}}
    catch(error){C.notify(error.message,true);}finally{button.disabled=false;}
  });
  $('refreshJobs').addEventListener('click',loadJobs);
  async function jobPipelines(){try{const rows=await C.request('/v1/pipelines');const selected=$('jobPipeline').value;$('jobPipeline').replaceChildren(...rows.map(row=>new Option(row.tags?.label||row.name,row.name)));if(rows.some(row=>row.name===selected))$('jobPipeline').value=selected;}catch(error){$('jobStatus').textContent=error.message;}}

  // OpenAPI is the source of truth for paths, parameter types and request bodies.
  let schema=null, endpoints=[], currentEndpoint=null, schemaPromise=null, requestController=null, lastCurl='', responseText='', responseDownload='';
  function resolve(value) {
    if(!value?.$ref)return value||{};
    if(!value.$ref.startsWith('#/'))return {};
    return value.$ref.slice(2).split('/').reduce((node,key)=>node?.[key.replace(/~1/g,'/').replace(/~0/g,'~')],schema)||{};
  }
  function example(spec,depth=0) {
    const item=resolve(spec);if(depth>4)return null;
    if(item.example!==undefined)return item.example;
    if(item.default!==undefined)return item.default;
    if(item.enum)return item.enum[0];
    if(item.anyOf||item.oneOf)return example((item.anyOf||item.oneOf).find(x=>x.type!=='null')||{},depth+1);
    if(item.allOf)return Object.assign({},...item.allOf.map(x=>example(x,depth+1)));
    if(item.properties)return Object.fromEntries(Object.entries(item.properties).filter(([,v])=>resolve(v).format!=='binary').map(([key,value])=>[key,example(value,depth+1)]));
    if(item.type==='array')return [];
    if(item.type==='boolean')return false;
    if(['number','integer'].includes(item.type))return item.minimum??0;
    if(item.type==='object')return {};
    return '';
  }
  async function ensureSchema() {
    if(schema)return;
    if(schemaPromise)return schemaPromise;
    schemaPromise=(async()=>{
      schema=await C.request('/openapi.json');
      endpoints=Object.entries(schema.paths).flatMap(([path,methods])=>Object.entries(methods).filter(([method])=>['get','post','put','patch','delete','head','options'].includes(method)).map(([method,operation])=>({path,method:method.toUpperCase(),operation,parameters:[...(methods.parameters||[]),...(operation.parameters||[])].map(resolve)})));
      renderEndpoints(); selectEndpoint(endpoints.findIndex(item=>item.path==='/healthz'));
    })();
    try{await schemaPromise;}catch(error){schema=null;$('endpointList').textContent=`Schema unavailable: ${error.message}`;throw error;}finally{schemaPromise=null;}
  }
  function renderEndpoints() {
    const query=$('endpointSearch').value.toLowerCase().trim();let count=0;
    $('endpointList').innerHTML=endpoints.map((item,index)=>{
      if(!`${item.method} ${item.path} ${item.operation.summary||''} ${(item.operation.tags||[]).join(' ')}`.toLowerCase().includes(query))return '';
      count++;return `<button type="button" class="endpoint-item${currentEndpoint===item?' selected':''}" data-endpoint="${index}" aria-label="${esc(`${item.method} ${item.path}`)}"><span class="method method-${item.method.toLowerCase()}">${item.method}</span><code>${esc(item.path)}</code><small>${esc(item.operation.summary||'')}</small></button>`;
    }).join('')||'<p class="footnote">No endpoints match this search.</p>';
    $('endpointCount').textContent=count;
  }
  function inputForParameter(parameter,index) {
    const spec=resolve(parameter.schema), id=`apiParam${index}`;
    const value=parameter.in==='path'?'':(spec.default??'');
    const required=parameter.required?' required':'';
    const control=spec.enum?`<select id="${id}"${required}><option value="">${parameter.required?'Choose a value':'Not set'}</option>${spec.enum.map(v=>`<option value="${esc(v)}"${String(v)===String(value)?' selected':''}>${esc(v)}</option>`).join('')}</select>`:`<input id="${id}" value="${esc(value)}" type="${['integer','number'].includes(spec.type)?'number':'text'}"${spec.type==='number'?' step="any"':''}${spec.minimum!=null?` min="${esc(spec.minimum)}"`:''}${spec.maximum!=null?` max="${esc(spec.maximum)}"`:''}${required} placeholder="${esc(parameter.description||spec.type||'Value')}">`;
    return `<label>${esc(parameter.name)} <small>${esc(parameter.in)}${parameter.required?' · required':''}</small>${control}</label>`;
  }
  function binaryField(spec) {
    const item=resolve(spec);
    return item.format==='binary'||Boolean(item.contentMediaType)||(item.anyOf||item.oneOf||[]).some(binaryField);
  }
  function requestContent(){return resolve(currentEndpoint?.operation.requestBody).content||{};}
  function selectEndpoint(index) {
    currentEndpoint=endpoints[index]||endpoints[0];if(!currentEndpoint)return;
    $('apiRequestError').textContent='';$('apiPath').value=currentEndpoint.path;$('apiMethod').textContent=currentEndpoint.method;$('apiMethod').className=`method method-${currentEndpoint.method.toLowerCase()}`;
    $('apiTag').textContent=(currentEndpoint.operation.tags||['PROGRAM API']).join(' / ');$('apiSummary').textContent=currentEndpoint.operation.summary||currentEndpoint.path;$('apiDescription').textContent=currentEndpoint.operation.description||'Fill in the request values, then send it to the current server.';
    $('apiParameters').innerHTML=currentEndpoint.parameters.map(inputForParameter).join('');
    $('apiBodyLabel').hidden=true;$('apiFileFields').replaceChildren();
    const content=requestContent();const json=content['application/json'];
    if(json){$('apiBodyLabel').hidden=false;$('apiBody').required=Boolean(resolve(currentEndpoint.operation.requestBody).required);$('apiBody').value=JSON.stringify(json.example??example(json.schema),null,2);}
    else{$('apiBody').required=false;$('apiBody').value='';}
    const multipart=content['multipart/form-data']||content['application/x-www-form-urlencoded'];
    if(multipart){const spec=resolve(multipart.schema);$('apiFileFields').innerHTML=Object.entries(spec.properties||{}).map(([key,value],i)=>{const field=resolve(value),required=(spec.required||[]).includes(key),binary=binaryField(field);return `<label>${esc(key)}${required?' · required':''}<input data-form-field="${esc(key)}" id="apiFile${i}" type="${binary?'file':'text'}"${required?' required':''}${!binary?` value="${esc(field.default??'')}"`:''}></label>`;}).join('');}
    renderEndpoints();
  }
  $('endpointSearch').addEventListener('input',renderEndpoints);
  $('endpointList').addEventListener('click',event=>{const button=event.target.closest('[data-endpoint]');if(button)selectEndpoint(Number(button.dataset.endpoint));});
  async function inspectApiPath(path) {
    C.url(path);
    await ensureSchema();
    const url=new URL(path,location.origin);
    let captures=[];
    const index=endpoints.findIndex(item=>{
      if(item.method!=='GET')return false;
      const names=[];
      const expression=item.path.split('/').map(segment=>{
        if(/^\{[^}]+\}$/.test(segment)){names.push(segment.slice(1,-1));return '([^/]+)';}
        return segment.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
      }).join('/');
      const match=url.pathname.match(new RegExp(`^${expression}$`));
      if(!match)return false;
      captures=names.map((name,i)=>[name,decodeURIComponent(match[i+1])]);return true;
    });
    if(index<0)throw new Error('This API path is not in the running schema.');
    selectEndpoint(index);
    for(const [i,param]of currentEndpoint.parameters.entries()){
      const value=param.in==='path'?captures.find(([name])=>name===param.name)?.[1]:param.in==='query'?url.searchParams.get(param.name):null;
      if(value!=null)$(`apiParam${i}`).value=value;
    }
    location.hash='api';
    $('apiForm').requestSubmit();
  }
  document.addEventListener('click',event=>{
    const link=event.target.closest('[data-api-path]');if(!link)return;
    event.preventDefault();inspectApiPath(link.dataset.apiPath).catch(error=>C.notify(error.message,true));
  });
  const shellQuote=value=>`'${String(value).replace(/'/g,"'\\''")}'`;
  function buildRequest() {
    let path=$('apiPath').value.trim();const headers=JSON.parse($('apiHeaders').value||'{}');
    if(!headers||Array.isArray(headers)||typeof headers!=='object'||Object.values(headers).some(value=>typeof value!=='string'))throw new Error('Request headers must be a JSON object with string values.');
    const query=new URLSearchParams();
    for(const [index,param] of (currentEndpoint?.parameters||[]).entries()) {
      const value=$(`apiParam${index}`).value;
      if(!value&&param.required)throw new Error(`Provide ${param.name}.`);
      if(!value)continue;
      if(param.in==='path')path=path.replaceAll(`{${param.name}}`,encodeURIComponent(value));
      if(param.in==='query')query.append(param.name,value);
      if(param.in==='header')headers[param.name]=value;
    }
    if(/\{[^}]+\}/.test(path))throw new Error('Fill in all path parameters.');
    if(path!=='/'&&!/^\/(v1(?:\/|$)|healthz(?:\?|$)|readyz(?:\?|$)|metrics(?:\?|$)|openapi\.json(?:\?|$))/.test(path))throw new Error('Choose a program API endpoint on this server.');
    C.url(path);if(query.size)path+=`${path.includes('?')?'&':'?'}${query}`;
    const method=$('apiMethod').textContent, options={method,headers,timeoutMs:90000,redirect:'manual'},curl=['curl','-X',method,shellQuote(C.url(path))];
    if(!$('apiBodyLabel').hidden&&!['GET','HEAD'].includes(method)) {
      const value=$('apiBody').value.trim();
      if(value){JSON.parse(value);headers['Content-Type']='application/json';options.body=value;curl.push('--data-raw',shellQuote(value));}
    }
    const fields=[...$('apiFileFields').querySelectorAll('[data-form-field]')];
    if(fields.length){const form=new FormData();for(const input of fields){const name=input.dataset.formField;if(input.type==='file'){if(input.files[0]){form.append(name,input.files[0]);curl.push('-F',shellQuote(`${name}=@${input.files[0].name}`));}}else if(input.value){form.append(name,input.value);curl.push('-F',shellQuote(`${name}=${input.value}`));}}if(requestContent()['application/x-www-form-urlencoded'])options.body=new URLSearchParams(form);else{options.body=form;for(const name of Object.keys(headers))if(name.toLowerCase()==='content-type')delete headers[name];}}
    for(const [key,value]of Object.entries(headers))curl.push('-H',shellQuote(`${key}: ${value}`));
    return{path,method,options,curl:curl.join(' ')};
  }
  $('apiForm').addEventListener('submit',async event=>{
    event.preventDefault();if(requestController)return;$('apiRequestError').textContent='';
    try {
      const request=buildRequest();lastCurl=request.curl;requestController=new AbortController();$('apiSend').disabled=true;$('apiCancel').hidden=false;
      $('apiResponseStatus').textContent='Sending…';$('apiResponseStatus').className='status-pill status-info';$('apiResponseTime').textContent='';
      const start=performance.now(),{response,body,blob}=await C.raw(request.path,{...request.options,signal:requestController.signal});
      if(responseDownload){URL.revokeObjectURL(responseDownload);responseDownload='';}
      $('apiDownload').hidden=!blob;
      if(blob){responseDownload=URL.createObjectURL(blob);$('apiDownload').href=responseDownload;$('apiDownload').download=new URL(request.path,location.origin).pathname.split('/').pop()||'response';}
      $('apiResponseTime').textContent=`${Math.round(performance.now()-start)} ms`;
      responseText=response.type==='opaqueredirect'?'This endpoint redirects to another interface. Use Services & runtime to open that interface.':typeof body==='string'?body:JSON.stringify(body,null,2);
      $('apiResponse').textContent=responseText.length>200000?`${responseText.slice(0,200000)}\n… Display limited to 200,000 characters.`:responseText;
      const label=response.type==='opaqueredirect'?'Redirect':`${response.status} ${response.statusText}`;
      $('apiResponseStatus').textContent=label;$('apiResponseStatus').className=`status-pill ${response.ok?'status-ok':'status-err'}`;
      $('apiResponseHeaders').textContent=[...response.headers].map(([key,value])=>`${key}: ${value}`).join('\n')||'No readable response headers.';
      const row=document.createElement('div');row.innerHTML=`<span class="method method-${request.method.toLowerCase()}">${request.method}</span><code>${esc(new URL(request.path,location.origin).pathname)}</code><span>${esc(label)}</span><time>${esc(new Date().toLocaleTimeString())}</time>`;
      if(!$('apiHistory').querySelector('div'))$('apiHistory').replaceChildren();$('apiHistory').prepend(row);while($('apiHistory').children.length>15)$('apiHistory').lastElementChild.remove();
    } catch(error){$('apiRequestError').textContent=error.message;$('apiResponse').textContent=error.message;responseText='';$('apiResponseStatus').textContent='Request failed';$('apiResponseStatus').className='status-pill status-err';}
    finally{requestController=null;$('apiSend').disabled=false;$('apiCancel').hidden=true;}
  });
  $('apiCancel').addEventListener('click',()=>requestController?.abort());
  async function copy(text){if(!text){C.notify('Send a request first.',true);return;}try{await navigator.clipboard.writeText(text);C.notify('Copied to clipboard.');}catch{C.notify('Clipboard access is unavailable. Select and copy the response text.',true);}}
  $('apiCopyResponse').addEventListener('click',()=>copy(responseText));
  $('apiCopyCurl').addEventListener('click',()=>{try{copy(lastCurl||buildRequest().curl);}catch(error){C.notify(error.message,true);}});

  async function loadGuide(){try{const data=await C.request('/v1/interfaces');$('guidePorts').innerHTML=(data.ports||[]).map(row=>`<tr><td>${esc(row.name)}</td><td><code>${esc(row.host)}</code></td><td>${esc(row.protocol)}</td><td><code>${esc(row.override||'—')}</code></td></tr>`).join('');}catch(error){$('guidePorts').innerHTML=`<tr><td colspan="4" class="form-error">${esc(error.message)}</td></tr>`;}}
  document.addEventListener('inferops:navigate',event=>{
    if(event.detail==='overview')refreshOverview();
    if(event.detail==='jobs'){jobPipelines();loadJobs();}
    if(event.detail==='api')ensureSchema().catch(error=>C.notify(error.message,true));
    if(event.detail==='guide')loadGuide();
  });
  document.addEventListener('inferops:refresh',()=>{
    refreshOverview();if(C.view==='jobs'){jobPipelines();loadJobs();}if(C.view==='guide')loadGuide();if(C.view==='api')ensureSchema().catch(error=>C.notify(error.message,true));
  });
  C.poll(()=>{if(C.view==='overview')return refreshOverview();if(C.view==='jobs')return loadJobs();});
  window.addEventListener('beforeunload',()=>{requestController?.abort();if(responseDownload)URL.revokeObjectURL(responseDownload);});
  navigate(true);if(C.view!=='overview')refreshOverview();
})();
