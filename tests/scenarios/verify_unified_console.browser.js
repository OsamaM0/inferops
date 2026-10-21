/* Run against an isolated tiny_classifier server with:
   playwright-cli -s=inferops-console run-code --filename=tests/scenarios/verify_unified_console.browser.js
   The scenario creates sources/jobs and changes browser-local preferences. */
async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const navigate = async view => {
    await page.evaluate(view => { location.hash = view; }, view);
    await page.waitForFunction(view => window.InferOps.view === view, view);
  };
  const apiSelect = async (method, path) => {
    await page.locator('#endpointSearch').fill(path);
    await page.locator('#endpointList').getByRole('button', {name:`${method} ${path}`, exact:true}).click();
  };
  const send = async expected => {
    const response = page.waitForResponse(response => response.url().includes('/v1/') || response.url().endsWith('/healthz'));
    await page.locator('#apiSend').click();
    await response;
    await page.waitForFunction(() => !document.querySelector('#apiSend').disabled);
    const status = await page.locator('#apiResponseStatus').textContent();
    assert(status.startsWith(String(expected)), `Expected API ${expected}, got ${status}: ${await page.locator('#apiRequestError').textContent()}`);
  };
  await page.goto('http://127.0.0.1:18083/ui/');
  await page.evaluate(() => window.InferOps.savePreferences({...window.InferOps.defaults}));
  await page.waitForFunction(() => document.querySelector('#domainSelect').options.length > 0);
  await page.locator('#connectionLabel').filter({hasText:'Engine ready'}).waitFor();
  await page.setViewportSize({width:1440,height:1100});
  await page.screenshot({path:'.run/verification/unified-console/overview-dark.png'});
  const documentIdentity = await page.evaluate(() => {window.consoleIdentity = Math.random();return window.consoleIdentity;});
  const views = ['overview','sources','pipeline','playground','jobs','activity-view','settings','services','api','preferences','guide'];
  for (const view of views) {
    await navigate(view);
    assert(await page.locator('[data-pane]:visible').count() === 1, `Multiple visible views: ${view}`);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Desktop overflow: ${view}`);
    assert(await page.evaluate(() => window.consoleIdentity) === documentIdentity, `Navigation reloaded the document: ${view}`);
  }
  await page.locator('#guidePorts tr').filter({hasText:'MinIO Console'}).waitFor();
  await navigate('preferences');
  await page.locator('[name="theme"]').selectOption('light');
  await page.locator('[name="accent"]').selectOption('violet');
  await page.locator('[name="density"]').selectOption('compact');
  await page.locator('[name="sourceLayout"]').selectOption('list');
  await page.locator('[name="refresh"]').selectOption('0');
  await page.reload();
  assert(await page.evaluate(() => document.documentElement.dataset.theme === 'light' && document.documentElement.dataset.accent === 'violet' && document.documentElement.dataset.density === 'compact'), 'Preferences did not persist');
  await navigate('settings');
  await page.waitForFunction(() => document.querySelector('#configEditor').value.length>0);
  await page.locator('#configMode').selectOption('yaml');
  const originalConfig=await page.locator('#configEditor').inputValue();
  const configEndpoint='/v1/configs/pipelines/tiny_classifier';
  try {
    await page.locator('#configEditor').fill(`${originalConfig}\n# console-browser-save-check\n`);
    assert((await page.locator('#configStatus').textContent()).includes('Unsaved'),'Unsaved changes were not reported');
    const saved=page.waitForResponse(response=>response.url().endsWith(configEndpoint)&&response.request().method()==='PUT');
    await page.keyboard.press('Control+s');
    assert((await saved).status()===200,'Keyboard save failed');
    await page.waitForFunction(()=>!document.querySelector('#configSaveReloadBtn').disabled);
    assert((await page.locator('#configStatus').textContent()).includes('Saved'),'Configuration save was not reported');
  } finally {
    await page.evaluate(async ({path,content})=>{const response=await fetch(path,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({content,reload:true})});if(!response.ok)throw new Error('Test config restore failed');},{path:configEndpoint,content:originalConfig});
  }
  await page.screenshot({path:'.run/verification/unified-console/configuration-light.png'});
  // Palette keyboard navigation stays in the same document.
  await page.keyboard.press('Control+k');
  await page.locator('#commandInput').fill('API explorer');
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.InferOps.view === 'api');
  await page.locator('#endpointList .endpoint-item').first().waitFor();
  await apiSelect('GET','/healthz'); await send(200);
  assert(JSON.parse(await page.locator('#apiResponse').textContent()).status === 'ok', 'Health request did not reach the server');
  await apiSelect('GET','/v1/pipelines/{name}');
  await page.locator('#apiParam0').fill('missing-browser-pipeline'); await send(404);
  await page.locator('#apiParam0').fill('tiny_classifier'); await send(200);
  await apiSelect('POST','/v1/sessions');
  await page.locator('#apiBody').fill(JSON.stringify({pipeline:'tiny_classifier',stream_id:'console-browser-session'})); await send(200);
  const session = JSON.parse(await page.locator('#apiResponse').textContent());
  await apiSelect('DELETE','/v1/sessions/{session_id}');
  await page.locator('#apiParam0').fill(session.session_id);
  await page.locator('.request-headers summary').click();
  await page.locator('#apiHeaders').fill(JSON.stringify({'X-Session-Capability':session.capability})); await send(200);
  await page.locator('#apiHeaders').fill('{}');
  const imageBase64 = await page.evaluate(() => {const canvas=document.createElement('canvas');canvas.width=canvas.height=64;const context=canvas.getContext('2d');context.fillStyle='#65c9a0';context.fillRect(0,0,64,64);return canvas.toDataURL('image/jpeg').split(',')[1];});
  await apiSelect('POST','/v1/infer/{pipeline}');
  await page.locator('#apiParam0').fill('tiny_classifier');
  await page.locator('#apiFileFields input[type="file"]').setInputFiles({name:'console-test.jpg',mimeType:'image/jpeg',buffer:Buffer.from(imageBase64,'base64')});
  await send(200);
  assert((await page.locator('#apiResponse').textContent()).includes('classifications'), 'Multipart inference failed');
  await page.screenshot({path:'.run/verification/unified-console/api-explorer.png'});
  // Real source registration, processing and preview.
  await navigate('sources');
  await page.locator('#addSource').click();
  const sourceId=`console-browser-${Date.now()}`;
  await page.locator('[name="camera_id"]').fill(sourceId);
  await page.locator('[name="name"]').fill('Browser verification');
  await page.locator('#sourceKind').selectOption('image');
  await page.locator('#sourceFile').setInputFiles({name:'console-test.jpg',mimeType:'image/jpeg',buffer:Buffer.from(imageBase64,'base64')});
  await page.locator('#saveSource').click();
  await page.locator('#sourceDialog').waitFor({state:'hidden'});
  const card=page.locator('.source-card').filter({hasText:sourceId});
  await card.waitFor();
  await card.locator('.toggle').click();
  await page.locator('#globalRefresh').click();
  await page.waitForFunction(async id => {const data=await (await fetch('/v1/cameras')).json();return data.cameras.find(row=>row.camera_id===id)?.processed>0;}, sourceId);
  await page.locator('#globalRefresh').click();
  await card.locator('img').waitFor({state:'visible'});
  assert(await card.locator('img').evaluate(image=>image.naturalWidth>0), 'Source preview did not decode');
  await card.locator('.history').click();
  await page.locator('#historyBody [data-api-path]').click();
  await page.waitForFunction(() => window.InferOps.view==='api' && !document.querySelector('#apiSend').disabled);
  assert((await page.locator('#apiResponse').textContent()).includes(sourceId), 'Inline result inspection failed');
  await apiSelect('GET','/v1/cameras/{camera_id}/preview.jpg');
  await page.locator('#apiParam0').fill(sourceId); await send(200);
  await page.locator('#apiDownload').waitFor({state:'visible'});
  assert((await page.locator('#apiDownload').getAttribute('href')).startsWith('blob:'),'Binary API response was not downloadable');
  await navigate('sources');
  // Manual mode does not make background source requests; explicit refresh works.
  let sourceRequests=0;
  const countSources=request=>{if(request.url().endsWith('/v1/cameras'))sourceRequests++;};
  page.on('request',countSources);
  await page.waitForTimeout(3100);
  assert(sourceRequests===0, `Manual refresh mode still polled ${sourceRequests} times`);
  await Promise.all([page.waitForResponse(response=>response.url().endsWith('/v1/cameras')),page.locator('#globalRefresh').click()]);
  assert(sourceRequests>0,'Manual refresh did not request sources');
  page.off('request',countSources);
  // Submit and inspect a successful asynchronous run.
  await navigate('jobs');
  await page.locator('#jobPipeline').selectOption('tiny_classifier');
  await page.locator('#jobPayload').fill(JSON.stringify({stream_id:'console-browser-job',image_base64:imageBase64}));
  await page.locator('#jobSubmit').click();
  await page.waitForFunction(()=>document.querySelector('#jobStatus').textContent.startsWith('Queued job'));
  const jobId=(await page.locator('#jobStatus').textContent()).split(' ').pop();
  await page.waitForFunction(async id=>(await (await fetch(`/v1/jobs/${id}`)).json()).status==='succeeded',jobId);
  await page.locator('#refreshJobs').click();
  await page.locator('#jobDetailStatus').filter({hasText:'succeeded'}).waitFor();
  assert((await page.locator('#jobDetail').textContent()).includes(jobId),'Job details were not loaded');
  await navigate('preferences'); await page.locator('#resetPreferences').click();
  await page.setViewportSize({width:390,height:844});
  for (const view of views) {
    await navigate(view);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Mobile overflow: ${view}`);
  }
  await page.locator('#menuToggle').click();
  assert(await page.locator('#menuToggle').getAttribute('aria-expanded')==='true','Mobile navigation did not open');
  await page.locator('[data-view="sources"]').click();
  assert(await page.locator('#menuToggle').getAttribute('aria-expanded')==='false','Mobile navigation did not close');
  await page.waitForFunction(() => window.InferOps.view==='sources' && document.querySelector('#sidebar').getBoundingClientRect().right<=0);
  await page.screenshot({path:'.run/verification/unified-console/sources-mobile.png'});
  // Remove only the registration created by this scenario; its history is retained.
  await page.evaluate(async id => {const response=await fetch(`/v1/cameras/${id}`,{method:'DELETE'});if(!response.ok)throw new Error('Test source cleanup failed');},sourceId);
  assert(errors.length===0,`Browser errors: ${errors.join('; ')}`);
  return {passed:true,views:views.length,preferencesPersist:true,api:['health','path parameters','404 response','session capability','multipart inference'],sourcePreview:true,jobs:true,configSave:true,manualRefresh:true,mobile:true,errors};
}
