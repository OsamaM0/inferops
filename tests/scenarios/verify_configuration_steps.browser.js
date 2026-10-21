/* Run only against serve_unified_console.py (disposable configs + tiny model):
   playwright-cli -s=inferops-config run-code --filename=tests/scenarios/verify_configuration_steps.browser.js
   This creates a pipeline in that server's temporary configuration directory. */
async page => {
  const origin = "http://127.0.0.1:18083", errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  await page.goto(`${origin}/ui/?configuration-scenario=${Date.now()}#settings`);
  await page.locator('#configForm [data-config-path="name"]').waitFor();
  await page.setViewportSize({width:1440, height:1100});
  await page.evaluate(() => window.InferOps.savePreferences({refresh:0}));
  const name = `inspection_${Date.now()}`;
  await page.locator("#configNewBtn").click();
  await page.locator("#configNewSection").selectOption("pipelines");
  await page.locator("#configNewName").fill(name);
  await page.locator('#configNewForm button[type="submit"]').click();
  await page.locator("#configBadge").filter({hasText:"New draft"}).waitFor();
  const addStep = async type => {
    await page.getByRole("combobox", {name:"stages new step type", exact:true}).selectOption(type);
    await page.getByRole("button", {name:"Add processing step", exact:true}).click();
  };
  await addStep("decode_any");
  await page.locator('[data-config-path="stages.0.params.every_n"]').fill("2");
  await addStep("infer");
  assert(await page.locator('[data-config-path="stages.0.params.every_n"]').inputValue() === "2", "Adding a step reset an existing value");
  assert(await page.locator('[data-config-path="stages.1.params.model"]').inputValue() === "tiny_classifier", "Logical model choice missing");
  await page.getByRole("button", {name:"Move up", exact:true}).click();
  assert(await page.locator('[data-config-path="stages.0.params.model"]').inputValue() === "tiny_classifier", "Reordering lost the model reference");
  assert(await page.locator('[data-config-path="stages.1.params.every_n"]').inputValue() === "2", "Reordering lost a numeric value");
  await page.getByRole("button", {name:"Move up", exact:true}).click();
  await addStep("__custom__");
  await page.getByRole("textbox", {name:"stages.2.stage", exact:true}).fill("inferops.adapters.outbound.stages.route_by:RouteBy");
  const customField = page.getByRole("textbox", {name:"stages.2.params new field", exact:true});
  await customField.fill("when");
  await customField.locator("..").getByRole("button", {name:"Add field", exact:true}).click();
  await page.locator('[data-config-path="stages.2.params.when"]').fill("True");
  const newField = page.getByRole("combobox", {name:"settings new field", exact:true});
  await newField.selectOption("sinks");
  await newField.locator("..").getByRole("button", {name:"Add field", exact:true}).click();
  await page.getByRole("combobox", {name:"sinks new step type", exact:true}).selectOption("sink_ws");
  await page.getByRole("button", {name:"Add output", exact:true}).click();
  assert(await page.locator('[data-config-path="sinks.0.params.channel"]').inputValue() === "live.{stream_id}", "Output defaults missing");
  await page.locator("#configValidateBtn").click();
  await page.locator("#configStatus").filter({hasText:"Configuration is valid"}).waitFor();
  assert((await page.request.get(`${origin}/v1/configs/pipelines/${name}`)).status() === 404, "Validation wrote a file");
  await page.locator("#configSaveReloadBtn").click();
  await page.locator("#configStatus").filter({hasText:"Saved and reloaded"}).waitFor();
  const config = await (await page.request.get(`${origin}/v1/configs/pipelines/${name}`)).json();
  assert(config.data.stages[0].params.every_n === 2, "A numeric setting became a string");
  assert(config.data.stages[1].params.model === "tiny_classifier", "Saved model reference is incorrect");
  assert(config.data.stages[2].params.when === "True", "Custom plugin parameters changed");
  assert(await page.locator("#configEditor").isHidden(), "A file editor was required");
  const image = await page.evaluate(() => {
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 32;
    return canvas.toDataURL("image/jpeg").split(",")[1];
  });
  const inference = await page.request.post(`${origin}/v1/infer/${name}`, {multipart:{
    image:{name:"sample.jpg", mimeType:"image/jpeg", buffer:Buffer.from(image, "base64")},
  }});
  assert(inference.ok(), `Created pipeline could not run: ${await inference.text()}`);
  assert((await inference.json()).some(item => item.results?.length), "Pipeline returned no model result");
  await page.setViewportSize({width:390, height:844});
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Mobile form overflows");
  const pluginField = await page.evaluate(() => {
    const container = document.createElement("div");
    const form = new window.InferOpsConfigForm(container, {
      data:{params:{model:"custom_checkpoint"}},
      schema:{type:"object", properties:{params:{type:"object", additionalProperties:true}}},
    }, {files:[{section:"models", logical_name:"tiny_classifier"}]}, () => {});
    const input = container.querySelector('[data-config-path="params.model"]');
    if (input.tagName !== "INPUT") return false;
    input.value = "another_checkpoint"; input.dispatchEvent(new Event("input"));
    return form.read().params.model === "another_checkpoint";
  });
  assert(pluginField, "An unrelated plugin field was restricted to InferOps model names");
  assert(errors.length === 0, errors.join("\n"));
  return {passed:true, pipeline:name, formOnly:true, typedParameters:true, reordering:true, customPlugin:true, inference:true, mobile:true, genericFields:true};
}
