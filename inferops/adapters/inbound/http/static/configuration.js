/* Schema-backed controls over the project's existing YAML documents. */
(() => {
  "use strict";
  const clone = value => JSON.parse(JSON.stringify(value));
  const title = key => String(key).replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const button = (text, action) => {
    const node = element("button", text, "button subtle");
    node.type = "button"; node.addEventListener("click", action); return node;
  };
  window.InferOpsConfigForm = class {
    constructor(container, entry, catalog, onChange) {
      this.container = container;
      this.data = clone(entry.data);
      this.schema = entry.schema;
      this.catalog = catalog;
      this.onChange = onChange;
      this.openGroups = new Map();
      this.render();
    }
    resolve(schema) {
      if (!schema?.$ref) return schema || {};
      const value = schema.$ref.slice(2).split("/").reduce((node, key) => node?.[key], this.schema);
      return {...value, ...Object.fromEntries(Object.entries(schema).filter(([key]) => key !== "$ref"))};
    }
    type(value) { return value === null ? "null" : Array.isArray(value) ? "array" : typeof value; }
    branches(schema) { return (schema.anyOf || schema.oneOf || []).map(item => this.resolve(item)); }
    matches(schema, value) {
      const type = this.type(value);
      return schema.type === type || (type === "number" && schema.type === "integer") ||
        (!schema.type && schema.properties && type === "object");
    }
    chosen(schema, value) {
      const resolved = this.resolve(schema);
      const branches = this.branches(resolved);
      return branches.length ? (branches.find(item => this.matches(item, value)) || branches[0]) : resolved;
    }
    initial(schema, depth = 0) {
      const item = this.resolve(schema);
      if (item.default !== undefined && item.default !== null) return clone(item.default);
      if (depth > 12) return null;
      const choices = this.branches(item);
      if (choices.length) return this.initial(choices.find(choice => choice.type !== "null") || choices[0], depth + 1);
      if (item.enum) return item.enum[0];
      if (item["x-reference"]) {
        const names = this.referenceOptions(item["x-reference"]);
        return names.length === 1 ? names[0] : "";
      }
      if (item.type === "object" || item.properties) {
        return Object.fromEntries((item.required || []).map(key => [key, this.initial(item.properties?.[key] || {}, depth + 1)]));
      }
      if (item.type === "array") return Array.from({length:item.minItems || 0}, () => this.initial(item.items || {}, depth + 1));
      if (item.type === "boolean") return false;
      if (["integer", "number"].includes(item.type)) return item.minimum ?? 0;
      if (item.type === "null") return null;
      return "";
    }
    changed(redraw = false) {
      this.onChange();
      if (redraw) this.render();
    }
    render() {
      for (const group of this.container.querySelectorAll("details[data-config-group]")) {
        this.openGroups.set(group.dataset.configGroup, group.open);
      }
      this.container.replaceChildren(this.control("Settings", this.data, this.schema,
        value => { this.data = value; }, [], true));
    }
    read() {
      for (const input of this.container.querySelectorAll("input, select, textarea")) {
        if (!input.checkValidity()) {
          for (let parent = input.parentElement; parent && parent !== this.container; parent = parent.parentElement) {
            if (parent.tagName === "DETAILS") parent.open = true;
          }
          input.reportValidity();
          throw new Error("Check the highlighted configuration field.");
        }
      }
      return clone(this.data);
    }
    referenceOptions(section) {
      return [...new Set((this.catalog?.files || []).filter(item => item.section === section).map(item => item.logical_name))];
    }
    options(path, schema = {}) {
      if (schema["x-reference"]) return this.referenceOptions(schema["x-reference"]);
      const key = path.at(-1);
      const parent = path.at(-2);
      let section;
      const stageParameter = parent === "params" && (path[0] === "source" && path.length === 3 ||
        ["stages", "sinks"].includes(path[0]) && path.length === 4);
      if (stageParameter) section = {model:"models", fallback_model:"models", fusion:"fusion", alerts:"alerts"}[key];
      if (typeof key === "number" && path.length === 2) section = {
        prewarm_models:"models", pipelines:"pipelines", prewarm_pipelines:"pipelines",
      }[parent];
      if (typeof key === "number" && parent === "model_filter" && path[0] === "tracking") section = "models";
      if (path.length === 2 && path[0] === "scenario" && key === "pipeline") section = "pipelines";
      if (!section) return [];
      return this.referenceOptions(section);
    }
    group(path) {
      const group = element("details", undefined, "config-group");
      group.dataset.configGroup = JSON.stringify(path);
      group.open = this.openGroups.get(group.dataset.configGroup) ?? path.length < 2;
      return group;
    }
    stageChoices(category) {
      return Object.entries(this.schema["x-stages"] || {}).filter(([, item]) => !item.deprecated && item.category === category);
    }
    stageParameters(object) {
      const params = this.schema["x-stages"]?.[object.stage]?.params || {type:"object", additionalProperties:true};
      if (object.stage === "infer" && object.params?.model_modes && Object.keys(object.params.model_modes).length) {
        return {...params, required:[]};
      }
      return params;
    }
    stageDefaults(object) {
      const schema = this.stageParameters(object), params = this.initial(schema);
      for (const [key, spec] of Object.entries(schema.properties || {})) {
        if (spec.default !== undefined) params[key] = this.initial(spec);
      }
      return params;
    }
    stageSelector(object, set, path) {
      const category = path[0] === "source" ? "source" : path[0] === "sinks" ? "sink" : "processor";
      const label = element("label", "Step type", "config-field"), select = element("select");
      const choices = this.stageChoices(category), known = this.schema["x-stages"]?.[object.stage];
      if (known && !choices.some(([name]) => name === object.stage)) choices.unshift([object.stage, known]);
      choices.forEach(([name, item]) => select.add(new Option(item.label, name)));
      select.add(new Option("Custom project stage", "__custom__"));
      select.value = known ? object.stage : "__custom__";
      select.setAttribute("aria-label", [...path, "stage type"].join("."));
      select.dataset.configPath = [...path, "stage"].join(".");
      select.addEventListener("change", () => {
        const previous = known ? object.stage : "__custom__";
        if (Object.keys(object.params || {}).length && !confirm("Changing this step replaces its parameter settings. Continue?")) {
          select.value = previous; return;
        }
        object.stage = select.value === "__custom__" ? "" : select.value;
        object.params = this.stageDefaults(object);
        set(object);
        this.openGroups.set(JSON.stringify([...path, "params"]), true);
        this.changed(true);
      });
      label.append(select);
      if (!known) {
        const plugin = element("input"); plugin.value = object.stage;
        plugin.placeholder = ".plugins.processing:MyStage";
        plugin.required = true; plugin.setAttribute("aria-label", [...path, "stage"].join("."));
        plugin.addEventListener("input", () => {object.stage = plugin.value; set(object); this.changed();});
        label.append(plugin);
      }
      return label;
    }
    control(key, value, spec, set, path, root = false) {
      const original = this.resolve(spec), schema = this.chosen(original, value);
      const type = schema.type || this.type(value);
      const container = element("div", undefined, "config-field");
      const labelText = root ? "Settings" : typeof key === "number" ? (value?.name ? title(value.name) : `Item ${key + 1}`)
        : (schema.title && !/(Spec|Decl|Settings)$/.test(schema.title) ? schema.title : title(key));
      const alternatives = this.branches(original);
      if (alternatives.length > 1 || (!schema.type && !schema.properties && !schema.enum && !this.options(path, schema).length)) {
        const select = element("select"); select.setAttribute("aria-label", `${path.join(".")} value type`);
        const choices = alternatives.length ? alternatives : ["string", "number", "boolean", "object", "array", "null"].map(type => ({type}));
        choices.forEach((choice, index) => select.add(new Option(title(choice.type || "object"), index)));
        const selected = choices.findIndex(choice => this.matches(choice, value));
        select.value = String(Math.max(0, selected));
        select.addEventListener("change", () => {set(this.initial(choices[Number(select.value)])); this.changed(true);});
        const row = element("div", undefined, "config-type"); row.append(element("small", `${labelText} · value type`), select);
        container.append(row);
      }
      if (type === "object" || schema.properties) {
        const object = value && typeof value === "object" && !Array.isArray(value) ? value : {};
        const group = root ? element("div") : this.group(path);
        if (!root) group.append(element("summary", labelText));
        const properties = schema.properties || {};
        const propertySchema = child => schema["x-stage-declaration"] && child === "params" ? this.stageParameters(object)
          : properties[child] || (typeof schema.additionalProperties === "object" ? schema.additionalProperties : {});
        for (const [child, item] of Object.entries(object)) {
          const row = element("div", undefined, "config-row");
          if (schema["x-stage-declaration"] && child === "stage") row.append(this.stageSelector(object, set, path));
          else row.append(this.control(child, item, {...propertySchema(child), "x-required":(schema.required || []).includes(child)},
            next => {Object.defineProperty(object, child, {value:next, writable:true, enumerable:true, configurable:true}); set(object);}, [...path, child]));
          if (!(schema.required || []).includes(child)) row.append(button("Remove", () => {delete object[child]; set(object); this.changed(true);}));
          group.append(row);
        }
        const missing = Object.keys(properties).filter(child => !Object.hasOwn(object, child) && !properties[child].deprecated);
        if (!Object.keys(object).length && !missing.length && schema.additionalProperties === false) {
          group.append(element("p", "This step has no additional settings.", "footnote"));
        }
        if (missing.length || schema.additionalProperties !== false && !Object.keys(properties).length) {
          const add = element("div", undefined, "config-add");
          const input = element(missing.length ? "select" : "input");
          input.setAttribute("aria-label", `${path.join(".") || "settings"} new field`);
          if (missing.length) missing.forEach(child => input.add(new Option(properties[child].title || title(child), child)));
          else input.placeholder = "Setting name";
          add.append(input, button("Add field", () => {
            const child = input.value.trim();
            if (!child || Object.hasOwn(object, child)) {input.setCustomValidity("Enter a new, unique setting name."); input.reportValidity(); return;}
            input.setCustomValidity("");
            const childSchema = propertySchema(child);
            Object.defineProperty(object, child, {value:this.initial(childSchema), writable:true, enumerable:true, configurable:true});
            set(object); this.changed(true);
          }));
          input.addEventListener("input", () => input.setCustomValidity(""));
          group.append(add);
        }
        container.append(group);
      } else if (type === "array") {
        const list = Array.isArray(value) ? value : [];
        const group = this.group(path);
        group.append(element("summary", `${labelText} · ${list.length} items`));
        list.forEach((item, index) => {
          const row = element("div", undefined, "config-row");
          row.append(this.control(index, item, schema.items || {}, next => {list[index] = next; set(list);}, [...path, index]));
          const actions = element("div", undefined, "config-item-actions");
          actions.append(button("Remove", () => {list.splice(index, 1); set(list); this.changed(true);}));
          if (index) actions.append(button("Move up", () => {[list[index - 1], list[index]] = [list[index], list[index - 1]]; set(list); this.changed(true);}));
          row.append(actions); group.append(row);
        });
        if (this.resolve(schema.items)["x-stage-declaration"]) {
          const row = element("div", undefined, "config-add"), select = element("select");
          const category = path[0] === "sinks" ? "sink" : "processor";
          this.stageChoices(category).forEach(([name, item]) => select.add(new Option(item.label, name)));
          select.add(new Option("Custom project stage", "__custom__"));
          if (category === "processor") select.value = "infer";
          select.setAttribute("aria-label", `${path.join(".")} new step type`);
          row.append(select, button(category === "sink" ? "Add output" : "Add processing step", () => {
            const alias = select.value === "__custom__" ? "" : select.value;
            const names = new Set([this.data.source, ...(this.data.stages || []), ...(this.data.sinks || [])].filter(Boolean).map(item => item.name));
            const base = alias || "custom_stage";
            let name = base, suffix = 2;
            while (names.has(name)) name = `${base}_${suffix++}`;
            const item = {name, stage:alias}; item.params = this.stageDefaults(item);
            list.push(item); set(list);
            this.openGroups.set(JSON.stringify([...path, list.length - 1]), true);
            this.openGroups.set(JSON.stringify([...path, list.length - 1, "params"]), true);
            this.changed(true);
          })); group.append(row);
        } else group.append(button("Add item", () => {list.push(this.initial(schema.items || {})); set(list); this.changed(true);}));
        container.append(group);
      } else if (type === "null") {
        container.append(element("p", `${labelText}: not set`, "footnote"));
      } else {
        const label = element("label", labelText);
        let input;
        const choices = schema.enum || this.options(path, schema);
        // Environment references remain editable strings even for numeric settings.
        const interpolation = typeof value === "string" && value.includes("${");
        if (choices.length && !interpolation) {
          input = element("select");
          if (!choices.map(String).includes(String(value))) input.add(new Option(String(value ?? ""), String(value ?? "")));
          choices.forEach(choice => input.add(new Option(String(choice), String(choice))));
          input.value = String(value ?? "");
          input.addEventListener("change", () => {set(schema.enum ? choices.find(choice => String(choice) === input.value) : input.value); this.changed();});
        } else {
          const secret = /secret|password|access_key/i.test(String(key));
          input = element(type === "string" && String(value).length > 120 && !secret ? "textarea" : "input");
          if (input.tagName === "INPUT") {
            input.type = interpolation ? "text" : type === "boolean" ? "checkbox" : ["integer", "number"].includes(type) ? "number" : "text";
            if (secret) input.type = "password";
          }
          if (input.type === "checkbox") input.checked = Boolean(value);
          else input.value = value ?? "";
          if (input.type === "number") {
            input.required = true; input.step = type === "integer" ? "1" : "any";
            if (schema.minimum != null) input.min = schema.minimum;
            if (schema.maximum != null) input.max = schema.maximum;
          }
          input.addEventListener("input", () => {
            if (input.type === "number" && !input.validity.valid) {this.onChange(); return;}
            set(input.type === "checkbox" ? input.checked : input.type === "number" ? Number(input.value) : input.value);
            this.changed();
          });
        }
        input.setAttribute("aria-label", path.join("."));
        if (schema["x-required"]) input.required = true;
        if (schema.minLength != null) input.minLength = schema.minLength;
        input.dataset.configPath = path.join(".");
        label.append(input); container.append(label);
      }
      if (schema.description) container.append(element("small", schema.description.split("\n")[0], "footnote"));
      return container;
    }
  };
})();
