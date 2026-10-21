# Architecture diagrams

Open [index.html](index.html) for the offline gallery with fit and zoom controls.
Use the [documentation tour](../README.md), [Hexagonal architecture](../architecture.md),
[full lifecycle](../lifecycle.md) and [Redpanda guide](../messaging.md) for the
source-linked explanations. The [package structure](../structure.md) maps the
source modules to their responsibilities.

## Reading order

| Drawing | Contents | Companion guide |
| --- | --- | --- |
| [12 — Package layout](12-package-layout.svg) | Six top-level package directories and inward dependencies | [Structure](../structure.md) |
| [10 — Shared engine](10-shared-engine.svg) | Inbound/outbound adapters, application services and owned state | [Architecture](../architecture.md) |
| [13 — Full lifecycle](13-lifecycle.svg) | Preparation, startup, sessions, processing, effects, close and recovery | [Lifecycle](../lifecycle.md) |
| [11 — Evidence recovery](11-evidence-and-recovery.svg) | Acceptance transaction, capture spool, verified artifacts and reconciliation | [Evidence lifecycle](../lifecycle.md#9-capture-and-complete-evidence) |
| [15 — Evidence states](15-evidence-states.svg) | Capture outcomes, retained-material retry and deletion tombstone | [Evidence states](../lifecycle.md#9-capture-and-complete-evidence) |
| [14 — Redpanda](14-redpanda.svg) | Topics, keys, transactional outbox/inbox and commit boundaries | [Messaging](../messaging.md) |
| [09 — Deployment](09-deployment.svg) | Compose services, ports, model serving, volumes and metrics | [Deployment](../../deploy/README.md) |

## Update and render

Each `.mmd` is the editable Mermaid source for the same-named `.svg`. Diagrams
show logical calls and ownership; an arrow is not automatically a source-code
import or a separate process. Deployment shows PostgreSQL, persistent-disk evidence and MediaMTX defaults, with optional MinIO/S3
and optional broker, monitoring and static workers. Redis is removed.

The checked-in SVGs were rendered with Mermaid CLI **11.12.0**. With that
version installed in a development tooling environment:

```bash
mmdc -i docs/architecture/14-redpanda.mmd \
     -o docs/architecture/14-redpanda.svg \
     -c docs/architecture/mermaid-config.json
```

For custom Chromium installations, supply the CLI's Puppeteer configuration
option. Mermaid and Chromium are documentation tools, not InferOps runtime
requirements. Commit matching source/SVG changes. If SVG dimensions change,
update the gallery image's `data-width` from its SVG `viewBox` width so Fit/100%
controls remain accurate.

Keep the gallery self-contained: no CDN, remote fonts or runtime Mermaid library.
After editing, open it locally and check desktop/mobile fit, zoom, navigation,
image loading and console errors. Verify relative links and Markdown examples
against the checked-in source.
