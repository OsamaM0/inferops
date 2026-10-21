# Static camera workers

The default remains one combined API with PostgreSQL, persistent-disk evidence and MediaMTX.
MinIO/S3 is optional for this single-owner deployment.
Static workers are optional: one API plus fixed camera shards share PostgreSQL,
S3 storage and a Triton server. Camera count, resolution and inference/evidence
rates are configurable; the example's 12-camera limit is provisional, not a
capacity recommendation. See [source control and capacity](source-control.md#capacity-what-the-calculation-means).

## Start a new isolated deployment

Prepare and qualify models first using [Triton serving](triton-serving.md).
Retain the qualified repository across worker restarts. Copy
`deploy/workers.example.yaml` to your own manifest and set its endpoints and
per-worker source limits. Credentials in camera URIs can use `${VARIABLE}`
references supplied to the owning worker. Keep the internal control token secret.

```bash
export INFEROPS_WORKER_TOKEN="$(openssl rand -hex 32)"
export INFEROPS_WORKER_MANIFEST_FILE=./deploy/workers.example.yaml
docker compose --project-directory . -f deploy/compose.yaml \
  -f deploy/compose/triton.yaml -f deploy/compose/workers.yaml \
  --profile storage --profile triton up -d --build
```

The overlay starts `camera-a` and `camera-b` with `restart: always`, separate
persistent spool volumes and the same model server. Extend services and the
manifest together for more shards. Register a source in the dashboard and select
its owner, or seed fixed source specifications in the manifest. Persisted UI
registrations resume when `autostart` is set. Moving an existing camera between
owners is deliberately unsupported; ownership cannot be changed by editing it.

Workers run `inferops worker cameras --manifest /path/workers.yaml --worker-id a`.
They process sources and evidence; they do not execute global jobs or claim the
delivery outbox. The API routes controls/previews to the owner and queries shared
history/artifacts directly. Worker endpoints require `X-Worker-Token`; expose
them only on the internal network. External API authentication remains the
existing deployment responsibility.

## Ownership and recovery

Schema version 2 adds lease/record-scope tables; version 3 adds delivery receipts.
Migrations are additive and use a PostgreSQL advisory transaction lock. A shard claims a database-time lease for
20 seconds, renews every 5 seconds and increments its generation on reacquisition.
Writes check the lease under a row lock in the same transaction as their effects.
The API scope is `api`; camera scopes are `camera-<worker_id>`.

Each process locks its own stable spool directory. Restart recovery paginates only
that scope's pending evidence and interrupts only its sessions/jobs. Artifact keys
include owner and generation; an old upload cannot overwrite the new owner's
objects. Old orphan objects are retained until a safe cleanup policy is selected.
The SQL lease does not make the local spool portable between hosts.

On uncertain/lost ownership, processing stops and readiness fails. A native
watchdog exits the process if no successful ownership renewal occurs for 60
seconds (`INFEROPS_WORKER_WATCHDOG_SECONDS`). Docker then restarts it. This handles
a stuck event loop as well as a process crash. A single offline camera reconnects
independently and does not trigger a whole-shard restart.

## Existing databases

Back up SQL, objects, project/configuration and spool material. Stop **all** old
combined and distributed owners before adoption. New roles refuse legacy unscoped
records. Select which initial scope owns that legacy bundle, then run:

```bash
inferops ownership adopt-legacy --scope api --owners-stopped
```

Use `INFEROPS_DATABASE_URL` for the intended PostgreSQL database. Adoption assigns
existing session/job/evidence documents without changing their object references;
it is idempotent and refuses live distributed leases. It does not move spool files.
Move the old spool contents into the selected scope's persistent spool directory
while all owners remain stopped, preserve permissions and verify recovery on a
copy before cutover. Splitting legacy captures across shards requires an explicit
migration; the command assigns one bundle to one scope.

## Operation and cleanup

Inspect `/readyz`, `/v1/workers`, `/v1/cameras`, camera history and `/metrics`.
An unreachable owner returns an explicit unavailable state; its previews/controls
cannot be served from another worker's local material. Completed shared artifacts
remain readable. Local notification, broker publication and each action retry
independently; only the API claims delivery rows.

Evidence cleanup is explicit and reviewable:

```bash
inferops evidence retention --before 2026-10-01T00:00:00Z --url http://localhost:8080 --worker-id a
# Add --apply only after reviewing the listed candidates.
inferops evidence prune-outbox --before 2026-10-01T00:00:00Z --url http://localhost:8080
```

The dashboard's Activity retention panel supports the same owner/cutoff preview.
Pending captures and pending delivery are protected and eligibility is rechecked
transactionally before tombstoning. Inbox identities are retained; no replay
horizon or automatic destructive retention policy is assumed.

Only one API owner is supported. API replicas, cross-host automatic reassignment,
direct multi-process CUDA sharing, hardware decoding and rollback to arbitrary
older single-owner releases are outside this layout. Stop owners and preserve
data before switching to a release proven compatible with schema version 3 and
pending spool records; volume deletion is not a rollback procedure.
