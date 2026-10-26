# Isolated backend integration evidence — 2026-10-27

This record covers one bounded local Docker integration attempt using synthetic data. It does not validate production sizing or claim that every service has an application adapter.

## Stack and isolation

- Compose project: `raama-int-20261027-a2339897`
- Images: PostgreSQL `17.11-alpine3.24`, RustFS `1.0.3`, OpenSearch `3.9.0`, Qdrant `v1.19.1`, Temporal `1.9.1`.
- Each published port was bound to `127.0.0.1` and allocated dynamically. Observed mappings were PostgreSQL `49182`, RustFS `49185`, OpenSearch `49186`, Qdrant `49189`, Temporal RPC `49180`, and Temporal UI `49181`.
- The Compose project used its own network and named volumes. It did not use the existing workbench ports or another project's services.
- The only document payloads were synthetic local test bytes. No CMB source bytes, research query, provider credential, or other financial source text was sent to a service or provider.

## Immutable image identities

Recorded with `docker image inspect --format '{{.RepoTags}}|{{.Id}}|{{.RepoDigests}}' ...` after the stack was removed:

| Image | Local image ID | Registry digest |
| --- | --- | --- |
| `postgres:17.11-alpine3.24` | `sha256:266d3c257594e195b309ae9d1805e54c788f1d7b375f56bf6c367aac6c917e7b` | `postgres@sha256:266d3c257594e195b309ae9d1805e54c788f1d7b375f56bf6c367aac6c917e7b` |
| `rustfs/rustfs:1.0.3` | `sha256:9ca9559533b537d10482fe923288b676ad08f92a7069d9a17327fbae3cab688b` | `rustfs/rustfs@sha256:9ca9559533b537d10482fe923288b676ad08f92a7069d9a17327fbae3cab688b` |
| `opensearchproject/opensearch:3.9.0` | `sha256:adfa61f85025d06b4aeb562e7e74fde7e31c437039c93c3862c17e9acebd6c7c` | `opensearchproject/opensearch@sha256:adfa61f85025d06b4aeb562e7e74fde7e31c437039c93c3862c17e9acebd6c7c` |
| `qdrant/qdrant:v1.19.1` | `sha256:12364fe851b9f17356fc88189fc06d1b521262e04659ec7345975b00c9246a10` | `qdrant/qdrant@sha256:12364fe851b9f17356fc88189fc06d1b521262e04659ec7345975b00c9246a10` |
| `temporalio/temporal:1.9.1` | `sha256:ad4c82c97bd12b417d1ea942610dbcd511afb250c4d5ed26c694009533df447e` | `temporalio/temporal@sha256:ad4c82c97bd12b417d1ea942610dbcd511afb250c4d5ed26c694009533df447e` |

## Initial attempt and evidence-based correction

The single startup command completed successfully in 155.9 seconds:

```powershell
docker compose --project-name raama-int-20261027-a2339897 `
  --env-file .tmp/industrial_rag_backend_integration_20261027/attempt_4452c76/secrets.env `
  -f .tmp/industrial_rag_backend_integration_20261027/attempt_4452c76/compose.yaml `
  up --detach --wait --wait-timeout 240
```

`docker compose ps --all` then showed all five containers up with the loopback mappings listed above.

The initial `smoke.py` reached the synthetic S3 round trip, PostgreSQL migration and connection, bundle ingestion, source object reopen, and outbox consume. It exited at its assertion because the smoke check expected `_outbox_verify_once()` to return `status == "done"`. The repository method returns a `committed` boolean; the outbox operation had already committed. The failed assertion did not trigger another ingestion or provider request.

The one evidence-based correction read the existing outbox status and consumer receipt from PostgreSQL and verified the already-ingested source bytes before continuing. It did not replay ingestion. It then built and published the synthetic reference-index release and ran the direct service checks below. Script hashes, including the initial and corrected scripts and Compose file, are retained in the local ignored attempt directory for comparison:

- Compose SHA-256: `4ed7f86a386688719f2b36e5a65fc7e103c712d969ca10d5beea6e9658d69101`.
- Initial smoke script SHA-256: `005dce2bdc4608c87f90ba59bc7bf982175d63b4b716ef0703d13db641acedea`.
- Corrected smoke script SHA-256: `c7479cf0ae392e6cb99552533540f884c0a15602e9051d62dd10240442c9e2ef`.
- Sanitized result JSON SHA-256: `1093cba359c718d9e7b583cd9fc76f506f86b6ecd0b4f5be4da7fb21418ae716`.

## Results

### Repository application adapters

- PostgreSQL adapter connected and applied its migration successfully.
- A synthetic `DocumentBundle` ingested through `PostgresIndustrialRagStore` with `S3CompatibleObjectStore` backed by RustFS.
- Content-addressed put/get returned the original bytes and matched SHA-256.
- The outbox consumer receipt and acknowledgement were present after the committed consume.
- A `sqlite_reference` receipt was verified; the release published and became the active release.

### Direct service checks

- OpenSearch returned HTTP 200 and reported version `3.9.0`; a temporary index was created, queried for one synthetic document, and deleted.
- Qdrant accepted a temporary two-dimensional collection and synthetic vector; the query returned point ID `1`; the collection was deleted.
- Temporal CLI cluster health returned `SERVING`; the development UI returned HTTP 200.

The OpenSearch and Qdrant checks used their REST APIs directly. They did not exercise repository adapters, release-bound receipts, or retrieval through the RAG service. Temporal was checked for service health only; no application workflow was started or recovered.

## Resource configuration and cleanup

- Host bindings were loopback-only and dynamically allocated.
- Compose declared no per-container CPU quota or memory limit, and no aggregate cgroup limit or runtime resource samples were recorded. OpenSearch alone had a JVM heap setting of `-Xms512m -Xmx512m`; this is not a container memory limit.
- Cleanup command: `docker compose --project-name raama-int-20261027-a2339897 --env-file .tmp/industrial_rag_backend_integration_20261027/attempt_4452c76/secrets.env -f .tmp/industrial_rag_backend_integration_20261027/attempt_4452c76/compose.yaml down --volumes --remove-orphans`.
- Cleanup exited `0` and reported all five containers removed, the three named volumes removed, and the Compose network removed. The ephemeral local credentials file was deleted after cleanup. Docker inspection afterward reported no such containers.

## Versioned reproduction path

The clean-clone probe is now in `scripts/backend_integration_smoke.py`, with the isolated service template at `deploy/compose/raama-backend-smoke.yml` and the public synthetic fixture at `tests/fixtures/backend_smoke_document.v1.json` (schema: `schemas/backend_smoke_document.v1.schema.json`). It does not read the historical ignored attempt directory. The Compose template binds dynamic host ports on loopback only and uses placeholder variables for the three credentials. The caller supplies a private `KEY=VALUE` file containing `POSTGRES_PASSWORD`, `RUSTFS_ACCESS_KEY`, and `RUSTFS_SECRET_KEY`; that file must stay outside version control and the probe never prints its contents.

Use a new, unique Compose project name for each run. Start the template with `docker compose --project-name <unique-project> --env-file <private-env-file> -f deploy/compose/raama-backend-smoke.yml up --detach --wait`, then run `python scripts/backend_integration_smoke.py --project-name <same-project> --env-file <private-env-file>`. The probe discovers the assigned ports from Compose, performs the synthetic adapter and direct REST checks, stops only that project's PostgreSQL service to confirm the adapter fails closed, starts it again, and verifies that the committed object, outbox receipt, source row, and active release remain available. It also checks Temporal's development UI and RPC health. It creates unique records and removes its temporary OpenSearch index and Qdrant collection. Finally, remove the isolated stack and volumes with the matching `docker compose ... down --volumes --remove-orphans` command and delete the private env file.

The probe injects a late validation failure after PostgreSQL has inserted synthetic document, version, and child rows inside the ingest transaction. It verifies those rows and the outbox event roll back together; the content-addressed object can remain until the isolated object-store volume is removed. It also fails an outbox handler, checks that no receipt is committed, then waits for lease expiry and verifies a single successful retry with duplicate acknowledgement prevented. A release missing its required backend receipt remains `building` and does not change the active pointer. A separate bounded PostgreSQL process outage after a completed commit verifies reconnect and durable state. This does not test failover to a second PostgreSQL node or PostgreSQL high availability. OpenSearch/Qdrant remain direct service checks rather than application adapters; Temporal remains a service health check rather than an application workflow.

### Portable probe execution — 2026-10-27

- Project `raama-smoke-df3117bc3396` ran the checked-in Compose template, synthetic fixture, and probe. PostgreSQL rolled back rows after an injected late invalid access-grant insert; a failing outbox handler left no receipt and recovered after lease expiry; a release missing its required backend receipt remained `building` with the active pointer unchanged. The valid PostgreSQL/S3 ingest verified object bytes, committed outbox receipt/ack, `sqlite_reference` receipt, and release publication. Stopping PostgreSQL returned the documented generic connection failure; after restart and dynamic-port rediscovery, source metadata, object bytes, outbox receipt, and active release were still present. The emitted adapter status still says `postgres_contract_suite_verified: false`.
- Direct OpenSearch 3.9.0 index/search/delete and Qdrant collection/upsert/query/delete passed; Temporal RPC reported `SERVING` and the UI returned HTTP 200. Those remain service checks with the adapter/workflow limits above.
- During probe development, an early attempt retained a stale dynamically assigned PostgreSQL host port after restart and another reached an OpenSearch socket before it accepted requests. Both stacks were removed. The final isolated execution passed after adding port rediscovery, bounded readiness polling, and the transaction rollback injection.
- The final Compose cleanup output confirmed five containers, three named volumes, and the project network removed. The ephemeral `private.env` file is absent. Sanitized result JSON remains local at `.tmp/raama-smoke-df3117bc3396/probe_result.json` with SHA-256 `56dddfb80d94868c17b4a2c4cf120811dedc157c9c321493f08e4c92d3ccd8ad`.
- Reproduction artifact SHA-256 values: probe `df1c7553c2c7938be9e8e3c56c2abf728882508e1d5842ccf18e61c0461a8b7b`, Compose `dc42e0dc0d6de518f3e549ddce05987e87f68f96cbfd31ef419bc5b4aee39ab7`, fixture `c3d1e33addd625932dad47ca9352d203dfbefc36cff1b6b5ae68f307780e4407`, and fixture schema `91a384ecb0b3b78d35ae27851a4082d2f7bc9f24dc2bf3a4241d0486781cdc53`.

### Parent-review follow-up probe — 2026-10-27

- A first revised startup reached PostgreSQL before it accepted connections; the temporary stack and private credentials were removed. The probe now retries adapter construction for up to 45 seconds. The next attempt exposed an accounting-basis mismatch in its new synthetic rollback fact; that stack and credentials were also removed. The final bounded run passed after the fixture fact was aligned with the declared document basis.
- Project `raama-smoke-3f67131299a7`, run `f52a211c9c8f`: injected late invalid-principal failure left zero document, version, span, block, fact, and outbox rows. The content-addressed object remained available through the rollback check.
- After successful outbox completion, the probe explicitly called `finish_outbox` again. The duplicate acknowledgement returned `False`, and the stored receipt JSON remained unchanged. A separate claim check found no completed event available to claim.
- The probe first published a valid release, then attempted a second release with its required OpenSearch receipt missing. The failed release remained `building` and the previously active release ID remained active.
- PostgreSQL outage/reconnect, RustFS content-addressed round trip, direct OpenSearch/Qdrant REST checks, and Temporal UI/RPC health passed. These do not establish a full PostgreSQL contract suite, OpenSearch/Qdrant application adapters, or a Temporal application workflow.
- Cleanup output confirmed all five containers, three named volumes, and the isolated network removed; the generated private env file is absent. Sanitized result JSON is `.tmp/industrial_rag_backend_integration_20261027/parent-review-followup-raama-smoke-3f67131299a7.json`, SHA-256 `6aee849130a051d256c9efd778fc8689b013d593c84008f62d556b94c229c563`. Probe SHA-256 `27aa574f45966975727a48e5ffa1bba73c4cbe4c2a19c24bfa1500b60540f14c`.

### OpenSearch application adapter verification — 2026-10-27

- The checked-in `OpenSearchTextIndexAdapter` now builds an immutable release-scoped block index, validates the exact source-backed records and index metadata before saving the OpenSearch receipt, and can serve the configured `bm25` channel through `IndustrialRagService`. The PostgreSQL migration now includes the continuation review/state tables required by the shared block reader.
- Project `raama-smoke-c842e539486c`, run `926e6c198412`: the synthetic release recorded both OpenSearch and SQLite reference receipts, published successfully, and returned source evidence through the release-bound BM25 application adapter. The same run passed the direct OpenSearch/Qdrant REST checks, PostgreSQL transaction/outbox/active-release/outage checks, RustFS checksum round trip, and Temporal health checks.
- A first adapter attempt exposed the missing PostgreSQL continuation tables; after adding them, a second attempt exposed OpenSearch startup readiness. The final run waited for the bounded HTTP readiness check before index creation. All attempted stacks were isolated and removed.
- Cleanup confirmed five containers, three named volumes and the network removed; private env file absent. Sanitized result JSON: `.tmp/industrial_rag_backend_integration_20261027/opensearch-app-raama-smoke-c842e539486c.json`, SHA-256 `9a99f43b6e00e8aeb04fb30115de68ab998b0e9669e9b9a374d888d5af45f19b`. Smoke script SHA-256 `a81020379702203ca256f5be96537a1e3c748fbc0489d1b7d79057fd3c27850a`; adapter SHA-256 `010456cb8a3d139fde363a0137bae48f08e06c214fb61c975276418479d3bd00`.
- A follow-up run `raama-smoke-f55f79e64aa0` / `76b713632989` repeated the positive receipt and BM25 path, then rejected a nonactive release, a revoked principal, and an `as_of` time before `first_available_at`. A request counter confirmed each negative was rejected before an OpenSearch `_search` request. Five containers, three volumes, network and private env file were removed. Result SHA-256 `de5bff6f143bc1abf49ec499a1729f5b1e245df0fac09ee5469f8b774a398fc4`; final probe SHA-256 `6b2b01de3ede999a59d980608e39e5490bd23f82bfdecc8fd15b86733286e53c`; final adapter SHA-256 `5dc4ed68ca9fb157a2b6ce4ef39dd0a4003e0fd7c1b036ef8cd1998bf8543393`.

## Remaining work

The release-bound Qdrant application adapter and local Temporal ingest/recovery path are covered by the follow-up runs below. Temporal index/evaluate/research activities and checkpoint controls are implemented, but still need execution against a real Temporal service with worker restart/replay coverage. Expand PostgreSQL contract and HA coverage beyond synthetic transaction, outbox concurrency/lease fencing, release publication, and bounded outage/reconnect. Validate release-bound Qwen1024 retrieval with actual pinned-model inference and a live Qdrant service, then continue the remaining A-G acceptance. This remains local integration evidence, not production acceptance.

### PostgreSQL concurrency and Temporal ingest-time follow-up — 2026-10-27

- The PostgreSQL outbox claim path now uses `READ COMMITTED` with the existing `FOR UPDATE SKIP LOCKED` row lock. A prior concurrent-claim probe showed that `SERIALIZABLE` could abort one independent claim with SQLSTATE 40001; the isolated rerun passed with two workers receiving distinct events. The probe also confirmed an expired lease generation cannot write a receipt, while the current generation can.
- `IndustrialRagStore.ingest_bundle` now selects `ingested_at` only after the source/version check inside the RAG transaction. Failed insertion rolls the timestamp back with its row changes. When a later activity retry finds a committed version, it compares the remaining immutable metadata and returns the timestamp already stored with that version. The timestamp is no longer frozen in a separate task artifact before database commit.
- A controlled local retry regression selected a time, failed the RAG transaction, then advanced the clock by two hours and retried. The committed row used the later successful time. The probe then simulated a worker failure after the RAG commit but before saving the task result; the next retry reused the committed row’s time without selecting a replacement.
- Final isolated synthetic run: project `raama-temporal-txn-1027-06`, run `ae91084ca315`, result `.tmp/industrial_rag_backend_integration_20261027/temporal-ingest-transaction-time-06-20261027.json`, SHA-256 `ba6ba9161d2bc95a1cfebda8907429d6a557104ba315efdc4ae0e8dce4cddd0a`. PostgreSQL continuation confirm/revoke, transaction-time recovery, ingest rollback, outbox retry/concurrency/lease fencing, release atomicity, outage/reconnect, OpenSearch/Qdrant release-bound synthetic search, coherent Qdrant tamper rejection, and Temporal source/metadata capture drift plus worker lost-ack recovery all passed. The generated credentials and all five service containers, three volumes, and project network were removed; the private env file is absent; provider calls were zero.
- Focused RAG/API/planning/Qwen/current-capture/live-agent/Temporal regression slice: 104 passed in 14.33s. Python compilation, `git diff --check`, and offline `uv lock --check` passed.
- Input SHA-256s: backend probe `6ed63f56e238ab415bc357fd86bdf62af1b5ef31211d112d61698c10e6d02406`; Temporal probe `1d0c9a4ffc8d7661cf92a1cf4622396dd609df71bbd35913d76eb63cdb584536`; repository `20de4cef8260d58d2d4f945f7b1de599b6116253a8df4811cea74219187c9cf3`; PostgreSQL adapter `5e195d89b36ca8a5efdce1c16e1cb548cb457a5566f304fd6fa9a751ac871ac6`; Temporal workflow `0103cc76cf9ab1fec3ad6c95e063cfaf0154c6188a0801bb440962a3ccfc395b`; ingest CLI `ef5ed0052c0f557da31bd3b2f9390bff35456ce96bfb7f9df95367b2849b46c6`; Temporal regression test `6d9aedd2a9b6564ad0e38515de4505c10ff42e6097c2376f3e276520e126f82d`; Compose `dc42e0dc0d6de518f3e549ddce05987e87f68f96cbfd31ef419bc5b4aee39ab7`; fixture `c3d1e33addd625932dad47ca9352d203dfbefc36cff1b6b5ae68f307780e4407`; `pyproject.toml` `c9e28f6ebf630e1f21578b11a62779e4e870c29a52c8d4f6304802df6cbd1285`; `uv.lock` `60422202eac55c7d22109e397687de908f1588b5237aba87a0ec2a0cfa2b3f5a`.

### PostgreSQL cache, permission-revision, and continuation idempotency — 2026-10-27

- Extended the actual PostgreSQL smoke to upsert and reread a query-cache payload, confirm that a no-op permission update does not bump the permission revision while revoke/restore each do, and repeat a continuation confirmation without changing its revision. The isolated PostgreSQL 17.11 run passed these checks alongside the previously recorded transaction rollback, committed ingest-time retry, concurrent outbox claim/lease fencing, continuation confirm/revoke, publication atomicity, outage/reconnect, and application retrieval/recovery checks. Temporal captured-input verification and worker lost-ack recovery also passed; provider calls were zero.
- Project `raama-pgcontract-1027-07`, run `6ee9387c8a8f`; result `.tmp/industrial_rag_backend_integration_20261027/postgres-contract-cache-permission-07-20261027.json`, SHA-256 `e8cf7d6110481f0be4893793b29b8d5b470bfcd02a9996041824a7ca05fc693d`. Updated smoke SHA-256 `a15a0b08cc1a312affbaeed5f8970bdfcc86b6adf3de0141e7cbf5a16e69a3ab`; updated PostgreSQL translation test SHA-256 `0eb63a6b3c684637d000209a8cb0ac070e9263afd91ab3b78a6a00e713f3cfec`.
- The result explicitly leaves `postgres_contract_suite_verified` false: these are bounded integration probes, not the complete PostgreSQL contract or HA suite. The private env file is absent. This task's saved output did not preserve the Compose cleanup exit status, but parent subsequently verified read-only that projects `raama-pgcontract-1027-07` and `raama-temporal-txn-1027-06` have no containers, volumes, or networks. Parent also verified the evidence hashes and the local checkpoint backup; cleanup was not inferred from the result JSON.
- Focused RAG/API/planning/Qwen/current-capture/live-agent/OpenSearch/Temporal regression slice: 104 passed in 14.23s. `py_compile`, `git diff --check`, and `uv lock --check --offline` passed. The offline lock check used a workspace-local cache because the default user cache was not accessible.

### Parent verification of transaction-bound ingest time — 2026-10-27

- Parent independently reviewed checkpoint `316efc5` and passed 68 affected checks in 11.80s. The original activity-to-CLI-to-parser/SQLite fault path was exercised: an injected error after selecting the timestamp and inserting the version row rolled back to zero versions; after advancing the controlled clock by two hours, the successful retry used the later commit time; a failure after RAG commit followed by another two-hour delay reused the committed timestamp. Captured source/metadata drift, captured-file tampering rejection, and completed retry reuse also passed.
- Evidence: `.tmp/industrial_rag_supervision_20261025/parent_reviews/316efc5-y86q7r7k/parent_transaction_timestamp_fixed_result.json`. This closes the F06-P1-01 transaction-time defect only. It did not use Temporal service, a model, or a provider, and does not complete the full A-G acceptance.
- Parent verified checkpoint `316efc5a8e76757ed654651b690a198865be364a` is backed up as `95f6f6d9570468be2dfdc6292b254e6706652b3f` (tree `164eac0dad8bfe1a39ff3e59ba7865c19cc37db8`), expected predecessor `0bbfadb`; non-force backup has 10 matching blobs/ref and zero `.tmp` content. Saved PostgreSQL artifacts `ba6ba9161d2bc95a1cfebda8907429d6a557104ba315efdc4ae0e8dce4cddd0a` and `e8cf7d6110481f0be4893793b29b8d5b470bfcd02a9996041824a7ca05fc693d` and six current input hashes matched. Parent's correctly located Docker inspection found both isolated projects empty.

### Temporal operation and lifecycle implementation — 2026-10-27

- Added durable local Temporal activity paths for release indexing, exact-byte offline manifest evaluation, and the multi-expert research harness. Evaluation and research inputs are captured before workflow submission and hash-verified when workers read them. Research replays only exact local response-cache hits through an offline-only Qwen client with a zero provider-call budget; a cache miss fails before any network request.
- Ingest and operation workflows support audited pause/resume/cancel signals at activity checkpoints. Task state records transitions and failures; failed or unknown tasks can be dead-lettered with an operator, reason, and command ID. Added recovery for resume arriving while a previously paused task is being restored.
- Focused state/Temporal/evaluation regression run: 44 passed in 3.94s. This includes mocked workflow control/release paths, activity persistence and retry, offline research cache behavior, durable cancellation/failure/dead-letter handling, and the paused-task restore race. Python compilation, CLI help, and `git diff --check` passed.
- These workflow orchestration checks use direct activities and a mocked Temporal workflow harness; no Temporal server or worker restart/replay run was performed for this slice. Provider calls were zero. A real Temporal service run and the broader A-G acceptance remain pending.

### Qdrant 1024-dimension application contract — 2026-10-27

- Added a release-bound application check using the real persistent-vector cache adapter and the pinned Qwen3 embedding identity at 1024 dimensions. The synthetic release built and verified a Qdrant receipt, published with both required backends, and returned source-backed dense evidence through `IndustrialRagService`. A principal with no permitted versions gets an empty result without a Qdrant search request. This also avoids asking a remote adapter to interpret an empty permission scope.
- The cached vector and Qdrant REST endpoint are synthetic fixtures; this does not demonstrate actual Qwen inference or a live Qdrant application run. The earlier low-level physical roundtrip remains separate evidence for three previously cached 1024-dimensional vectors. No provider calls were made.
- New adapter check: 1 passed in 0.34s. The broader focused RAG/API/planning/OpenSearch/Qdrant/Temporal/state/evaluation regression run passed 112 tests in 13.50s; Python compilation and `git diff --check` passed. A live release-bound Qwen/Qdrant run remains pending.

### F06-P2-02 terminal-state recovery and committed-result reconciliation — 2026-10-27

- Closed the artifact-before-terminal-state recovery order that let a persisted operation artifact promote a dead-lettered task back to completed. Initialization now returns cancelled/failed/unknown/dead-letter/paused state before any result restoration and marks an existing result artifact as explicit reconciliation evidence. Operation and ingest activities refuse automatic replay from non-running states. The workflow result/query and dead-letter response identify the artifact needing operator reconciliation.
- Added a transactional completion fence to `HarnessStateStore.complete_task` for Temporal callers and made late checkpoints leave cancelled, failed, unknown, and dead-letter stages intact. A committed database/object-store effect remains represented by the immutable task artifact; it is not silently treated as task completion after operator action.
- Reproduced both paths locally. Positive: index publication and task completion commit, then the activity acknowledgement is lost; replay returns the stored result without rebuilding the release. Negative: the release publishes and the artifact persists, task completion fails, the task is failed then dead-lettered, and later initialization/activity replay preserve `dead_letter`, expose `reconciliation_required`, and do not add a completed event. A cancelled task with a saved artifact and the corresponding ingest guard are also covered.
- Focused state/Temporal/RAG/API/planning/OpenSearch/Qdrant/evaluation regression slice: 118 passed in 14.16s. Python compilation and `git diff --check` passed. No Temporal service, model inference, or provider was used; real worker restart/replay remains pending.
