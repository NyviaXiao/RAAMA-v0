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

## Remaining work

Expand the PostgreSQL contract suite beyond these synthetic transaction, lease recovery, release publication, and bounded outage/reconnect cases. Implement and validate OpenSearch/Qdrant application adapters with release-bound receipts. Connect Temporal to the application workflow and exercise durable recovery. This smoke is local integration evidence, not production acceptance.
