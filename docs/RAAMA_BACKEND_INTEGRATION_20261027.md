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

## Remaining work

Run the full PostgreSQL contract and failure-recovery suite. Implement and validate OpenSearch/Qdrant application adapters with release-bound receipts. Connect Temporal to the application workflow and exercise durable recovery. This smoke is local integration evidence, not production acceptance.
