"""Run a bounded local Qwen-1024/Qdrant release and permission smoke.

The source database and persistent vector cache are opened read-only and
backed up to a fresh scratch directory. Document vectors must already exist in
that copied cache. One local query embedding is allowed; no provider client is
created by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.industrial_rag.models import CorpusReleaseSpec, QueryPlan  # noqa: E402
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore  # noqa: E402
from adaptive_mas.industrial_rag.qdrant_adapter import (  # noqa: E402
    QdrantBackendError,
    QdrantRestClient,
    QdrantVectorIndexAdapter,
)
from adaptive_mas.industrial_rag.qwen_models import (  # noqa: E402
    EMBEDDING_REPO,
    EMBEDDING_REVISION,
    LocalQwenDenseAdapter,
    Qwen3EmbeddingRuntime,
)
from adaptive_mas.industrial_rag.repository import IndustrialRagStore  # noqa: E402
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService  # noqa: E402
from adaptive_mas.research_provenance import stable_hash  # noqa: E402
from adaptive_mas.runtime_limits import process_thread_limit  # noqa: E402


EXPECTED_SOURCE_RELEASE = "current-capture-qwen-cmb-traced-20261026"
EXPECTED_SOURCE_VECTORS = 901
EXPECTED_DIMENSION = 1024
MAX_CPU_CORES = 20
MAX_MEMORY_BYTES = 200_000_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sqlite_backup(source_path: Path, destination_path: Path) -> None:
    source_uri = source_path.as_uri() + "?mode=ro"
    with sqlite3.connect(source_uri, uri=True, timeout=30) as source:
        with sqlite3.connect(destination_path, timeout=30) as destination:
            source.backup(destination)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite run artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _parse_cpu_max(value: str) -> tuple[int, int]:
    pieces = value.split()
    if len(pieces) != 2:
        raise ValueError("cpu-max must be a cgroup v2 'quota period' pair")
    quota, period = (int(piece) for piece in pieces)
    if quota <= 0 or period <= 0:
        raise ValueError("cpu-max quota and period must be positive")
    return quota, period


def _source_assets_fingerprint(
    source_database: Path,
    vector_cache: Path,
    run_identity_path: Path,
    qwen_assets: Path,
) -> dict[str, object]:
    embedding_dir = qwen_assets / "Qwen--Qwen3-Embedding-0.6B"
    reranker_dir = qwen_assets / "Qwen--Qwen3-Reranker-0.6B"
    paths = {
        "source_database_sha256": source_database,
        "vector_cache_sha256": vector_cache,
        "run_identity_sha256": run_identity_path,
        "qwen_asset_manifest_sha256": qwen_assets / "raama_qwen_models.json",
        "embedding_config_sha256": embedding_dir / "config.json",
        "embedding_weights_sha256": embedding_dir / "model.safetensors",
        "reranker_config_sha256": reranker_dir / "config.json",
        "reranker_weights_sha256": reranker_dir / "model.safetensors",
    }
    hashes = {key: _sha256_file(path.resolve(strict=True)) for key, path in paths.items()}
    run_identity = json.loads(run_identity_path.read_text(encoding="utf-8"))
    model_identity = (
        run_identity.get("runtime_model_identity", {})
        .get("embedding_cache_identity", {})
        .get("model_identity")
    )
    if not isinstance(model_identity, dict):
        raise RuntimeError("source run identity does not contain its Qwen embedding identity")
    if (
        model_identity.get("model") != EMBEDDING_REPO
        or model_identity.get("revision") != EMBEDDING_REVISION
        or model_identity.get("dimension") != EXPECTED_DIMENSION
    ):
        raise RuntimeError("source run identity differs from the pinned Qwen 1024 embedding runtime")
    return {
        "hashes": hashes,
        "source_runtime_model_identity_sha256": run_identity.get("runtime_model_identity_sha256"),
        "source_embedding_identity": model_identity,
        "script_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _release_backend_rows(store: IndustrialRagStore, release_id: str) -> list[dict[str, object]]:
    with store._connect() as connection:
        rows = connection.execute(
            "SELECT backend,physical_index_handle,backend_version,source_manifest_sha256,"
            "index_manifest_sha256,document_count,record_count,embedding_model,embedding_dimension "
            "FROM rag_release_backends WHERE release_id=? ORDER BY backend",
            (release_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-container-id", required=True)
    parser.add_argument("--qdrant-image-id", required=True)
    parser.add_argument("--qdrant-host-nano-cpus", type=int, required=True)
    parser.add_argument("--qdrant-host-memory-bytes", type=int, required=True)
    parser.add_argument("--qdrant-pids-limit", type=int, required=True)
    parser.add_argument("--qdrant-cpu-max", required=True)
    parser.add_argument("--qdrant-memory-max", type=int, required=True)
    parser.add_argument("--qdrant-loopback-binding", required=True)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--vector-cache", type=Path, required=True)
    parser.add_argument("--run-identity", type=Path, required=True)
    parser.add_argument("--qwen-assets", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--security-code", default="600036")
    parser.add_argument("--principal-id", default="local-public")
    parser.add_argument("--query", default="2024年度营业收入是多少，较2023年度增长多少？")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9-]{1,48}", args.run_id):
        raise ValueError("run-id must contain 1-48 ASCII letters, digits, or hyphens")
    if not re.fullmatch(r"\d{6}", args.security_code):
        raise ValueError("security-code must contain six digits")
    if not 1 <= args.top_k <= 100:
        raise ValueError("top-k must be between 1 and 100")
    parsed_url = urlsplit(args.qdrant_url)
    if (
        parsed_url.scheme != "http"
        or parsed_url.hostname not in {"127.0.0.1", "localhost"}
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.path not in {"", "/"}
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError("Qdrant endpoint must be an unauthenticated loopback HTTP address")
    quota, period = _parse_cpu_max(args.qdrant_cpu_max)
    cgroup_cores = quota / period
    if (
        args.qdrant_host_nano_cpus <= 0
        or args.qdrant_host_nano_cpus > MAX_CPU_CORES * 1_000_000_000
        or args.qdrant_host_memory_bytes <= 0
        or args.qdrant_host_memory_bytes > MAX_MEMORY_BYTES
        or args.qdrant_pids_limit < 1
        or cgroup_cores > MAX_CPU_CORES
        or args.qdrant_memory_max <= 0
        or args.qdrant_memory_max > MAX_MEMORY_BYTES
        or args.qdrant_host_memory_bytes != args.qdrant_memory_max
    ):
        raise ValueError("Qdrant host and cgroup limits are missing, inconsistent, or above project caps")
    if not args.qdrant_loopback_binding.startswith("127.0.0.1:"):
        raise ValueError("Qdrant published port must bind to loopback only")

    source_database = args.source_database.resolve(strict=True)
    vector_cache = args.vector_cache.resolve(strict=True)
    run_identity_path = args.run_identity.resolve(strict=True)
    qwen_assets = args.qwen_assets.resolve(strict=True)
    scratch_dir = args.scratch_dir.resolve()
    if scratch_dir.exists():
        raise FileExistsError(f"refusing to reuse scratch directory: {scratch_dir}")
    scratch_dir.mkdir(parents=True)

    initial_fingerprints = _source_assets_fingerprint(
        source_database, vector_cache, run_identity_path, qwen_assets,
    )
    scratch_database = scratch_dir / "industrial_rag.sqlite3"
    scratch_cache = scratch_dir / "dense_vector_cache.sqlite3"
    _sqlite_backup(source_database, scratch_database)
    LocalQwenDenseAdapter.copy_persistent_cache(vector_cache, scratch_cache)

    store = IndustrialRagStore(
        scratch_database,
        LocalContentAddressedStore(scratch_dir / "objects"),
    )
    source_release_id = store.active_release_id()
    if source_release_id != EXPECTED_SOURCE_RELEASE:
        raise RuntimeError("copied source database is not the expected published CMB capture release")
    source_spec, source_spec_sha256 = store.release_spec(source_release_id)
    version_ids = store.release_versions(source_release_id)
    if len(version_ids) != 1:
        raise RuntimeError("the pinned CMB capture is expected to contain one source version")

    embedding = Qwen3EmbeddingRuntime(qwen_assets)
    if embedding.dimension != EXPECTED_DIMENSION:
        raise RuntimeError("pinned local Qwen embedding model is not 1024-dimensional")
    if embedding.cache_identity != initial_fingerprints["source_embedding_identity"]:
        raise RuntimeError("Qwen assets do not match the model identity captured by the source run")
    document_embedding_inputs = 0
    query_embedding_inputs = 0
    query_vector_sha256: str | None = None
    original_embed = embedding.embed

    def tracked_embed(texts: list[str], *, query: bool = False, control: Any = None) -> np.ndarray:
        nonlocal document_embedding_inputs, query_embedding_inputs, query_vector_sha256
        result = original_embed(texts, query=query, control=control)
        if query:
            query_embedding_inputs += len(texts)
            if len(texts) == 1:
                query_vector_sha256 = hashlib.sha256(np.asarray(result[0], dtype="<f4").tobytes()).hexdigest()
        else:
            document_embedding_inputs += len(texts)
        return result

    embedding.embed = tracked_embed  # type: ignore[method-assign]
    dense = LocalQwenDenseAdapter(
        store, embedding, cache_path=scratch_cache, require_cached_vectors=True,
    )
    adapter = QdrantVectorIndexAdapter(store, QdrantRestClient(args.qdrant_url, timeout_seconds=30), dense)

    release_id = f"qwen1024-qdrant-{args.run_id.lower()}"
    source_manifest_sha256 = stable_hash(store.source_manifest(version_ids))
    text_index = store.preview_reference_index_manifest(release_id, version_ids)
    vector_index = adapter.preview_vector_index_manifest(release_id, version_ids)
    verified_vector_count = int(vector_index["record_count"])
    if verified_vector_count != EXPECTED_SOURCE_VECTORS:
        raise RuntimeError(f"expected {EXPECTED_SOURCE_VECTORS} verified source vectors, found {verified_vector_count}")
    scratch_cache_sha256_before = _sha256_file(scratch_cache)
    runtime_config = {
        "schema_version": "raama.financial-rag.qwen1024-qdrant-run-config.v1",
        "run_id": args.run_id,
        "source_release_id": source_release_id,
        "source_release_spec_sha256": source_spec_sha256,
        "target_release_id": release_id,
        "security_code": args.security_code,
        "principal_id": args.principal_id,
        "query_sha256": hashlib.sha256(args.query.encode("utf-8")).hexdigest(),
        "channels": ["dense"],
        "top_k": args.top_k,
        "time_mode": "strict_forward",
        "reranker_enabled": False,
        "provider_requests": 0,
    }
    qdrant_client = adapter.client
    server_info = qdrant_client.request("GET", "/")
    qdrant_server_version = server_info.get("version")
    if qdrant_server_version != "1.19.1":
        raise RuntimeError("Qdrant server is not the pinned local 1.19.1 service")
    preflight = {
        "schema_version": "raama.financial-rag.qwen1024-qdrant-preflight.v1",
        "run_config": runtime_config,
        "run_config_sha256": stable_hash(runtime_config),
        "source_and_model_fingerprints": initial_fingerprints,
        "source_database_copy_sha256": _sha256_file(scratch_database),
        "copied_cache_sha256": scratch_cache_sha256_before,
        "copied_cache_verified_vectors": verified_vector_count,
        "source_manifest_sha256": source_manifest_sha256,
        "qdrant": {
            "url": args.qdrant_url,
            "server_version": qdrant_server_version,
            "container_id": args.qdrant_container_id,
            "image_id": args.qdrant_image_id,
            "host_nano_cpus": args.qdrant_host_nano_cpus,
            "host_memory_bytes": args.qdrant_host_memory_bytes,
            "pids_limit": args.qdrant_pids_limit,
            "cpu_cgroup": args.qdrant_cpu_max,
            "cpu_cgroup_cores": cgroup_cores,
            "memory_cgroup_bytes": args.qdrant_memory_max,
            "loopback_binding": args.qdrant_loopback_binding,
            "telemetry_disabled": True,
        },
        "python_process_thread_limit": process_thread_limit(),
        "inference_device_contract": "CPU; Qwen runtime loads the model with .to('cpu')",
        "source_copy_mode": "sqlite backup from mode=ro source; original source and cache hashes rechecked after run",
    }
    _write_json(scratch_dir / "preflight.json", preflight)

    release_spec = CorpusReleaseSpec(
        release_id=release_id,
        source_manifest_sha256=source_manifest_sha256,
        parser_version=str(source_spec["parser_version"]),
        chunker_version=str(source_spec["chunker_version"]),
        embedding_model=adapter.embedding_model,
        embedding_dimension=adapter.dimension,
        required_backends=("qdrant", "sqlite_reference"),
        text_index_manifest_sha256=str(text_index["index_manifest_sha256"]),
        vector_index_manifest_sha256=str(vector_index["index_manifest_sha256"]),
    )
    store.create_release(release_spec, version_ids)
    reference_receipt = store.build_reference_index_receipt(release_id)
    reference_ready = store.mark_backend_ready(reference_receipt, store.verify_reference_index)
    qdrant_ready = adapter.mark_release_ready(release_id)
    publication = store.publish_release(
        release_id,
        actor_id="qwen1024-qdrant-offline-smoke",
        reason="One bounded local Qwen 1024/Qdrant application retrieval on a copied public-source release.",
    )

    query_count = 0
    original_request = qdrant_client.request

    def tracked_request(method: str, path: str, body: object | None = None) -> dict[str, Any]:
        nonlocal query_count
        if method.upper() == "POST" and path.endswith("/points/query"):
            query_count += 1
        return original_request(method, path, body)

    qdrant_client.request = tracked_request  # type: ignore[method-assign]
    plan = QueryPlan(
        plan_id=f"qwen1024-query-{args.run_id.lower()}",
        query=args.query,
        security_code=args.security_code,
        as_of=datetime.now(timezone.utc),
        principal_id=args.principal_id,
        task_type="comparison",
        release_id=release_id,
        channels=("dense",),
        top_k=args.top_k,
    )
    search_result = IndustrialRagService(
        store,
        channel_adapters={"dense": adapter},
        allow_retrospective_diagnostic=False,
    ).search(plan, use_cache=False, use_reranker=False)
    retrieved = search_result.get("retrieved")
    if not isinstance(retrieved, list) or not retrieved:
        raise RuntimeError("release-bound Qdrant application search returned no source evidence")
    dense_status = search_result.get("channels", {}).get("dense", {})
    if dense_status.get("implementation") != adapter.implementation or query_count != 1:
        raise RuntimeError("application search did not exercise the release-bound Qdrant adapter exactly once")
    if document_embedding_inputs != 0 or query_embedding_inputs != 1 or not query_vector_sha256:
        raise RuntimeError("expected one local query embedding and zero document embeddings")
    if str(getattr(embedding._model, "device", "")) != "cpu":
        raise RuntimeError("local Qwen embedding runtime did not load on CPU")

    with store._connect() as connection:
        source_version = connection.execute(
            "SELECT document_id,first_available_at FROM rag_document_versions WHERE version_id=?",
            (version_ids[0],),
        ).fetchone()
    if source_version is None:
        raise RuntimeError("source version disappeared from the copied database")
    document_id = str(source_version["document_id"])
    first_available_at = datetime.fromisoformat(str(source_version["first_available_at"]))

    def expect_rejection(name: str, expected_exception: type[Exception], operation: Callable[[], object]) -> dict[str, object]:
        query_count_before = query_count
        query_embeddings_before = query_embedding_inputs
        try:
            operation()
        except expected_exception as error:
            if query_count != query_count_before or query_embedding_inputs != query_embeddings_before:
                raise RuntimeError(f"negative case {name} reached Qdrant or embedded a query") from error
            return {
                "case": name,
                "result": "rejected",
                "exception": type(error).__name__,
                "qdrant_query_delta": 0,
                "query_embedding_delta": 0,
            }
        raise RuntimeError(f"negative case {name} unexpectedly passed")

    negative_cases: list[dict[str, object]] = []
    negative_cases.append(expect_rejection(
        "inactive_release_binding",
        QdrantBackendError,
        lambda: adapter.search_with_control(replace(plan, release_id=source_release_id), version_ids, None),
    ))
    store.set_access(args.principal_id, document_id, can_read=False)
    try:
        negative_cases.append(expect_rejection(
            "revoked_principal",
            PermissionError,
            lambda: adapter.search_with_control(plan, version_ids, None),
        ))
    finally:
        store.set_access(args.principal_id, document_id, can_read=True)
    negative_cases.append(expect_rejection(
        "strict_forward_before_available_at",
        PermissionError,
        lambda: adapter.search_with_control(
            replace(plan, as_of=first_available_at - timedelta(seconds=1)), version_ids, None,
        ),
    ))

    release_summaries = store.release_summaries()
    active_release = next(
        row for row in release_summaries["releases"] if row["release_id"] == release_id
    )
    backend_rows = _release_backend_rows(store, release_id)
    qdrant_backend_row = next((row for row in backend_rows if row["backend"] == "qdrant"), None)
    if (
        publication.get("status") != "published"
        or store.active_release_id() != release_id
        or active_release["status"] != "published"
        or set(active_release["ready_backends"]) != {"qdrant", "sqlite_reference"}
        or qdrant_backend_row is None
        or qdrant_backend_row["record_count"] != EXPECTED_SOURCE_VECTORS
        or qdrant_backend_row["embedding_dimension"] != EXPECTED_DIMENSION
        or qdrant_backend_row["index_manifest_sha256"] != vector_index["index_manifest_sha256"]
    ):
        raise RuntimeError("published release receipt does not match the verified Qdrant index")
    verify_receipt = adapter.verify_release_index(str(qdrant_backend_row["physical_index_handle"]))
    if (
        verify_receipt.get("record_count") != EXPECTED_SOURCE_VECTORS
        or verify_receipt.get("embedding_dimension") != EXPECTED_DIMENSION
        or verify_receipt.get("index_manifest_sha256") != vector_index["index_manifest_sha256"]
    ):
        raise RuntimeError("post-publication Qdrant verification did not match its release receipt")

    source_database_sha256_after = _sha256_file(source_database)
    vector_cache_sha256_after = _sha256_file(vector_cache)
    scratch_cache_sha256_after = _sha256_file(scratch_cache)
    if (
        source_database_sha256_after != initial_fingerprints["hashes"]["source_database_sha256"]
        or vector_cache_sha256_after != initial_fingerprints["hashes"]["vector_cache_sha256"]
        or scratch_cache_sha256_after != scratch_cache_sha256_before
    ):
        raise RuntimeError("source or copied dense-vector cache changed during the bounded run")

    result = {
        "schema_version": "raama.financial-rag.qwen1024-qdrant-release-smoke-result.v1",
        "run_id": args.run_id,
        "preflight_sha256": _sha256_file(scratch_dir / "preflight.json"),
        "run_config_sha256": stable_hash(runtime_config),
        "source_release_id": source_release_id,
        "source_release_spec_sha256": source_spec_sha256,
        "source_version_ids": version_ids,
        "source_manifest_sha256": source_manifest_sha256,
        "release_id": release_id,
        "release_spec_sha256": store.release_spec(release_id)[1],
        "text_index_manifest_sha256": text_index["index_manifest_sha256"],
        "vector_index_manifest_sha256": vector_index["index_manifest_sha256"],
        "embedding_model": adapter.embedding_model,
        "embedding_dimension": adapter.dimension,
        "verified_cached_document_vectors": verified_vector_count,
        "document_embedding_inputs": document_embedding_inputs,
        "query_embedding_inputs": query_embedding_inputs,
        "query_vector_sha256": query_vector_sha256,
        "model_device": str(embedding._model.device),
        "qdrant_server_version": qdrant_server_version,
        "qdrant_query_count": query_count,
        "reference_backend_ready": bool(reference_ready.get("ready")),
        "qdrant_backend_ready": bool(qdrant_ready.get("ready")),
        "publication_status": publication.get("status"),
        "active_release_id_after_search": store.active_release_id(),
        "verified_qdrant_backend_receipt": qdrant_backend_row,
        "post_publication_backend_verification": dict(verify_receipt),
        "search_status": search_result.get("status"),
        "search_retrieved_count": len(retrieved),
        "dense_candidate_count": dense_status.get("candidate_count"),
        "top_evidence": [
            {
                "evidence_id": item.get("evidence_id"),
                "source_id": item.get("source_id"),
                "version_id": item.get("version_id"),
                "block_id": item.get("block_id"),
                "page_number": item.get("page_number"),
                "score": item.get("score"),
            }
            for item in retrieved[: args.top_k]
            if isinstance(item, dict)
        ],
        "negative_cases": negative_cases,
        "source_database_unchanged": source_database_sha256_after == initial_fingerprints["hashes"]["source_database_sha256"],
        "source_vector_cache_unchanged": vector_cache_sha256_after == initial_fingerprints["hashes"]["vector_cache_sha256"],
        "copied_vector_cache_unchanged": scratch_cache_sha256_after == scratch_cache_sha256_before,
        "provider_requests": 0,
        "provider_client_instantiated": False,
        "reranker_enabled": False,
        "synthetic_fixture": False,
        "scratch_only_release": True,
    }
    _write_json(scratch_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
