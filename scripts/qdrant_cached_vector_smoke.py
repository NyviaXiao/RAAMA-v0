"""Round-trip up to three verified 1024-dimension local Qwen cache vectors through Qdrant."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore  # noqa: E402
from adaptive_mas.industrial_rag.qdrant_adapter import QdrantRestClient, QdrantVectorIndexAdapter  # noqa: E402
from adaptive_mas.industrial_rag.qwen_models import LocalQwenDenseAdapter, Qwen3EmbeddingRuntime  # noqa: E402
from adaptive_mas.industrial_rag.repository import IndustrialRagStore  # noqa: E402
from adaptive_mas.research_provenance import stable_hash  # noqa: E402


def _sqlite_backup(source_path: Path, destination_path: Path) -> None:
    source_uri = source_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source:
        with sqlite3.connect(destination_path) as destination:
            source.backup(destination)


def _result_vector(point: dict[str, Any]) -> np.ndarray:
    vector = point.get("vector")
    if not isinstance(vector, list):
        raise AssertionError("Qdrant did not return a dense point vector")
    result = np.asarray(vector, dtype="<f4")
    if result.ndim != 1 or result.shape != (1024,) or not np.isfinite(result).all():
        raise AssertionError("Qdrant returned a malformed 1024-dimension vector")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--vector-cache", type=Path, required=True)
    parser.add_argument("--qwen-assets", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.limit <= 3:
        raise ValueError("cached-vector smoke is bounded to one through three blocks")
    source_database = args.source_database.resolve(strict=True)
    vector_cache = args.vector_cache.resolve(strict=True)
    qwen_assets = args.qwen_assets.resolve(strict=True)
    scratch_dir = args.scratch_dir.resolve()
    scratch_dir.mkdir(parents=True, exist_ok=False)
    scratch_database = scratch_dir / "source.sqlite3"
    scratch_cache = scratch_dir / "vectors.sqlite3"
    _sqlite_backup(source_database, scratch_database)
    _sqlite_backup(vector_cache, scratch_cache)

    store = IndustrialRagStore(
        scratch_database, LocalContentAddressedStore(scratch_dir / "unused-objects"),
    )
    embedding = Qwen3EmbeddingRuntime(qwen_assets)
    if embedding.dimension != 1024:
        raise AssertionError("pinned local Qwen embedding model is not 1024-dimensional")
    dense = LocalQwenDenseAdapter(
        store, embedding, cache_path=scratch_cache, require_cached_vectors=True,
    )
    adapter = QdrantVectorIndexAdapter(store, QdrantRestClient(args.qdrant_url), dense)
    with store._connect() as connection:
        version = connection.execute(
            "SELECT version_id FROM rag_document_versions "
            "WHERE parser_status='parsed' AND parser_confidence>=0.6 ORDER BY version_id LIMIT 1"
        ).fetchone()
    if version is None:
        raise AssertionError("source database has no parsed version available for cached-vector verification")
    version_id = str(version["version_id"])
    rows = [
        row for row in store.read_blocks([version_id])
        if row.get("quarantine_reason") is None and str(row["text"]).strip()
    ][:args.limit]
    if not rows:
        raise AssertionError("selected source version has no searchable blocks")
    block_keys = [(str(row["version_id"]), str(row["block_id"])) for row in rows]
    cached = dense.cached_vectors([version_id], block_keys=block_keys)
    if len(cached) != len(rows):
        raise AssertionError("verified local Qwen cache did not cover every selected source block")

    run_id = uuid.uuid4().hex[:12]
    smoke_id = f"qwen-cache-{run_id}"
    collection = f"raama_qwen_cached_smoke_{run_id}"
    identity_sha256 = stable_hash(embedding.cache_identity)
    client = adapter.client
    metadata = {
        "schema_version": "raama.financial-rag.qdrant-cached-vector-smoke.v1",
        "smoke_id": smoke_id,
        "embedding_model": adapter.embedding_model,
        "embedding_dimension": adapter.dimension,
        "model_identity_sha256": identity_sha256,
    }
    points: list[dict[str, object]] = []
    expected_vectors: dict[str, np.ndarray] = {}
    for row, vector in cached:
        vector = np.asarray(vector, dtype="<f4")
        point_id = adapter._point_id(str(row["version_id"]), str(row["block_id"]))
        vector_sha256 = adapter._vector_sha256(vector)
        payload: dict[str, object] = {
            **metadata,
            "version_id": str(row["version_id"]),
            "block_id": str(row["block_id"]),
            "source_id": str(row["source_id"]),
            "content_sha256": str(row["content_sha256"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
            "vector_sha256": vector_sha256,
        }
        payload["record_sha256"] = stable_hash(payload)
        points.append({"id": point_id, "vector": vector.tolist(), "payload": payload})
        expected_vectors[point_id] = vector

    collection_created = False
    try:
        client.request("PUT", f"/collections/{collection}", {
            "vectors": {"size": 1024, "distance": "Cosine"},
            "metadata": metadata,
        })
        collection_created = True
        client.request("PUT", f"/collections/{collection}/points?wait=true", {"points": points})
        scroll = client.request("POST", f"/collections/{collection}/points/scroll", {
            "limit": args.limit,
            "with_payload": True,
            "with_vector": True,
        })
        result = scroll.get("result")
        stored = result.get("points") if isinstance(result, dict) else None
        if not isinstance(stored, list) or len(stored) != len(points):
            raise AssertionError("Qdrant scroll did not return all bounded cached-vector points")
        max_abs_error = 0.0
        seen: set[str] = set()
        expected_by_id = {str(point["id"]): point for point in points}
        for item in stored:
            if not isinstance(item, dict):
                raise AssertionError("Qdrant scroll returned a malformed point")
            point_id = str(item.get("id"))
            expected = expected_by_id.get(point_id)
            if expected is None or point_id in seen or item.get("payload") != expected["payload"]:
                raise AssertionError("Qdrant scroll payload differs from the verified local cache record")
            actual_vector = _result_vector(item)
            expected_vector = expected_vectors[point_id]
            max_abs_error = max(max_abs_error, float(np.max(np.abs(actual_vector - expected_vector))))
            cosine = float(np.dot(actual_vector, expected_vector) / (
                np.linalg.norm(actual_vector) * np.linalg.norm(expected_vector)
            ))
            if cosine < 0.999999 or abs(float(np.linalg.norm(actual_vector)) - 1.0) > 1e-5:
                raise AssertionError("Qdrant vector round trip exceeded cosine or normalization tolerance")
            seen.add(point_id)
        if seen != set(expected_vectors):
            raise AssertionError("Qdrant point identities differ from the selected cached blocks")

        query_id = next(iter(expected_vectors))
        query = client.request("POST", f"/collections/{collection}/points/query", {
            "query": expected_vectors[query_id].tolist(),
            "limit": len(points),
            "with_payload": True,
            "with_vector": True,
            "filter": {"must": [{"key": "smoke_id", "match": {"value": smoke_id}}]},
        })
        query_result = query.get("result")
        hits = query_result.get("points") if isinstance(query_result, dict) else None
        hit_ids = {str(hit.get("id")) for hit in hits if isinstance(hit, dict)} if isinstance(hits, list) else set()
        if query_id not in hit_ids:
            raise AssertionError("Qdrant query did not retrieve the selected cached source vector")
        return_value = {
            "schema_version": "raama.financial-rag.qdrant-cached-vector-smoke-result.v1",
            "run_id": run_id,
            "embedding_model": adapter.embedding_model,
            "embedding_dimension": 1024,
            "verified_cache_vectors_loaded": len(cached),
            "document_embedding_inference": 0,
            "provider_requests": 0,
            "physical_points_written": len(points),
            "scroll_roundtrip_verified": True,
            "maximum_absolute_vector_error": max_abs_error,
            "cosine_similarity_at_least": 0.999999,
            "cached_vector_query_returned_target": True,
            "ephemeral_collection_deleted": True,
        }
    finally:
        if collection_created:
            client.request("DELETE", f"/collections/{collection}")
    rendered = json.dumps(return_value, ensure_ascii=False, sort_keys=True, indent=2)
    if args.result_file is not None:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
