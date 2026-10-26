from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from adaptive_mas.industrial_rag.models import (
    Block,
    CorpusReleaseSpec,
    Document,
    DocumentBundle,
    DocumentVersion,
    QueryPlan,
    Span,
)
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore
from adaptive_mas.industrial_rag.qdrant_adapter import (
    QdrantRequestError,
    QdrantVectorIndexAdapter,
)
from adaptive_mas.industrial_rag.qwen_models import (
    EMBEDDING_REPO,
    EMBEDDING_REVISION,
    LocalQwenDenseAdapter,
)
from adaptive_mas.industrial_rag.repository import IndustrialRagStore
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService
from adaptive_mas.research_provenance import stable_hash


class SyntheticEmbedding:
    dimension = 1024
    cache_identity = {
        "model": EMBEDDING_REPO,
        "revision": EMBEDDING_REVISION,
        "dimension": dimension,
        "runtime": "synthetic-contract-fixture",
    }

    def __init__(self) -> None:
        self.vector = np.zeros(self.dimension, dtype=np.float32)
        self.vector[0] = 1.0

    def embed(self, texts, *, query=False, control=None):
        assert query
        if control is not None:
            control.check()
        return np.repeat(self.vector[None, :], len(texts), axis=0)


class SyntheticQdrantClient:
    base_url = "http://synthetic-qdrant"

    def __init__(self) -> None:
        self.metadata = None
        self.vector_config = None
        self.points: dict[str, dict[str, object]] = {}
        self.query_count = 0

    def request(self, method, path, body=None):
        if method == "GET" and path == "/":
            return {"version": "synthetic-qdrant-1"}
        if method == "GET" and path.startswith("/collections/"):
            if self.metadata is None:
                raise QdrantRequestError(404, method, path)
            return {"result": {
                "config": {
                    "params": {"vectors": self.vector_config},
                    "metadata": self.metadata,
                },
                "points_count": len(self.points),
            }}
        if method == "PUT" and path.startswith("/collections/") and path.count("/") == 2:
            assert isinstance(body, dict)
            self.metadata = body["metadata"]
            self.vector_config = body["vectors"]
            return {"result": True}
        if method == "PUT" and path.endswith("/points?wait=true"):
            assert isinstance(body, dict)
            self.points.update({str(point["id"]): dict(point) for point in body["points"]})
            return {"result": {"status": "completed"}}
        if method == "POST" and path.endswith("/points/scroll"):
            return {"result": {"points": list(self.points.values()), "next_page_offset": None}}
        if method == "POST" and path.endswith("/points/query"):
            self.query_count += 1
            assert isinstance(body, dict)
            query = np.asarray(body["query"], dtype=np.float32)
            hits = []
            for point in self.points.values():
                vector = np.asarray(point["vector"], dtype=np.float32)
                hits.append({**point, "score": float(np.dot(query, vector))})
            hits.sort(key=lambda row: (-float(row["score"]), str(row["id"])))
            return {"result": {"points": hits[: int(body["limit"]) ]}}
        raise AssertionError(f"unexpected synthetic Qdrant request: {method} {path}")


def test_qwen_1024_qdrant_release_receipt_publication_and_permission_scoped_retrieval(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    first_available = now - timedelta(days=3)
    captured = now - timedelta(days=2)
    ingested = now - timedelta(days=1)
    payload = b"Synthetic report states revenue of 100 million yuan for 2025."
    text = payload.decode("utf-8")
    content_sha256 = hashlib.sha256(payload).hexdigest()
    document = Document(
        document_id="qdrant-document", title="Synthetic annual report", document_type="text",
        security_code="600036", entity_name="Synthetic issuer", report_period="2025",
        accounting_basis="consolidated",
    )
    version = DocumentVersion(
        version_id="qdrant-version", source_id="qdrant-source", content_sha256=content_sha256,
        source_url="https://example.invalid/qdrant-fixture", rights_note="Synthetic fixture only",
        published_at=first_available, first_available_at=first_available, captured_at=captured,
        ingested_at=ingested, time_basis="strict_forward", parser_name="synthetic-parser",
        parser_version="1", parser_confidence=0.95, parser_status="parsed",
    )
    span = Span("qdrant-span", version.version_id, 1, 0, len(text), text)
    block = Block(
        "qdrant-block", version.version_id, text, "paragraph", page_number=1,
        char_start=0, char_end=len(text), span_id=span.span_id,
    )
    store = IndustrialRagStore(
        tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"),
    )
    store.ingest_bundle(DocumentBundle(
        document, version, payload, spans=(span,), blocks=(block,), read_principals=("local-public",),
    ))
    version_ids = [version.version_id]
    release_id = "qwen-1024-qdrant-release"
    dense = LocalQwenDenseAdapter(
        store,
        SyntheticEmbedding(),
        cache_path=tmp_path / "qwen-vectors.sqlite3",
        require_cached_vectors=True,
    )
    for row in store.read_blocks(version_ids):
        key, identity = dense._vector_key(row, str(row["text"]))
        dense._write_persistent_vector(key, identity, dense.embedding.vector)
    assert dense._cache_connection is not None
    dense._cache_connection.commit()
    assert dense.verify_persistent_cache(version_ids) == 1
    client = SyntheticQdrantClient()
    adapter = QdrantVectorIndexAdapter(store, client, dense)
    vector_manifest = adapter.preview_vector_index_manifest(release_id, version_ids)
    source_manifest_sha256 = stable_hash(store.source_manifest(version_ids))
    text_manifest_sha256 = str(
        store.preview_reference_index_manifest(release_id, version_ids)["index_manifest_sha256"]
    )
    store.create_release(CorpusReleaseSpec(
        release_id=release_id,
        source_manifest_sha256=source_manifest_sha256,
        parser_version="synthetic-parser@1",
        chunker_version="synthetic-chunker@1",
        embedding_model=f"{EMBEDDING_REPO}@{EMBEDDING_REVISION}",
        embedding_dimension=1024,
        required_backends=("sqlite_reference", "qdrant"),
        text_index_manifest_sha256=text_manifest_sha256,
        vector_index_manifest_sha256=str(vector_manifest["index_manifest_sha256"]),
    ), version_ids)

    reference_receipt = store.build_reference_index_receipt(release_id)
    store.mark_backend_ready(reference_receipt, store.verify_reference_index)
    qdrant_ready = adapter.mark_release_ready(release_id)
    assert qdrant_ready["ready"] is True
    assert qdrant_ready["missing_backends"] == []
    assert client.vector_config["size"] == 1024
    assert store.publish_release(release_id, actor_id="synthetic-operator", reason="Exercise Qdrant app path.")["status"] == "published"

    plan = QueryPlan(
        plan_id="qdrant-query", query="2025 revenue", security_code="600036", as_of=now,
        principal_id="local-public", task_type="fact", report_periods=("2025",),
        release_id=release_id, channels=("dense",), top_k=3,
    )
    service = IndustrialRagService(store, channel_adapters={"dense": adapter})
    result = service.search(
        plan, use_cache=False, use_reranker=False, fusion_strategy="first_available_channel",
    )
    evidence = result["retrieved"]
    assert result["status"] == "available"
    assert result["channels"]["dense"]["candidate_count"] == 1
    assert len(evidence) == 1
    assert evidence[0]["channel"] == "dense"
    assert evidence[0]["block_id"] == block.block_id
    assert evidence[0]["citation"]["embedding_model"] == EMBEDDING_REPO
    assert evidence[0]["citation"]["embedding_revision"] == EMBEDDING_REVISION
    assert client.query_count == 1

    unauthorized = QueryPlan(
        plan_id="qdrant-query-denied", query="2025 revenue", security_code="600036", as_of=now,
        principal_id="no-access", task_type="fact", report_periods=("2025",),
        release_id=release_id, channels=("dense",), top_k=3,
    )
    unauthorized_result = service.search(
        unauthorized, use_cache=False, use_reranker=False,
        fusion_strategy="first_available_channel",
    )
    assert unauthorized_result["status"] == "no_evidence"
    assert unauthorized_result["retrieved"] == []
    assert unauthorized_result["channels"]["dense"]["candidate_count"] == 0
    assert client.query_count == 1
