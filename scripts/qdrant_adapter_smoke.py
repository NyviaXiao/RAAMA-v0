"""Run a bounded synthetic SQLite-to-Qdrant application retrieval probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.industrial_rag.models import (  # noqa: E402
    Block,
    CorpusReleaseSpec,
    Document,
    DocumentBundle,
    DocumentVersion,
    QueryPlan,
    Span,
)
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore  # noqa: E402
from adaptive_mas.industrial_rag.qdrant_adapter import (  # noqa: E402
    QdrantBackendError,
    QdrantRestClient,
    QdrantVectorIndexAdapter,
)
from adaptive_mas.industrial_rag.repository import IndustrialRagStore  # noqa: E402
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService  # noqa: E402
from adaptive_mas.research_provenance import stable_hash  # noqa: E402


class SyntheticEmbedding:
    dimension = 2
    cache_identity = {
        "model": "synthetic-qdrant-smoke-embedding",
        "revision": "fixture-v1",
        "dimension": 2,
    }

    def embed(self, texts: list[str], *, query: bool = False, control: Any = None) -> np.ndarray:
        if control is not None:
            control.check()
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class SyntheticCachedDenseAdapter:
    def __init__(self, store: IndustrialRagStore) -> None:
        self.store = store
        self.embedding = SyntheticEmbedding()
        self.cache_identity = {"model_identity": self.embedding.cache_identity}

    def cached_vectors(
        self, version_ids: list[str], *, block_keys: list[tuple[str, str]] | None = None,
    ) -> list[tuple[dict[str, object], np.ndarray]]:
        selected = set(block_keys) if block_keys is not None else None
        return [
            (dict(row), np.asarray([1.0, 0.0], dtype=np.float32))
            for row in self.store.read_blocks(version_ids)
            if str(row["text"]).strip()
            and (selected is None or (str(row["version_id"]), str(row["block_id"])) in selected)
        ]


def _synthetic_bundle(run_id: str) -> DocumentBundle:
    now = datetime.now(timezone.utc)
    published = now - timedelta(minutes=5)
    payload = "Synthetic report says revenue was 10 million yuan.".encode("utf-8")
    text = payload.decode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    document = Document(
        document_id=f"smoke-document-{run_id}", title="Synthetic smoke report", document_type="text",
        security_code="000001", entity_name="Synthetic issuer", report_period="2025",
        accounting_basis="consolidated",
    )
    version = DocumentVersion(
        version_id=f"smoke-version-{run_id}", source_id=f"smoke-source-{run_id}",
        content_sha256=digest, source_url="https://example.invalid/synthetic",
        rights_note="Synthetic fixture only", published_at=published,
        first_available_at=published, captured_at=now - timedelta(minutes=4),
        ingested_at=now - timedelta(minutes=3), time_basis="strict_forward",
        parser_name="synthetic", parser_version="1", parser_confidence=1.0, parser_status="parsed",
    )
    span = Span(f"smoke-span-{run_id}", version.version_id, 1, 0, len(text), text)
    block = Block(
        f"smoke-block-{run_id}", version.version_id, text, "paragraph", page_number=1,
        char_start=0, char_end=len(text), parser_confidence=1.0, span_id=span.span_id,
    )
    return DocumentBundle(
        document, version, payload, spans=(span,), blocks=(block,), read_principals=("local-public",),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("-", "").isalnum():
        raise ValueError("run-id must contain only ASCII letters, digits, and hyphens")
    run_id = args.run_id.lower()
    release_id = f"qdrant-smoke-{run_id}"
    bundle = _synthetic_bundle(run_id)

    with tempfile.TemporaryDirectory(prefix="raama-qdrant-smoke-") as temporary:
        root = Path(temporary)
        store = IndustrialRagStore(root / "rag.sqlite3", LocalContentAddressedStore(root / "objects"))
        store.ingest_bundle(bundle)
        version_ids = [bundle.version.version_id]
        client = QdrantRestClient(args.qdrant_url)
        adapter = QdrantVectorIndexAdapter(store, client, SyntheticCachedDenseAdapter(store))
        vector_manifest = adapter.preview_vector_index_manifest(release_id, version_ids)
        source_hash = stable_hash(store.source_manifest(version_ids))
        text_hash = str(store.preview_reference_index_manifest(release_id, version_ids)["index_manifest_sha256"])
        spec = CorpusReleaseSpec(
            release_id, source_hash, "synthetic@1", "raama-financial-chunker-v1",
            str(vector_manifest["embedding_model"]), int(vector_manifest["embedding_dimension"]),
            ("qdrant",), text_hash, str(vector_manifest["index_manifest_sha256"]),
        )
        store.create_release(spec, version_ids)
        query_requests = 0
        original_request = client.request

        def tracked_request(method: str, path: str, body: object | None = None) -> dict[str, Any]:
            nonlocal query_requests
            if path.endswith("/points/query"):
                query_requests += 1
            return original_request(method, path, body)

        client.request = tracked_request  # type: ignore[method-assign]
        readiness = adapter.mark_release_ready(release_id)
        publication = store.publish_release(release_id, actor_id="qdrant-smoke", reason="Synthetic Qdrant adapter acceptance.")
        plan = QueryPlan(
            plan_id=f"qdrant-query-{run_id}", query="What was revenue?", security_code="000001",
            as_of=datetime.now(timezone.utc), principal_id="local-public", task_type="narrative",
            report_periods=("2025",), release_id=release_id, channels=("dense",), top_k=5,
        )
        searched = IndustrialRagService(
            store, channel_adapters={"dense": adapter}, allow_retrospective_diagnostic=False,
        ).search(plan, use_cache=False)
        if not searched["retrieved"] or searched["channels"]["dense"]["implementation"] != adapter.implementation:
            raise AssertionError("release-bound Qdrant application search returned no source evidence")
        completed_queries = query_requests

        handle = adapter._collection_handle(release_id)
        scroll = client.request(
            "POST", f"/collections/{handle}/points/scroll",
            {"limit": 1, "with_payload": True, "with_vector": True},
        )
        stored_points = scroll.get("result", {}).get("points", [])
        if not stored_points:
            raise AssertionError("Qdrant smoke collection did not retain its indexed point")
        point = stored_points[0]
        forged_vector = [0.0, 1.0]
        forged_payload = dict(point["payload"])
        forged_payload.pop("record_sha256")
        forged_payload["vector_sha256"] = adapter._vector_sha256(forged_vector)
        forged_payload["record_sha256"] = stable_hash(forged_payload)
        if (
            forged_payload["vector_sha256"] != adapter._vector_sha256(forged_vector)
            or forged_payload["record_sha256"] != stable_hash({
                key: value for key, value in forged_payload.items() if key != "record_sha256"
            })
        ):
            raise AssertionError("Qdrant smoke tamper fixture is not internally self-consistent")
        client.request(
            "PUT", f"/collections/{handle}/points?wait=true",
            {"points": [{"id": point["id"], "vector": forged_vector, "payload": forged_payload}]},
        )
        try:
            adapter.search_with_control(plan, version_ids, None)
        except QdrantBackendError:
            pass
        else:
            raise AssertionError("Qdrant accepted a forged vector with matching payload hashes")
        if query_requests != completed_queries + 1:
            raise AssertionError("Qdrant tamper check did not exercise the application query path")
        completed_queries = query_requests

        try:
            adapter.search_with_control(replace(plan, release_id=f"{release_id}-inactive"), version_ids, None)
        except QdrantBackendError:
            pass
        else:
            raise AssertionError("Qdrant accepted a release other than the active publication")
        store.set_access("local-public", bundle.document.document_id, can_read=False)
        try:
            try:
                adapter.search_with_control(plan, version_ids, None)
            except PermissionError:
                pass
            else:
                raise AssertionError("Qdrant returned a source to a revoked principal")
        finally:
            store.set_access("local-public", bundle.document.document_id, can_read=True)
        before_available = replace(plan, as_of=bundle.version.first_available_at - timedelta(seconds=1))
        try:
            adapter.search_with_control(before_available, version_ids, None)
        except PermissionError:
            pass
        else:
            raise AssertionError("Qdrant returned a source before its recorded availability time")
        if query_requests != completed_queries:
            raise AssertionError("a release, ACL, or as-of rejection issued an external Qdrant query")
        if not readiness["ready"] or publication["status"] != "published":
            raise AssertionError("Qdrant receipt and corpus release publication did not complete")
        print(json.dumps({
            "schema_version": "raama.financial-rag.qdrant-adapter-smoke-result.v1",
            "run_id": run_id,
            "release_id": release_id,
            "qdrant_receipt_ready": True,
            "release_bound_dense_search": True,
            "coherent_vector_payload_tamper_rejected": True,
            "nonactive_release_rejected_before_query": True,
            "revoked_principal_rejected_before_query": True,
            "pre_availability_query_rejected_before_query": True,
            "query_request_count": query_requests,
            "source_text_is_synthetic": True,
        }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
