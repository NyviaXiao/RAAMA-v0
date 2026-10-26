from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from adaptive_mas.industrial_rag.models import (
    Block,
    CorpusReleaseSpec,
    Document,
    DocumentBundle,
    DocumentVersion,
    Span,
)
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore
from adaptive_mas.industrial_rag.opensearch_adapter import (
    OpenSearchRequestError,
    OpenSearchTextIndexAdapter,
)
from adaptive_mas.industrial_rag.repository import IndustrialRagStore
from adaptive_mas.research_provenance import stable_hash


class SyntheticOpenSearchClient:
    def __init__(self) -> None:
        self.base_url = "http://synthetic-opensearch"
        self.username = None
        self.metadata: dict[str, object] | None = None
        self.documents: dict[str, dict[str, object]] = {}

    def request(self, method: str, path: str, body: object | bytes | None = None) -> dict[str, object]:
        if method == "GET" and path == "/":
            return {"version": {"number": "synthetic-opensearch"}}
        if method == "GET" and path.endswith("/_mapping"):
            if self.metadata is None:
                raise OpenSearchRequestError(404, method, path)
            handle = path.removeprefix("/").removesuffix("/_mapping")
            return {handle: {"mappings": {"_meta": self.metadata}}}
        if method == "PUT" and path.count("/") == 1:
            assert isinstance(body, dict)
            mappings = body["mappings"]
            assert isinstance(mappings, dict)
            self.metadata = mappings["_meta"]
            return {"acknowledged": True}
        if method == "POST" and path == "/_bulk":
            assert isinstance(body, bytes)
            lines = body.decode("utf-8").splitlines()
            for offset in range(0, len(lines), 2):
                action = json.loads(lines[offset])["index"]
                self.documents[str(action["_id"])] = json.loads(lines[offset + 1])
            return {"errors": False}
        if method == "POST" and path.endswith("/_refresh"):
            return {}
        if method == "GET" and path.endswith("/_count"):
            return {"count": len(self.documents)}
        if method == "POST" and path.endswith("/_mget"):
            assert isinstance(body, dict)
            return {"docs": [
                {"found": identifier in self.documents, "_source": self.documents.get(identifier)}
                for identifier in body["ids"]
            ]}
        raise AssertionError(f"unexpected synthetic OpenSearch request: {method} {path}")


def test_receipt_preserves_non_null_release_embedding_dimension(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    available = now - timedelta(minutes=20)
    payload = "Synthetic annual report paragraph".encode("utf-8")
    text = payload.decode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    document = Document(
        document_id="synthetic-document", title="Synthetic report", document_type="text",
        security_code="600036", entity_name="Synthetic issuer", report_period="2025",
        accounting_basis="consolidated",
    )
    version = DocumentVersion(
        version_id="synthetic-version", source_id="synthetic-source", content_sha256=digest,
        source_url="https://example.invalid/synthetic", rights_note="Synthetic test fixture",
        published_at=available, first_available_at=available, captured_at=now - timedelta(minutes=10),
        ingested_at=now - timedelta(minutes=5), time_basis="strict_forward",
        parser_name="synthetic", parser_version="1", parser_confidence=1.0, parser_status="parsed",
    )
    span = Span("synthetic-span", version.version_id, 1, 0, len(text), text)
    block = Block(
        "synthetic-block", version.version_id, text, "paragraph", page_number=1,
        char_start=0, char_end=len(text), parser_confidence=1.0, span_id=span.span_id,
    )
    bundle = DocumentBundle(
        document, version, payload, spans=(span,), blocks=(block,), read_principals=("local-public",),
    )
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(bundle)
    versions = [version.version_id]
    release_id = "release-hybrid-embedding"
    source_hash = stable_hash(store.source_manifest(versions))
    text_hash = str(store.preview_reference_index_manifest(release_id, versions)["index_manifest_sha256"])
    store.create_release(CorpusReleaseSpec(
        release_id, source_hash, "parser-test", "chunker-test", "Qwen3-Embedding-0.6B", 1024,
        ("opensearch",), text_hash, hashlib.sha256(b"synthetic vector manifest").hexdigest(),
    ), versions)

    adapter = OpenSearchTextIndexAdapter(store, SyntheticOpenSearchClient())
    receipt = adapter.build_release_index(release_id)
    assert receipt.embedding_dimension == 1024
    ready = store.mark_backend_ready(receipt, adapter.verify_release_index)
    assert ready["ready"] is True
    assert ready["missing_backends"] == []
