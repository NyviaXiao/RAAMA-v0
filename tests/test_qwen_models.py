from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from adaptive_mas.industrial_rag import qwen_models
from adaptive_mas.industrial_rag.models import QueryPlan
from adaptive_mas.industrial_rag.qwen_models import LocalQwenDenseAdapter, Qwen3EmbeddingRuntime


def test_last_token_pool_handles_mixed_left_padding_and_matches_unpadded_rows() -> None:
    # Hidden values encode their token index so the selected last real token is explicit.
    hidden = torch.tensor([
        [[0.0], [1.0], [2.0], [3.0]],
        [[10.0], [11.0], [12.0], [13.0]],
    ])
    mixed_left_mask = torch.tensor([
        [0, 0, 1, 1],
        [1, 1, 1, 1],
    ])
    pooled = Qwen3EmbeddingRuntime._last_token_pool(hidden, mixed_left_mask, torch)
    assert pooled.tolist() == [[3.0], [13.0]]
    assert pooled[0].tolist() == hidden[0, 3].tolist()


def test_last_token_pool_handles_right_padding() -> None:
    hidden = torch.tensor([
        [[0.0], [1.0], [2.0], [3.0]],
        [[10.0], [11.0], [12.0], [13.0]],
    ])
    right_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ])
    pooled = Qwen3EmbeddingRuntime._last_token_pool(hidden, right_mask, torch)
    assert pooled.tolist() == [[1.0], [13.0]]


def test_qwen_asset_manifest_rejects_modified_weights(tmp_path) -> None:
    for repo_id in (qwen_models.EMBEDDING_REPO, qwen_models.RERANKER_REPO):
        model_dir = qwen_models._model_dir(tmp_path, repo_id)
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
    weights = qwen_models._model_dir(tmp_path, qwen_models.EMBEDDING_REPO) / "weights.safetensors"
    weights.write_bytes(b"locked weights")
    manifest = {
        "schema_version": "raama.qwen-assets.v1",
        "embedding": {"repo_id": qwen_models.EMBEDDING_REPO, "revision": qwen_models.EMBEDDING_REVISION, "license": "apache-2.0"},
        "reranker": {"repo_id": qwen_models.RERANKER_REPO, "revision": qwen_models.RERANKER_REVISION, "license": "apache-2.0"},
        "files": qwen_models._inventory(tmp_path),
    }
    (tmp_path / qwen_models.MODEL_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    runtime = qwen_models.Qwen3EmbeddingRuntime(tmp_path)
    qwen_models._require_model_files(tmp_path, verify_hashes=True)

    weights.write_bytes(b"replaced weights")
    with pytest.raises(RuntimeError, match="SHA-256 integrity check"):
        runtime._load()


class _FakeEmbedding:
    dimension = 3

    def __init__(self, *, instruction_sha256: str = "instruction-a") -> None:
        self.cache_identity = {
            "model": "fake-embedding",
            "revision": "revision-a",
            "asset_manifest_sha256": "manifest-a",
            "instruction_sha256": instruction_sha256,
            "dimension": self.dimension,
        }
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def embed(self, texts, *, query=False, control=None):
        self.calls.append((tuple(texts), query))
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0, 0.0] if query or text == "first block" else [0.0, 1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


class _FakeBlockStore:
    def __init__(self) -> None:
        self.rows = [
            {
                "block_id": "block-1", "version_id": "version-1", "source_id": "source-1",
                "content_sha256": "a" * 64, "text": "first block", "security_code": "600036",
                "document_id": "document-1", "page_number": 1, "time_basis": "strict_forward",
                "payload_json": "{}",
            },
            {
                "block_id": "block-2", "version_id": "version-1", "source_id": "source-1",
                "content_sha256": "a" * 64, "text": "second block", "security_code": "600036",
                "document_id": "document-1", "page_number": 2, "time_basis": "strict_forward",
                "payload_json": "{}",
            },
        ]

    def read_blocks(self, _version_ids):
        return list(self.rows)


def _dense_plan() -> QueryPlan:
    return QueryPlan(
        plan_id="dense-plan-1",
        query="revenue 2024",
        security_code="600036",
        as_of=datetime(2026, 10, 25, tzinfo=timezone.utc),
        principal_id="local-public",
        task_type="fact",
        report_periods=("2024",),
        channels=("dense",),
        top_k=2,
    )


def test_dense_vectors_are_reused_only_for_verified_source_and_model_identity(tmp_path) -> None:
    store = _FakeBlockStore()
    cache_path = tmp_path / "dense-vectors.sqlite3"
    first_embedding = _FakeEmbedding()
    first_events: list[dict[str, object]] = []
    first = LocalQwenDenseAdapter(store, first_embedding, cache_path=cache_path)
    first_results = first.search_with_trace(_dense_plan(), ["version-1"], None, first_events.append)
    assert len(first_results) == 2
    assert first_embedding.calls[0][1] is True
    assert first_embedding.calls[1][1] is False
    assert first_events[-1]["stage"] == "dense_search_completed"
    assert len(first_events[-1]["ranked_candidates"]) == 2

    second_embedding = _FakeEmbedding()
    second_events: list[dict[str, object]] = []
    second = LocalQwenDenseAdapter(store, second_embedding, cache_capacity=1, cache_path=cache_path)
    second_results = second.search_with_trace(_dense_plan(), ["version-1"], None, second_events.append)
    assert [item.evidence_id for item in second_results] == [item.evidence_id for item in first_results]
    assert second_embedding.calls == [(('revenue 2024',), True)]
    assert second_events[-1]["cache"]["persistent_hits"] == 2
    assert len(second._vectors) == 1

    changed_model = _FakeEmbedding(instruction_sha256="instruction-b")
    changed = LocalQwenDenseAdapter(store, changed_model, cache_path=cache_path)
    changed.search_with_trace(_dense_plan(), ["version-1"], None, lambda _event: None)
    assert changed_model.calls[1][1] is False

    store.rows[0]["content_sha256"] = "b" * 64
    changed_source_embedding = _FakeEmbedding()
    changed_source_events: list[dict[str, object]] = []
    LocalQwenDenseAdapter(store, changed_source_embedding, cache_path=cache_path).search_with_trace(
        _dense_plan(), ["version-1"], None, changed_source_events.append,
    )
    assert any(not query for _texts, query in changed_source_embedding.calls)
    assert changed_source_events[-1]["cache"]["persistent_hits"] == 1


def test_required_dense_cache_is_preflighted_and_never_reembeds_source_blocks(tmp_path) -> None:
    store = _FakeBlockStore()
    cache_path = tmp_path / "dense-vectors.sqlite3"
    LocalQwenDenseAdapter(store, _FakeEmbedding(), cache_path=cache_path).search_with_trace(
        _dense_plan(), ["version-1"], None, lambda _event: None,
    )
    copied_cache_path = tmp_path / "copied-dense-vectors.sqlite3"
    LocalQwenDenseAdapter.copy_persistent_cache(cache_path, copied_cache_path)

    embedding = _FakeEmbedding()
    adapter = LocalQwenDenseAdapter(
        store, embedding, cache_path=copied_cache_path, require_cached_vectors=True,
    )
    assert adapter.verify_persistent_cache(["version-1"]) == 2
    adapter.search_with_trace(_dense_plan(), ["version-1"], None, lambda _event: None)
    assert embedding.calls == [(('revenue 2024',), True)]


def test_required_dense_cache_fails_closed_before_document_embedding_on_miss(tmp_path) -> None:
    store = _FakeBlockStore()
    embedding = _FakeEmbedding()
    adapter = LocalQwenDenseAdapter(
        store, embedding, cache_path=tmp_path / "empty-cache.sqlite3", require_cached_vectors=True,
    )
    with pytest.raises(RuntimeError, match="missing or invalid for 2 source block"):
        adapter.verify_persistent_cache(["version-1"])
    with pytest.raises(RuntimeError, match="missing or invalid for 2 source block"):
        adapter.search_with_trace(_dense_plan(), ["version-1"], None, lambda _event: None)
    assert embedding.calls == [(('revenue 2024',), True)]


def test_cached_vectors_can_scope_verification_to_release_indexed_blocks(tmp_path) -> None:
    store = _FakeBlockStore()
    cache_path = tmp_path / "dense-vectors.sqlite3"
    LocalQwenDenseAdapter(store, _FakeEmbedding(), cache_path=cache_path).search_with_trace(
        _dense_plan(), ["version-1"], None, lambda _event: None,
    )
    adapter = LocalQwenDenseAdapter(store, _FakeEmbedding(), cache_path=cache_path, require_cached_vectors=True)

    vectors = adapter.cached_vectors(["version-1"], block_keys=[("version-1", "block-2")])
    assert [str(row["block_id"]) for row, _vector in vectors] == ["block-2"]
    with pytest.raises(RuntimeError, match="missing or invalid for 1 source block"):
        adapter.cached_vectors(["version-1"], block_keys=[("version-1", "missing-block")])


def test_dense_vector_cache_detects_blob_tampering_and_recomputes(tmp_path) -> None:
    store = _FakeBlockStore()
    cache_path = tmp_path / "dense-vectors.sqlite3"
    first = LocalQwenDenseAdapter(store, _FakeEmbedding(), cache_path=cache_path)
    first.search_with_trace(_dense_plan(), ["version-1"], None, lambda _event: None)

    connection = qwen_models.sqlite3.connect(cache_path)
    connection.execute("UPDATE dense_vectors SET vector_blob=? WHERE block_id='block-1'", (b"tampered",))
    connection.commit()
    connection.close()

    embedding = _FakeEmbedding()
    events: list[dict[str, object]] = []
    LocalQwenDenseAdapter(store, embedding, cache_path=cache_path).search_with_trace(
        _dense_plan(), ["version-1"], None, events.append,
    )
    assert any(not query for _texts, query in embedding.calls)
    assert events[-1]["cache"]["persistent_hits"] == 1
