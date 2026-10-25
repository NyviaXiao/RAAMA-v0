from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from adaptive_mas.industrial_rag import qwen_models
from adaptive_mas.industrial_rag.qwen_models import Qwen3EmbeddingRuntime


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
