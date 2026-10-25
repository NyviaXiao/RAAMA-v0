"""Pinned CPU Qwen embedding and reranking adapters for local retrieval."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

from adaptive_mas.industrial_rag.models import EvidenceResult, QueryPlan
from adaptive_mas.runtime_limits import apply_thread_limits, process_thread_limit


apply_thread_limits()

import numpy as np


EMBEDDING_REPO = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RERANKER_REPO = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_MANIFEST = "raama_qwen_models.json"
RETRIEVAL_INSTRUCTION = (
    "Given a Chinese A-share financial research query, retrieve passages that provide "
    "source-backed evidence. Match issuer, report period, accounting basis, unit, and metric."
)
RERANK_MAX_LENGTH = 4096
EMBED_MAX_LENGTH = 8192
EMBED_BATCH_SIZE = 8
RERANK_BATCH_SIZE = 1


def _model_dir(root: Path, repo_id: str) -> Path:
    return root / repo_id.replace("/", "--")


def _inventory(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != MODEL_MANIFEST):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for part in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(part)
        entries.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": digest.hexdigest()})
    return entries


def download_qwen_models(asset_dir: str | Path) -> dict[str, object]:
    """Fetch the two immutable Hugging Face revisions into a local-only directory."""
    apply_thread_limits()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to fetch the pinned Qwen models") from error
    root = Path(asset_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for repo_id, revision in ((EMBEDDING_REPO, EMBEDDING_REVISION), (RERANKER_REPO, RERANKER_REVISION)):
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=_model_dir(root, repo_id))
    manifest: dict[str, object] = {
        "schema_version": "raama.qwen-assets.v1",
        "embedding": {"repo_id": EMBEDDING_REPO, "revision": EMBEDDING_REVISION, "license": "apache-2.0"},
        "reranker": {"repo_id": RERANKER_REPO, "revision": RERANKER_REVISION, "license": "apache-2.0"},
        "files": _inventory(root),
    }
    path = root / MODEL_MANIFEST
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"asset_dir": str(root), "manifest_path": str(path), "file_count": len(manifest["files"])}


def _require_model_files(root: Path, *, verify_hashes: bool = False) -> None:
    try:
        manifest = json.loads((root / MODEL_MANIFEST).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        raise RuntimeError("pinned Qwen files are missing; run the explicit retrieval-model fetch command first") from None
    if (
        manifest.get("schema_version") != "raama.qwen-assets.v1"
        or manifest.get("embedding") != {"repo_id": EMBEDDING_REPO, "revision": EMBEDDING_REVISION, "license": "apache-2.0"}
        or manifest.get("reranker") != {"repo_id": RERANKER_REPO, "revision": RERANKER_REVISION, "license": "apache-2.0"}
        or not manifest.get("files")
    ):
        raise RuntimeError("Qwen model manifest does not match the pinned model contract")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("Qwen model manifest has no asset inventory")
    for repo_id in (EMBEDDING_REPO, RERANKER_REPO):
        if not (_model_dir(root, repo_id) / "config.json").is_file():
            raise RuntimeError(f"pinned local model files are incomplete for {repo_id}")
    if verify_hashes:
        actual = _inventory(root)
        if actual != entries:
            raise RuntimeError("Qwen assets failed the recorded path, size, or SHA-256 integrity check")


class Qwen3EmbeddingRuntime:
    def __init__(self, asset_dir: str | Path) -> None:
        self.asset_dir = Path(asset_dir).resolve()
        _require_model_files(self.asset_dir)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        _require_model_files(self.asset_dir, verify_hashes=True)
        apply_thread_limits(apply_torch=True)
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("install the document-ai optional dependencies to run Qwen inference") from error
        model_path = _model_dir(self.asset_dir, EMBEDDING_REPO)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
        self._model = AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32).to("cpu").eval()
        self._torch = (torch, functional)

    @staticmethod
    def _last_token_pool(hidden: Any, mask: Any, torch: Any) -> Any:
        if bool(torch.all(mask[:, -1] == 1)):
            return hidden[:, -1]
        lengths = mask.sum(dim=1) - 1
        indexes = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[indexes, lengths]

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        self._load()
        torch, functional = self._torch
        prepared = [
            f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery:{text}" if query else text
            for text in texts
        ]
        vectors: list[np.ndarray] = []
        for start in range(0, len(prepared), EMBED_BATCH_SIZE):
            batch = prepared[start:start + EMBED_BATCH_SIZE]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=EMBED_MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = self._model(**tokens)
                pooled = self._last_token_pool(output.last_hidden_state, tokens["attention_mask"], torch)
                normalized = functional.normalize(pooled, p=2, dim=1)
            vectors.append(normalized.to(dtype=torch.float32, device="cpu").numpy())
        if not vectors:
            dimension = int(getattr(self._model.config, "hidden_size", 1024))
            return np.empty((0, dimension), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


class Qwen3RerankerRuntime:
    def __init__(self, asset_dir: str | Path) -> None:
        self.asset_dir = Path(asset_dir).resolve()
        _require_model_files(self.asset_dir)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        _require_model_files(self.asset_dir, verify_hashes=True)
        apply_thread_limits(apply_torch=True)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("install the document-ai optional dependencies to run Qwen inference") from error
        model_path = _model_dir(self.asset_dir, RERANKER_REPO)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32,
        ).to("cpu").eval()
        self._tokenizer.padding_side = "left"
        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        if self._yes_id is None or self._no_id is None or self._yes_id == self._tokenizer.unk_token_id or self._no_id == self._tokenizer.unk_token_id:
            raise RuntimeError("Qwen reranker tokenizer does not expose the expected yes/no tokens")
        self._torch = torch

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self._load()
        torch = self._torch
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        pair_texts = [
            f"<Instruct>: {RETRIEVAL_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
            for document in documents
        ]
        prefix_ids = self._tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self._tokenizer.encode(suffix, add_special_tokens=False)
        scores: list[float] = []
        max_content = RERANK_MAX_LENGTH - len(prefix_ids) - len(suffix_ids)
        for start in range(0, len(pair_texts), RERANK_BATCH_SIZE):
            encoded = self._tokenizer(
                pair_texts[start:start + RERANK_BATCH_SIZE],
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_content,
            )
            for index, input_ids in enumerate(encoded["input_ids"]):
                encoded["input_ids"][index] = prefix_ids + input_ids + suffix_ids
            tokens = self._tokenizer.pad(encoded, padding=True, return_tensors="pt")
            with torch.inference_mode():
                logits = self._model(**tokens).logits[:, -1, :]
                batch_scores = logits[:, self._yes_id] - logits[:, self._no_id]
            scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
        return scores


class LocalQwenDenseAdapter:
    """In-memory dense reference adapter over the already permission-filtered blocks."""

    implementation = "local_in_memory_qwen3_embedding_0.6b"

    def __init__(self, store: Any, embedding: Qwen3EmbeddingRuntime, *, cache_capacity: int = 2048) -> None:
        self.store = store
        self.embedding = embedding
        self.cache_capacity = max(1, cache_capacity)
        self._vectors: OrderedDict[str, np.ndarray] = OrderedDict()

    def __call__(self, plan: QueryPlan, version_ids: Sequence[str]) -> Sequence[EvidenceResult]:
        rows = [
            row for row in self.store.read_blocks(version_ids)
            if str(row["security_code"] or "") == plan.security_code and str(row["text"]).strip()
        ]
        if not rows:
            return []
        query_text = plan.query + (" " + " ".join(plan.aliases) if plan.aliases else "")
        query_vector = self.embedding.embed([query_text], query=True)[0]
        from adaptive_mas.industrial_rag.retrieval import _citation

        ranked = []
        batch_rows: list[Any] = []
        batch_keys: list[str] = []
        for row in rows:
            text = str(row["text"])
            key = f"{row['version_id']}:{row['block_id']}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            vector = self._vectors.get(key)
            if vector is not None:
                self._vectors.move_to_end(key)
                ranked.append((float(np.dot(query_vector, vector)), row))
                continue
            batch_rows.append(row)
            batch_keys.append(key)
            if len(batch_rows) == EMBED_BATCH_SIZE:
                self._score_missing_batch(batch_rows, batch_keys, query_vector, ranked)
                batch_rows, batch_keys = [], []
        if batch_rows:
            self._score_missing_batch(batch_rows, batch_keys, query_vector, ranked)
        ranked.sort(key=lambda item: (-item[0], str(item[1]["block_id"])))
        return [
            EvidenceResult(
                evidence_id=f"ev-dense-{row['block_id']}", channel="dense",
                document_id=str(row["document_id"]), version_id=str(row["version_id"]),
                source_id=str(row["source_id"]), content_sha256=str(row["content_sha256"]),
                page_number=int(row["page_number"]) if row["page_number"] is not None else None,
                block_id=str(row["block_id"]), fact_id=None, text=str(row["text"]),
                score=score, source_time_basis=str(row["time_basis"]),
                citation={**_citation(row), "embedding_model": EMBEDDING_REPO,
                          "embedding_revision": EMBEDDING_REVISION,
                          "ranking_score_kind": "cosine_similarity", "scores_are_probabilities": False},
            )
            for score, row in ranked[:plan.top_k * 5]
        ]

    def _score_missing_batch(self, rows: list[Any], keys: list[str], query_vector: np.ndarray, ranked: list[Any]) -> None:
        vectors = self.embedding.embed([str(row["text"]) for row in rows])
        for row, key, vector in zip(rows, keys, vectors):
            self._vectors[key] = vector
            self._vectors.move_to_end(key)
            ranked.append((float(np.dot(query_vector, vector)), row))
        while len(self._vectors) > self.cache_capacity:
            self._vectors.popitem(last=False)


class LocalQwenReranker:
    implementation = "local_cpu_qwen3_reranker_0.6b"

    def __init__(self, runtime: Qwen3RerankerRuntime) -> None:
        self.runtime = runtime

    def __call__(self, plan: QueryPlan, evidence: Sequence[EvidenceResult]) -> Sequence[EvidenceResult]:
        scores = self.runtime.score(plan.query, [item.text for item in evidence])
        paired = []
        for item, score in zip(evidence, scores):
            citation = {
                **item.citation,
                "reranker_model": RERANKER_REPO,
                "reranker_revision": RERANKER_REVISION,
                "ranking_score_kind": "yes_no_logit_difference",
                "scores_are_probabilities": False,
            }
            paired.append((score, EvidenceResult(
                evidence_id=item.evidence_id, channel=item.channel,
                document_id=item.document_id, version_id=item.version_id,
                source_id=item.source_id, content_sha256=item.content_sha256,
                page_number=item.page_number, block_id=item.block_id, fact_id=item.fact_id,
                text=item.text, score=score, source_time_basis=item.source_time_basis,
                citation=citation,
            )))
        paired.sort(key=lambda value: (-value[0], value[1].evidence_id))
        return [item for _score, item in paired]


def create_local_qwen_adapters(store: Any, asset_dir: str | Path) -> tuple[dict[str, Any], LocalQwenReranker]:
    root = Path(asset_dir).resolve()
    _require_model_files(root)
    embedding = Qwen3EmbeddingRuntime(root)
    reranker = Qwen3RerankerRuntime(root)
    return {"dense": LocalQwenDenseAdapter(store, embedding)}, LocalQwenReranker(reranker)
