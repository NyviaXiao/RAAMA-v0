"""Pinned CPU Qwen embedding and reranking adapters for local retrieval."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Sequence

from adaptive_mas.industrial_rag.models import EvidenceResult, QueryPlan, SearchControl
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

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return int(self._model.config.hidden_size)
        config = json.loads((_model_dir(self.asset_dir, EMBEDDING_REPO) / "config.json").read_text(encoding="utf-8"))
        return int(config["hidden_size"])

    @property
    def cache_identity(self) -> dict[str, object]:
        manifest = self.asset_dir / MODEL_MANIFEST
        return {
            "model": EMBEDDING_REPO,
            "revision": EMBEDDING_REVISION,
            "asset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "instruction_sha256": hashlib.sha256(RETRIEVAL_INSTRUCTION.encode("utf-8")).hexdigest(),
            "pooling": "masked_last_token_l2_v1",
            "max_length": EMBED_MAX_LENGTH,
            "dimension": self.dimension,
        }

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

    def embed(
        self,
        texts: Sequence[str],
        *,
        query: bool = False,
        control: SearchControl | None = None,
    ) -> np.ndarray:
        if control is not None:
            control.check()
        self._load()
        if control is not None:
            control.check()
        torch, functional = self._torch
        prepared = [
            f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery:{text}" if query else text
            for text in texts
        ]
        vectors: list[np.ndarray] = []
        for start in range(0, len(prepared), EMBED_BATCH_SIZE):
            if control is not None:
                control.check()
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
            if control is not None:
                control.check()
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

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        control: SearchControl | None = None,
        on_batch: Callable[[int, list[float], float], None] | None = None,
        on_batch_start: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        if control is not None:
            control.check()
        self._load()
        if control is not None:
            control.check()
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
            if control is not None:
                control.check()
            if on_batch_start is not None:
                on_batch_start(start, min(start + RERANK_BATCH_SIZE, len(pair_texts)))
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
            batch_started = time.perf_counter()
            with torch.inference_mode():
                logits = self._model(**tokens).logits[:, -1, :]
                batch_scores = logits[:, self._yes_id] - logits[:, self._no_id]
            values = [float(value) for value in batch_scores.detach().cpu().tolist()]
            scores.extend(values)
            if on_batch is not None:
                on_batch(start, values, time.perf_counter() - batch_started)
            if control is not None:
                control.check()
        return scores


class LocalQwenDenseAdapter:
    """Dense retrieval over permission-filtered blocks with an identity-bound local vector cache."""

    implementation = "local_cached_qwen3_embedding_0.6b"
    cache_schema = "raama.qwen-dense-vectors.v1"

    def __init__(
        self,
        store: Any,
        embedding: Qwen3EmbeddingRuntime,
        *,
        cache_capacity: int = 2048,
        cache_path: str | Path | None = None,
        require_cached_vectors: bool = False,
    ) -> None:
        self.store = store
        self.embedding = embedding
        self.cache_capacity = max(1, cache_capacity)
        self.require_cached_vectors = bool(require_cached_vectors)
        self._vectors: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_path = Path(cache_path).resolve() if cache_path is not None else None
        self._cache_identity = self._build_cache_identity()
        self._cache_identity_sha256 = hashlib.sha256(
            json.dumps(self._cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._cache_connection: sqlite3.Connection | None = None
        if self._cache_path is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_connection = sqlite3.connect(self._cache_path, timeout=30)
            self._cache_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dense_vectors (
                    cache_key TEXT PRIMARY KEY,
                    cache_schema TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_content_sha256 TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    block_text_sha256 TEXT NOT NULL,
                    model_identity_sha256 TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_dtype TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    vector_sha256 TEXT NOT NULL
                )
                """
            )
            self._cache_connection.commit()

    @property
    def cache_identity(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "cache_schema": self.cache_schema,
            "model_identity": self._cache_identity,
        }

    def _build_cache_identity(self) -> dict[str, object]:
        identity = getattr(self.embedding, "cache_identity", None)
        if not isinstance(identity, dict):
            raise ValueError("Qwen embedding runtime must expose a versioned cache identity")
        if identity.get("dimension") != self.embedding.dimension:
            raise ValueError("Qwen embedding cache identity dimension differs from the runtime")
        return dict(identity)

    def _vector_key(self, row: Any, text: str) -> tuple[str, dict[str, object]]:
        metadata: dict[str, object] = {
            "cache_schema": self.cache_schema,
            "source_id": str(row["source_id"]),
            "source_content_sha256": str(row["content_sha256"]),
            "block_id": str(row["block_id"]),
            "block_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "model_identity_sha256": self._cache_identity_sha256,
            "dimension": int(self.embedding.dimension),
        }
        key = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return key, metadata

    def _remember_vector(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = vector
        self._vectors.move_to_end(key)
        while len(self._vectors) > self.cache_capacity:
            self._vectors.popitem(last=False)

    def _read_persistent_vector(self, key: str, metadata: dict[str, object]) -> np.ndarray | None:
        if self._cache_connection is None:
            return None
        row = self._cache_connection.execute(
            """
            SELECT cache_schema, source_id, source_content_sha256, block_id, block_text_sha256,
                   model_identity_sha256, dimension, vector_dtype, vector_blob, vector_sha256
            FROM dense_vectors WHERE cache_key=?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        expected = (
            metadata["cache_schema"], metadata["source_id"], metadata["source_content_sha256"],
            metadata["block_id"], metadata["block_text_sha256"], metadata["model_identity_sha256"],
            metadata["dimension"],
        )
        if tuple(row[:7]) != expected or row[7] != "float32-le":
            return None
        payload = bytes(row[8])
        dimension = int(metadata["dimension"])
        if len(payload) != dimension * np.dtype("<f4").itemsize:
            return None
        if hashlib.sha256(payload).hexdigest() != row[9]:
            return None
        vector = np.frombuffer(payload, dtype="<f4").copy()
        if not np.isfinite(vector).all() or not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-3):
            return None
        return vector

    @staticmethod
    def copy_persistent_cache(source_path: str | Path, destination_path: str | Path) -> Path:
        """Create a consistent SQLite backup of an existing dense-vector cache."""
        source = Path(source_path).resolve(strict=True)
        destination = Path(destination_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"persistent Qwen vector cache is not a file: {source}")
        if source == destination:
            return destination
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite dense-vector cache: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_uri = source.as_uri() + "?mode=ro"
        try:
            with closing(sqlite3.connect(source_uri, uri=True, timeout=30)) as source_connection:
                with closing(sqlite3.connect(destination, timeout=30)) as destination_connection:
                    source_connection.backup(destination_connection)
                    destination_connection.commit()
        except Exception:
            if destination.exists():
                destination.unlink()
            raise
        return destination

    def _write_persistent_vector(self, key: str, metadata: dict[str, object], vector: np.ndarray) -> None:
        if self._cache_connection is None:
            return
        payload = np.asarray(vector, dtype="<f4").tobytes(order="C")
        self._cache_connection.execute(
            """
            INSERT OR REPLACE INTO dense_vectors (
                cache_key, cache_schema, source_id, source_content_sha256, block_id,
                block_text_sha256, model_identity_sha256, dimension, vector_dtype,
                vector_blob, vector_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key, metadata["cache_schema"], metadata["source_id"], metadata["source_content_sha256"],
                metadata["block_id"], metadata["block_text_sha256"], metadata["model_identity_sha256"],
                metadata["dimension"], "float32-le", payload, hashlib.sha256(payload).hexdigest(),
            ),
        )

    def __call__(self, plan: QueryPlan, version_ids: Sequence[str]) -> Sequence[EvidenceResult]:
        return self._search(plan, version_ids, None)

    def search_with_control(
        self,
        plan: QueryPlan,
        version_ids: Sequence[str],
        control: SearchControl,
    ) -> Sequence[EvidenceResult]:
        return self._search(plan, version_ids, control)

    def search_with_trace(
        self,
        plan: QueryPlan,
        version_ids: Sequence[str],
        control: SearchControl | None,
        trace_callback: Callable[[dict[str, object]], None],
    ) -> Sequence[EvidenceResult]:
        return self._search(plan, version_ids, control, trace_callback=trace_callback)

    def _search(
        self,
        plan: QueryPlan,
        version_ids: Sequence[str],
        control: SearchControl | None,
        *,
        trace_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> Sequence[EvidenceResult]:
        search_started = time.perf_counter()
        if control is not None:
            control.check()
        rows = [
            row for row in self.store.read_blocks(version_ids)
            if str(row["security_code"] or "") == plan.security_code and str(row["text"]).strip()
        ]
        if control is not None:
            control.check()
        if not rows:
            if trace_callback is not None:
                trace_callback({"stage": "dense_search_completed", "candidate_count": 0, "elapsed_seconds": round(time.perf_counter() - search_started, 3)})
            return []
        query_text = plan.query + (" " + " ".join(plan.aliases) if plan.aliases else "")
        query_embedding_started = time.perf_counter()
        if trace_callback is not None:
            trace_callback({
                "stage": "dense_query_embedding_started",
                "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                "dimension": int(self.embedding.dimension),
            })
        query_model_started = time.perf_counter()
        query_vector = self.embedding.embed([query_text], query=True, control=control)[0]
        query_model_elapsed = time.perf_counter() - query_model_started
        if trace_callback is not None:
            trace_callback({
                "stage": "dense_query_embedding_completed",
                "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                "dimension": int(query_vector.shape[0]),
                "model_elapsed_seconds": round(query_model_elapsed, 3),
                "elapsed_seconds": round(time.perf_counter() - query_embedding_started, 3),
            })
        from adaptive_mas.industrial_rag.retrieval import _citation

        ranked = []
        missing_rows: list[Any] = []
        missing_keys: list[str] = []
        missing_metadata: list[dict[str, object]] = []
        persistent_cache_hits = 0
        memory_cache_hits = 0
        new_embeddings = 0
        for row in rows:
            if control is not None:
                control.check()
            text = str(row["text"])
            key, metadata = self._vector_key(row, text)
            vector = self._vectors.get(key)
            if vector is not None:
                self._vectors.move_to_end(key)
                memory_cache_hits += 1
                ranked.append((float(np.dot(query_vector, vector)), row))
                continue
            vector = self._read_persistent_vector(key, metadata)
            if vector is not None:
                persistent_cache_hits += 1
                self._remember_vector(key, vector)
                ranked.append((float(np.dot(query_vector, vector)), row))
                continue
            missing_rows.append(row)
            missing_keys.append(key)
            missing_metadata.append(metadata)
        if trace_callback is not None and ranked:
            ranked_cached = sorted(ranked, key=lambda item: (-item[0], str(item[1]["block_id"])))
            trace_callback({
                "stage": "dense_cached_candidates_scored",
                "rank_scope": "all_verified_cached_candidates",
                "candidate_count": len(ranked_cached),
                "cache": {"persistent_hits": persistent_cache_hits, "memory_hits": memory_cache_hits},
                "ranked_candidates": [
                    self._ranked_candidate(index, score, row)
                    for index, (score, row) in enumerate(ranked_cached, start=1)
                ],
                "elapsed_seconds": round(time.perf_counter() - search_started, 3),
            })
        if missing_rows and self.require_cached_vectors:
            raise RuntimeError(f"persistent Qwen vector cache is missing or invalid for {len(missing_rows)} source block(s)")
        for start in range(0, len(missing_rows), EMBED_BATCH_SIZE):
            batch_rows = missing_rows[start:start + EMBED_BATCH_SIZE]
            batch_keys = missing_keys[start:start + EMBED_BATCH_SIZE]
            batch_metadata = missing_metadata[start:start + EMBED_BATCH_SIZE]
            if trace_callback is not None:
                trace_callback(self._batch_started_trace(batch_rows, search_started, len(rows)))
            before_count = len(ranked)
            batch_model_started = time.perf_counter()
            self._score_missing_batch(batch_rows, batch_keys, batch_metadata, query_vector, ranked, control=control)
            batch_model_elapsed = time.perf_counter() - batch_model_started
            new_embeddings += len(ranked) - before_count
            if trace_callback is not None:
                event = self._batch_trace("dense_embedding_batch_completed", ranked, batch_rows, batch_keys, search_started, len(rows))
                event["embedding_batch_elapsed_seconds"] = round(batch_model_elapsed, 3)
                trace_callback(event)
        ranked.sort(key=lambda item: (-item[0], str(item[1]["block_id"])))
        if trace_callback is not None:
            trace_callback({
                "stage": "dense_search_completed",
                "candidate_count": len(ranked),
                "ranked_candidates": [self._ranked_candidate(index, score, row) for index, (score, row) in enumerate(ranked, start=1)],
                "cache": {
                    "persistent_hits": persistent_cache_hits,
                    "memory_hits": memory_cache_hits,
                    "new_embeddings": new_embeddings,
                    "persistent_cache_enabled": self._cache_connection is not None,
                },
                "elapsed_seconds": round(time.perf_counter() - search_started, 3),
            })
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

    def _score_missing_batch(
        self,
        rows: list[Any],
        keys: list[str],
        metadata: list[dict[str, object]],
        query_vector: np.ndarray,
        ranked: list[Any],
        *,
        control: SearchControl | None = None,
    ) -> None:
        if control is not None:
            control.check()
        vectors = self.embedding.embed([str(row["text"]) for row in rows], control=control)
        if control is not None:
            control.check()
        if vectors.ndim != 2 or vectors.shape != (len(rows), int(self.embedding.dimension)):
            raise ValueError("Qwen embedding output shape differs from the identity-bound vector dimension")
        if not np.isfinite(vectors).all():
            raise ValueError("Qwen embedding output contains non-finite values")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
            raise ValueError("Qwen dense retrieval requires normalized document vectors")
        try:
            for row, key, identity, vector in zip(rows, keys, metadata, vectors):
                self._write_persistent_vector(key, identity, vector)
            if self._cache_connection is not None:
                self._cache_connection.commit()
        except Exception:
            if self._cache_connection is not None:
                self._cache_connection.rollback()
            raise
        for row, key, vector in zip(rows, keys, vectors):
            self._remember_vector(key, vector)
            ranked.append((float(np.dot(query_vector, vector)), row))

    def cached_vectors(
        self,
        version_ids: Sequence[str],
        *,
        block_keys: Sequence[tuple[str, str]] | None = None,
    ) -> list[tuple[Any, np.ndarray]]:
        """Return verified persistent source vectors without invoking document embedding inference."""
        if self._cache_connection is None:
            raise RuntimeError("a persistent Qwen vector cache is required for this source run")
        rows = [row for row in self.store.read_blocks(version_ids) if str(row["text"]).strip()]
        required_keys = set(block_keys) if block_keys is not None else None
        if required_keys is not None:
            rows = [row for row in rows if (str(row["version_id"]), str(row["block_id"])) in required_keys]
        verified: list[tuple[Any, np.ndarray]] = []
        missing = 0
        found_keys: set[tuple[str, str]] = set()
        for row in rows:
            text = str(row["text"])
            key, metadata = self._vector_key(row, text)
            vector = self._read_persistent_vector(key, metadata)
            if vector is None:
                missing += 1
            else:
                verified.append((row, vector))
                found_keys.add((str(row["version_id"]), str(row["block_id"])))
        if required_keys is not None:
            missing = len(required_keys - found_keys)
        if missing:
            raise RuntimeError(f"persistent Qwen vector cache is missing or invalid for {missing} source block(s)")
        return verified

    def verify_persistent_cache(self, version_ids: Sequence[str]) -> int:
        """Validate source-block vectors without invoking document embedding inference."""
        return len(self.cached_vectors(version_ids))

    @staticmethod
    def _ranked_candidate(rank: int, score: float, row: Any) -> dict[str, object]:
        return {
            "rank": rank,
            "evidence_id": f"ev-dense-{row['block_id']}",
            "block_id": str(row["block_id"]),
            "version_id": str(row["version_id"]),
            "source_id": str(row["source_id"]),
            "source_content_sha256": str(row["content_sha256"]),
            "block_text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
            "page_number": int(row["page_number"]) if row["page_number"] is not None else None,
            "score": float(score),
        }

    def _batch_trace(
        self,
        stage: str,
        ranked: list[Any],
        batch_rows: list[Any],
        batch_keys: list[str],
        search_started: float,
        total_candidates: int,
    ) -> dict[str, object]:
        rank_by_key = {
            self._vector_key(row, str(row["text"]))[0]: rank
            for rank, (_score, row) in enumerate(
                sorted(ranked, key=lambda item: (-item[0], str(item[1]["block_id"]))), start=1
            )
        }
        score_by_key = {
            self._vector_key(row, str(row["text"]))[0]: score
            for score, row in ranked
        }
        return {
            "stage": stage,
            "rank_scope": "all_candidates_scored_by_event_time",
            "completed_candidate_count": len(ranked),
            "total_candidate_count": total_candidates,
            "batch_candidates": [
                self._ranked_candidate(rank_by_key[key], score_by_key[key], row)
                for row, key in zip(batch_rows, batch_keys)
            ],
            "elapsed_seconds": round(time.perf_counter() - search_started, 3),
        }

    @staticmethod
    def _batch_started_trace(rows: list[Any], search_started: float, total_candidates: int) -> dict[str, object]:
        return {
            "stage": "dense_embedding_batch_started",
            "total_candidate_count": total_candidates,
            "batch_candidates": [
                {
                    "block_id": str(row["block_id"]),
                    "version_id": str(row["version_id"]),
                    "source_id": str(row["source_id"]),
                    "source_content_sha256": str(row["content_sha256"]),
                    "block_text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
                    "page_number": int(row["page_number"]) if row["page_number"] is not None else None,
                }
                for row in rows
            ],
            "elapsed_seconds": round(time.perf_counter() - search_started, 3),
        }


class LocalQwenReranker:
    implementation = "local_cpu_qwen3_reranker_0.6b"

    def __init__(self, runtime: Qwen3RerankerRuntime) -> None:
        self.runtime = runtime

    @property
    def cache_identity(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "revision": RERANKER_REVISION,
            "configuration": {
                "model": RERANKER_REPO,
                "instruction_sha256": hashlib.sha256(RETRIEVAL_INSTRUCTION.encode("utf-8")).hexdigest(),
                "score": "yes_no_logit_difference",
                "max_length": RERANK_MAX_LENGTH,
            },
        }

    def __call__(self, plan: QueryPlan, evidence: Sequence[EvidenceResult]) -> Sequence[EvidenceResult]:
        return self._rerank(plan, evidence, None)

    def rerank_with_control(
        self,
        plan: QueryPlan,
        evidence: Sequence[EvidenceResult],
        control: SearchControl,
    ) -> Sequence[EvidenceResult]:
        return self._rerank(plan, evidence, control)

    def rerank_with_trace(
        self,
        plan: QueryPlan,
        evidence: Sequence[EvidenceResult],
        control: SearchControl | None,
        trace_callback: Callable[[dict[str, object]], None],
    ) -> Sequence[EvidenceResult]:
        return self._rerank(plan, evidence, control, trace_callback=trace_callback)

    def _rerank(
        self,
        plan: QueryPlan,
        evidence: Sequence[EvidenceResult],
        control: SearchControl | None,
        *,
        trace_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> Sequence[EvidenceResult]:
        def on_batch(start: int, scores: list[float], elapsed: float) -> None:
            if trace_callback is not None:
                trace_callback({
                    "stage": "reranker_batch_completed",
                    "score_kind": "yes_no_logit_difference",
                    "batch_offset": start,
                    "batch_candidates": [
                        {"rank_before_reranking": start + offset + 1, "evidence_id": evidence[start + offset].evidence_id, "score": score}
                        for offset, score in enumerate(scores)
                    ],
                    "elapsed_seconds": round(elapsed, 3),
                })

        def on_batch_start(start: int, end: int) -> None:
            if trace_callback is not None:
                trace_callback({
                    "stage": "reranker_batch_started",
                    "batch_offset": start,
                    "candidate_ids": [evidence[index].evidence_id for index in range(start, end)],
                    "candidate_text_sha256": [
                        hashlib.sha256(evidence[index].text.encode("utf-8")).hexdigest()
                        for index in range(start, end)
                    ],
                    "score_kind": "yes_no_logit_difference",
                })

        scores = self.runtime.score(
            plan.query,
            [item.text for item in evidence],
            control=control,
            on_batch=on_batch if trace_callback is not None else None,
            on_batch_start=on_batch_start if trace_callback is not None else None,
        )
        paired = []
        for item, score in zip(evidence, scores):
            if control is not None:
                control.check()
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
