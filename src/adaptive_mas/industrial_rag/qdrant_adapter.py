"""Release-bound Qdrant vector indexing and permission-scoped dense retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from adaptive_mas.industrial_rag.models import EvidenceResult, IndexBuildReceipt, QueryPlan, SearchControl
from adaptive_mas.industrial_rag.repository import IndustrialRagStore
from adaptive_mas.industrial_rag.retrieval import _citation
from adaptive_mas.research_provenance import stable_hash


class QdrantBackendError(RuntimeError):
    """A Qdrant request or immutable release-index check failed."""


class QdrantRequestError(QdrantBackendError):
    def __init__(self, status_code: int, method: str, path: str) -> None:
        super().__init__(f"Qdrant returned HTTP {status_code} for {method} {path}")
        self.status_code = status_code


class QdrantRestClient:
    """Small JSON client with bounded timeouts and sanitized errors."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 15,
    ) -> None:
        from urllib.parse import urlsplit

        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Qdrant URL must be an absolute HTTP(S) endpoint without embedded credentials or query data")
        if api_key is not None and not api_key:
            raise ValueError("Qdrant API key must not be empty")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Qdrant timeout must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds

    def request(self, method: str, path: str, body: object | None = None) -> dict[str, Any]:
        if not path.startswith("/") or ".." in path.split("/"):
            raise ValueError("Qdrant request path must be an absolute collection-relative path")
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key is not None:
            headers["api-key"] = self._api_key
        data = None if body is None else json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as error:
            raise QdrantRequestError(error.code, method.upper(), path) from None
        except (URLError, OSError):
            raise QdrantBackendError(f"Qdrant endpoint did not complete {method.upper()} {path}") from None
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise QdrantBackendError(f"Qdrant returned invalid JSON for {method.upper()} {path}") from None
        if not isinstance(value, dict):
            raise QdrantBackendError(f"Qdrant returned an unexpected response shape for {method.upper()} {path}")
        return value


class QdrantVectorIndexAdapter:
    """Build and query an immutable release-scoped Qdrant vector collection."""

    implementation = "raama-qdrant-release-dense"
    index_schema_version = "raama.financial-rag.qdrant-release-index.v1"

    def __init__(self, store: IndustrialRagStore, client: QdrantRestClient, dense_adapter: Any) -> None:
        self.store = store
        self.client = client
        self.dense_adapter = dense_adapter
        self.embedding = dense_adapter.embedding
        self.dimension = int(self.embedding.dimension)
        identity = dense_adapter.cache_identity.get("model_identity")
        if not isinstance(identity, dict) or identity.get("dimension") != self.dimension:
            raise ValueError("Qdrant adapter requires a dimension-matched persistent Qwen vector identity")
        self.model_identity = dict(identity)
        self.model_identity_sha256 = stable_hash(self.model_identity)
        model = str(identity.get("model", ""))
        revision = str(identity.get("revision", ""))
        if not model or not revision:
            raise ValueError("Qdrant adapter requires a pinned embedding model and revision")
        self.embedding_model = f"{model}@{revision}"
        self.cache_identity = {
            "implementation": self.implementation,
            "version": "1",
            "endpoint": client.base_url,
            "index_schema": self.index_schema_version,
            "embedding_model": self.embedding_model,
            "model_identity_sha256": self.model_identity_sha256,
        }

    @staticmethod
    def _collection_handle(release_id: str) -> str:
        return "raama_financial_rag_" + hashlib.sha256(release_id.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _point_id(version_id: str, block_id: str) -> str:
        digest = hashlib.sha256(f"{version_id}\0{block_id}".encode("utf-8")).hexdigest()
        return str(uuid.UUID(hex=digest[:32]))

    @staticmethod
    def _vector_sha256(vector: object) -> str:
        values = np.asarray(vector, dtype="<f4")
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise QdrantBackendError("Qdrant vector is not a finite one-dimensional float vector")
        if not np.isclose(float(np.linalg.norm(values)), 1.0, atol=1e-3):
            raise QdrantBackendError("Qdrant vectors must be L2 normalized")
        return hashlib.sha256(values.tobytes(order="C")).hexdigest()

    def _block_payload(
        self,
        row: Mapping[str, object],
        *,
        release_id: str,
        source_manifest_sha256: str,
        index_manifest_sha256: str,
        vector_sha256: str,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.index_schema_version,
            "release_id": release_id,
            "source_manifest_sha256": source_manifest_sha256,
            "index_manifest_sha256": index_manifest_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "document_id": str(row["document_id"]),
            "version_id": str(row["version_id"]),
            "source_id": str(row["source_id"]),
            "content_sha256": str(row["content_sha256"]),
            "block_id": str(row["block_id"]),
            "block_type": str(row["block_type"]),
            "page_number": int(row["page_number"]) if row.get("page_number") is not None else None,
            "time_basis": str(row["time_basis"]),
            "security_code": str(row["security_code"] or ""),
            "text": str(row["text"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
            "vector_sha256": vector_sha256,
        }
        payload["record_sha256"] = stable_hash(payload)
        return payload

    def _release_inputs(
        self, release_id: str, version_ids: Sequence[str],
    ) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        if not version_ids or len(version_ids) != len(set(version_ids)):
            raise ValueError("Qdrant release index requires a nonempty unique version list")
        source_manifest = self.store.source_manifest(version_ids)
        source_hash = stable_hash(source_manifest)
        source_rows = {
            (str(row["version_id"]), str(row["block_id"])): dict(row)
            for row in self.store.read_blocks(version_ids)
            if row.get("quarantine_reason") is None and str(row["text"]).strip()
        }
        cached_rows = self.dense_adapter.cached_vectors(version_ids, block_keys=list(source_rows))
        vectors_by_key = {
            (str(row["version_id"]), str(row["block_id"])): vector
            for row, vector in cached_rows
            if (str(row["version_id"]), str(row["block_id"])) in source_rows
        }
        if set(vectors_by_key) != set(source_rows):
            raise QdrantBackendError("verified persistent vectors do not cover the release's searchable source blocks")
        point_manifest: list[dict[str, object]] = []
        points: list[dict[str, object]] = []
        for key, row in sorted(source_rows.items(), key=lambda item: (item[0][0], item[0][1])):
            vector = np.asarray(vectors_by_key[key], dtype="<f4")
            if vector.shape != (self.dimension,):
                raise QdrantBackendError("cached vector dimension differs from the pinned Qwen runtime")
            vector_sha256 = self._vector_sha256(vector)
            point_id = self._point_id(*key)
            point_manifest.append({
                "point_id": point_id,
                "version_id": key[0],
                "block_id": key[1],
                "source_id": str(row["source_id"]),
                "content_sha256": str(row["content_sha256"]),
                "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
                "vector_sha256": vector_sha256,
            })
            points.append({"id": point_id, "vector": vector.tolist(), "row": row, "vector_sha256": vector_sha256})
        index_data: dict[str, object] = {
            "schema_version": self.index_schema_version,
            "release_id": release_id,
            "backend": "qdrant",
            "source_manifest_sha256": source_hash,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.dimension,
            "model_identity_sha256": self.model_identity_sha256,
            "document_count": len({str(row["document_id"]) for row in source_rows.values()}),
            "record_count": len(points),
            "point_manifest_sha256": stable_hash(point_manifest),
        }
        return {**index_data, "index_manifest_sha256": stable_hash(index_data)}, points, {"source_manifest_sha256": source_hash}

    def preview_vector_index_manifest(self, release_id: str, version_ids: Sequence[str]) -> dict[str, object]:
        """Hash verified cached vectors so the immutable release can bind this physical index."""
        index_data, _, _ = self._release_inputs(release_id, version_ids)
        return index_data

    def _expected_metadata(self, index_data: Mapping[str, object]) -> dict[str, object]:
        return {
            "schema_version": self.index_schema_version,
            "release_id": index_data["release_id"],
            "source_manifest_sha256": index_data["source_manifest_sha256"],
            "index_manifest_sha256": index_data["index_manifest_sha256"],
            "embedding_model": index_data["embedding_model"],
            "embedding_dimension": index_data["embedding_dimension"],
            "model_identity_sha256": index_data["model_identity_sha256"],
            "document_count": index_data["document_count"],
            "record_count": index_data["record_count"],
        }

    def _collection_info(self, handle: str) -> dict[str, object]:
        if not re.fullmatch(r"raama_financial_rag_[0-9a-f]{32}", handle):
            raise QdrantBackendError("Qdrant collection handle is malformed")
        response = self.client.request("GET", f"/collections/{handle}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise QdrantBackendError("Qdrant collection details are missing")
        return result

    def _ensure_collection(self, handle: str, metadata: Mapping[str, object]) -> None:
        try:
            actual = self._collection_info(handle)
        except QdrantRequestError as error:
            if error.status_code != 404:
                raise
            self.client.request("PUT", f"/collections/{handle}", {
                "vectors": {"size": self.dimension, "distance": "Cosine"},
                "metadata": dict(metadata),
            })
            actual = self._collection_info(handle)
        config = actual.get("config")
        params = config.get("params") if isinstance(config, dict) else None
        vectors = params.get("vectors") if isinstance(params, dict) else None
        stored_metadata = config.get("metadata") if isinstance(config, dict) else None
        if (
            not isinstance(vectors, dict)
            or vectors.get("size") != self.dimension
            or vectors.get("distance") != "Cosine"
            or stored_metadata != dict(metadata)
        ):
            raise QdrantBackendError("Qdrant collection configuration differs from the immutable release")

    def _write_points(self, handle: str, index_data: Mapping[str, object], points: Sequence[Mapping[str, object]]) -> None:
        for offset in range(0, len(points), 256):
            batch = points[offset:offset + 256]
            qdrant_points = []
            for point in batch:
                row = point["row"]
                assert isinstance(row, dict)
                qdrant_points.append({
                    "id": point["id"], "vector": point["vector"],
                    "payload": self._block_payload(
                        row,
                        release_id=str(index_data["release_id"]),
                        source_manifest_sha256=str(index_data["source_manifest_sha256"]),
                        index_manifest_sha256=str(index_data["index_manifest_sha256"]),
                        vector_sha256=str(point["vector_sha256"]),
                    ),
                })
            self.client.request("PUT", f"/collections/{handle}/points?wait=true", {"points": qdrant_points})

    def _verify_points(
        self, handle: str, index_data: Mapping[str, object], points: Sequence[Mapping[str, object]],
    ) -> None:
        expected = {
            str(point["id"]): (
                self._block_payload(
                    point["row"],  # type: ignore[arg-type]
                    release_id=str(index_data["release_id"]),
                    source_manifest_sha256=str(index_data["source_manifest_sha256"]),
                    index_manifest_sha256=str(index_data["index_manifest_sha256"]),
                    vector_sha256=str(point["vector_sha256"]),
                ),
                str(point["vector_sha256"]),
            )
            for point in points
        }
        found: set[str] = set()
        offset: object | None = None
        while True:
            request: dict[str, object] = {"limit": 256, "with_payload": True, "with_vector": True}
            if offset is not None:
                request["offset"] = offset
            response = self.client.request("POST", f"/collections/{handle}/points/scroll", request)
            result = response.get("result")
            rows = result.get("points") if isinstance(result, dict) else None
            if not isinstance(rows, list):
                raise QdrantBackendError("Qdrant scroll returned an invalid point collection")
            for item in rows:
                if not isinstance(item, dict):
                    raise QdrantBackendError("Qdrant scroll returned a malformed point")
                point_id = str(item.get("id"))
                expected_record = expected.get(point_id)
                vector = item.get("vector")
                if (
                    expected_record is None
                    or point_id in found
                    or item.get("payload") != expected_record[0]
                    or self._vector_sha256(vector) != expected_record[1]
                ):
                    raise QdrantBackendError("Qdrant physical point differs from its source-backed release record")
                found.add(point_id)
                if len(found) > len(expected):
                    raise QdrantBackendError("Qdrant physical collection contains points outside the release manifest")
            next_offset = result.get("next_page_offset") if isinstance(result, dict) else None
            if next_offset is None:
                break
            if next_offset == offset:
                raise QdrantBackendError("Qdrant point scroll did not advance its page cursor")
            offset = next_offset
        if found != set(expected):
            raise QdrantBackendError("Qdrant physical point count differs from the immutable release manifest")

    def _server_version(self) -> str:
        response = self.client.request("GET", "/")
        version = response.get("version")
        if not isinstance(version, str) or not version:
            raise QdrantBackendError("Qdrant did not return a server version")
        return version

    def build_release_index(self, release_id: str) -> IndexBuildReceipt:
        summaries = self.store.release_summaries()
        release = next((row for row in summaries["releases"] if row["release_id"] == release_id), None)
        if release is None:
            raise FileNotFoundError("corpus release not found")
        spec, _ = self.store.release_spec(release_id)
        if "qdrant" not in set(spec["required_backends"]):
            raise QdrantBackendError("corpus release does not require the Qdrant backend")
        index_data, points, _ = self._release_inputs(release_id, self.store.release_versions(release_id))
        if (
            index_data["index_manifest_sha256"] != spec["vector_index_manifest_sha256"]
            or index_data["source_manifest_sha256"] != spec["source_manifest_sha256"]
            or index_data["embedding_model"] != spec["embedding_model"]
            or index_data["embedding_dimension"] != spec["embedding_dimension"]
        ):
            raise QdrantBackendError("Qdrant vectors differ from the immutable corpus release specification")
        handle = self._collection_handle(release_id)
        metadata = self._expected_metadata(index_data)
        if release["status"] == "building":
            self._ensure_collection(handle, metadata)
            self._write_points(handle, index_data, points)
        self._verify_collection(handle, metadata, index_data, points)
        return IndexBuildReceipt(
            release_id=release_id, backend="qdrant", physical_index_handle=handle,
            backend_version=self._server_version(), source_manifest_sha256=str(index_data["source_manifest_sha256"]),
            index_manifest_sha256=str(index_data["index_manifest_sha256"]),
            document_count=int(index_data["document_count"]), record_count=int(index_data["record_count"]),
            embedding_model=str(index_data["embedding_model"]), embedding_dimension=int(index_data["embedding_dimension"]),
            built_at=datetime.now(timezone.utc),
        )

    def _verify_collection(
        self,
        handle: str,
        metadata: Mapping[str, object],
        index_data: Mapping[str, object],
        points: Sequence[Mapping[str, object]],
    ) -> None:
        actual = self._collection_info(handle)
        config = actual.get("config")
        params = config.get("params") if isinstance(config, dict) else None
        vectors = params.get("vectors") if isinstance(params, dict) else None
        stored_metadata = config.get("metadata") if isinstance(config, dict) else None
        if (
            not isinstance(vectors, dict)
            or vectors.get("size") != self.dimension
            or vectors.get("distance") != "Cosine"
            or stored_metadata != dict(metadata)
            or actual.get("points_count") != len(points)
        ):
            raise QdrantBackendError("Qdrant collection metadata, vector parameters, or point count differs from the release")
        self._verify_points(handle, index_data, points)

    def verify_release_index(self, physical_index_handle: str) -> Mapping[str, object]:
        result = self._collection_info(physical_index_handle)
        config = result.get("config")
        metadata = config.get("metadata") if isinstance(config, dict) else None
        release_id = metadata.get("release_id") if isinstance(metadata, dict) else None
        if not isinstance(release_id, str) or physical_index_handle != self._collection_handle(release_id):
            raise QdrantBackendError("Qdrant collection is not bound to the requested corpus release")
        spec, _ = self.store.release_spec(release_id)
        index_data, points, _ = self._release_inputs(release_id, self.store.release_versions(release_id))
        expected_metadata = self._expected_metadata(index_data)
        if (
            index_data["index_manifest_sha256"] != spec["vector_index_manifest_sha256"]
            or index_data["source_manifest_sha256"] != spec["source_manifest_sha256"]
            or index_data["embedding_model"] != spec["embedding_model"]
            or index_data["embedding_dimension"] != spec["embedding_dimension"]
            or metadata != expected_metadata
        ):
            raise QdrantBackendError("Qdrant collection metadata differs from the immutable corpus release")
        self._verify_collection(physical_index_handle, expected_metadata, index_data, points)
        return {
            "backend": "qdrant", "physical_index_handle": physical_index_handle,
            "backend_version": self._server_version(), "source_manifest_sha256": index_data["source_manifest_sha256"],
            "index_manifest_sha256": index_data["index_manifest_sha256"],
            "document_count": index_data["document_count"], "record_count": index_data["record_count"],
            "embedding_model": index_data["embedding_model"], "embedding_dimension": index_data["embedding_dimension"],
        }

    def mark_release_ready(self, release_id: str) -> dict[str, object]:
        receipt = self.build_release_index(release_id)
        return self.store.mark_backend_ready(receipt, self.verify_release_index)

    def __call__(self, plan: QueryPlan, version_ids: Sequence[str]) -> Sequence[EvidenceResult]:
        return self.search_with_control(plan, version_ids, None)

    def search_with_control(
        self, plan: QueryPlan, version_ids: Sequence[str], control: SearchControl | None,
    ) -> Sequence[EvidenceResult]:
        if control is not None:
            control.check()
        release_id = plan.release_id or self.store.active_release_id()
        if not release_id or self.store.active_release_id() != release_id:
            raise QdrantBackendError("Qdrant query is not bound to the active corpus release")
        versions = set(self.store.release_versions(release_id))
        if not version_ids or len(version_ids) != len(set(version_ids)) or not set(version_ids).issubset(versions):
            raise QdrantBackendError("Qdrant query versions are outside the active release")
        permitted = self.store.permitted_versions(
            plan.principal_id, plan.security_code, plan.as_of,
            time_mode=plan.time_mode, report_periods=plan.report_periods,
            accounting_basis=plan.accounting_basis, release_id=release_id,
        )
        permitted_ids = {str(row["version_id"]) for row in permitted}
        if not set(version_ids).issubset(permitted_ids):
            raise PermissionError("Qdrant version scope exceeds current principal/time/release permissions")
        spec, _ = self.store.release_spec(release_id)
        if (
            "qdrant" not in set(spec["required_backends"])
            or spec["embedding_model"] != self.embedding_model
            or spec["embedding_dimension"] != self.dimension
        ):
            raise QdrantBackendError("Qdrant runtime identity differs from the active release")
        source_manifest = self.store.source_manifest(self.store.release_versions(release_id))
        source_hash = stable_hash(source_manifest)
        expected_metadata = {
            "schema_version": self.index_schema_version,
            "release_id": release_id,
            "source_manifest_sha256": source_hash,
            "index_manifest_sha256": spec["vector_index_manifest_sha256"],
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.dimension,
            "model_identity_sha256": self.model_identity_sha256,
        }
        collection = self._collection_info(self._collection_handle(release_id))
        config = collection.get("config")
        params = config.get("params") if isinstance(config, dict) else None
        vectors = params.get("vectors") if isinstance(params, dict) else None
        metadata = config.get("metadata") if isinstance(config, dict) else None
        release_blocks = [
            row for row in self.store.read_blocks(self.store.release_versions(release_id))
            if row.get("quarantine_reason") is None and str(row["text"]).strip()
        ]
        expected_metadata["document_count"] = len({str(row["document_id"]) for row in release_blocks})
        expected_metadata["record_count"] = len(release_blocks)
        if (
            metadata != expected_metadata
            or not isinstance(vectors, dict)
            or vectors.get("size") != self.dimension
            or vectors.get("distance") != "Cosine"
            or collection.get("points_count") != len(release_blocks)
        ):
            raise QdrantBackendError("Qdrant collection differs from the active release metadata")
        if control is not None:
            control.check()
        query_text = plan.query + (" " + " ".join(plan.aliases) if plan.aliases else "")
        query_vectors = self.embedding.embed([query_text], query=True, control=control)
        if query_vectors.shape != (1, self.dimension) or not np.isfinite(query_vectors).all():
            raise QdrantBackendError("Qwen query embedding differs from the pinned Qdrant vector dimension")
        query_vector = np.asarray(query_vectors[0], dtype="<f4")
        self._vector_sha256(query_vector)
        if control is not None:
            control.check()
        handle = self._collection_handle(release_id)
        response = self.client.request("POST", f"/collections/{handle}/points/query", {
            "query": query_vector.tolist(),
            "limit": min(plan.top_k * 5, 500),
            "with_payload": True,
            "with_vector": True,
            "filter": {"must": [
                {"key": "release_id", "match": {"value": release_id}},
                {"key": "version_id", "match": {"any": list(version_ids)}},
                {"key": "security_code", "match": {"value": plan.security_code}},
            ]},
        })
        result = response.get("result")
        hits = result.get("points") if isinstance(result, dict) else None
        if not isinstance(hits, list):
            raise QdrantBackendError("Qdrant returned an invalid query result collection")
        local_rows = self.store.read_blocks(version_ids)
        local_by_id = {
            (str(row["version_id"]), str(row["block_id"])): dict(row)
            for row in local_rows if row.get("quarantine_reason") is None and str(row["text"]).strip()
        }
        output: list[EvidenceResult] = []
        for hit in hits[: plan.top_k * 5]:
            if not isinstance(hit, dict) or not isinstance(hit.get("payload"), dict):
                raise QdrantBackendError("Qdrant returned a malformed point result")
            payload = dict(hit["payload"])
            key = (str(payload.get("version_id")), str(payload.get("block_id")))
            row = local_by_id.get(key)
            if row is None or str(hit.get("id")) != self._point_id(*key):
                raise QdrantBackendError("Qdrant result is outside the permitted release source")
            expected_static = self._block_payload(
                row, release_id=release_id, source_manifest_sha256=source_hash,
                index_manifest_sha256=str(spec["vector_index_manifest_sha256"]),
                vector_sha256=str(payload.get("vector_sha256", "")),
            )
            vector = hit.get("vector")
            if (
                payload != expected_static
                or self._vector_sha256(vector) != payload["vector_sha256"]
                or str(row["security_code"] or "") != plan.security_code
            ):
                raise QdrantBackendError("Qdrant hit differs from its verified release source record")
            score_value = hit.get("score")
            if isinstance(score_value, bool) or not isinstance(score_value, (int, float)) or not math.isfinite(float(score_value)):
                raise QdrantBackendError("Qdrant returned an invalid similarity score")
            output.append(EvidenceResult(
                evidence_id=f"ev-dense-{row['block_id']}", channel="dense",
                document_id=str(row["document_id"]), version_id=str(row["version_id"]),
                source_id=str(row["source_id"]), content_sha256=str(row["content_sha256"]),
                page_number=int(row["page_number"]) if row.get("page_number") is not None else None,
                block_id=str(row["block_id"]), fact_id=None, text=str(row["text"]),
                score=float(score_value), source_time_basis=str(row["time_basis"]),
                citation={**_citation(row), "embedding_model": self.model_identity["model"],
                          "embedding_revision": self.model_identity["revision"],
                          "ranking_score_kind": "cosine_similarity", "scores_are_probabilities": False},
            ))
        if control is not None:
            control.check()
        return output
