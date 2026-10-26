"""Release-bound OpenSearch text indexing and BM25 retrieval adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from adaptive_mas.industrial_rag.models import EvidenceResult, IndexBuildReceipt, QueryPlan, SearchControl
from adaptive_mas.industrial_rag.repository import IndustrialRagStore
from adaptive_mas.industrial_rag.retrieval import _citation
from adaptive_mas.research_provenance import stable_hash


class OpenSearchBackendError(RuntimeError):
    """An OpenSearch request or immutable release-index check failed."""


class OpenSearchRequestError(OpenSearchBackendError):
    def __init__(self, status_code: int, method: str, path: str) -> None:
        super().__init__(f"OpenSearch returned HTTP {status_code} for {method} {path}")
        self.status_code = status_code


class OpenSearchRestClient:
    """Small JSON/NDJSON client with bounded timeouts and sanitized errors."""

    def __init__(
        self,
        base_url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        timeout_seconds: float = 15,
    ) -> None:
        from urllib.parse import urlsplit

        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OpenSearch URL must be an absolute HTTP(S) endpoint without embedded credentials or query data")
        if (username is None) != (password is None) or (username is not None and (not username or not password)):
            raise ValueError("OpenSearch basic authentication requires both username and password")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("OpenSearch timeout must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.timeout_seconds = timeout_seconds
        self._authorization = (
            "Basic " + base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
            if username is not None and password is not None else None
        )

    def request(self, method: str, path: str, body: object | bytes | None = None) -> dict[str, Any]:
        if not path.startswith("/") or ".." in path.split("/"):
            raise ValueError("OpenSearch request path must be an absolute index-relative path")
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._authorization is not None:
            headers["Authorization"] = self._authorization
        data: bytes | None
        if isinstance(body, bytes):
            data = body
            headers["Content-Type"] = "application/x-ndjson" if path.endswith("/_bulk") else "application/json"
        elif body is None:
            data = None
        else:
            data = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as error:
            raise OpenSearchRequestError(error.code, method.upper(), path) from None
        except (URLError, OSError):
            raise OpenSearchBackendError(f"OpenSearch endpoint did not complete {method.upper()} {path}") from None
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OpenSearchBackendError(f"OpenSearch returned an invalid JSON response for {method.upper()} {path}") from None
        if not isinstance(value, dict):
            raise OpenSearchBackendError(f"OpenSearch returned an unexpected response shape for {method.upper()} {path}")
        return value


class OpenSearchTextIndexAdapter:
    """Build and query an immutable release-scoped OpenSearch block index."""

    implementation = "raama-opensearch-release-bm25"
    index_schema_version = "raama.financial-rag.opensearch-release-index.v1"

    def __init__(self, store: IndustrialRagStore, client: OpenSearchRestClient) -> None:
        self.store = store
        self.client = client
        self.cache_identity = {
            "implementation": self.implementation,
            "version": "1",
            "configuration": {
                "endpoint": client.base_url,
                "account": client.username or "anonymous",
                "index_schema": self.index_schema_version,
                "analyzer": "cjk",
            },
        }

    @staticmethod
    def _index_handle(release_id: str) -> str:
        return "raama-financial-rag-" + hashlib.sha256(release_id.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _record_id(version_id: str, block_id: str) -> str:
        return hashlib.sha256(f"{version_id}\0{block_id}".encode("utf-8")).hexdigest()

    def _release_inputs(self, release_id: str) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], str]:
        spec, _spec_hash = self.store.release_spec(release_id)
        if "opensearch" not in set(spec.get("required_backends", [])):
            raise OpenSearchBackendError("corpus release does not require the OpenSearch backend")
        version_ids = self.store.release_versions(release_id)
        expected_index = self.store.preview_reference_index_manifest(release_id, version_ids)
        source_manifest = self.store.source_manifest(version_ids)
        source_hash = stable_hash(source_manifest)
        blocks = [dict(row) for row in self.store.read_blocks(version_ids) if row.get("quarantine_reason") is None]
        if len(blocks) != int(expected_index["record_count"]):
            raise OpenSearchBackendError("local searchable blocks differ from the release-bound text-index manifest")
        records: list[dict[str, object]] = []
        for row in blocks:
            record: dict[str, object] = {
                "document_id": str(row["document_id"]),
                "version_id": str(row["version_id"]),
                "source_id": str(row["source_id"]),
                "content_sha256": str(row["content_sha256"]),
                "block_id": str(row["block_id"]),
                "block_type": str(row["block_type"]),
                "page_number": int(row["page_number"]) if row.get("page_number") is not None else None,
                "time_basis": str(row["time_basis"]),
                "text": str(row["text"]),
            }
            record["record_sha256"] = stable_hash(record)
            records.append(record)
        records.sort(key=lambda item: (str(item["version_id"]), int(item["page_number"] or 0), str(item["block_id"])))
        return expected_index, records, {"source_manifest_sha256": source_hash}, str(spec["embedding_model"])

    def _metadata(self, handle: str) -> dict[str, object]:
        if not re.fullmatch(r"raama-financial-rag-[0-9a-f]{32}", handle):
            raise OpenSearchBackendError("OpenSearch release index handle is malformed")
        mappings = self.client.request("GET", f"/{handle}/_mapping")
        index_value = mappings.get(handle)
        if not isinstance(index_value, dict):
            raise OpenSearchBackendError("OpenSearch release index mapping is missing")
        mapping = index_value.get("mappings")
        if not isinstance(mapping, dict) or not isinstance(mapping.get("_meta"), dict):
            raise OpenSearchBackendError("OpenSearch release index metadata is missing")
        return dict(mapping["_meta"])

    def _expected_meta(
        self,
        release_id: str,
        expected_index: Mapping[str, object],
        source_manifest: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": self.index_schema_version,
            "release_id": release_id,
            "source_manifest_sha256": source_manifest["source_manifest_sha256"],
            "index_manifest_sha256": expected_index["index_manifest_sha256"],
            "document_count": expected_index["document_count"],
            "record_count": expected_index["record_count"],
            "analyzer": "cjk",
        }

    def _ensure_index(self, handle: str, expected_meta: Mapping[str, object]) -> None:
        try:
            actual_meta = self._metadata(handle)
        except OpenSearchRequestError as error:
            if error.status_code != 404:
                raise
            self.client.request("PUT", f"/{handle}", {
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {
                    "_meta": dict(expected_meta),
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "version_id": {"type": "keyword"},
                        "source_id": {"type": "keyword"},
                        "content_sha256": {"type": "keyword"},
                        "block_id": {"type": "keyword"},
                        "block_type": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "time_basis": {"type": "keyword"},
                        "text": {"type": "text", "analyzer": "cjk"},
                        "record_sha256": {"type": "keyword"},
                    },
                },
            })
            return
        if actual_meta != dict(expected_meta):
            raise OpenSearchBackendError("existing OpenSearch index is bound to a different release or manifest")

    def _write_records(self, handle: str, records: Sequence[Mapping[str, object]]) -> None:
        for offset in range(0, len(records), 500):
            lines: list[str] = []
            for record in records[offset:offset + 500]:
                identifier = self._record_id(str(record["version_id"]), str(record["block_id"]))
                lines.append(json.dumps({"index": {"_index": handle, "_id": identifier}}, separators=(",", ":")))
                lines.append(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
            if not lines:
                continue
            response = self.client.request("POST", "/_bulk", ("\n".join(lines) + "\n").encode("utf-8"))
            if response.get("errors") is not False:
                raise OpenSearchBackendError("OpenSearch bulk indexing reported one or more failed records")

    def _verify_documents(self, handle: str, records: Sequence[Mapping[str, object]]) -> None:
        count = self.client.request("GET", f"/{handle}/_count").get("count")
        if isinstance(count, bool) or count != len(records):
            raise OpenSearchBackendError("OpenSearch physical record count differs from the immutable release manifest")
        for offset in range(0, len(records), 500):
            expected_batch = records[offset:offset + 500]
            ids = [self._record_id(str(record["version_id"]), str(record["block_id"])) for record in expected_batch]
            response = self.client.request("POST", f"/{handle}/_mget", {"ids": ids})
            docs = response.get("docs")
            if not isinstance(docs, list) or len(docs) != len(expected_batch):
                raise OpenSearchBackendError("OpenSearch multi-get returned an incomplete release index page")
            for expected, actual in zip(expected_batch, docs, strict=True):
                if not isinstance(actual, dict) or actual.get("found") is not True or actual.get("_source") != dict(expected):
                    raise OpenSearchBackendError("OpenSearch physical record differs from its source-backed release record")

    def _server_version(self) -> str:
        info = self.client.request("GET", "/")
        version = info.get("version")
        number = version.get("number") if isinstance(version, dict) else None
        if not isinstance(number, str) or not number:
            raise OpenSearchBackendError("OpenSearch did not return a server version")
        return number

    def build_release_index(self, release_id: str) -> IndexBuildReceipt:
        summaries = self.store.release_summaries()
        release = next((row for row in summaries["releases"] if row["release_id"] == release_id), None)
        if release is None:
            raise FileNotFoundError("corpus release not found")
        expected_index, records, source_manifest, embedding_model = self._release_inputs(release_id)
        handle = self._index_handle(release_id)
        expected_meta = self._expected_meta(release_id, expected_index, source_manifest)
        if release["status"] != "building":
            self._ensure_index(handle, expected_meta)
            self._verify_documents(handle, records)
        else:
            self._ensure_index(handle, expected_meta)
            self._write_records(handle, records)
            self.client.request("POST", f"/{handle}/_refresh")
            self._verify_documents(handle, records)
        return IndexBuildReceipt(
            release_id=release_id,
            backend="opensearch",
            physical_index_handle=handle,
            backend_version=self._server_version(),
            source_manifest_sha256=str(source_manifest["source_manifest_sha256"]),
            index_manifest_sha256=str(expected_index["index_manifest_sha256"]),
            document_count=int(expected_index["document_count"]),
            record_count=int(expected_index["record_count"]),
            embedding_model=embedding_model,
            embedding_dimension=None,
            built_at=datetime.now(timezone.utc),
        )

    def verify_release_index(self, physical_index_handle: str) -> Mapping[str, object]:
        metadata = self._metadata(physical_index_handle)
        release_id = metadata.get("release_id")
        if not isinstance(release_id, str):
            raise OpenSearchBackendError("OpenSearch index is not bound to a corpus release")
        expected_index, records, source_manifest, embedding_model = self._release_inputs(release_id)
        expected_meta = self._expected_meta(release_id, expected_index, source_manifest)
        if metadata != expected_meta:
            raise OpenSearchBackendError("OpenSearch index metadata differs from the current immutable release")
        self._verify_documents(physical_index_handle, records)
        spec, _ = self.store.release_spec(release_id)
        return {
            "backend": "opensearch",
            "physical_index_handle": physical_index_handle,
            "backend_version": self._server_version(),
            "source_manifest_sha256": source_manifest["source_manifest_sha256"],
            "index_manifest_sha256": expected_index["index_manifest_sha256"],
            "document_count": expected_index["document_count"],
            "record_count": expected_index["record_count"],
            "embedding_model": embedding_model,
            "embedding_dimension": spec["embedding_dimension"],
        }

    def mark_release_ready(self, release_id: str) -> dict[str, object]:
        receipt = self.build_release_index(release_id)
        return self.store.mark_backend_ready(receipt, self.verify_release_index)

    def __call__(self, plan: QueryPlan, version_ids: Sequence[str]) -> Sequence[EvidenceResult]:
        return self.search_with_control(plan, version_ids, None)

    def search_with_control(
        self,
        plan: QueryPlan,
        version_ids: Sequence[str],
        control: SearchControl | None,
    ) -> Sequence[EvidenceResult]:
        if control is not None:
            control.check()
        release_id = plan.release_id or self.store.active_release_id()
        if not release_id or self.store.active_release_id() != release_id:
            raise OpenSearchBackendError("OpenSearch query is not bound to the active corpus release")
        versions = set(self.store.release_versions(release_id))
        if not version_ids or not set(version_ids).issubset(versions):
            raise OpenSearchBackendError("OpenSearch query versions are outside the active release")
        permitted = self.store.permitted_versions(
            plan.principal_id,
            plan.security_code,
            plan.as_of,
            time_mode=plan.time_mode,
            report_periods=plan.report_periods,
            accounting_basis=plan.accounting_basis,
            release_id=release_id,
        )
        permitted_ids = {str(row["version_id"]) for row in permitted}
        if not set(version_ids).issubset(permitted_ids):
            raise PermissionError("OpenSearch version scope exceeds the current principal/time/release permissions")
        expected_index, records, source_manifest, _ = self._release_inputs(release_id)
        handle = self._index_handle(release_id)
        if self._metadata(handle) != self._expected_meta(release_id, expected_index, source_manifest):
            raise OpenSearchBackendError("OpenSearch index metadata differs from the active release")
        physical_count = self.client.request("GET", f"/{handle}/_count").get("count")
        if isinstance(physical_count, bool) or physical_count != len(records):
            raise OpenSearchBackendError("OpenSearch physical record count differs from the active release")
        if control is not None:
            control.check()
        query_text = " ".join((plan.query, *plan.aliases))
        response = self.client.request("POST", f"/{handle}/_search", {
            "size": min(plan.top_k * 5, 500),
            "track_total_hits": False,
            "query": {
                "bool": {
                    "must": [{"match": {"text": {"query": query_text, "operator": "or"}}}],
                    "filter": [{"terms": {"version_id": list(version_ids)}}],
                },
            },
        })
        hits_obj = response.get("hits")
        hits = hits_obj.get("hits") if isinstance(hits_obj, dict) else None
        if not isinstance(hits, list):
            raise OpenSearchBackendError("OpenSearch returned an invalid search-hit collection")
        local_rows = self.store.read_blocks(version_ids)
        local_by_id = {
            (str(row["version_id"]), str(row["block_id"])): dict(row)
            for row in local_rows if row.get("quarantine_reason") is None
        }
        expected_by_key = {(str(record["version_id"]), str(record["block_id"])): dict(record) for record in records}
        output: list[EvidenceResult] = []
        for hit in hits[: plan.top_k * 5]:
            if not isinstance(hit, dict) or not isinstance(hit.get("_source"), dict):
                raise OpenSearchBackendError("OpenSearch returned a malformed release-index hit")
            indexed = hit["_source"]
            key = (str(indexed.get("version_id")), str(indexed.get("block_id")))
            expected = expected_by_key.get(key)
            row = local_by_id.get(key)
            if (
                key[0] not in set(version_ids)
                or expected is None
                or row is None
                or indexed != expected
                or str(row["content_sha256"]) != str(indexed.get("content_sha256"))
                or str(row["text"]) != str(indexed.get("text"))
            ):
                raise OpenSearchBackendError("OpenSearch hit is not an exact member of the permitted release source")
            score_value = hit.get("_score")
            score = float(score_value) if isinstance(score_value, (int, float)) and math.isfinite(float(score_value)) else 0.0
            output.append(EvidenceResult(
                evidence_id=f"ev-block-{row['block_id']}",
                channel="bm25",
                document_id=str(row["document_id"]),
                version_id=str(row["version_id"]),
                source_id=str(row["source_id"]),
                content_sha256=str(row["content_sha256"]),
                page_number=int(row["page_number"]) if row.get("page_number") is not None else None,
                block_id=str(row["block_id"]),
                fact_id=None,
                text=str(row["text"]),
                score=score,
                source_time_basis=str(row["time_basis"]),
                citation=_citation(row),
            ))
        if control is not None:
            control.check()
        return output
