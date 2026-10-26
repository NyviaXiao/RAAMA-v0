from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from adaptive_mas.industrial_rag.models import QueryPlan, SearchCancelled, SearchControl
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore
from adaptive_mas.industrial_rag.repository import IndustrialRagStore
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService
from adaptive_mas.industrial_rag.web import LocalRagHTTPServer, RagRequestHandler
from adaptive_mas.runtime_limits import http_thread_limit


def _cli(repo_root: Path, env: dict[str, str], *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "adaptive_mas.industrial_rag.cli", *arguments],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def _request_json(url: str, value: dict[str, object] | None = None) -> dict[str, object]:
    if value is None:
        request = Request(url)
    else:
        request = Request(
            url,
            data=json.dumps(value, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_cli_ingest_release_status_and_live_local_api_flow(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    source = tmp_path / "synthetic_public_financial_table.csv"
    metadata_path = tmp_path / "synthetic_metadata.json"
    database = tmp_path / "rag.sqlite3"
    objects = tmp_path / "objects"
    scratch = tmp_path / "parser-scratch"
    source.write_text(
        "项目,2024年,单位\n营业收入,123456,人民币元\n营业收入,120000,人民币元\n营业收入,110000,人民币元\n净利润,789,人民币元\n",
        encoding="utf-8",
        newline="",
    )
    first_available = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    metadata = {
        "document_id": "synthetic-doc-1",
        "version_id": "synthetic-version-1",
        "source_id": "synthetic-source-1",
        "title": "SYNTHETIC API FIXTURE ONLY — 2024 financial table",
        "document_type": "synthetic_fixture",
        "security_code": "600036",
        "entity_name": "synthetic issuer fixture",
        "report_period": "2024",
        "accounting_basis": "synthetic basis fixture",
        "source_url": "https://example.invalid/synthetic/financial-table.csv",
        "rights_note": "Synthetic test fixture only; not an issuer filing.",
        "published_at": first_available,
        "first_available_at": first_available,
        "time_basis": "strict_forward",
        "read_principals": ["local-public"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    ingested = _cli(
        repo_root, env,
        "ingest", "--source", str(source), "--metadata", str(metadata_path),
        "--database", str(database), "--objects", str(objects), "--scratch", str(scratch),
    )
    assert ingested["status"] == "ingested"
    assert ingested["parser"]["status"] == "parsed"

    released = _cli(
        repo_root, env,
        "release", "--release-id", "synthetic-release-1", "--version-id", "synthetic-version-1",
        "--database", str(database), "--objects", str(objects), "--reason", "isolated API contract test",
    )
    assert released["status"] == "published"
    assert released["release"]["release_id"] == "synthetic-release-1"

    status = _cli(repo_root, env, "status", "--database", str(database), "--objects", str(objects))
    assert status["release"]["active_release_id"] == "synthetic-release-1"

    store = IndustrialRagStore(database, LocalContentAddressedStore(objects))
    server = LocalRagHTTPServer(("127.0.0.1", 0), RagRequestHandler)
    assert server._ordinary_worker_limit == max(1, http_thread_limit() - 1)
    assert server._cancel_worker_limit == 1
    assert server._total_worker_limit == max(2, http_thread_limit())
    server.context = {
        "store": store,
        "service": IndustrialRagService(store),
        "principal_id": "local-public",
        "continuation_review_principal_id": "local-public",
        "time_mode": "strict_forward",
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        health = _request_json(base_url + "/api/health")
        assert health["ok"] is True
        assert health["release"]["active_release_id"] == "synthetic-release-1"
        assert health["continuation_review_enabled"] is True
        continuation_query = urlencode({
            "security_code": "600036", "as_of": datetime.now(timezone.utc).isoformat(),
        })
        assert _request_json(base_url + "/api/continuations?" + continuation_query)["candidates"] == []
        try:
            _request_json(base_url + "/api/continuations/review", {
                "security_code": "600036",
                "as_of": datetime.now(timezone.utc).isoformat(),
                "child_table_id": "caller-selected-table",
                "action": "confirm",
                "content_sha256": "0" * 64,
                "reviewer_id": "attacker-controlled",
            })
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("reviewer identity must be fixed by server configuration")

        query_scope = {
            "security_code": "600036",
            "query": "2024年营业收入",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "task_type": "fact",
            "channels": ["bm25"],
            "top_k": 5,
        }
        result = _request_json(base_url + "/api/search", query_scope)
        assert result["query_binding"]["release_id"] == "synthetic-release-1"
        assert result["retrieved"]
        assert any("营业收入" in item["text"] for item in result["retrieved"])

        default_page = _request_json(base_url + "/api/search/page", query_scope)
        assert default_page["pagination"]["page_size"] == query_scope["top_k"]
        empty_page = _request_json(base_url + "/api/search/page", {
            "security_code": "600036",
            "query": "no matching terms zzzzz",
            "as_of": query_scope["as_of"],
            "task_type": "fact",
            "channels": ["bm25"],
        })
        assert empty_page["retrieved"] == []
        assert empty_page["pagination"]["page_size"] == 10
        assert empty_page["pagination"]["next_cursor"] is None

        first_page = _request_json(base_url + "/api/search/page", {**query_scope, "page_size": 1})
        first_cursor = first_page["pagination"]["next_cursor"]
        assert first_cursor
        assert first_page["pagination"]["total_items"] >= 2
        try:
            _request_json(base_url + "/api/search/page", {
                **query_scope,
                "query": "2024年净利润",
                "page_size": 1,
                "cursor": first_cursor,
            })
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("a cursor must not be reusable with another query")
        tampered_cursor = ("A" if first_cursor[0] != "A" else "B") + first_cursor[1:]
        try:
            _request_json(base_url + "/api/search/page", {
                **query_scope, "page_size": 1, "cursor": tampered_cursor,
            })
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("a modified cursor signature must be rejected")

        paged_items = list(first_page["retrieved"])
        page = first_page
        while page["pagination"]["next_cursor"] is not None:
            cursor = page["pagination"]["next_cursor"]
            page = _request_json(base_url + "/api/search/page", {
                **query_scope, "page_size": 1, "cursor": cursor,
            })
            assert page["pagination"]["offset"] == len(paged_items)
            paged_items.extend(page["retrieved"])
            assert len(paged_items) <= first_page["pagination"]["total_items"]
        assert len(paged_items) == first_page["pagination"]["total_items"]
        assert [item["evidence_id"] for item in paged_items] == [
            item["evidence_id"] for item in result["retrieved"]
        ]

        prior_service = server.context["service"]
        server.context["service"] = IndustrialRagService(store)
        try:
            _request_json(base_url + "/api/search/page", {
                **query_scope, "page_size": 1, "cursor": first_cursor,
            })
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("a cursor from a previous service snapshot must be rejected")
        finally:
            server.context["service"] = prior_service

        research = _request_json(base_url + "/api/research", query_scope)
        assert research["status"] == "candidate_only"
        assert research["coverage"]
        assert research["coverage"][0]["status"] == "candidate_only"
        assert research["supporting_evidence"] == []
        assert all(item["citation"].get("numeric_eligible") is False for item in research["candidate_evidence"])

        table_item = next(
            item for item in result["retrieved"]
            if item["citation"].get("table_id") is not None
        )
        table_row = _request_json(base_url + "/api/table-row", {
            **{key: query_scope[key] for key in ("security_code", "query", "as_of")},
            "table_id": table_item["citation"]["table_id"],
            "row_start": table_item["citation"]["row_start"],
            "row_end": table_item["citation"]["row_end"],
        })
        assert table_row["rows"]

        original_url = base_url + "/api/original/synthetic-doc-1/synthetic-version-1?" + urlencode({
            key: query_scope[key] for key in ("security_code", "query", "as_of")
        })
        with urlopen(original_url, timeout=5) as response:
            assert response.read() == source.read_bytes()

        service = server.context["service"]
        class RevokingDenseAdapter:
            implementation = "synthetic_acl_revoker"
            cache_identity = {
                "implementation": "synthetic_acl_revoker",
                "revision": "test-v1",
                "configuration": {"purpose": "mid-search ACL change contract"},
            }

            def __call__(self, plan, version_ids):
                return []

            def search_with_control(self, plan, version_ids, control):
                store.set_access("local-public", "synthetic-doc-1", can_read=False)
                return []

        prior_dense_adapter = service.channel_adapters.get("dense")
        service.channel_adapters["dense"] = RevokingDenseAdapter()
        try:
            _request_json(base_url + "/api/search", {
                **query_scope, "query": "permission revocation during search", "channels": ["dense"],
            })
        except HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("a permission revocation during retrieval must fail closed")
        finally:
            store.set_access("local-public", "synthetic-doc-1", can_read=True)
            if prior_dense_adapter is None:
                service.channel_adapters.pop("dense", None)
            else:
                service.channel_adapters["dense"] = prior_dense_adapter

        prior_search = service.search

        def revoke_after_search(
            plan, *, use_cache=True, rrf_rank_constant=60, use_reranker=True, control=None,
        ):
            response = prior_search(
                plan,
                use_cache=use_cache,
                rrf_rank_constant=rrf_rank_constant,
                use_reranker=use_reranker,
                control=control,
            )
            store.set_access("local-public", "synthetic-doc-1", can_read=False)
            return response

        service.search = revoke_after_search
        try:
            _request_json(base_url + "/api/research", query_scope)
        except HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("research must reject evidence revoked after a completed subsearch")
        finally:
            service.search = prior_search
            store.set_access("local-public", "synthetic-doc-1", can_read=True)

        request_started = threading.Event()
        release_request = threading.Event()

        class BlockingDenseAdapter:
            implementation = "synthetic_blocking_adapter"
            cache_identity = {
                "implementation": "synthetic_blocking_adapter",
                "revision": "test-v1",
                "configuration": {"purpose": "http backpressure contract"},
            }

            def __call__(self, plan, version_ids):
                return self.search_with_control(plan, version_ids, None)

            def search_with_control(self, plan, version_ids, control):
                request_started.set()
                while not release_request.wait(0.01):
                    if control is not None:
                        control.check()
                return []

        server.context["service"].channel_adapters["dense"] = BlockingDenseAdapter()
        server._request_slots = threading.BoundedSemaphore(1)
        blocking_query = {**query_scope, "channels": ["dense"]}
        cancel_request_id = "synthetic-cancel-request-1"
        background_response: dict[str, object] = {}

        def send_blocked_request() -> None:
            try:
                background_response["status"] = _request_json(base_url + "/api/search", {
                    **blocking_query,
                    "request_id": cancel_request_id,
                })["status"]
            except HTTPError as error:
                background_response["status"] = error.code

        background_request = threading.Thread(target=send_blocked_request)
        background_request.start()
        assert request_started.wait(2)
        try:
            for _ in range(3):
                try:
                    _request_json(base_url + "/api/search", blocking_query)
                except HTTPError as error:
                    assert error.code == 503
                else:
                    raise AssertionError("saturated ordinary requests must receive bounded 503 backpressure")
            assert server._cancel_slots.acquire(timeout=2)
            try:
                try:
                    _request_json(base_url + "/api/search", blocking_query)
                except HTTPError as error:
                    assert error.code == 503
                else:
                    raise AssertionError("requests beyond the ordinary and reserved pools must receive 503")
            finally:
                server._cancel_slots.release()
            cancel_ack = _request_json(base_url + "/api/cancel", {"request_id": cancel_request_id})
            assert cancel_ack["status"] == "cancellation_requested"
        finally:
            release_request.set()
        background_request.join(timeout=5)
        assert not background_request.is_alive()
        assert background_response.get("status") == 409

        acl_page = _request_json(base_url + "/api/search/page", {**query_scope, "page_size": 1})
        acl_cursor = acl_page["pagination"]["next_cursor"]
        assert acl_cursor
        store.set_access("local-public", "synthetic-doc-1", can_read=False)
        try:
            _request_json(base_url + "/api/search/page", {
                **query_scope, "page_size": 1, "cursor": acl_cursor,
            })
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("a cursor must be invalidated after a permission change")

        _cli(
            repo_root, env,
            "release", "--release-id", "synthetic-release-2", "--version-id", "synthetic-version-1",
            "--database", str(database), "--objects", str(objects), "--reason", "cursor snapshot invalidation contract",
        )
        try:
            _request_json(base_url + "/api/search/page", {
                **query_scope, "page_size": 1, "cursor": first_cursor,
            })
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("a cursor must not cross an active release snapshot")

        cancelled = threading.Event()
        cancelled.set()
        cancelled_plan = QueryPlan(
            plan_id="cancelled-search", query=str(query_scope["query"]), security_code="600036",
            as_of=str(query_scope["as_of"]), principal_id="local-public", task_type="fact",
            channels=("bm25",), top_k=5,
        )
        try:
            service = server.context["service"]
            service.search(cancelled_plan, control=SearchControl(cancellation_event=cancelled))
        except SearchCancelled:
            pass
        else:
            raise AssertionError("a cancelled retrieval must stop before returning cached or new results")

        server.context["search_timeout_seconds"] = 0
        try:
            _request_json(base_url + "/api/search", query_scope)
        except HTTPError as error:
            assert error.code == 504
        else:
            raise AssertionError("an expired API search deadline must return HTTP 504")

        try:
            _request_json(base_url + "/api/search", {**query_scope, "principal_id": "untrusted"})
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("caller-controlled principal_id must be rejected")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
