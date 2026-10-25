from __future__ import annotations

import hashlib
import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from adaptive_mas.industrial_rag.models import (
    Block,
    ContractError,
    CorpusReleaseSpec,
    Document,
    DocumentBundle,
    DocumentVersion,
    EvidenceResult,
    FinancialFact,
    QueryPlan,
    Section,
    Span,
)
from adaptive_mas.agents.harness import _fundamental_handlers
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore
from adaptive_mas.industrial_rag.parser import (
    ParseError,
    UnsupportedFormat,
    normalize_financial_cell,
    parse_isolated,
    records_from_parse,
    validate_cross_page_table_continuation,
)
from adaptive_mas.industrial_rag.postgres import _row_factory, translate_sqlite_contract
from adaptive_mas.industrial_rag.repository import IndustrialRagStore, ImmutableConflict, ReleaseNotReady
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService
from adaptive_mas.research_provenance import stable_hash


def _bundle(
    *,
    source_id: str = "cninfo-123456",
    version_id: str = "version-1",
    document_id: str = "document-1",
    payload: bytes = "2024年度营业收入 337,488 人民币百万元".encode(),
    principal: str = "analyst",
    time_basis: str = "strict_forward",
    first_available_at: datetime | None = None,
    captured_at: datetime | None = None,
    ingested_at: datetime | None = None,
    metric: str = "营业收入",
    raw_value: str = "337,488",
    normalized_value: str = "337488",
) -> DocumentBundle:
    now = datetime.now(timezone.utc)
    first = first_available_at or now - timedelta(minutes=20)
    captured = captured_at or now - timedelta(minutes=10)
    ingested = ingested_at or now - timedelta(minutes=5)
    digest = hashlib.sha256(payload).hexdigest()
    document = Document(
        document_id=document_id,
        title="招商银行股份有限公司2024年度报告摘要",
        document_type="annual_report_summary",
        security_code="600036",
        entity_name="招商银行股份有限公司",
        report_period="2024",
        accounting_basis="本集团",
    )
    version = DocumentVersion(
        version_id=version_id,
        source_id=source_id,
        content_sha256=digest,
        source_url="https://static.cninfo.com.cn/finalpage/2025-03-26/1222895153.PDF",
        rights_note="Public disclosure, local engineering diagnosis only; no redistribution license asserted.",
        published_at=first,
        first_available_at=first,
        captured_at=captured,
        ingested_at=ingested,
        time_basis=time_basis,  # type: ignore[arg-type]
        parser_name="test-structure-parser",
        parser_version="test-1",
        parser_confidence=0.92,
    )
    section = Section(f"{version_id}-section-1", version_id, "财务指标", 1, start_page=3, end_page=3)
    span = Span(f"{version_id}-span-1", version_id, 3, 0, len(payload.decode()), payload.decode())
    block = Block(
        f"{version_id}-block-1", version_id, payload.decode(), "paragraph", section_id=f"{version_id}-section-1",
        page_number=3, char_start=0, char_end=len(payload.decode()),
        parser_confidence=0.92, span_id=f"{version_id}-span-1",
    )
    fact = FinancialFact(
        fact_id=f"{version_id}-fact-1", version_id=version_id, security_code="600036",
        entity_name="招商银行股份有限公司", metric=metric, raw_value=raw_value,
        normalized_value=normalized_value, unit="人民币百万元", multiplier="1",
        currency="CNY", report_period="2024", consolidation="本集团",
        span_ids=(f"{version_id}-span-1",), parser_version="test-1", confidence=0.95,
        review_status="reviewed",
    )
    return DocumentBundle(document, version, payload, (section,), (span,), (block,), facts=(fact,), read_principals=(principal,))


def _consume_all(store: IndustrialRagStore) -> None:
    while True:
        event = store.consume_outbox_once(
            "sqlite-reference-indexer-v1",
            lambda event_key, event_type, payload: {
                "handler_key": event_key,
                "event_type": event_type,
                "verified_content_sha256": hashlib.sha256(
                    store.object_store.get(str(payload["content_sha256"]))
                ).hexdigest(),
            },
        )
        if event is None:
            return


def _publish(store: IndustrialRagStore, version_ids: list[str], release_id: str) -> None:
    source_hash = stable_hash(store.source_manifest(version_ids))
    text_manifest = str(store.preview_reference_index_manifest(release_id, version_ids)["index_manifest_sha256"])
    vector_manifest = stable_hash({"release_id": release_id, "backend": "qdrant", "status": "unavailable", "points": []})
    spec = CorpusReleaseSpec(
        release_id=release_id,
        source_manifest_sha256=source_hash,
        parser_version="pinned:test-1",
        chunker_version="chunker-test-1",
        embedding_model="none",
        embedding_dimension=None,
        required_backends=("sqlite_reference",),
        text_index_manifest_sha256=text_manifest,
        vector_index_manifest_sha256=vector_manifest,
    )
    store.create_release(spec, version_ids)
    receipt = store.build_reference_index_receipt(release_id)
    store.mark_backend_ready(receipt, store.verify_reference_index)
    store.publish_release(release_id, actor_id="test-operator", reason="isolated contract test")


def _research_one_bundle(tmp_path, bundle: DocumentBundle, query: str, *, periods: tuple[str, ...] = ("2024",)) -> dict[str, object]:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, [bundle.version.version_id], "release-1")
    plan = replace(_plan(), query=query, report_periods=periods)
    return IndustrialRagService(store).research(plan)


def _plan(*, as_of: datetime | None = None, principal: str = "analyst", time_mode: str = "strict_forward", release_id: str | None = None, channels: tuple[str, ...] = ("structured_facts", "bm25")) -> QueryPlan:
    return QueryPlan(
        plan_id="query-plan-1", query="2024年度营业收入是多少", security_code="600036",
        as_of=as_of or datetime.now(timezone.utc), principal_id=principal, task_type="fact",
        report_periods=("2024",), release_id=release_id, channels=channels,
        time_mode=time_mode,  # type: ignore[arg-type]
    )


def test_sqlite_ingest_release_query_and_citation_are_source_bound(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    bundle = _bundle()
    assert store.ingest_bundle(bundle)["status"] == "ingested"
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")

    result = IndustrialRagService(store).search(_plan())
    assert result["status"] == "available"
    assert result["query_binding"]["release_id"] == "release-1"
    fact = next(item for item in result["retrieved"] if item["fact_id"] == "version-1-fact-1")
    assert fact["citation"]["source_id"] == "cninfo-123456"
    assert fact["citation"]["source_url"].startswith("https://static.cninfo.com.cn/")
    assert fact["citation"]["content_sha256"] == hashlib.sha256(bundle.payload).hexdigest()
    assert fact["citation"]["page_number"] == 3
    assert fact["citation"]["numeric_eligible"] is True
    span = IndustrialRagService(store).read_span(_plan(), "version-1-span-1")
    assert span["text"] == bundle.payload.decode()
    assert store._get_document_bytes("document-1", "version-1", "analyst") == bundle.payload


@pytest.mark.parametrize(
    ("query", "source_metric", "payload"),
    [
        ("2024年净利润", "利润总额", "2024年度利润总额 178,652"),
        ("2024年基本每股收益", "稀释每股收益", "2024年度稀释每股收益 5.66"),
        ("2024年主营业务收入", "营业总收入", "2024年度营业总收入 337,488"),
    ],
)
def test_metric_coverage_rejects_financially_distinct_source_labels(tmp_path, query, source_metric, payload) -> None:
    result = _research_one_bundle(
        tmp_path,
        _bundle(metric=source_metric, raw_value="178652", normalized_value="178652", payload=payload.encode()),
        query,
    )
    assert result["coverage"][0]["status"] == "candidate_only"
    assert result["coverage"][0]["candidate_metrics"] == [
        "total_profit" if source_metric == "利润总额" else
        "diluted_eps" if source_metric == "稀释每股收益" else
        "total_operating_revenue"
    ]
    assert result["supporting_evidence"] == []
    assert all(item["citation"].get("numeric_eligible") is False for item in result["candidate_evidence"])


def test_page_level_metric_mention_without_a_corresponding_value_is_candidate_only(tmp_path) -> None:
    result = _research_one_bundle(
        tmp_path,
        _bundle(
            metric="营业收入",
            payload="2024年度净利润相关说明未列出对应金额；营业收入为337,488。".encode(),
        ),
        "2024年净利润",
    )
    assert result["coverage"][0]["status"] == "candidate_only"
    assert result["supporting_evidence"] == []
    assert result["candidate_evidence"]


def test_coverage_preserves_parent_metric_period_unit_and_consolidation_basis(tmp_path) -> None:
    result = _research_one_bundle(
        tmp_path,
        _bundle(
            metric="归属于母公司股东的净利润",
            raw_value="148,391",
            normalized_value="148391",
            payload="2024年度归属于本行股东的净利润为148,391 人民币百万元。".encode(),
        ),
        "2024年合并归母净利润（人民币百万元）",
    )
    coverage = result["coverage"][0]
    assert coverage["status"] == "located"
    assert coverage["requested_metric"] == "归母净利润"
    assert coverage["normalized_report_period"] == "2024"
    assert coverage["requested_unit"] == "人民币百万元"
    assert coverage["requested_accounting_basis"] == "consolidated"
    assert coverage["observed_units"] == ["人民币百万元"]
    assert coverage["observed_accounting_bases"] == ["本集团"]
    assert coverage["observed_report_periods"] == ["2024"]
    assert coverage["observed_metrics"] == ["parent_net_profit"]
    assert coverage["candidate_metrics"] == []
    assert result["supporting_evidence"][0]["query_numeric_eligible"] is True


@pytest.mark.parametrize(
    ("query", "periods"),
    [
        ("2024年1-3月营业收入", ()),
        ("2024年人民币元营业收入", ("2024",)),
    ],
)
def test_period_or_unit_mismatch_stays_candidate_only(tmp_path, query, periods) -> None:
    result = _research_one_bundle(tmp_path, _bundle(), query, periods=periods)
    assert result["coverage"][0]["status"] == "candidate_only"
    assert result["supporting_evidence"] == []
    assert result["candidate_evidence"]


def test_cache_is_bound_to_current_grants_and_span_or_download_rechecks_access(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    bundle = _bundle()
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")
    service = IndustrialRagService(store)
    plan = _plan()
    first = service.search(plan)
    cached = service.search(plan)
    assert first["retrieved"] and cached["cache"]["hit"] is True
    prior_revision = int(cached["permission_revision"])

    new_revision = store.set_access("analyst", "document-1", can_read=False)
    assert new_revision > prior_revision
    denied = service.search(plan)
    assert denied["retrieved"] == []
    assert denied["cache"]["hit"] is False
    with pytest.raises(PermissionError):
        service.read_span(plan, "version-1-span-1")
    with pytest.raises(PermissionError):
        service.download_original(plan, "document-1", "version-1")


def test_search_cache_identity_includes_aliases_and_runtime_revisions(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    bundle = _bundle()
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")

    class DenseStub:
        implementation = "dense-v1"

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, plan, version_ids):
            self.calls += 1
            assert "version-1" in version_ids
            return [EvidenceResult(
                evidence_id="dense-result", channel="dense", document_id="document-1",
                version_id="version-1", source_id="cninfo-123456",
                content_sha256=bundle.version.content_sha256,
                page_number=3, block_id="version-1-block-1", fact_id=None,
                text=" ".join(plan.aliases), score=1.0, source_time_basis="strict_forward",
            )]

    class RerankerStub:
        implementation = "reranker-v1"

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, plan, evidence):
            self.calls += 1
            return evidence

    dense = DenseStub()
    reranker = RerankerStub()
    service = IndustrialRagService(store, channel_adapters={"dense": dense}, reranker=reranker)
    plan = replace(_plan(channels=("dense",)), aliases=("营业收入",))
    first = service.search(plan)
    cached_repeat = service.search(replace(plan, plan_id="query-plan-retry"))
    assert cached_repeat["cache"]["hit"] is True
    assert cached_repeat["plan_id"] == "query-plan-retry"
    assert dense.calls == 1
    assert reranker.calls == 1

    with_different_alias = service.search(replace(plan, aliases=("主营业务收入",)))
    assert first["cache"]["key_sha256"] != with_different_alias["cache"]["key_sha256"]
    assert dense.calls == 2
    assert reranker.calls == 2

    dense.implementation = "dense-v2"
    with_new_adapter = service.search(replace(plan, aliases=("主营业务收入",)))
    assert with_new_adapter["cache"]["hit"] is False
    assert with_new_adapter["cache"]["key_sha256"] != with_different_alias["cache"]["key_sha256"]
    assert dense.calls == 3

    reranker.implementation = "reranker-v2"
    with_new_reranker = service.search(replace(plan, aliases=("主营业务收入",)))
    assert with_new_reranker["cache"]["hit"] is False
    assert with_new_reranker["cache"]["key_sha256"] != with_new_adapter["cache"]["key_sha256"]
    assert reranker.calls == 4


def test_strict_time_requires_capture_and_ingestion_by_cutoff_but_proxy_is_labelled(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    now = datetime.now(timezone.utc)
    late_capture = _bundle(
        source_id="cninfo-late", version_id="version-late", document_id="document-late",
        first_available_at=now - timedelta(days=2),
        captured_at=now - timedelta(minutes=3), ingested_at=now - timedelta(minutes=2),
    )
    proxy = _bundle(
        source_id="cninfo-proxy", version_id="version-proxy", document_id="document-proxy",
        payload=b"2024 annual report proxy document", time_basis="retrospective_proxy",
        first_available_at=now - timedelta(days=2), captured_at=now - timedelta(minutes=3),
        ingested_at=now - timedelta(minutes=2),
    )
    store.ingest_bundle(late_capture)
    store.ingest_bundle(proxy)
    _consume_all(store)
    _publish(store, ["version-late", "version-proxy"], "release-mixed")
    service = IndustrialRagService(store)
    strict = service.search(_plan(as_of=now - timedelta(minutes=4), time_mode="strict_forward"))
    assert strict["retrieved"] == []
    diagnostic = service.search(_plan(as_of=now - timedelta(days=1), time_mode="retrospective_diagnostic"))
    assert diagnostic["query_binding"]["time_mode"] == "retrospective_diagnostic"
    assert any(item["source_time_basis"] == "retrospective_proxy" for item in diagnostic["retrieved"])
    assert any("retrospective source timing" in item for item in diagnostic["warnings"])


def test_timezone_fields_are_normalized_and_bad_timestamp_order_is_rejected() -> None:
    now = datetime.now(timezone.utc)
    version = _bundle(
        first_available_at=(now - timedelta(minutes=20)).astimezone(timezone(timedelta(hours=8))),
        captured_at=(now - timedelta(minutes=10)).astimezone(timezone(timedelta(hours=8))),
        ingested_at=(now - timedelta(minutes=5)).astimezone(timezone(timedelta(hours=8))),
    ).version
    assert version.first_available_at.tzinfo == timezone.utc
    assert version.captured_at.tzinfo == timezone.utc
    with pytest.raises(ContractError, match="captured_at cannot precede"):
        _bundle(captured_at=now - timedelta(days=2))


def test_release_manifest_requires_verified_physical_receipts_and_atomic_all_backends(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(_bundle())
    _consume_all(store)
    source_hash = stable_hash(store.source_manifest(["version-1"]))
    text_hash = str(store.preview_reference_index_manifest("release-1", ["version-1"])["index_manifest_sha256"])
    vector_hash = hashlib.sha256(b"vector manifest").hexdigest()
    spec = CorpusReleaseSpec(
        "release-1", source_hash, "parser-1", "chunker-1", "test-embedding-model", 4,
        ("sqlite_reference", "qdrant"), text_hash, vector_hash,
    )
    store.create_release(spec, ["version-1"])
    receipt = store.build_reference_index_receipt("release-1")
    with pytest.raises(ImmutableConflict, match="release-bound"):
        forged = receipt.__class__(
            **{**asdict(receipt), "physical_index_handle": "sqlite-reference:forged"}
        )
        store.mark_backend_ready(forged, store.verify_reference_index)
    store.mark_backend_ready(receipt, store.verify_reference_index)
    with pytest.raises(ReleaseNotReady, match="qdrant"):
        store.publish_release("release-1", actor_id="test-operator", reason="missing vector backend")
    assert store.active_release_id() is None


def test_same_source_identity_cannot_be_reused_with_changed_bytes(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(_bundle())
    with pytest.raises(ImmutableConflict, match="changed bytes or metadata"):
        store.ingest_bundle(_bundle(source_id="cninfo-123456", version_id="version-2", payload=b"changed source bytes"))


def test_fact_ingestion_requires_existing_page_locatable_span_and_matching_basis(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    bundle = _bundle()
    missing_span_fact = replace(bundle.facts[0], span_ids=("missing-span",))
    with pytest.raises(ContractError, match="existing, nonempty, locatable span"):
        store.ingest_bundle(replace(bundle, facts=(missing_span_fact,)))

    unlocated_span = replace(bundle.spans[0], page_number=None)
    with pytest.raises(ContractError, match="existing, nonempty, locatable span"):
        store.ingest_bundle(replace(bundle, spans=(unlocated_span,)))

    mismatched_basis = replace(bundle.facts[0], consolidation="母公司")
    with pytest.raises(ContractError, match="accounting basis"):
        store.ingest_bundle(replace(bundle, facts=(mismatched_basis,)))


def test_fact_citation_uses_the_fact_span_page_and_lists_all_supporting_spans(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    bundle = _bundle()
    page_eight = Span("version-1-span-8", "version-1", 8, 0, 12, "营业收入表列数值")
    fact = replace(bundle.facts[0], span_ids=(page_eight.span_id,))
    bundle = replace(bundle, spans=(*bundle.spans, page_eight), facts=(fact,))
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")

    result = IndustrialRagService(store).search(_plan())
    cited = next(item for item in result["retrieved"] if item["fact_id"] == fact.fact_id)
    assert cited["page_number"] == 8
    assert cited["citation"]["span_id"] == page_eight.span_id
    assert cited["citation"]["supporting_spans"] == [{
        "span_id": page_eight.span_id,
        "document_id": "document-1",
        "version_id": "version-1",
        "source_id": "cninfo-123456",
        "source_url": bundle.version.source_url,
        "content_sha256": bundle.version.content_sha256,
        "page_number": 8,
        "char_start": 0,
        "char_end": 12,
    }]


def test_reused_source_version_rejects_changed_document_or_child_metadata(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    bundle = _bundle()
    store.ingest_bundle(bundle)
    with pytest.raises(ImmutableConflict, match="source/version identity"):
        store.ingest_bundle(_bundle(document_id="different-document"))

    changed_fact = replace(bundle.facts[0], raw_value="337,489", normalized_value="337489")
    with pytest.raises(ImmutableConflict, match="parsed child records"):
        store.ingest_bundle(replace(bundle, facts=(changed_fact,)))


def test_outbox_expiry_reclaim_duplicate_ack_and_idempotent_handler(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(_bundle())
    first = store.claim_outbox("worker-one", lease_seconds=0.01)
    assert first is not None
    time.sleep(0.02)
    second = store.claim_outbox("worker-one", lease_seconds=10)
    assert second is not None and second["event_key"] == first["event_key"]
    assert second["lease_generation"] > first["lease_generation"]
    with pytest.raises(RuntimeError, match="no longer owns"):
        store.finish_outbox(
            str(first["event_key"]), "worker-one", {"late": True},
            lease_generation=int(first["lease_generation"]),
        )
    assert store.finish_outbox(
        str(second["event_key"]), "worker-one", {"event": second["event_key"]},
        lease_generation=int(second["lease_generation"]),
    ) is True
    assert store.finish_outbox(
        str(second["event_key"]), "worker-one", {"event": second["event_key"]},
        lease_generation=int(second["lease_generation"]),
    ) is False


def test_content_addressed_store_parallel_same_bytes_is_atomic(tmp_path) -> None:
    store = LocalContentAddressedStore(tmp_path / "objects")
    payload = b"same immutable official source bytes"
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(lambda _index: store.put(payload), range(24)))
    expected = hashlib.sha256(payload).hexdigest()
    assert set(hashes) == {expected}
    assert store.get(expected) == payload


def test_decimal_facts_reject_nan_and_infinity_and_preserve_blank_dash_zero_percent_and_negative() -> None:
    common = dict(
        version_id="version-1", security_code="600036", entity_name="招商银行股份有限公司",
        metric="营业收入", raw_value="1", unit="元", multiplier="1", currency="CNY",
        report_period="2024", consolidation="本集团", span_ids=("span-1",),
        parser_version="p1", confidence=0.9,
    )
    with pytest.raises(ContractError):
        FinancialFact(fact_id="fact-nan", normalized_value="NaN", **common)
    with pytest.raises(ContractError):
        FinancialFact(fact_id="fact-inf", normalized_value="Infinity", **common)
    with pytest.raises(ContractError):
        FinancialFact(fact_id="fact-bad-multiplier", normalized_value="1", multiplier="Infinity", **{key: value for key, value in common.items() if key != "multiplier"})
    assert normalize_financial_cell("")["value_kind"] == "blank"
    assert normalize_financial_cell("—")["value_kind"] == "dash"
    assert normalize_financial_cell("0")["value_kind"] == "zero"
    assert normalize_financial_cell("0%")["value_kind"] == "percent"
    assert normalize_financial_cell("(1,234.50)")["normalized_value"] == "-1234.50"
    assert normalize_financial_cell("(1.25%)")["normalized_value"] == "-1.25"
    assert normalize_financial_cell("25bp")["value_kind"] == "basis_points"


def test_isolated_parser_supports_safe_text_csv_and_rejects_unsupported_or_entity_xml(tmp_path) -> None:
    scratch = tmp_path / "scratch"
    parsed = parse_isolated("2024年度营业收入为337,488百万元。".encode(), "source.md", scratch_root=scratch)
    assert parsed["status"] == "parsed"
    assert "337,488" in parsed["blocks"][1]["text"]
    csv_parsed = parse_isolated(b"Metric,Value\nRevenue,337488\n", "source.csv", scratch_root=scratch)
    assert csv_parsed["tables"] and len(csv_parsed["cells"]) == 4
    with pytest.raises(UnsupportedFormat):
        parse_isolated(b"legacy workbook", "source.xls", scratch_root=scratch)
    blocked_xml = parse_isolated(b"<!DOCTYPE x [<!ENTITY e SYSTEM 'file:///x'>]><x>&e;</x>", "source.xbrl", scratch_root=scratch)
    assert blocked_xml["status"] == "quarantined"
    assert "prohibited DTD/entity" in blocked_xml["error"]


def test_table_row_groups_keep_headers_cells_and_exact_source_spans_distinct(tmp_path) -> None:
    source = '指标,2024,2023\r\n营业收入,"(1,234.50%)",0\r\n净利润,—,25bp\r\n'.encode()
    parsed = parse_isolated(source, "financial.csv", scratch_root=tmp_path / "scratch")
    assert parsed["status"] == "parsed"
    table = parsed["tables"][0]
    cells = parsed["cells"]
    negative_percent = next(cell for cell in cells if cell["row_index"] == 1 and cell["column_index"] == 1)
    assert negative_percent["value_kind"] == "percent"
    assert negative_percent["normalized_value"] == "-1234.50"
    zero = next(cell for cell in cells if cell["row_index"] == 1 and cell["column_index"] == 2)
    assert zero["value_kind"] == "zero"
    dash = next(cell for cell in cells if cell["row_index"] == 2 and cell["column_index"] == 1)
    assert dash["value_kind"] == "dash"
    basis_points = next(cell for cell in cells if cell["row_index"] == 2 and cell["column_index"] == 2)
    assert basis_points["value_kind"] == "basis_points"

    row_groups = [block for block in parsed["blocks"] if block["block_type"] == "table_row_group"]
    assert len(row_groups) == 2
    assert all("列头：指标 | 2024 | 2023" in block["text"] for block in row_groups)
    assert row_groups[0]["row_start"] == row_groups[0]["row_end"] == 1
    assert row_groups[0]["next_block_id"] == row_groups[1]["block_id"]
    records = records_from_parse(parsed, "version-rows")
    stored_row = next(block for block in records["blocks"] if block.block_type == "table_row_group")
    assert stored_row.table_id == f"tbl-version-rows-{table['table_id']}"
    assert stored_row.span_id is None
    assert stored_row.char_start is None and stored_row.char_end is None

    markdown = parse_isolated(
        "| 指标 | 2024 | 2023 |\n| --- | ---: | ---: |\n| 营业收入 | 100 | 90 |\n".encode(),
        "financial.md",
        scratch_root=tmp_path / "scratch",
    )
    assert markdown["tables"] and len(markdown["cells"]) == 6
    markdown_group = next(block for block in markdown["blocks"] if block["block_type"] == "table_row_group")
    assert "数据行：营业收入 | 100 | 90" in markdown_group["text"]


def test_isolated_office_parser_rejects_external_relationships(tmp_path) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "word/_rels/document.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="external" Target="https://example.invalid/" TargetMode="External"/>'
            '</Relationships>',
        )
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>',
        )
    result = parse_isolated(payload.getvalue(), "external.docx", scratch_root=tmp_path / "scratch")
    assert result["status"] == "quarantined"
    assert "external relationships" in result["error"]


def _office_package(extension: str, macro_entry: str | None = None) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        if extension == ".docx":
            archive.writestr(
                "word/document.xml",
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>Annual report</w:t></w:r></w:p></w:body></w:document>',
            )
        elif extension == ".pptx":
            archive.writestr(
                "ppt/slides/slide1.xml",
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                '<p:spTree><a:t>Annual report</a:t></p:spTree></p:sld>',
            )
        else:
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Revenue</t></is></c></row>'
                '</sheetData></worksheet>',
            )
        if macro_entry is not None:
            archive.writestr(macro_entry, b"marker only; never executed")
    return payload.getvalue()


@pytest.mark.parametrize("extension", [".docx", ".pptx", ".xlsx"])
@pytest.mark.parametrize("macro_entry", ["word/vbaProject.bin", "xl/VBAproject.BIN"])
def test_isolated_office_parser_rejects_case_insensitive_macro_markers(tmp_path, extension, macro_entry) -> None:
    result = parse_isolated(
        _office_package(extension, macro_entry),
        f"macro-marker{extension}",
        scratch_root=tmp_path / "scratch",
    )
    assert result["status"] == "quarantined"
    assert "VBA macros" in result["error"]


@pytest.mark.parametrize("extension", [".docx", ".pptx", ".xlsx"])
def test_isolated_office_parser_keeps_macro_free_documents_parseable(tmp_path, extension) -> None:
    result = parse_isolated(
        _office_package(extension),
        f"regular{extension}",
        scratch_root=tmp_path / "scratch",
    )
    assert result["status"] == "parsed"


def test_table_row_expansion_rechecks_access_and_returns_exact_cell_citations(tmp_path) -> None:
    source = b"Metric,2024,2023\nRevenue,100,90\n"
    parsed = parse_isolated(source, "financial.csv", scratch_root=tmp_path / "scratch")
    base = _bundle(payload=source)
    version = replace(
        base.version,
        parser_name=str(parsed["parser_name"]),
        parser_version=str(parsed["parser_version"]),
        parser_confidence=float(parsed["parser_confidence"]),
    )
    records = records_from_parse(parsed, version.version_id)
    bundle = replace(
        base,
        version=version,
        sections=records["sections"],
        spans=records["spans"],
        blocks=records["blocks"],
        tables=records["tables"],
        cells=records["cells"],
        facts=(),
    )
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, [version.version_id], "release-1")
    service = IndustrialRagService(store)
    plan = replace(_plan(release_id="release-1"), query="Revenue 2024", top_k=10)
    result = service.search(plan)
    row_hit = next(item for item in result["retrieved"] if item["citation"].get("table_id"))
    expanded = service.read_table_row(
        plan,
        str(row_hit["citation"]["table_id"]),
        int(row_hit["citation"]["row_start"]),
        int(row_hit["citation"]["row_end"]),
    )
    assert "2024: 100" in expanded["text"]
    assert "2023: 90" in expanded["text"]
    assert expanded["citation"]["source_id"] == version.source_id
    assert expanded["citation"]["supporting_cells"]

    store.set_access("analyst", "document-1", can_read=False)
    with pytest.raises(PermissionError):
        service.read_table_row(
            plan,
            str(row_hit["citation"]["table_id"]),
            int(row_hit["citation"]["row_start"]),
            int(row_hit["citation"]["row_end"]),
        )


def test_cross_page_merge_is_not_automatic_without_matching_basis_and_header() -> None:
    prior = {"title": "财务指标", "header": ["2024", "2023"], "security_code": "600036", "report_period": "2024", "unit": "人民币百万元", "page_end": 3}
    next_page = {"title": "财务指标", "header": ["2024", "2023"], "security_code": "600036", "report_period": "2024", "unit": "人民币百万元", "page_start": 4, "continued_marker": True}
    verified = validate_cross_page_table_continuation(prior, next_page)
    assert verified["decision"] == "verified_continuation"
    assert verified["automatic_merge_allowed"] is True
    next_page["unit"] = "人民币元"
    rejected = validate_cross_page_table_continuation(prior, next_page)
    assert rejected["automatic_merge_allowed"] is False


def test_old_release_snapshot_is_stable_and_rollback_is_complete(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    first = _bundle()
    store.ingest_bundle(first)
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")
    original = IndustrialRagService(store).search(_plan())
    future = _bundle(
        source_id="cninfo-234567", version_id="version-2", document_id="document-2",
        payload="2024年度净利润 99".encode(), metric="净利润", raw_value="99", normalized_value="99",
    )
    store.ingest_bundle(future)
    _consume_all(store)
    _publish(store, ["version-1", "version-2"], "release-2")
    old_snapshot = IndustrialRagService(store).search(_plan(release_id="release-1"))
    new_snapshot = IndustrialRagService(store).search(_plan(release_id="release-2"))
    assert old_snapshot["retrieved"] == original["retrieved"]
    assert {item["version_id"] for item in new_snapshot["retrieved"]} >= {"version-1", "version-2"}
    rolled_back = store.rollback_release("release-1", actor_id="test-operator", reason="restore full previous corpus snapshot")
    assert rolled_back["active_release_id"] == "release-1"
    assert store.release_summaries()["active_release_id"] == "release-1"
    seed = store.rollback_release("seed", actor_id="test-operator", reason="return to initial state")
    assert seed["active_release_id"] == "seed"
    assert store.active_release_id() is None


def test_sqlite_to_postgres_transaction_and_queue_translation_is_explicit() -> None:
    assert translate_sqlite_contract("BEGIN IMMEDIATE") == "BEGIN ISOLATION LEVEL SERIALIZABLE"
    query = translate_sqlite_contract("SELECT * FROM rag_outbox WHERE status='pending' OR (status='processing' AND lease_until<?) ORDER BY event_id LIMIT 1")
    assert query.endswith("LIMIT 1 FOR UPDATE SKIP LOCKED")
    assert "%s" in query
    assert translate_sqlite_contract("INSERT OR IGNORE INTO rag_access_grants VALUES(?,?,1,?)").endswith("ON CONFLICT DO NOTHING")


def test_postgres_row_factory_handles_no_result_statements_and_selects() -> None:
    class Cursor:
        description = None

    cursor = Cursor()
    make_row = _row_factory(cursor)
    # DDL and INSERT/UPDATE without RETURNING produce no rows to convert.
    assert callable(make_row)
    assert cursor.description is None

    class Column:
        def __init__(self, name: str) -> None:
            self.name = name

    # SELECT metadata is populated after the cursor factory was constructed.
    cursor.description = (Column("version_id"), Column("content_sha256"))
    row = make_row(("v1", "a" * 64))
    assert row["version_id"] == "v1"
    assert row[1] == "a" * 64


def test_harness_fundamental_tools_search_and_reopen_pinned_rag_evidence(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(_bundle())
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")
    service = IndustrialRagService(store)
    tools, handlers, catalog = _fundamental_handlers(
        None,
        datetime.now(timezone.utc),
        industrial_rag_service=service,
        industrial_rag_principal_id="analyst",
        industrial_rag_release_id="release-1",
    )
    tool_names = {tool["function"]["name"] for tool in tools}
    assert "search_indexed_financial_documents" in tool_names
    assert "read_indexed_financial_evidence" in tool_names

    result = handlers["search_indexed_financial_documents"]({
        "question": "2024年度营业收入是多少",
        "symbols": ["600036"],
        "report_periods": ["2024"],
        "task_type": "fact",
    })
    assert result["status"] == "available"
    evidence = next(item for item in result["retrieved"] if item["financial_fact"].get("fact_id"))
    evidence_id = evidence["evidence_id"]
    assert evidence_id in catalog
    assert catalog[evidence_id]["vintage_verified"] is True
    reopened = handlers["read_indexed_financial_evidence"]({"evidence_id": evidence_id})
    assert reopened["text_excerpt"] == _bundle().payload.decode()
    assert reopened["citation"]["release_id"] == "release-1"

    repeated = handlers["search_indexed_financial_documents"]({
        "question": "2024年度营业收入是多少",
        "symbols": ["600036"],
        "report_periods": ["2024"],
        "task_type": "fact",
    })
    assert repeated["status"] == "no_new_evidence"
    assert repeated["retrieved"] == []
    assert repeated["reused_evidence_ids"]

    store.set_access("analyst", "document-1", can_read=False)
    with pytest.raises(PermissionError):
        handlers["read_indexed_financial_evidence"]({"evidence_id": evidence_id})
