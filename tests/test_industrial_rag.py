from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import threading
import time
import uuid
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
    Table,
    TableCell,
)
from adaptive_mas.agents.harness import _fundamental_handlers
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore
from adaptive_mas.industrial_rag.parser import (
    ParseError,
    UnsupportedFormat,
    normalize_financial_cell,
    annotate_table_continuation_candidates,
    parse_isolated,
    records_from_parse,
    validate_cross_page_table_continuation,
)
from adaptive_mas.industrial_rag.postgres import PostgresIndustrialRagStore, _row_factory, translate_sqlite_contract
from adaptive_mas.industrial_rag.repository import IndustrialRagStore, ImmutableConflict, ReleaseNotReady
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService, _period_matches_request
from adaptive_mas.research_provenance import stable_hash


@pytest.fixture(params=("sqlite", "postgresql"), ids=("sqlite", "postgresql"))
def shared_store_contract(request, tmp_path):
    """Run the same metadata/read contract against SQLite and an isolated PG schema."""
    object_store = LocalContentAddressedStore(tmp_path / "objects")
    if request.param == "sqlite":
        yield IndustrialRagStore(tmp_path / "rag.sqlite3", object_store)
        return

    dsn = os.environ.get("RAAMA_POSTGRES_CONTRACT_DSN")
    if not dsn:
        pytest.skip("set RAAMA_POSTGRES_CONTRACT_DSN to a dedicated raama_contract_test database")
    try:
        import psycopg
        from psycopg import sql
    except ImportError as error:
        pytest.fail(f"PostgreSQL contract DSN is set but psycopg is unavailable: {error}")
    connection_options = psycopg.conninfo.conninfo_to_dict(dsn)
    if connection_options.get("dbname") != "raama_contract_test":
        pytest.fail("PostgreSQL contract tests require the dedicated raama_contract_test database")
    if connection_options.get("host") not in {"127.0.0.1", "::1"}:
        pytest.fail("PostgreSQL contract tests require a loopback host")
    if connection_options.get("hostaddr") not in {None, "127.0.0.1", "::1"}:
        pytest.fail("PostgreSQL contract tests require a loopback host address")

    schema = f"raama_contract_{uuid.uuid4().hex}"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    isolated_dsn = psycopg.conninfo.make_conninfo(dsn, options=f"-csearch_path={schema}")
    try:
        yield PostgresIndustrialRagStore(isolated_dsn, object_store)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


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


def test_research_trace_emits_ranked_channel_candidates_before_result_validation(tmp_path) -> None:
    events: list[dict[str, object]] = []
    bundle = _bundle()
    result = _research_one_bundle(
        tmp_path,
        bundle,
        "2024年度营业收入是多少",
        trace_callback=events.append,
    )
    assert result["status"] == "available"
    completed_channels = [event for event in events if event.get("stage") == "channel_completed"]
    assert completed_channels
    assert all("candidates" in event for event in completed_channels)
    assert any(event.get("stage") == "search_completed" for event in events)


def test_research_candidate_depth_preserves_each_requested_fact_period(tmp_path) -> None:
    base = _bundle(
        payload="2023年度营业收入339,123；2024年度营业收入337,488".encode(),
        raw_value="337,488",
        normalized_value="337488",
    )
    fact_2024 = base.facts[0]
    fact_2023 = replace(
        fact_2024,
        fact_id="version-1-fact-2023",
        raw_value="339,123",
        normalized_value="339123",
        report_period="2023",
    )
    bundle = replace(base, facts=(fact_2023, fact_2024))
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, [bundle.version.version_id], "release-1")
    plan = replace(
        _plan(),
        query="2024年度营业收入是多少，和2023年度相比如何？",
        report_periods=("2023", "2024"),
        task_type="comparison",
        top_k=2,
    )

    result = IndustrialRagService(store).research(
        plan,
        max_searches=2,
        evidence_limit=8,
        table_expansion_limit=0,
        use_reranker=False,
    )

    coverage = result["coverage"]
    by_period = {str(row["report_period"]): row for row in coverage}
    assert set(by_period) == {"2023", "2024"}
    assert all(row["numeric_claim_ready"] is True for row in by_period.values())
    fact_ids = {
        str(item["citation"]["fact_id"])
        for item in result["supporting_evidence"]
        if item.get("citation", {}).get("fact_id")
    }
    assert fact_ids == {"version-1-fact-2023", "version-1-fact-1"}
    assert result["planner"]["requested_result_limit"] == 2
    assert result["planner"]["candidate_depth_per_search"] == 3


def _research_one_bundle(
    tmp_path,
    bundle: DocumentBundle,
    query: str,
    *,
    periods: tuple[str, ...] = ("2024",),
    trace_callback=None,
) -> dict[str, object]:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, [bundle.version.version_id], "release-1")
    plan = replace(_plan(), query=query, report_periods=periods)
    return IndustrialRagService(store).research(plan, trace_callback=trace_callback)


def _plan(*, as_of: datetime | None = None, principal: str = "analyst", time_mode: str = "strict_forward", release_id: str | None = None, channels: tuple[str, ...] = ("structured_facts", "bm25")) -> QueryPlan:
    return QueryPlan(
        plan_id="query-plan-1", query="2024年度营业收入是多少", security_code="600036",
        as_of=as_of or datetime.now(timezone.utc), principal_id=principal, task_type="fact",
        report_periods=("2024",), release_id=release_id, channels=channels,
        time_mode=time_mode,  # type: ignore[arg-type]
    )


def test_annual_period_matching_rejects_interim_dates_and_flow_date_ambiguity() -> None:
    assert _period_matches_request("2024", "2024", metric="operating_revenue")
    assert _period_matches_request("2024", "2024-12-31", metric="total_assets")
    assert _period_matches_request("2024", "2024-12-31", metric="debt_ratio")
    assert not _period_matches_request("2024", "2024-03-31", metric="total_assets")
    assert not _period_matches_request("2024", "2024-06-30", metric="total_assets")
    assert not _period_matches_request("2024", "2024-12-31", metric="operating_revenue")
    assert _period_matches_request("2024-12-31", "2024-12-31", metric="operating_revenue")


def test_unknown_accounting_basis_fact_is_source_visible_but_not_numeric_eligible(tmp_path) -> None:
    unknown_basis = "not_stated_in_target_table"
    base = _bundle()
    bundle = replace(
        base,
        document=replace(base.document, accounting_basis=unknown_basis),
        facts=(replace(base.facts[0], consolidation=unknown_basis),),
    )
    unspecified_basis_dir = tmp_path / "unspecified-basis"
    unspecified_basis_dir.mkdir()

    result = _research_one_bundle(
        unspecified_basis_dir,
        bundle,
        "2024年度营业收入是多少人民币百万元",
    )

    fact = next(item for item in result["retrieved"] if item.get("fact_id") == "version-1-fact-1")
    assert fact["citation"]["accounting_basis"] == unknown_basis
    assert fact["citation"]["accounting_basis_status"] == "unknown"
    assert fact["citation"]["source_observation_eligible"] is True
    assert fact["citation"]["numeric_eligible"] is False
    coverage = result["coverage"][0]
    assert coverage["status"] == "located"
    assert coverage["claim_status"] == "review_required"
    assert coverage["numeric_claim_ready"] is False
    assert {tuple(sorted(item.items())) for item in coverage["observed_accounting_basis_details"]} == {
        tuple(sorted({"raw": unknown_basis, "status": "unknown"}.items()))
    }

    explicit_basis_dir = tmp_path / "explicit-basis"
    explicit_basis_dir.mkdir()
    explicit_basis = _research_one_bundle(
        explicit_basis_dir,
        bundle,
        "合并2024年度营业收入是多少人民币百万元",
    )
    assert explicit_basis["coverage"][0]["status"] != "located"
    assert explicit_basis["numeric_claim_ready"] is False


@pytest.mark.parametrize(
    ("source_period", "prior_period"),
    (("2024-03-31", "2023-03-31"), ("2024-06-30", "2023-06-30")),
)
def test_annual_flow_research_does_not_cover_an_interim_table_date(
    tmp_path, source_period: str, prior_period: str,
) -> None:
    base = _bundle()
    table_id = "version-1-revenue-period-table"
    table = Table(
        table_id,
        "version-1",
        "营业收入",
        1,
        unit_text="千元",
        accounting_basis="consolidated",
        report_period="2024",
        page_start=3,
        page_end=3,
    )
    cells = (
        TableCell("revenue-r0-c0", table_id, "version-1", 0, 0, "项目", page_number=3),
        TableCell("revenue-r0-c1", table_id, "version-1", 0, 1, source_period, page_number=3),
        TableCell("revenue-r0-c2", table_id, "version-1", 0, 2, prior_period, page_number=3),
        TableCell("revenue-r1-c0", table_id, "version-1", 1, 0, "营业收入", page_number=3),
        TableCell(
            "revenue-r1-c1", table_id, "version-1", 1, 1, "100", "100", "number",
            page_number=3, unit="千元", accounting_basis="consolidated", report_period=source_period,
        ),
        TableCell(
            "revenue-r1-c2", table_id, "version-1", 1, 2, "90", "90", "number",
            page_number=3, unit="千元", accounting_basis="consolidated", report_period=prior_period,
        ),
    )
    block = Block(
        "version-1-revenue-row",
        "version-1",
        f"列头：项目 | {source_period} | {prior_period}\n数据行：营业收入 | 100 | 90",
        block_type="table_row_group",
        page_number=3,
        table_id=table_id,
        row_start=1,
        row_end=1,
    )
    bundle = replace(base, tables=(table,), cells=cells, blocks=(block,), facts=())

    result = _research_one_bundle(tmp_path, bundle, "2024年度合并营业收入是多少千元")

    coverage = result["coverage"][0]
    assert coverage["normalized_report_period"] == "2024"
    assert coverage["status"] != "located"
    assert coverage["observed_report_periods"] == []
    assert result["numeric_claim_ready"] is False


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


def test_local_search_ablation_switches_change_row_group_and_fusion_behavior(tmp_path) -> None:
    base = _bundle()
    row_group = Block(
        "version-1-semantic-row",
        "version-1",
        "列头：2024年度营业收入\n数据行：营业收入 | 337,488",
        block_type="table_row_group",
        page_number=3,
        table_id="version-1-semantic-table",
        row_start=1,
        row_end=1,
    )
    bundle = replace(base, blocks=(*base.blocks, row_group))
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")
    service = IndustrialRagService(store)
    plan = replace(
        _plan(release_id="release-1", channels=("bm25", "structured_facts")),
        query="2024年度营业收入是多少",
        top_k=10,
    )

    baseline = service.search(plan, use_cache=False, use_reranker=False)
    without_row_groups = service.search(
        plan, use_cache=False, use_reranker=False, row_group_chunking=False,
    )
    first_channel = service.search(
        plan, use_cache=False, use_reranker=False, fusion_strategy="first_available_channel",
    )

    assert any(item["citation"].get("table_id") == row_group.table_id for item in baseline["retrieved"])
    assert all(item["citation"].get("table_id") != row_group.table_id for item in without_row_groups["retrieved"])
    assert first_channel["fusion"]["method"] == "fusion_disabled_first_available_channel"
    assert first_channel["fusion"]["selected_channel"] == "bm25"


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
        def __init__(self) -> None:
            self.calls = 0
            self.version = "v1"

        @property
        def cache_identity(self):
            return {
                "implementation": "test-dense",
                "version": self.version,
                "configuration": {"fixture": "one-source-table"},
            }

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
        def __init__(self) -> None:
            self.calls = 0
            self.version = "v1"

        @property
        def cache_identity(self):
            return {
                "implementation": "test-reranker",
                "version": self.version,
                "configuration": {"fixture": "identity-ranking"},
            }

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

    dense.version = "v2"
    with_new_adapter = service.search(replace(plan, aliases=("主营业务收入",)))
    assert with_new_adapter["cache"]["hit"] is False
    assert with_new_adapter["cache"]["key_sha256"] != with_different_alias["cache"]["key_sha256"]
    assert dense.calls == 3

    reranker.version = "v2"
    with_new_reranker = service.search(replace(plan, aliases=("主营业务收入",)))
    assert with_new_reranker["cache"]["hit"] is False
    assert with_new_reranker["cache"]["key_sha256"] != with_new_adapter["cache"]["key_sha256"]
    assert reranker.calls == 4

    policy_search = service.search(
        replace(plan, aliases=("营业收入",)),
        rrf_rank_constant=35,
        use_reranker=False,
    )
    assert policy_search["cache"]["key_sha256"] != first["cache"]["key_sha256"]
    assert policy_search["fusion"]["rank_constant"] == 35
    assert policy_search["reranking"]["status"] == "disabled"
    assert reranker.calls == 4
    policy_repeat = service.search(
        replace(plan, plan_id="policy-retry", aliases=("营业收入",)),
        rrf_rank_constant=35,
        use_reranker=False,
    )
    assert policy_repeat["cache"]["hit"] is True
    assert policy_repeat["plan_id"] == "policy-retry"


def test_search_skips_persistent_cache_for_unversioned_configured_adapters(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(_bundle())
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")
    plan = _plan(channels=("dense",))

    def adapter_for(text: str):
        def adapter(query_plan, version_ids):
            assert "version-1" in version_ids
            return [EvidenceResult(
                evidence_id=f"ev-{text}", channel="dense", document_id="document-1",
                version_id="version-1", source_id="cninfo-123456",
                content_sha256=_bundle().version.content_sha256,
                page_number=3, block_id="version-1-block-1", fact_id=None,
                text=text, score=1.0, source_time_basis="strict_forward",
            )]
        return adapter

    first = IndustrialRagService(store, channel_adapters={"dense": adapter_for("A")}).search(plan)
    second = IndustrialRagService(store, channel_adapters={"dense": adapter_for("B")}).search(plan)

    assert first["retrieved"][0]["text"] == "A"
    assert second["retrieved"][0]["text"] == "B"
    assert first["cache"]["stored"] is False
    assert second["cache"]["stored"] is False
    assert second["cache"]["reason"] == "configured_runtime_identity_incomplete"
    assert "key_sha256" not in second["cache"]


def test_configured_bm25_adapter_replaces_local_reference_channel(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    bundle = _bundle()
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, ["version-1"], "release-1")

    class Bm25Stub:
        implementation = "test-release-bound-bm25"
        cache_identity = {
            "implementation": implementation,
            "version": "v1",
            "configuration": {"fixture": "one-source-block"},
        }

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, plan, version_ids):
            self.calls += 1
            assert plan.channels == ("bm25",)
            assert version_ids == ["version-1"]
            return [EvidenceResult(
                evidence_id="ev-external-bm25", channel="bm25", document_id="document-1",
                version_id="version-1", source_id="cninfo-123456",
                content_sha256=bundle.version.content_sha256, page_number=3,
                block_id="version-1-block-1", fact_id=None, text="external release-bound hit",
                score=1.0, source_time_basis="strict_forward",
            )]

    adapter = Bm25Stub()
    service = IndustrialRagService(store, channel_adapters={"bm25": adapter})
    result = service.search(_plan(channels=("bm25",)), use_cache=False)

    assert adapter.calls == 1
    assert result["channels"]["bm25"]["implementation"] == adapter.implementation
    assert result["retrieved"][0]["text"] == "external release-bound hit"


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


def test_late_document_does_not_change_an_existing_as_of_search_result(tmp_path) -> None:
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=2)
    early = _bundle(
        source_id="cninfo-early", version_id="version-early", document_id="document-early",
        payload=b"2024 annual report revenue 100 CNY million",
        first_available_at=cutoff - timedelta(days=2),
        captured_at=cutoff - timedelta(minutes=1),
        ingested_at=cutoff - timedelta(seconds=30),
    )
    store.ingest_bundle(early)
    _consume_all(store)
    _publish(store, [early.version.version_id], "release-before-late-source")
    plan = replace(
        _plan(as_of=cutoff, release_id="release-before-late-source"),
        query="2024年度营业收入是多少",
        channels=("structured_facts", "bm25"),
    )
    before = IndustrialRagService(store).search(plan, use_cache=False)
    before_signature = [
        (item["evidence_id"], item["source_id"], item["text"], item["rank_score"])
        for item in before["retrieved"]
    ]
    assert before_signature

    late = _bundle(
        source_id="cninfo-late", version_id="version-late", document_id="document-late",
        payload=b"2024 annual report revenue 999 CNY million",
        first_available_at=cutoff - timedelta(days=2),
        captured_at=cutoff + timedelta(minutes=1),
        ingested_at=cutoff + timedelta(minutes=2),
    )
    store.ingest_bundle(late)
    _consume_all(store)
    _publish(store, [early.version.version_id, late.version.version_id], "release-after-late-source")
    after = IndustrialRagService(store).search(
        replace(plan, release_id="release-after-late-source"),
        use_cache=False,
    )
    after_signature = [
        (item["evidence_id"], item["source_id"], item["text"], item["rank_score"])
        for item in after["retrieved"]
    ]

    assert after_signature == before_signature
    assert all(item["source_id"] != late.version.source_id for item in after["retrieved"])


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


def test_search_prioritizes_exact_metric_period_table_row_over_noisy_page_text(tmp_path) -> None:
    source = "指标,2024年,2023年\r\n营业收入（千元）,100,90\r\n".encode()
    parsed = parse_isolated(source, "financial.csv", scratch_root=tmp_path / "scratch")
    base = _bundle(payload=source)
    version = replace(
        base.version,
        parser_name=str(parsed["parser_name"]),
        parser_version=str(parsed["parser_version"]),
        parser_confidence=float(parsed["parser_confidence"]),
    )
    records = records_from_parse(parsed, version.version_id)
    noisy_blocks = tuple(
        Block(
            block_id=f"noise-query-{index}", version_id=version.version_id,
            text="宁德时代2024年年度报告第二节主要会计数据表列示的2024年营业收入是多少",
            block_type="paragraph", page_number=2,
        )
        for index in range(20)
    )
    bundle = replace(
        base,
        version=version,
        document=replace(base.document, title="Test 2024 annual report", entity_name="Test issuer"),
        sections=records["sections"],
        spans=records["spans"],
        blocks=records["blocks"] + noisy_blocks,
        tables=records["tables"],
        cells=records["cells"],
        facts=(),
    )
    store = IndustrialRagStore(tmp_path / "rag.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, [version.version_id], "release-1")
    plan = replace(
        _plan(release_id="release-1", channels=("bm25",)),
        query="宁德时代2024年年度报告第二节主要会计数据表列示的2024年营业收入是多少",
        task_type="table",
        top_k=10,
    )

    result = IndustrialRagService(store).search(plan)

    assert result["retrieved"]
    first = result["retrieved"][0]
    assert first["citation"]["table_id"] == f"tbl-{version.version_id}-{parsed['tables'][0]['table_id']}"
    assert first["citation"]["row_start"] == first["citation"]["row_end"] == 1


def test_cross_page_merge_is_not_automatic_without_matching_basis_and_header() -> None:
    prior = {"title": "财务指标", "header": ["2024", "2023"], "security_code": "600036", "report_period": "2024", "unit": "人民币百万元", "page_end": 3}
    next_page = {"title": "财务指标", "header": ["2024", "2023"], "security_code": "600036", "report_period": "2024", "unit": "人民币百万元", "page_start": 4, "continued_marker": True}
    verified = validate_cross_page_table_continuation(prior, next_page)
    assert verified["decision"] == "verified_continuation"
    assert verified["automatic_merge_allowed"] is True
    next_page["unit"] = "人民币元"
    rejected = validate_cross_page_table_continuation(prior, next_page)
    assert rejected["automatic_merge_allowed"] is False


def test_docling_continuation_candidates_require_adjacent_matching_tables_and_compatible_context() -> None:
    def detect(child_page: int = 115, child_columns: int = 3, child_context: str = "") -> dict[str, object]:
        tables = [
            {"table_id": "prior-table", "page_start": 114, "page_end": 114, "header_rows": 1, "title": None},
            {
                "table_id": "child-table", "page_start": child_page, "page_end": child_page,
                "header_rows": 1, "title": None, "row_group_quarantine_reason": "column_headers_not_detected",
            },
        ]
        cells = [
            {"table_id": "prior-table", "row_index": 0, "column_index": 0, "raw_text": "项目"},
            {"table_id": "prior-table", "row_index": 0, "column_index": 1, "raw_text": "期末余额"},
            {"table_id": "prior-table", "row_index": 0, "column_index": 2, "raw_text": "期初余额"},
        ]
        cells.extend(
            {"table_id": "child-table", "row_index": 0, "column_index": column, "raw_text": f"value-{column}"}
            for column in range(child_columns)
        )
        pages = [
            {"page_number": 114, "text": "1、合并资产负债表\n编制单位：Example Issuer\n2024 年 12 月 31 日\n单位为：千元"},
            {"page_number": child_page, "text": child_context},
        ]
        annotate_table_continuation_candidates(tables, cells, pages)
        candidate = tables[1].get("continuation_candidate")
        assert isinstance(candidate, dict)
        return candidate

    candidate = detect()
    assert candidate["eligible_for_review"] is True
    assert candidate["previous_context"]["entity_name"] == "Example Issuer"
    assert candidate["previous_context"]["unit"] == "千元"
    assert candidate["previous_context"]["accounting_basis"] == "consolidated"
    assert candidate["header_labels"] == {0: "项目", 1: "期末余额", 2: "期初余额"}

    nonadjacent = detect(child_page=116)
    assert nonadjacent["checks"]["adjacent_pages"] is False
    assert nonadjacent["eligible_for_review"] is False

    mismatched = detect(child_columns=2)
    assert mismatched["checks"]["matching_column_count"] is False
    assert mismatched["eligible_for_review"] is False

    conflicting = detect(
        child_context="编制单位：Different Issuer\n2023 年 12 月 31 日\n单位为：人民币元",
    )
    assert set(conflicting["metadata_conflicts"]) == {"entity_name", "report_period", "unit"}
    assert conflicting["checks"]["metadata_compatible"] is False
    assert conflicting["eligible_for_review"] is False


def test_continuation_review_is_durable_revocable_and_preserves_both_page_citations(tmp_path) -> None:
    base = _bundle()
    context_text = "1、合并资产负债表\n编制单位：招商银行股份有限公司\n2024 年 12 月 31 日\n单位为：千元\n项目 | 期末余额 | 期初余额"
    prior_id, child_id = "version-1-table-prior", "version-1-table-child"
    candidate = {
        "candidate_id": f"cont-{prior_id}-{child_id}",
        "previous_table_id": prior_id,
        "current_table_id": child_id,
        "previous_page": 114,
        "current_page": 115,
        "column_count": 3,
        "header_labels": {"0": "项目", "1": "期末余额", "2": "期初余额"},
        "previous_context": {
            "title": "合并资产负债表", "entity_name": "招商银行股份有限公司",
            "report_period": "2024 年 12 月 31 日", "unit": "千元", "accounting_basis": "consolidated",
        },
        "current_context": {},
        "metadata_conflicts": {},
        "checks": {key: True for key in (
            "adjacent_pages", "unique_predecessor", "matching_column_count",
            "child_headers_missing", "predecessor_headers_present", "metadata_compatible",
        )},
        "eligible_for_review": True,
        "review_status": "proposed",
        "parser_header_rows": 1,
        "context_span_id": "version-1-span-context",
    }
    prior = Table(prior_id, "version-1", None, 1, page_start=114, page_end=114)
    child = Table(child_id, "version-1", None, 0, page_start=115, page_end=115, continuation_candidate=candidate)
    cells = (
        TableCell("prior-r0-c0", prior_id, "version-1", 0, 0, "项目", page_number=114),
        TableCell("prior-r0-c1", prior_id, "version-1", 0, 1, "期末余额", page_number=114),
        TableCell("prior-r0-c2", prior_id, "version-1", 0, 2, "期初余额", page_number=114),
        TableCell("child-r15-c0", child_id, "version-1", 15, 0, "资产总计", page_number=115),
        TableCell("child-r15-c1", child_id, "version-1", 15, 1, "786,658,123", "786658123", "number", page_number=115),
        TableCell("child-r15-c2", child_id, "version-1", 15, 2, "717,168,041", "717168041", "number", page_number=115),
    )
    context_span = Span("version-1-span-context", "version-1", 114, 0, len(context_text), context_text)
    blocks = (
        Block("version-1-context-block", "version-1", context_text, page_number=114,
              char_start=0, char_end=len(context_text), span_id=context_span.span_id),
        Block("version-1-child-row", "version-1", "列头：\n数据行：资产总计 | 786,658,123 | 717,168,041",
              block_type="table_row_group", page_number=115, table_id=child_id,
              row_start=15, row_end=15, quarantine_reason="column_headers_not_detected"),
    ) + tuple(
        Block(f"version-1-noise-{index}", "version-1", "2024年年报资产总计相关说明", page_number=116 + index)
        for index in range(3)
    )
    bundle = replace(
        base,
        blocks=blocks,
        spans=base.spans + (context_span,),
        tables=(prior, child),
        cells=cells,
        facts=(),
        read_principals=("analyst", "continuation-reviewer"),
    )
    store = IndustrialRagStore(tmp_path / "continuation.sqlite3", LocalContentAddressedStore(tmp_path / "objects"))
    store.ingest_bundle(bundle)
    _consume_all(store)
    _publish(store, [bundle.version.version_id], "release-1")
    service = IndustrialRagService(store)
    query_plan = replace(
        _plan(release_id="release-1", channels=("bm25",)),
        query="2024年合并资产负债表资产总计",
    )
    review_plan = replace(query_plan, principal_id="continuation-reviewer")
    before = service.search(query_plan)
    assert not any(item["citation"].get("table_id") == child_id for item in before["retrieved"])
    before_page = service.search_page(query_plan, page_size=1)
    stale_cursor = before_page["pagination"]["next_cursor"]
    assert stale_cursor
    with pytest.raises(PermissionError):
        service.read_table_row(query_plan, child_id, 15, 15)
    with pytest.raises(PermissionError):
        service.review_continuation_candidate(
            replace(query_plan, principal_id="untrusted-reviewer"), child_id,
            action="confirm", content_sha256=bundle.version.content_sha256,
        )

    listed = service.list_continuation_candidates(review_plan)["candidates"]
    reviewed = next(item for item in listed if item["current_table_id"] == child_id)
    assert reviewed["review_status"] == "proposed"
    revision_before = store.continuation_revision()
    service.review_continuation_candidate(
        review_plan, child_id, action="confirm", content_sha256=bundle.version.content_sha256,
        note="reviewed source page pair",
    )
    assert store.continuation_revision() == revision_before + 1
    with pytest.raises(ValueError, match="continuation review change"):
        service.search_page(query_plan, page_size=1, cursor=stale_cursor)

    after = service.search(query_plan)
    assert after["continuation_revision"] == revision_before + 1
    assert after["cache"]["key_sha256"] != before["cache"]["key_sha256"]
    row_hit = next(item for item in after["retrieved"] if item["citation"].get("table_id") == child_id)
    expanded = service.read_table_row(query_plan, child_id, 15, 15)
    assert row_hit["page_number"] == 115
    assert expanded["citation"]["page_number"] == 115
    assert expanded["citation"]["supporting_spans"][0]["page_number"] == 114
    assert any(cell["page_number"] == 114 for cell in expanded["citation"]["supporting_cells"])
    assert any(cell["cell_id"] == "child-r15-c1" and cell["page_number"] == 115 for cell in expanded["citation"]["supporting_cells"])

    research_plan = replace(
        query_plan,
        query="截至2024年报第115页的合并资产负债表资产总计是多少千元",
        report_periods=("2024",),
    )
    research = service.research(research_plan)
    coverage = research["coverage"][0]
    assert research["status"] == "review_required"
    assert research["numeric_claim_ready"] is False
    assert coverage["status"] == "located"
    assert coverage["normalized_report_period"] == "2024"
    assert coverage["observed_report_periods"] == ["2024-12-31"]
    assert coverage["observed_units"] == ["千元"]
    assert coverage["observed_accounting_bases"] == ["consolidated"]
    assert coverage["claim_status"] == "review_required"
    assert len(coverage["observations"]) == 1
    observation = coverage["observations"][0]
    assert observation["source_metric_label"] == "资产总计"
    assert observation["normalized_value"] == "786658123"
    assert observation["numeric_claim_eligible"] is False
    assert observation["report_period"] == "2024-12-31"
    assert observation["unit"] == "千元"
    assert observation["accounting_basis"] == "consolidated"
    assert observation["source_location"]["cell_id"] == "child-r15-c1"
    assert observation["source_location"]["page_number"] == 115
    assert coverage["source_locations"][0]["supporting_pages"] == [114, 115]
    assert research["stop_policy"]["stop_reason"] == "coverage_located_but_numeric_claim_requires_review"

    service.review_continuation_candidate(
        review_plan, child_id, action="revoke", content_sha256=bundle.version.content_sha256,
        note="review revoked",
    )
    assert store.continuation_revision() == revision_before + 2
    after_revoke = service.search(query_plan)
    assert after_revoke["cache"]["key_sha256"] != after["cache"]["key_sha256"]
    assert not any(item["citation"].get("table_id") == child_id for item in after_revoke["retrieved"])
    assert not any(block["block_id"] == "version-1-child-row" for block in store.read_blocks(["version-1"]))
    with pytest.raises(PermissionError):
        service.read_table_row(query_plan, child_id, 15, 15)
    final_candidate = next(
        item for item in service.list_continuation_candidates(review_plan)["candidates"]
        if item["current_table_id"] == child_id
    )
    assert final_candidate["review_status"] == "revoked"


def test_continuation_review_rejects_a_predecessor_from_another_source_version(tmp_path) -> None:
    child_base = _bundle(
        version_id="version-1", document_id="document-1", source_id="source-1",
        payload=b"child source version",
    )
    other_base = _bundle(
        version_id="version-2", document_id="document-2", source_id="source-2",
        payload=b"different predecessor source version",
    )
    prior_id, child_id = "version-2-table-prior", "version-1-table-child"
    candidate = {
        "candidate_id": f"cont-{prior_id}-{child_id}",
        "previous_table_id": prior_id,
        "current_table_id": child_id,
        "previous_page": 114,
        "current_page": 115,
        "column_count": 3,
        "header_labels": {"0": "项目", "1": "期末余额", "2": "期初余额"},
        "previous_context": {"title": "合并资产负债表", "unit": "千元"},
        "current_context": {},
        "metadata_conflicts": {},
        "checks": {key: True for key in (
            "adjacent_pages", "unique_predecessor", "matching_column_count",
            "child_headers_missing", "predecessor_headers_present", "metadata_compatible",
        )},
        "eligible_for_review": True,
        "review_status": "proposed",
        "parser_header_rows": 1,
    }
    prior = Table(prior_id, "version-2", None, 1, page_start=114, page_end=114)
    prior_cells = tuple(
        TableCell(f"prior-r0-c{column}", prior_id, "version-2", 0, column, label, page_number=114)
        for column, label in enumerate(("项目", "期末余额", "期初余额"))
    )
    child = Table(child_id, "version-1", None, 0, page_start=115, page_end=115, continuation_candidate=candidate)
    child_cells = (
        TableCell("child-r15-c0", child_id, "version-1", 15, 0, "资产总计", page_number=115),
        TableCell("child-r15-c1", child_id, "version-1", 15, 1, "786,658,123", "786658123", "number", page_number=115),
        TableCell("child-r15-c2", child_id, "version-1", 15, 2, "717,168,041", "717168041", "number", page_number=115),
    )
    child_block = Block(
        "version-1-child-row", "version-1", "列头：\n数据行：资产总计 | 786,658,123 | 717,168,041",
        block_type="table_row_group", page_number=115, table_id=child_id,
        row_start=15, row_end=15, quarantine_reason="column_headers_not_detected",
    )
    reviewer = "continuation-reviewer"
    child_bundle = replace(
        child_base, tables=(child,), cells=child_cells, blocks=child_base.blocks + (child_block,), facts=(),
        read_principals=("analyst", reviewer),
    )
    other_bundle = replace(
        other_base, tables=(prior,), cells=prior_cells, facts=(),
        read_principals=("analyst", reviewer),
    )
    store = IndustrialRagStore(tmp_path / "cross-source.sqlite3", LocalContentAddressedStore(tmp_path / "cross-source-objects"))
    store.ingest_bundle(child_bundle)
    store.ingest_bundle(other_bundle)
    _consume_all(store)
    _publish(store, ["version-1", "version-2"], "release-1")
    service = IndustrialRagService(store)
    review_plan = replace(_plan(release_id="release-1", principal=reviewer), query="continuation review")
    found = next(
        item for item in service.list_continuation_candidates(review_plan)["candidates"]
        if item["current_table_id"] == child_id
    )
    assert found["checks"]["same_source_version"] is False
    assert found["eligible_for_review"] is False
    with pytest.raises(ValueError, match="predecessor is no longer valid"):
        service.review_continuation_candidate(
            review_plan, child_id, action="confirm", content_sha256=child_bundle.version.content_sha256,
        )
    with pytest.raises(PermissionError, match="quarantined"):
        service.read_table_row(review_plan, child_id, 15, 15)
    assert not any(block["block_id"] == child_block.block_id for block in store.read_blocks(["version-1"]))


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
    assert translate_sqlite_contract("INSERT OR REPLACE INTO rag_query_cache VALUES(?,?,?)") == (
        "INSERT INTO rag_query_cache VALUES(%s, %s, %s) ON CONFLICT(cache_key) DO UPDATE "
        "SET payload_json=EXCLUDED.payload_json, created_at=EXCLUDED.created_at"
    )
    assert translate_sqlite_contract(
        "SELECT json_extract(t.payload_json,'$.continuation_candidate.previous_table_id') FROM rag_tables t"
    ) == "SELECT (t.payload_json::jsonb #>> '{continuation_candidate,previous_table_id}') FROM rag_tables t"


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


def _shared_contract_bundle() -> DocumentBundle:
    base = _bundle(
        source_id="contract-source-1",
        version_id="contract-version-1",
        document_id="contract-document-1",
    )
    table = Table(
        "contract-table-1",
        base.version.version_id,
        "2024年度营业收入",
        1,
        unit_text="人民币百万元",
        accounting_basis="本集团",
        report_period="2024",
        page_start=3,
        page_end=3,
    )
    cells = (
        TableCell("contract-cell-r0c0", table.table_id, base.version.version_id, 0, 0, "项目", page_number=3),
        TableCell("contract-cell-r0c1", table.table_id, base.version.version_id, 0, 1, "营业收入", page_number=3),
    )
    return replace(base, tables=(table,), cells=cells)


def test_shared_store_contract_ingest_reads_and_current_permission(shared_store_contract) -> None:
    store = shared_store_contract
    bundle = _shared_contract_bundle()

    first = store.ingest_bundle(bundle)
    repeated = store.ingest_bundle(bundle)
    assert first["status"] == "ingested"
    assert repeated["status"] == "reused"
    assert repeated["ingested_at"] == first["ingested_at"]

    facts = store.read_facts([bundle.version.version_id])
    assert len(facts) == 1
    assert facts[0]["fact_id"] == "contract-version-1-fact-1"
    blocks = store.read_blocks([bundle.version.version_id])
    assert [block["block_id"] for block in blocks] == ["contract-version-1-block-1"]
    spans = store.read_spans([bundle.spans[0].span_id])
    assert len(spans) == 1
    assert spans[0]["text"] == bundle.spans[0].text
    reopened_row = store._read_table_row("contract-table-1", 0, 0)
    assert reopened_row is not None
    assert [cell["raw_text"] for cell in reopened_row["cells"]] == ["项目", "营业收入"]
    assert store._get_document_bytes(
        bundle.document.document_id, bundle.version.version_id, "analyst",
    ) == bundle.payload

    _consume_all(store)
    _publish(store, [bundle.version.version_id], "shared-read-release")
    as_of = datetime.now(timezone.utc) + timedelta(seconds=5)
    permitted = store.permitted_versions(
        "analyst", "600036", as_of,
        time_mode="strict_forward", report_periods=("2024",),
        accounting_basis="本集团", release_id="shared-read-release",
    )
    assert [row["version_id"] for row in permitted] == [bundle.version.version_id]
    assert store.quarantine_summary()["items"] == []
    cache_payload = {"release_id": "shared-read-release", "results": [{"version_id": bundle.version.version_id}]}
    store.cache_put("shared-contract-cache", cache_payload)
    assert store.cache_get("shared-contract-cache") == cache_payload
    replacement_payload = {**cache_payload, "revision": 2}
    store.cache_put("shared-contract-cache", replacement_payload)
    assert store.cache_get("shared-contract-cache") == replacement_payload

    before_revision = store.permission_revision()
    after_revision = store.set_access("analyst", bundle.document.document_id, can_read=False)
    assert after_revision > before_revision
    assert store.permitted_versions(
        "analyst", "600036", as_of,
        time_mode="strict_forward", report_periods=("2024",),
        accounting_basis="本集团", release_id="shared-read-release",
    ) == []
    with pytest.raises(PermissionError):
        store._get_document_bytes(bundle.document.document_id, bundle.version.version_id, "analyst")

    changed_fact = replace(bundle.facts[0], raw_value="337,489", normalized_value="337489")
    with pytest.raises(ImmutableConflict, match="parsed child records"):
        store.ingest_bundle(replace(bundle, facts=(changed_fact,)))


def test_shared_store_contract_rolls_back_partial_child_insert(shared_store_contract) -> None:
    store = shared_store_contract
    bundle = _shared_contract_bundle()
    duplicated_span = replace(bundle.spans[0], text=bundle.spans[0].text + " duplicate")
    invalid_bundle = replace(bundle, spans=(*bundle.spans, duplicated_span))
    integrity_errors: tuple[type[BaseException], ...] = (sqlite3.IntegrityError,)
    if isinstance(store, PostgresIndustrialRagStore):
        import psycopg

        integrity_errors = (*integrity_errors, psycopg.IntegrityError)

    with pytest.raises(integrity_errors):
        store.ingest_bundle(invalid_bundle)

    with pytest.raises(FileNotFoundError):
        store.source_manifest([bundle.version.version_id])
    assert store.read_facts([bundle.version.version_id]) == []
    assert store.ingest_bundle(bundle)["status"] == "ingested"


def test_shared_store_contract_stale_outbox_lease_cannot_ack(shared_store_contract) -> None:
    store = shared_store_contract
    bundle = _shared_contract_bundle()
    store.ingest_bundle(bundle)

    first = store.claim_outbox("shared-contract-worker", lease_seconds=0.01)
    assert first is not None
    time.sleep(0.02)
    second = store.claim_outbox("shared-contract-worker", lease_seconds=10)
    assert second is not None
    assert second["event_key"] == first["event_key"]
    assert second["lease_generation"] > first["lease_generation"]
    with pytest.raises(RuntimeError, match="no longer owns"):
        store.finish_outbox(
            str(first["event_key"]),
            "shared-contract-worker",
            {"stale": True},
            lease_generation=int(first["lease_generation"]),
        )
    assert store.finish_outbox(
        str(second["event_key"]),
        "shared-contract-worker",
        {"current": True},
        lease_generation=int(second["lease_generation"]),
    ) is True
    assert store.finish_outbox(
        str(second["event_key"]),
        "shared-contract-worker",
        {"current": True},
        lease_generation=int(second["lease_generation"]),
    ) is False


def test_shared_store_contract_release_rollback_restores_prior_snapshot(shared_store_contract) -> None:
    store = shared_store_contract
    first_bundle = _shared_contract_bundle()
    store.ingest_bundle(first_bundle)
    _consume_all(store)
    _publish(store, [first_bundle.version.version_id], "contract-release-1")

    second_bundle = _bundle(
        source_id="contract-source-2",
        version_id="contract-version-2",
        document_id="contract-document-2",
        payload=b"2024 annual revenue 337489 CNY million",
    )
    store.ingest_bundle(second_bundle)
    _consume_all(store)
    _publish(store, [second_bundle.version.version_id], "contract-release-2")
    assert store.active_release_id() == "contract-release-2"

    store.rollback_release("contract-release-1", actor_id="contract-test", reason="shared adapter contract")

    assert store.active_release_id() == "contract-release-1"
    assert store.release_versions("contract-release-1") == [first_bundle.version.version_id]


def test_shared_store_contract_concurrent_same_version_ingest_is_idempotent(shared_store_contract) -> None:
    store = shared_store_contract
    bundle = _shared_contract_bundle()
    start_barrier = threading.Barrier(2)

    def ingest() -> dict[str, object]:
        start_barrier.wait(timeout=5)
        if isinstance(store, PostgresIndustrialRagStore):
            return store.ingest_bundle(bundle, ingested_at_factory=lambda: bundle.version.ingested_at)
        return store.ingest_bundle(bundle)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: ingest(), range(2)))

    assert sorted(str(outcome["status"]) for outcome in outcomes) == ["ingested", "reused"]
    assert len(store.read_facts([bundle.version.version_id])) == 1
    assert store.claim_outbox("shared-concurrent-consumer") is not None
    assert store.claim_outbox("shared-concurrent-consumer") is None


def test_shared_store_contract_concurrent_outbox_claims_are_exclusive(shared_store_contract) -> None:
    store = shared_store_contract
    first_bundle = _shared_contract_bundle()
    second_bundle = _bundle(
        source_id="contract-outbox-source-2",
        version_id="contract-outbox-version-2",
        document_id="contract-outbox-document-2",
        payload=b"second concurrent outbox event",
    )
    store.ingest_bundle(first_bundle)
    store.ingest_bundle(second_bundle)
    start_barrier = threading.Barrier(2)

    def claim(worker_id: str) -> dict[str, object] | None:
        start_barrier.wait(timeout=5)
        return store.claim_outbox(worker_id, lease_seconds=10)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ("shared-claim-worker-1", "shared-claim-worker-2")))

    assert all(claim is not None for claim in claims)
    claimed = [claim for claim in claims if claim is not None]
    assert len({claim["event_id"] for claim in claimed}) == 2
    assert len({claim["event_key"] for claim in claimed}) == 2
    for claim in claimed:
        assert store.finish_outbox(
            str(claim["event_key"]),
            str(claim["consumer_id"]),
            {"claimed_by": claim["consumer_id"]},
            lease_generation=int(claim["lease_generation"]),
        ) is True
    assert store.claim_outbox("shared-claim-worker-after-ack") is None


def test_shared_store_contract_reads_populated_quarantine_summary(shared_store_contract) -> None:
    store = shared_store_contract
    source = _shared_contract_bundle()
    quarantined_version = replace(
        source.version,
        parser_status="quarantined",
        parser_error_code="unsupported_format",
        parser_error="Parser does not support this source format.",
    )
    bundle = replace(
        source,
        version=quarantined_version,
        sections=(),
        spans=(),
        blocks=(),
        tables=(),
        cells=(),
        facts=(),
    )

    assert store.ingest_bundle(bundle)["status"] == "ingested"
    summary = store.quarantine_summary(limit=1)

    assert summary["count"] == 1
    assert summary["limit"] == 1
    item = summary["items"][0]
    assert set(item) == {
        "quarantine_id", "version_id", "error_code", "reason", "state", "created_at",
        "document_id", "title", "security_code", "source_id", "content_sha256", "source_url",
    }
    assert item["quarantine_id"] == "q-contract-version-1"
    assert item["version_id"] == bundle.version.version_id
    assert item["error_code"] == "unsupported_format"
    assert item["reason"] == "Parser does not support this source format."
    assert item["state"] == "open"
    assert item["document_id"] == bundle.document.document_id
    assert item["security_code"] == bundle.document.security_code
    assert item["source_id"] == bundle.version.source_id
    assert item["content_sha256"] == bundle.version.content_sha256
    assert item["source_url"] == bundle.version.source_url
    assert store.release_summaries()["open_quarantine_count"] == 1
    with pytest.raises(ValueError, match="quarantine limit"):
        store.quarantine_summary(limit=0)
    with pytest.raises(ValueError, match="quarantine limit"):
        store.quarantine_summary(limit=501)


def test_shared_store_contract_new_version_requires_immutable_document_metadata(shared_store_contract) -> None:
    store = shared_store_contract
    first = replace(
        _shared_contract_bundle(),
        sections=(), spans=(), blocks=(), tables=(), cells=(), facts=(),
    )
    assert store.ingest_bundle(first)["status"] == "ingested"
    _consume_all(store)

    second_payload = b"new version payload with changed document title"
    second_version = replace(
        first.version,
        version_id="contract-version-2",
        source_id="contract-source-2",
        content_sha256=hashlib.sha256(second_payload).hexdigest(),
    )
    changed_document = replace(first.document, title="changed document identity title")
    changed_second = replace(first, document=changed_document, version=second_version, payload=second_payload)
    with pytest.raises(ImmutableConflict, match="document identity was reused with changed metadata"):
        store.ingest_bundle(changed_second)

    with pytest.raises(FileNotFoundError):
        store.source_manifest([second_version.version_id])
    assert store.read_blocks([second_version.version_id]) == []
    assert store.read_facts([second_version.version_id]) == []
    assert store.claim_outbox("metadata-contract-consumer") is None
    # The object-store CAS write precedes the SQL transaction; a rejected
    # metadata identity may leave this exact content-addressed orphan behind.
    assert store.object_store.get(second_version.content_sha256) == second_payload

    valid_second = replace(changed_second, document=first.document)
    assert store.ingest_bundle(valid_second)["status"] == "ingested"
    assert store.ingest_bundle(valid_second)["status"] == "reused"
