"""Canonical, backend-neutral financial document and query contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal


class ContractError(ValueError):
    """A persisted RAG record does not satisfy its immutable schema contract."""


def validate_security_code(value: str | None) -> str | None:
    if value is None:
        return None
    if not re.fullmatch(r"\d{6}", value):
        raise ContractError("security code must contain exactly six digits")
    return value


def validate_sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractError("sha256 must be 64 lowercase hexadecimal characters")
    return value


def validate_identifier(value: str, label: str = "identifier") -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value):
        raise ContractError(f"{label} is not a safe immutable identifier")
    return value


def parse_datetime(value: datetime | str | None, label: str, *, required: bool = False) -> datetime | None:
    if value is None:
        if required:
            raise ContractError(f"{label} is required")
        return None
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    title: str
    document_type: str
    security_code: str | None = None
    entity_name: str | None = None
    report_period: str | None = None
    accounting_basis: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.document_id, "document_id")
        validate_security_code(self.security_code)
        if not self.title.strip() or len(self.title) > 500:
            raise ContractError("document title must contain 1..500 characters")
        if not self.document_type.strip() or len(self.document_type) > 64:
            raise ContractError("document_type must contain 1..64 characters")


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    version_id: str
    source_id: str
    content_sha256: str
    source_url: str
    rights_note: str
    first_available_at: datetime | str | None
    captured_at: datetime | str
    ingested_at: datetime | str
    published_at: datetime | str | None = None
    valid_from: datetime | str | None = None
    valid_to: datetime | str | None = None
    revised_at: datetime | str | None = None
    time_basis: Literal["strict_forward", "retrospective_proxy", "unknown"] = "unknown"
    parser_name: str = "unparsed"
    parser_version: str = "unavailable"
    parser_confidence: float | None = None
    parser_status: Literal["parsed", "quarantined", "unavailable"] = "parsed"
    parser_error_code: str | None = None
    parser_error: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.version_id, "version_id")
        validate_identifier(self.source_id, "source_id")
        validate_sha256(self.content_sha256)
        if not self.source_url.strip() or not self.rights_note.strip():
            raise ContractError("source_url and rights_note are required provenance")
        if self.time_basis not in {"strict_forward", "retrospective_proxy", "unknown"}:
            raise ContractError("time_basis must distinguish strict, retrospective, and unknown")
        first = parse_datetime(self.first_available_at, "first_available_at")
        captured = parse_datetime(self.captured_at, "captured_at", required=True)
        ingested = parse_datetime(self.ingested_at, "ingested_at", required=True)
        published = parse_datetime(self.published_at, "published_at")
        valid_from = parse_datetime(self.valid_from, "valid_from")
        valid_to = parse_datetime(self.valid_to, "valid_to")
        revised = parse_datetime(self.revised_at, "revised_at")
        for name, moment in (
            ("first_available_at", first), ("captured_at", captured), ("ingested_at", ingested),
            ("published_at", published), ("valid_from", valid_from), ("valid_to", valid_to),
            ("revised_at", revised),
        ):
            object.__setattr__(self, name, moment)
        if self.time_basis == "strict_forward" and first is None:
            raise ContractError("strict_forward records require a verified first_available_at")
        if self.time_basis == "unknown" and first is not None:
            raise ContractError("unknown time basis cannot claim first_available_at")
        if first is not None and captured < first:
            raise ContractError("captured_at cannot precede first_available_at")
        if ingested < captured:
            raise ContractError("ingested_at cannot precede captured_at")
        if first is not None and published is not None and first < published:
            raise ContractError("first_available_at cannot precede published_at")
        for label, moment in (("first_available_at", first), ("published_at", published)):
            if moment is not None and moment > captured:
                raise ContractError(f"{label} cannot be after captured_at")
        if valid_from is not None and valid_to is not None and valid_from >= valid_to:
            raise ContractError("valid_to must be later than valid_from")
        if self.parser_confidence is not None and not 0 <= self.parser_confidence <= 1:
            raise ContractError("parser_confidence must be between zero and one")
        if self.parser_status == "quarantined" and not self.parser_error_code:
            raise ContractError("quarantined document versions require a parser error code")


@dataclass(frozen=True, slots=True)
class Section:
    section_id: str
    version_id: str
    title: str
    level: int
    parent_section_id: str | None = None
    start_page: int | None = None
    end_page: int | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.section_id, "section_id")
        validate_identifier(self.version_id, "version_id")
        if self.level < 1 or self.level > 12:
            raise ContractError("section level must be between one and twelve")
        if (self.start_page is None) != (self.end_page is None):
            raise ContractError("section page range must be complete")
        if self.start_page is not None and (self.start_page < 1 or self.end_page < self.start_page):
            raise ContractError("section page range is invalid")


@dataclass(frozen=True, slots=True)
class Span:
    span_id: str
    version_id: str
    page_number: int | None
    char_start: int
    char_end: int
    text: str
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.span_id, "span_id")
        validate_identifier(self.version_id, "version_id")
        if self.page_number is not None and self.page_number < 1:
            raise ContractError("page_number must be positive")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ContractError("character offsets are invalid")
        if self.bbox is not None and (len(self.bbox) != 4 or self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]):
            raise ContractError("bbox must be an ordered x0,y0,x1,y1 tuple")


@dataclass(frozen=True, slots=True)
class Block:
    block_id: str
    version_id: str
    text: str
    block_type: str = "paragraph"
    section_id: str | None = None
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    char_start: int | None = None
    char_end: int | None = None
    parent_block_id: str | None = None
    previous_block_id: str | None = None
    next_block_id: str | None = None
    table_id: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    parser_confidence: float | None = None
    quarantine_reason: str | None = None
    span_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.block_id, "block_id")
        validate_identifier(self.version_id, "version_id")
        if self.page_number is not None and self.page_number < 1:
            raise ContractError("page_number must be positive")
        if (self.char_start is None) != (self.char_end is None):
            raise ContractError("block character range must be complete")
        if self.char_start is not None and (self.char_start < 0 or self.char_end < self.char_start):
            raise ContractError("block character offsets are invalid")
        if self.parser_confidence is not None and not 0 <= self.parser_confidence <= 1:
            raise ContractError("parser_confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class Table:
    table_id: str
    version_id: str
    title: str | None
    header_rows: int
    continued_from_table_id: str | None = None
    continuation_confidence: float | None = None
    unit_text: str | None = None
    accounting_basis: str | None = None
    report_period: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.table_id, "table_id")
        validate_identifier(self.version_id, "version_id")
        if self.header_rows < 0:
            raise ContractError("header_rows cannot be negative")
        if self.continuation_confidence is not None and not 0 <= self.continuation_confidence <= 1:
            raise ContractError("continuation_confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class TableCell:
    cell_id: str
    table_id: str
    version_id: str
    row_index: int
    column_index: int
    raw_text: str
    normalized_value: str | None = None
    value_kind: Literal["text", "number", "percent", "basis_points", "zero", "blank", "dash", "unknown"] = "text"
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    unit: str | None = None
    multiplier: str | None = None
    currency: str | None = None
    accounting_basis: str | None = None
    report_period: str | None = None
    footnote_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.cell_id, "cell_id")
        if self.row_index < 0 or self.column_index < 0:
            raise ContractError("table coordinates must be nonnegative")
        if self.value_kind in {"number", "percent", "basis_points", "zero"} and self.normalized_value is not None:
            try:
                numeric = Decimal(self.normalized_value)
            except InvalidOperation as error:
                raise ContractError("normalized numeric value must be a Decimal string") from error
            if not numeric.is_finite():
                raise ContractError("normalized numeric value must be finite")
        if self.value_kind in {"number", "percent", "basis_points", "zero"} and self.normalized_value is None:
            raise ContractError("numeric table cells require a normalized Decimal string")
        if self.value_kind in {"blank", "dash"} and self.normalized_value is not None:
            raise ContractError("blank and dash cells must remain nonnumeric")
        if self.multiplier is not None:
            try:
                multiplier = Decimal(self.multiplier)
            except InvalidOperation as error:
                raise ContractError("cell multiplier must be a Decimal string") from error
            if not multiplier.is_finite() or multiplier <= 0:
                raise ContractError("cell multiplier must be finite and positive")
        if self.value_kind == "zero" and Decimal(str(self.normalized_value)) != 0:
            raise ContractError("zero cell must normalize to exactly zero")
        if self.value_kind == "blank" and self.raw_text.strip():
            raise ContractError("blank cell must not contain source text")
        if self.value_kind == "dash" and self.raw_text.strip() not in {"-", "—", "–", "－", "--", "——"}:
            raise ContractError("dash cell must preserve a dash character")


@dataclass(frozen=True, slots=True)
class FinancialFact:
    """normalized_value stays in `unit`; multiply it for base-unit arithmetic."""
    fact_id: str
    version_id: str
    security_code: str
    entity_name: str
    metric: str
    raw_value: str
    normalized_value: str
    unit: str
    multiplier: str
    currency: str
    report_period: str
    consolidation: str
    span_ids: tuple[str, ...]
    parser_version: str
    confidence: float
    review_status: Literal["unreviewed", "reviewed", "rejected"] = "unreviewed"
    restatement_of_fact_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.fact_id, "fact_id")
        validate_identifier(self.version_id, "version_id")
        validate_security_code(self.security_code)
        try:
            normalized = Decimal(self.normalized_value)
            multiplier = Decimal(self.multiplier)
        except InvalidOperation as error:
            raise ContractError("fact value and multiplier must be Decimal strings") from error
        if not normalized.is_finite() or not multiplier.is_finite() or multiplier <= 0:
            raise ContractError("fact value must be finite and multiplier must be finite and positive")
        if not self.metric.strip() or not self.entity_name.strip() or not self.report_period.strip():
            raise ContractError("fact metric, entity, and report_period are required")
        if not self.span_ids:
            raise ContractError("financial fact must cite at least one source span")
        if not 0 <= self.confidence <= 1:
            raise ContractError("fact confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class DocumentBundle:
    document: Document
    version: DocumentVersion
    payload: bytes
    sections: tuple[Section, ...] = ()
    spans: tuple[Span, ...] = ()
    blocks: tuple[Block, ...] = ()
    tables: tuple[Table, ...] = ()
    cells: tuple[TableCell, ...] = ()
    facts: tuple[FinancialFact, ...] = ()
    read_principals: tuple[str, ...] = ("local-public",)


@dataclass(frozen=True, slots=True)
class CorpusReleaseSpec:
    release_id: str
    source_manifest_sha256: str
    parser_version: str
    chunker_version: str
    embedding_model: str
    embedding_dimension: int | None
    required_backends: tuple[str, ...]
    text_index_manifest_sha256: str
    vector_index_manifest_sha256: str

    def __post_init__(self) -> None:
        validate_identifier(self.release_id, "release_id")
        for digest in (self.source_manifest_sha256, self.text_index_manifest_sha256, self.vector_index_manifest_sha256):
            validate_sha256(digest)
        if not self.required_backends or len(set(self.required_backends)) != len(self.required_backends):
            raise ContractError("a corpus release requires unique, explicit backends")
        if self.embedding_dimension is not None and self.embedding_dimension < 1:
            raise ContractError("embedding_dimension must be positive when specified")
        if not self.parser_version or not self.chunker_version or not self.embedding_model:
            raise ContractError("release must pin parser, chunker, and embedding model versions")


@dataclass(frozen=True, slots=True)
class IndexBuildReceipt:
    release_id: str
    backend: str
    physical_index_handle: str
    backend_version: str
    source_manifest_sha256: str
    index_manifest_sha256: str
    document_count: int
    record_count: int
    embedding_model: str
    embedding_dimension: int | None
    built_at: datetime | str

    def __post_init__(self) -> None:
        validate_identifier(self.release_id, "release_id")
        if not self.backend or not self.physical_index_handle or not self.backend_version:
            raise ContractError("index receipt requires a physical handle and backend version")
        validate_sha256(self.source_manifest_sha256)
        validate_sha256(self.index_manifest_sha256)
        if self.document_count < 0 or self.record_count < 0:
            raise ContractError("index receipt counts cannot be negative")
        if self.embedding_dimension is not None and self.embedding_dimension < 1:
            raise ContractError("embedding_dimension must be positive")
        object.__setattr__(self, "built_at", parse_datetime(self.built_at, "built_at", required=True))


@dataclass(frozen=True, slots=True)
class QueryPlan:
    plan_id: str
    query: str
    security_code: str
    as_of: datetime | str
    principal_id: str
    task_type: Literal["fact", "comparison", "table", "narrative", "refutation", "unknown"]
    accounting_basis: str | None = None
    report_periods: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    release_id: str | None = None
    channels: tuple[str, ...] = ("structured_facts", "bm25")
    top_k: int = 10
    time_mode: Literal["strict_forward", "retrospective_diagnostic"] = "strict_forward"

    def __post_init__(self) -> None:
        validate_identifier(self.plan_id, "plan_id")
        validate_security_code(self.security_code)
        normalized_as_of = parse_datetime(self.as_of, "as_of", required=True)
        object.__setattr__(self, "as_of", normalized_as_of)
        if not self.query.strip() or len(self.query) > 2000:
            raise ContractError("query must contain 1..2000 characters")
        if not self.principal_id or len(self.principal_id) > 128:
            raise ContractError("principal_id is required")
        if self.task_type not in {"fact", "comparison", "table", "narrative", "refutation", "unknown"}:
            raise ContractError("task_type is unsupported")
        if self.top_k < 1 or self.top_k > 100:
            raise ContractError("top_k must be between 1 and 100")
        allowed = {"structured_facts", "bm25", "toc_page", "dense", "vision", "source_relations"}
        if not self.channels or set(self.channels) - allowed:
            raise ContractError("query channels contain an unsupported value")
        if self.time_mode not in {"strict_forward", "retrospective_diagnostic"}:
            raise ContractError("time_mode is invalid")


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    evidence_id: str
    channel: str
    document_id: str
    version_id: str
    source_id: str
    content_sha256: str
    page_number: int | None
    block_id: str | None
    fact_id: str | None
    text: str
    score: float
    source_time_basis: str
    citation: dict[str, object] = field(default_factory=dict)
