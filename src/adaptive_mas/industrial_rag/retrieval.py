"""Permission-first retrieval, calibrated score separation, and immutable citations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

from adaptive_mas.industrial_rag.models import EvidenceResult, QueryPlan, parse_datetime
from adaptive_mas.industrial_rag.repository import IndustrialRagStore, ReleaseNotReady, _json
from adaptive_mas.research_provenance import stable_hash


class BackendUnavailable(RuntimeError):
    """A requested retrieval backend is not configured; no substitute was selected."""


def financial_tokens(text: str) -> list[str]:
    """Deterministic Chinese/Latin reference analyzer; not a production language model."""
    tokens: list[str] = []
    for run in re.findall(r"[\u3400-\u9fff]+|[A-Za-z]+|\d+(?:\.\d+)?", text.casefold()):
        if re.fullmatch(r"\d+(?:\.\d+)?", run):
            tokens.append(run)
        elif re.fullmatch(r"[a-z]+", run):
            tokens.append(run)
        else:
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    return tokens


def _citation(block: Mapping[str, object], *, fact: Mapping[str, object] | None = None) -> dict[str, object]:
    # sqlite3.Row supports key lookup but not Mapping.get().
    block = dict(block)
    fact = dict(fact) if fact is not None else None
    citation: dict[str, object] = {
        "document_id": block.get("document_id"),
        "version_id": block.get("version_id"),
        "source_id": block.get("source_id"),
        "source_url": block.get("source_url"),
        "published_at": block.get("published_at"),
        "first_available_at": block.get("first_available_at"),
        "captured_at": block.get("captured_at"),
        "ingested_at": block.get("ingested_at"),
        "content_sha256": block.get("content_sha256"),
        "page_number": block.get("page_number"),
        "block_id": block.get("block_id"),
        "time_basis": block.get("time_basis"),
        "title": block.get("title"),
        "security_code": block.get("security_code"),
    }
    payload = json.loads(str(block.get("payload_json", "{}")))
    for key in ("bbox", "char_start", "char_end", "section_id", "table_id", "row_start", "row_end"):
        if payload.get(key) is not None:
            citation[key] = payload[key]
    if payload.get("span_id") is not None:
        citation["span_id"] = payload["span_id"]
    if fact is not None:
        citation.update({
            "fact_id": fact.get("fact_id"),
            "metric": fact.get("metric"),
            "raw_value": fact.get("raw_value"),
            "normalized_value": fact.get("normalized_value"),
            "unit": fact.get("unit"),
            "multiplier": fact.get("multiplier"),
            "currency": fact.get("currency"),
            "report_period": fact.get("report_period"),
            "accounting_basis": fact.get("consolidation"),
            "parser_confidence": fact.get("confidence"),
            "review_status": fact.get("review_status"),
        })
    return citation


def _bm25(query: str, documents: Sequence[Mapping[str, object]], *, limit: int, k1: float = 1.2, b: float = 0.75) -> list[tuple[Mapping[str, object], float]]:
    query_terms = list(dict.fromkeys(financial_tokens(query)))
    if not query_terms or not documents:
        return []
    tokenized = [financial_tokens(str(document.get("text", ""))) for document in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(len(lengths), 1)
    document_frequency: Counter[str] = Counter()
    term_frequencies: list[Counter[str]] = []
    for tokens in tokenized:
        frequency = Counter(tokens)
        term_frequencies.append(frequency)
        document_frequency.update(frequency.keys())
    scored: list[tuple[Mapping[str, object], float]] = []
    count = len(documents)
    for index, frequency in enumerate(term_frequencies):
        score = 0.0
        for term in query_terms:
            tf = frequency.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * lengths[index] / max(average_length, 1))
            score += idf * tf * (k1 + 1) / denominator
        if score > 0:
            scored.append((documents[index], score))
    return sorted(scored, key=lambda item: (-item[1], str(item[0].get("block_id"))))[:limit]


def reciprocal_rank_fusion(
    channels: Mapping[str, Sequence[EvidenceResult]], *, rank_constant: int = 60, limit: int = 50
) -> tuple[list[EvidenceResult], dict[str, object]]:
    """Fuse ranks only; the returned rank score is not a confidence or probability."""
    if rank_constant < 1 or limit < 1:
        raise ValueError("rank_constant and limit must be positive")
    fused: dict[str, tuple[float, EvidenceResult, set[str]]] = {}
    channel_ranks: dict[str, list[dict[str, object]]] = {}
    for channel, items in channels.items():
        channel_ranks[channel] = []
        seen: set[str] = set()
        for rank, item in enumerate(items, start=1):
            key = item.fact_id or item.block_id or item.evidence_id
            channel_ranks[channel].append({"evidence_id": item.evidence_id, "rank": rank})
            if key in seen:
                continue
            seen.add(key)
            contribution = 1.0 / (rank_constant + rank)
            if key not in fused:
                fused[key] = (contribution, item, {channel})
            else:
                current, prior, sources = fused[key]
                sources.add(channel)
                fused[key] = (current + contribution, prior, sources)
    ranked = sorted(fused.values(), key=lambda row: (-row[0], row[1].evidence_id))[:limit]
    results = [
        EvidenceResult(
            evidence_id=item.evidence_id, channel="rrf", document_id=item.document_id,
            version_id=item.version_id, source_id=item.source_id, content_sha256=item.content_sha256,
            page_number=item.page_number, block_id=item.block_id, fact_id=item.fact_id,
            text=item.text, score=score, source_time_basis=item.source_time_basis,
            citation={**item.citation, "rank_fusion_channels": sorted(channels_used), "rank_score_only": True},
        )
        for score, item, channels_used in ranked
    ]
    return results, {"method": "reciprocal_rank_fusion", "rank_constant": rank_constant, "channel_ranks": channel_ranks, "score_is_probability": False}


class IndustrialRagService:
    """Shared source-bound retrieval service used by CLI, API, and financial specialists."""

    def __init__(
        self,
        store: IndustrialRagStore,
        *,
        channel_adapters: Mapping[str, Callable[[QueryPlan, Sequence[str]], Sequence[EvidenceResult]]] | None = None,
        reranker: Callable[[QueryPlan, Sequence[EvidenceResult]], Sequence[EvidenceResult]] | None = None,
        allow_retrospective_diagnostic: bool = True,
    ) -> None:
        self.store = store
        self.channel_adapters = dict(channel_adapters or {})
        self.reranker = reranker
        self.allow_retrospective_diagnostic = allow_retrospective_diagnostic

    @staticmethod
    def route(task_type: str) -> tuple[str, ...]:
        routes = {
            "fact": ("structured_facts", "bm25"),
            "comparison": ("structured_facts", "bm25", "source_relations"),
            "table": ("structured_facts", "bm25", "toc_page"),
            "narrative": ("bm25", "dense"),
            "refutation": ("structured_facts", "bm25", "source_relations"),
            "unknown": ("structured_facts", "bm25"),
        }
        if task_type not in routes:
            raise ValueError("task_type is unsupported")
        return routes[task_type]

    def search(self, plan: QueryPlan, *, use_cache: bool = True) -> dict[str, object]:
        if plan.time_mode == "retrospective_diagnostic" and not self.allow_retrospective_diagnostic:
            raise PermissionError("retrospective diagnostic queries are disabled by server policy")
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise ReleaseNotReady("no published corpus release is active")
        release_spec, release_hash = self.store.release_spec(release_id)
        as_of = parse_datetime(plan.as_of, "as_of", required=True)
        if as_of > datetime.now(timezone.utc):
            raise ValueError("as_of cannot be in the future")
        permitted = self.store.permitted_versions(
            plan.principal_id, plan.security_code, as_of, time_mode=plan.time_mode,
            report_periods=plan.report_periods, accounting_basis=plan.accounting_basis,
            release_id=release_id,
        )
        version_ids = [str(row["version_id"]) for row in permitted]
        permission_revision = self.store.permission_revision()
        safe_plan = {
            "query_sha256": hashlib.sha256(plan.query.encode("utf-8")).hexdigest(),
            "security_code": plan.security_code, "as_of": as_of.isoformat(),
            "principal_id": plan.principal_id, "permission_revision": permission_revision,
            "task_type": plan.task_type, "accounting_basis": plan.accounting_basis,
            "report_periods": list(plan.report_periods), "release_id": release_id,
            "release_spec_sha256": release_hash, "channels": list(plan.channels),
            "top_k": plan.top_k, "time_mode": plan.time_mode,
        }
        cache_key = stable_hash(safe_plan)
        cached = self.store.cache_get(cache_key) if use_cache else None
        if cached is not None:
            return {**cached, "cache": {"hit": True, "key_sha256": cache_key}, "current_permission_revision": permission_revision}

        channel_results: dict[str, list[EvidenceResult]] = {}
        channel_status: dict[str, dict[str, object]] = {}
        blocks = self.store.read_blocks(version_ids)
        facts = self.store.read_facts(version_ids)
        version_by_id = {str(row["version_id"]): row for row in permitted}

        for channel in plan.channels:
            if channel == "structured_facts":
                channel_results[channel] = self._fact_results(plan, facts, version_by_id)
                channel_status[channel] = {"status": "available", "implementation": "typed_fact_records", "candidate_count": len(channel_results[channel])}
            elif channel == "bm25":
                channel_results[channel] = self._block_results(plan, blocks, version_by_id)
                channel_status[channel] = {"status": "available", "implementation": "local_bm25_reference_chinese_bigram_analyzer", "candidate_count": len(channel_results[channel]), "production_opensearch": "unavailable unless explicitly configured"}
            elif channel == "toc_page":
                channel_results[channel] = self._toc_results(plan, blocks, version_by_id)
                channel_status[channel] = {"status": "available", "implementation": "section_and_page_block_reference", "candidate_count": len(channel_results[channel])}
            elif channel in self.channel_adapters:
                adapted = list(self.channel_adapters[channel](plan, version_ids))
                channel_results[channel] = adapted[: plan.top_k * 5]
                channel_status[channel] = {"status": "available", "implementation": "configured_external_adapter", "candidate_count": len(adapted)}
            else:
                channel_results[channel] = []
                channel_status[channel] = {"status": "unavailable", "reason": self._unavailable_reason(channel)}

        fused, fusion_trace = reciprocal_rank_fusion(channel_results, limit=plan.top_k * 5)
        rerank_status: dict[str, object]
        if self.reranker is not None and fused:
            reranked = list(self.reranker(plan, fused))
            allowed_ids = {item.evidence_id for item in fused}
            if any(item.evidence_id not in allowed_ids for item in reranked):
                raise ValueError("reranker returned evidence outside the retrieved candidate set")
            fused = reranked[: plan.top_k]
            rerank_status = {"status": "available", "implementation": "configured_cross_encoder_adapter", "scores_are_probabilities": False}
        else:
            fused = fused[: plan.top_k]
            rerank_status = {"status": "unavailable", "reason": "no independently versioned cross-encoder reranker configured"}

        status = "available" if fused and all(item["status"] != "unavailable" for item in channel_status.values()) else "partial" if fused else "no_evidence"
        response: dict[str, object] = {
            "schema_version": "adaptive-mas.financial-rag-search.v1",
            "plan_id": plan.plan_id,
            "status": status,
            "query_binding": {"security_code": plan.security_code, "as_of": as_of.isoformat(), "time_mode": plan.time_mode, "release_id": release_id, "release_spec_sha256": release_hash},
            "answer_policy": {"source_time_basis": "citation-specific", "unreviewed_facts_are_numeric_eligible": False, "scores_are_confidence": False, "abstain_when_no_eligible_evidence": True},
            "channels": channel_status,
            "retrieved": [self._serialize(item) for item in fused],
            "fusion": fusion_trace,
            "reranking": rerank_status,
            "permission_revision": permission_revision,
            "cache": {"hit": False, "key_sha256": cache_key},
            "warnings": [
                str(value["reason"]) for value in channel_status.values() if value.get("status") == "unavailable"
            ] + (["retrospective source timing is a diagnostic proxy, not verified historical availability"] if any(row["time_basis"] == "retrospective_proxy" for row in permitted) else []),
        }
        self.store.cache_put(cache_key, response)
        return response

    def read_span(self, plan: QueryPlan, span_id: str) -> dict[str, object]:
        """Expand a cited page span only after rechecking present-day permission and as-of scope."""
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise PermissionError("evidence is not currently available to this query")
        if plan.time_mode == "retrospective_diagnostic" and not self.allow_retrospective_diagnostic:
            raise PermissionError("retrospective diagnostic queries are disabled by server policy")
        as_of = parse_datetime(plan.as_of, "as_of", required=True)
        if as_of > datetime.now(timezone.utc):
            raise ValueError("as_of cannot be in the future")
        versions = self.store.permitted_versions(
            plan.principal_id, plan.security_code, as_of, time_mode=plan.time_mode,
            report_periods=plan.report_periods, accounting_basis=plan.accounting_basis,
            release_id=release_id,
        )
        permitted_ids = {str(row["version_id"]) for row in versions}
        record = self.store._get_span(span_id)
        if record is None or str(record["version_id"]) not in permitted_ids:
            raise PermissionError("evidence is not currently available to this query")
        return {
            "span_id": span_id,
            "text": record["text"],
            "citation": {
                "document_id": record["document_id"], "version_id": record["version_id"],
                "source_id": record["source_id"], "source_url": record["source_url"],
                "published_at": record["published_at"], "first_available_at": record["first_available_at"],
                "captured_at": record["captured_at"], "ingested_at": record["ingested_at"],
                "content_sha256": record["content_sha256"],
                "page_number": record["page_number"], "security_code": record["security_code"],
                "title": record["title"], "as_of": as_of.isoformat(), "release_id": release_id,
            },
            "permission_revision": self.store.permission_revision(),
        }

    def read_table_row(self, plan: QueryPlan, table_id: str, row_start: int, row_end: int) -> dict[str, object]:
        """Expand a table row group from stored cells after rechecking access, release, and time scope."""
        if not isinstance(table_id, str) or not table_id or len(table_id) > 256:
            raise ValueError("table_id must be a nonempty bounded string")
        if (
            isinstance(row_start, bool) or isinstance(row_end, bool)
            or not isinstance(row_start, int) or not isinstance(row_end, int)
            or row_start < 0 or row_end < row_start or row_end - row_start >= 20
        ):
            raise ValueError("table row range must contain one to twenty nonnegative rows")
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise PermissionError("table evidence is not currently available to this query")
        if plan.time_mode == "retrospective_diagnostic" and not self.allow_retrospective_diagnostic:
            raise PermissionError("retrospective diagnostic queries are disabled by server policy")
        as_of = parse_datetime(plan.as_of, "as_of", required=True)
        if as_of > datetime.now(timezone.utc):
            raise ValueError("as_of cannot be in the future")
        versions = self.store.permitted_versions(
            plan.principal_id, plan.security_code, as_of, time_mode=plan.time_mode,
            report_periods=plan.report_periods, accounting_basis=plan.accounting_basis,
            release_id=release_id,
        )
        record = self.store._read_table_row(table_id, row_start, row_end)
        if record is None or str(record["table"]["version_id"]) not in {str(row["version_id"]) for row in versions}:
            raise PermissionError("table evidence is not currently available to this query")
        table = record["table"]
        cells = record["cells"]
        header_rows = int(table["header_rows"])
        headers: dict[int, list[str]] = {}
        data: dict[int, list[dict[str, object]]] = {}
        for cell in cells:
            row_index, column_index = int(cell["row_index"]), int(cell["column_index"])
            raw_text = str(cell["raw_text"])
            if row_index < header_rows:
                headers.setdefault(column_index, []).append(raw_text)
            elif row_start <= row_index <= row_end:
                data.setdefault(row_index, []).append(cell)
        if not data:
            raise FileNotFoundError("requested table row group is absent")
        ordered_headers = {
            column: " / ".join(value for value in values if value.strip())
            for column, values in headers.items()
        }
        output_rows = []
        used_cells = [cell for cell in cells if int(cell["row_index"]) < header_rows]
        for row_index in sorted(data):
            ordered_cells = sorted(data[row_index], key=lambda cell: int(cell["column_index"]))
            used_cells.extend(ordered_cells)
            output_rows.append({
                "row_index": row_index,
                "values": [
                    {
                        "cell_id": json.loads(str(cell["payload_json"])).get("cell_id"),
                        "column_index": int(cell["column_index"]),
                        "column_header": ordered_headers.get(int(cell["column_index"])),
                        "raw_text": str(cell["raw_text"]),
                        "normalized_value": cell["normalized_value"],
                        "value_kind": str(cell["value_kind"]),
                        "unit": cell["unit"] or table["unit_text"],
                        "multiplier": cell["multiplier"],
                        "currency": cell["currency"],
                    }
                    for cell in ordered_cells
                ],
            })
        header_text = " | ".join(
            f"{column}:{label}" for column, label in sorted(ordered_headers.items())
        )
        row_texts = [
            " | ".join(
                f"{ordered_headers.get(int(cell['column_index']), str(cell['column_index']))}: {cell['raw_text']}"
                for cell in sorted(row_cells, key=lambda cell: int(cell["column_index"]))
            )
            for _, row_cells in sorted(data.items())
        ]
        pages = [int(cell["page_number"]) for cell in used_cells if cell["page_number"] is not None]
        citation = {
            "document_id": table["document_id"], "version_id": table["version_id"],
            "source_id": table["source_id"], "source_url": table["source_url"],
            "published_at": table["published_at"], "first_available_at": table["first_available_at"],
            "captured_at": table["captured_at"], "ingested_at": table["ingested_at"],
            "content_sha256": table["content_sha256"], "time_basis": table["time_basis"],
            "page_number": min(pages) if pages else table["page_start"],
            "table_id": table_id, "row_start": row_start, "row_end": row_end,
            "unit": table["unit_text"], "accounting_basis": table["table_accounting_basis"],
            "report_period": table["table_report_period"], "as_of": as_of.isoformat(),
            "release_id": release_id,
            "supporting_cells": [
                {
                    "cell_id": json.loads(str(cell["payload_json"])).get("cell_id"),
                    "row_index": int(cell["row_index"]),
                    "column_index": int(cell["column_index"]),
                    "page_number": cell["page_number"],
                    "raw_text": str(cell["raw_text"]),
                }
                for cell in used_cells
            ],
        }
        text = f"列头：{header_text}\n" + "\n".join(row_texts)
        return {
            "table_id": table_id, "row_start": row_start, "row_end": row_end,
            "title": table["table_title"], "header_rows": header_rows,
            "rows": output_rows, "text": text, "citation": citation,
            "permission_revision": self.store.permission_revision(),
        }

    def download_original(self, plan: QueryPlan, document_id: str, version_id: str) -> bytes:
        """Download is bound to the same current grant, release and temporal policy as search."""
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise PermissionError("document is not currently available to this query")
        if plan.time_mode == "retrospective_diagnostic" and not self.allow_retrospective_diagnostic:
            raise PermissionError("retrospective diagnostic queries are disabled by server policy")
        as_of = parse_datetime(plan.as_of, "as_of", required=True)
        if as_of > datetime.now(timezone.utc):
            raise ValueError("as_of cannot be in the future")
        versions = self.store.permitted_versions(
            plan.principal_id, plan.security_code, as_of, time_mode=plan.time_mode,
            report_periods=plan.report_periods, accounting_basis=plan.accounting_basis,
            release_id=release_id,
        )
        allowed = {str(row["version_id"]): str(row["document_id"]) for row in versions}
        if allowed.get(version_id) != document_id:
            raise PermissionError("document is not currently available to this query")
        return self.store._get_document_bytes(document_id, version_id, plan.principal_id)

    @staticmethod
    def _unavailable_reason(channel: str) -> str:
        reasons = {
            "dense": "no real embedding model and persistent vector backend are configured; lexical matching was not substituted",
            "vision": "no page image encoder or multivector index is configured",
            "source_relations": "no time-filtered source relation index is configured",
        }
        return reasons.get(channel, "requested channel has no configured backend")

    def _fact_results(self, plan: QueryPlan, facts: Sequence[Mapping[str, object]], versions: Mapping[str, Mapping[str, object]]) -> list[EvidenceResult]:
        query_tokens = set(financial_tokens(plan.query + " " + " ".join(plan.aliases)))
        rows: list[tuple[float, EvidenceResult]] = []
        referenced_span_ids = [
            str(span_id)
            for fact in facts
            for span_id in json.loads(str(fact["payload_json"])).get("span_ids", [])
        ]
        span_by_id = {str(row["span_id"]): row for row in self.store.read_spans(referenced_span_ids)}
        for fact in facts:
            if str(fact["security_code"]) != plan.security_code:
                continue
            if plan.report_periods and str(fact["report_period"]) not in plan.report_periods:
                continue
            if plan.accounting_basis and str(fact["consolidation"]) != plan.accounting_basis:
                continue
            terms = set(financial_tokens(str(fact["metric"]))) | set(financial_tokens(str(fact["raw_value"])))
            if not terms.intersection(query_tokens):
                continue
            span_ids = [str(value) for value in json.loads(str(fact["payload_json"])).get("span_ids", [])]
            supporting_spans = [span_by_id[span_id] for span_id in span_ids if span_id in span_by_id]
            spans_are_valid = (
                len(supporting_spans) == len(span_ids)
                and bool(supporting_spans)
                and all(
                    str(span["version_id"]) == str(fact["version_id"])
                    and str(span["source_id"]) == str(fact["source_id"])
                    and span["page_number"] is not None
                    and bool(str(span["text"]).strip())
                    and int(span["char_end"]) > int(span["char_start"])
                    for span in supporting_spans
                )
            )
            if supporting_spans:
                primary_span = supporting_spans[0]
                citation = _citation(primary_span, fact=fact)
                citation["span_id"] = str(primary_span["span_id"])
            else:
                # Legacy rows without resolvable spans stay visible as nonnumeric evidence.
                citation = _citation({
                    "document_id": fact["document_id"], "version_id": fact["version_id"],
                    "source_id": fact["source_id"], "source_url": fact["source_url"],
                    "published_at": fact["published_at"], "first_available_at": fact["first_available_at"],
                    "captured_at": fact["captured_at"], "ingested_at": fact["ingested_at"],
                    "content_sha256": fact["content_sha256"], "time_basis": fact["time_basis"],
                    "title": fact["title"], "security_code": plan.security_code,
                    "page_number": None, "payload_json": "{}",
                }, fact=fact)
            citation["span_ids"] = span_ids
            citation["supporting_spans"] = [
                {
                    "span_id": str(span["span_id"]),
                    "document_id": str(span["document_id"]),
                    "version_id": str(span["version_id"]),
                    "source_id": str(span["source_id"]),
                    "source_url": str(span["source_url"]),
                    "content_sha256": str(span["content_sha256"]),
                    "page_number": int(span["page_number"]) if span["page_number"] is not None else None,
                    "char_start": int(span["char_start"]),
                    "char_end": int(span["char_end"]),
                }
                for span in supporting_spans
            ]
            citation["point_in_time_eligible"] = fact["time_basis"] == "strict_forward"
            citation["numeric_eligible"] = spans_are_valid and fact["review_status"] == "reviewed" and float(fact["confidence"]) >= 0.9
            text = f"{fact['metric']}：{fact['raw_value']} {fact['unit']}；规范值={fact['normalized_value']} {fact['unit']}；报告期={fact['report_period']}；口径={fact['consolidation']}"
            result = EvidenceResult(
                evidence_id=f"ev-fact-{fact['fact_id']}", channel="structured_facts",
                document_id=str(fact["document_id"]), version_id=str(fact["version_id"]),
                source_id=str(fact["source_id"]), content_sha256=str(fact["content_sha256"]),
                page_number=citation.get("page_number"), block_id=None, fact_id=str(fact["fact_id"]),
                text=text, score=float(fact["confidence"]), source_time_basis=str(fact["time_basis"]), citation=citation,
            )
            score = len(terms.intersection(query_tokens)) + float(fact["confidence"])
            rows.append((score, result))
        return [item for _, item in sorted(rows, key=lambda row: (-row[0], row[1].evidence_id))[: plan.top_k * 5]]

    def _block_results(self, plan: QueryPlan, blocks: Sequence[Mapping[str, object]], versions: Mapping[str, Mapping[str, object]]) -> list[EvidenceResult]:
        scoped = [dict(row) | {"text": str(row["text"])} for row in blocks if row["version_id"] in versions]
        ranked = _bm25(plan.query + " " + " ".join(plan.aliases), scoped, limit=plan.top_k * 5)
        return [
            EvidenceResult(
                evidence_id=f"ev-block-{row['block_id']}", channel="bm25",
                document_id=str(row["document_id"]), version_id=str(row["version_id"]),
                source_id=str(row["source_id"]), content_sha256=str(row["content_sha256"]),
                page_number=int(row["page_number"]) if row.get("page_number") is not None else None,
                block_id=str(row["block_id"]), fact_id=None, text=str(row["text"]), score=score,
                source_time_basis=str(row["time_basis"]), citation=_citation(row),
            )
            for row, score in ranked
        ]

    def _toc_results(self, plan: QueryPlan, blocks: Sequence[Mapping[str, object]], versions: Mapping[str, Mapping[str, object]]) -> list[EvidenceResult]:
        query_tokens = set(financial_tokens(plan.query))
        candidates = []
        for row in blocks:
            if row["version_id"] not in versions or row["block_type"] not in {"heading", "table_caption", "page_header"}:
                continue
            overlap = len(query_tokens.intersection(financial_tokens(str(row["text"]))))
            if overlap:
                candidates.append((overlap, row))
        return [
            EvidenceResult(
                evidence_id=f"ev-toc-{row['block_id']}", channel="toc_page",
                document_id=str(row["document_id"]), version_id=str(row["version_id"]),
                source_id=str(row["source_id"]), content_sha256=str(row["content_sha256"]),
                page_number=int(row["page_number"]) if row["page_number"] is not None else None,
                block_id=str(row["block_id"]), fact_id=None, text=str(row["text"]), score=float(overlap),
                source_time_basis=str(row["time_basis"]), citation=_citation(row),
            )
            for overlap, row in sorted(candidates, key=lambda pair: (-pair[0], str(pair[1]["block_id"])))[:plan.top_k * 5]
        ]

    @staticmethod
    def _serialize(result: EvidenceResult) -> dict[str, object]:
        return {
            "evidence_id": result.evidence_id, "channel": result.channel,
            "document_id": result.document_id, "version_id": result.version_id,
            "source_id": result.source_id, "content_sha256": result.content_sha256,
            "page_number": result.page_number, "block_id": result.block_id,
            "fact_id": result.fact_id, "text": result.text, "rank_score": result.score,
            "source_time_basis": result.source_time_basis, "citation": result.citation,
        }
