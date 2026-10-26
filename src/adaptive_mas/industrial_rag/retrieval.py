"""Permission-first retrieval, calibrated score separation, and immutable citations."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import re
import secrets
import uuid
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Callable, Mapping, Sequence

from adaptive_mas.industrial_rag.models import EvidenceResult, QueryPlan, SearchControl, parse_datetime
from adaptive_mas.industrial_rag.planning import (
    accounting_basis_signature,
    metric_for_source_label,
    normalize_period,
    plan_research,
    requested_accounting_basis,
    requested_metric_specs,
    requested_periods,
    requested_unit,
    unit_signature,
)
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
    for key in ("bbox", "char_start", "char_end", "section_id", "table_id", "row_start", "row_end", "continuation", "supporting_cells", "supporting_spans"):
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


def _decimal_value(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def _period_matches_request(
    requested_period: str | None,
    observed_period: str | None,
    *,
    metric: str | None = None,
) -> bool:
    if requested_period is None:
        return True
    if observed_period == requested_period:
        return True
    return bool(
        re.fullmatch(r"(?:19|20)\d{2}", requested_period)
        and metric in {"total_assets", "debt_ratio"}
        and observed_period == requested_period + "-12-31"
    )


def _financial_observations(item: Mapping[str, object]) -> list[dict[str, object]]:
    """Return source-located metric values; page-level text is never a value label."""
    citation = item.get("citation")
    if not isinstance(citation, Mapping):
        citation = {}
    fact_metric = metric_for_source_label(citation.get("metric"))
    if fact_metric is not None:
        value = _decimal_value(citation.get("normalized_value"))
        if value is None:
            return []
        period = citation.get("report_period")
        unit = citation.get("unit")
        multiplier = citation.get("multiplier", "1")
        basis = citation.get("accounting_basis")
        return [{
            "metric": fact_metric,
            "period": normalize_period(period),
            "unit": unit,
            "unit_signature": unit_signature(unit, multiplier),
            "multiplier": str(multiplier),
            "basis": basis,
            "basis_signature": accounting_basis_signature(basis),
            "currency": citation.get("currency"),
            "value": value,
            "numeric_eligible": citation.get("numeric_eligible") is True,
        }]

    expanded = item.get("reopened_table_row")
    if not isinstance(expanded, Mapping) or not isinstance(expanded.get("rows"), list):
        return []
    reopened_citation = expanded.get("citation")
    if isinstance(reopened_citation, Mapping):
        citation = reopened_citation
    supporting_cells = citation.get("supporting_cells", [])
    source_cells = {
        (cell.get("row_index"), cell.get("column_index")): cell
        for cell in supporting_cells
        if isinstance(cell, Mapping)
    } if isinstance(supporting_cells, list) else {}
    observations: list[dict[str, object]] = []
    for row in expanded["rows"]:
        if not isinstance(row, Mapping) or not isinstance(row.get("values"), list):
            continue
        cells = [cell for cell in row["values"] if isinstance(cell, Mapping)]
        row_label = next((
            str(cell.get("raw_text", "")).strip()
            for cell in cells if cell.get("column_index") == 0
        ), "")
        metric = metric_for_source_label(row_label)
        if metric is None:
            continue
        for cell in cells:
            if cell.get("column_index") == 0 or cell.get("value_kind") not in {"number", "percent", "basis_points", "zero"}:
                continue
            value = _decimal_value(cell.get("normalized_value"))
            if value is None:
                continue
            row_index, column_index = row.get("row_index"), cell.get("column_index")
            source_cell = source_cells.get((row_index, column_index), {})
            unit = cell.get("unit") or source_cell.get("unit") or citation.get("unit")
            multiplier = cell.get("multiplier") or source_cell.get("multiplier") or "1"
            basis = source_cell.get("accounting_basis") or citation.get("accounting_basis")
            column_header = str(cell.get("column_header") or "")
            period = (
                normalize_period(column_header) if re.search(r"(?:19|20)\d{2}", column_header) else None
            ) or (
                source_cell.get("report_period")
                or citation.get("report_period")
            )
            period_key = normalize_period(period)
            opening_balance_year = re.fullmatch(r"((?:19|20)\d{2})(?:-12-31)?", period_key or "")
            if "期初" in column_header and opening_balance_year:
                period_key = f"{int(opening_balance_year.group(1)) - 1}-12-31"
            observations.append({
                "metric": metric,
                "period": period_key,
                "unit": unit,
                "unit_signature": unit_signature(unit, multiplier),
                "multiplier": str(multiplier),
                "basis": basis,
                "basis_signature": accounting_basis_signature(basis),
                "currency": cell.get("currency") or source_cell.get("currency"),
                "value": value,
                "numeric_eligible": False,
                "source_metric_label": row_label,
                "raw_value": cell.get("raw_text"),
                "source_cell_id": cell.get("cell_id"),
                "source_row_index": row_index,
                "source_column_index": column_index,
                "source_page_number": source_cell.get("page_number"),
                "column_header": cell.get("column_header"),
            })
    return observations


def _component_cache_identity(component: object) -> dict[str, object] | None:
    identity = getattr(component, "cache_identity", None)
    if not isinstance(identity, Mapping):
        return None
    serialized = dict(identity)
    implementation = serialized.get("implementation")
    version = serialized.get("version") or serialized.get("revision")
    configuration = serialized.get("configuration")
    if (
        not isinstance(implementation, str) or not implementation.strip()
        or not isinstance(version, str) or not version.strip()
        or not isinstance(configuration, (Mapping, str))
        or (isinstance(configuration, Mapping) and not configuration)
        or (isinstance(configuration, str) and not configuration.strip())
    ):
        return None
    json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return serialized


def _encode_search_cursor(value: Mapping[str, object], signing_key: bytes) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    signature = hmac.new(signing_key, encoded.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _decode_search_cursor(value: object, signing_key: bytes) -> dict[str, object]:
    if not isinstance(value, str) or len(value) > 4161:
        raise ValueError("search cursor is malformed")
    match = re.fullmatch(r"([A-Za-z0-9_-]{1,4096})\.([0-9a-f]{64})", value)
    if match is None:
        raise ValueError("search cursor is malformed")
    encoded, signature = match.groups()
    expected_signature = hmac.new(signing_key, encoded.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("search cursor signature is invalid")
    try:
        padding = "=" * ((4 - len(encoded) % 4) % 4)
        decoded = json.loads(base64.b64decode(encoded + padding, altchars=b"-_", validate=True).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("search cursor is malformed") from None
    if not isinstance(decoded, dict) or set(decoded) != {
        "schema", "release_id", "cache_key", "permission_revision", "continuation_revision", "offset", "page_size",
    }:
        raise ValueError("search cursor is malformed")
    if (
        decoded.get("schema") != "adaptive-mas.financial-rag-search-cursor.v2"
        or not isinstance(decoded.get("release_id"), str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", decoded["release_id"])
        or not isinstance(decoded.get("cache_key"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", decoded["cache_key"])
        or isinstance(decoded.get("permission_revision"), bool)
        or not isinstance(decoded.get("permission_revision"), int)
        or decoded["permission_revision"] < 0
        or isinstance(decoded.get("continuation_revision"), bool)
        or not isinstance(decoded.get("continuation_revision"), int)
        or decoded["continuation_revision"] < 0
        or isinstance(decoded.get("offset"), bool)
        or not isinstance(decoded.get("offset"), int)
        or decoded["offset"] < 1
        or isinstance(decoded.get("page_size"), bool)
        or not isinstance(decoded.get("page_size"), int)
        or not 1 <= decoded["page_size"] <= 100
    ):
        raise ValueError("search cursor is malformed")
    return decoded


def _bm25(
    query: str,
    documents: Sequence[Mapping[str, object]],
    *,
    limit: int,
    k1: float = 1.2,
    b: float = 0.75,
    control: SearchControl | None = None,
) -> list[tuple[Mapping[str, object], float]]:
    query_terms = list(dict.fromkeys(financial_tokens(query)))
    if not query_terms or not documents:
        return []
    tokenized = []
    for document in documents:
        if control is not None:
            control.check()
        tokenized.append(financial_tokens(str(document.get("text", ""))))
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
        if control is not None:
            control.check()
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


def _matches_exact_table_row_query(plan: QueryPlan, block: Mapping[str, object]) -> bool:
    """Recognize a table row whose source label and column header match the query."""
    if block.get("block_type") != "table_row_group":
        return False
    metrics = {
        item.metric for item in requested_metric_specs(plan.query)
        if item.match_status == "exact"
    }
    if not metrics:
        return False
    text = str(block.get("text", ""))
    header, separator, data = text.partition("\n")
    if not separator or "数据行：" not in data:
        return False
    row_label = data.split("数据行：", 1)[1].split("|", 1)[0].strip()
    if metric_for_source_label(row_label) not in metrics:
        return False
    periods = requested_periods(plan.query, plan.report_periods)
    if not periods:
        return True
    source_periods = {
        normalize_period(part.strip())
        for part in re.split(r"[|\n/]+", header.partition("列头：")[2])
    }
    source_periods.discard(None)
    requested = {normalize_period(period) for period in periods}
    requested.discard(None)
    return bool(source_periods.intersection(requested))


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
        self._cursor_signing_key = secrets.token_bytes(32)

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

    def search(
        self,
        plan: QueryPlan,
        *,
        use_cache: bool = True,
        rrf_rank_constant: int = 60,
        use_reranker: bool = True,
        fusion_strategy: str = "rrf",
        row_group_chunking: bool = True,
        control: SearchControl | None = None,
    ) -> dict[str, object]:
        if control is not None:
            control.check()
        if isinstance(rrf_rank_constant, bool) or not isinstance(rrf_rank_constant, int) or not 1 <= rrf_rank_constant <= 200:
            raise ValueError("rrf_rank_constant must be an integer within 1..200")
        if not isinstance(use_reranker, bool):
            raise ValueError("use_reranker must be a boolean")
        if fusion_strategy not in {"rrf", "first_available_channel"}:
            raise ValueError("fusion_strategy is unsupported")
        if not isinstance(row_group_chunking, bool):
            raise ValueError("row_group_chunking must be a boolean")
        if plan.time_mode == "retrospective_diagnostic" and not self.allow_retrospective_diagnostic:
            raise PermissionError("retrospective diagnostic queries are disabled by server policy")
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise ReleaseNotReady("no published corpus release is active")
        release_spec, release_hash = self.store.release_spec(release_id)
        as_of = parse_datetime(plan.as_of, "as_of", required=True)
        if as_of > datetime.now(timezone.utc):
            raise ValueError("as_of cannot be in the future")
        permission_revision_before = self.store.permission_revision()
        continuation_revision_before = self.store.continuation_revision()
        permitted = self.store.permitted_versions(
            plan.principal_id, plan.security_code, as_of, time_mode=plan.time_mode,
            report_periods=plan.report_periods, accounting_basis=plan.accounting_basis,
            release_id=release_id,
        )
        version_ids = [str(row["version_id"]) for row in permitted]
        permission_revision = self.store.permission_revision()
        continuation_revision = self.store.continuation_revision()
        if permission_revision != permission_revision_before or continuation_revision != continuation_revision_before:
            raise PermissionError("query state changed while resolving the query scope")
        builtin_channel_identities = {
            "structured_facts": "typed_fact_records_v1",
            "bm25": "local_bm25_chinese_bigrams_v1",
            "toc_page": "section_page_reference_v1",
        }
        channel_implementations: dict[str, object] = {}
        runtime_cacheable = True
        for channel in plan.channels:
            if channel in self.channel_adapters:
                identity = _component_cache_identity(self.channel_adapters[channel])
                channel_implementations[channel] = identity
                runtime_cacheable = runtime_cacheable and identity is not None
            else:
                channel_implementations[channel] = builtin_channel_identities.get(channel, "unavailable")
        reranker_enabled = self.reranker is not None and use_reranker
        if not reranker_enabled:
            reranker_identity: object = "disabled"
        else:
            reranker_identity = _component_cache_identity(self.reranker)
            runtime_cacheable = runtime_cacheable and reranker_identity is not None
        safe_plan = {
            "cache_schema_version": "adaptive-mas.financial-rag-search-cache.v5",
            "query_sha256": hashlib.sha256(plan.query.encode("utf-8")).hexdigest(),
            "aliases": list(plan.aliases),
            "security_code": plan.security_code, "as_of": as_of.isoformat(),
            "principal_id": plan.principal_id, "permission_revision": permission_revision,
            "continuation_revision": continuation_revision,
            "task_type": plan.task_type, "accounting_basis": plan.accounting_basis,
            "report_periods": list(plan.report_periods), "release_id": release_id,
            "release_spec_sha256": release_hash, "channels": list(plan.channels),
            "top_k": plan.top_k, "time_mode": plan.time_mode,
            "channel_implementations": channel_implementations,
            "reranker_implementation": reranker_identity,
            "rrf_rank_constant": rrf_rank_constant,
            "reranker_enabled": reranker_enabled,
            "fusion_strategy": fusion_strategy,
            "row_group_chunking": row_group_chunking,
        }
        cache_key = stable_hash(safe_plan)
        cached = self.store.cache_get(cache_key) if use_cache and runtime_cacheable else None
        if cached is not None:
            if self.store.permission_revision() != permission_revision or self.store.continuation_revision() != continuation_revision:
                raise PermissionError("query state changed while resolving the cached result")
            if control is not None:
                control.check()
            return {
                **cached,
                "plan_id": plan.plan_id,
                "cache": {"hit": True, "stored": True, "key_sha256": cache_key},
                "current_permission_revision": permission_revision,
                "current_continuation_revision": continuation_revision,
            }

        channel_results: dict[str, list[EvidenceResult]] = {}
        channel_status: dict[str, dict[str, object]] = {}
        if self.store.permission_revision() != permission_revision or self.store.continuation_revision() != continuation_revision:
            raise PermissionError("query state changed while retrieval was running")
        if control is not None:
            control.check()
        blocks = self.store.read_blocks(version_ids)
        if not row_group_chunking:
            blocks = [row for row in blocks if row.get("block_type") != "table_row_group"]
        if control is not None:
            control.check()
        facts = self.store.read_facts(version_ids)
        if control is not None:
            control.check()
        version_by_id = {str(row["version_id"]): row for row in permitted}

        for channel in plan.channels:
            if control is not None:
                control.check()
            if channel == "structured_facts":
                channel_results[channel] = self._fact_results(plan, facts, version_by_id, control=control)
                channel_status[channel] = {"status": "available", "implementation": "typed_fact_records", "candidate_count": len(channel_results[channel])}
            elif channel == "bm25":
                channel_results[channel] = self._block_results(plan, blocks, version_by_id, control=control)
                channel_status[channel] = {"status": "available", "implementation": "local_bm25_reference_chinese_bigram_analyzer", "candidate_count": len(channel_results[channel]), "production_opensearch": "unavailable unless explicitly configured"}
            elif channel == "toc_page":
                channel_results[channel] = self._toc_results(plan, blocks, version_by_id)
                channel_status[channel] = {"status": "available", "implementation": "section_and_page_block_reference", "candidate_count": len(channel_results[channel])}
            elif channel in self.channel_adapters:
                adapter = self.channel_adapters[channel]
                controlled_search = getattr(adapter, "search_with_control", None)
                adapted = list(
                    controlled_search(plan, version_ids, control)
                    if control is not None and callable(controlled_search)
                    else adapter(plan, version_ids)
                )
                if control is not None:
                    control.check()
                channel_results[channel] = adapted[: plan.top_k * 5]
                channel_status[channel] = {
                    "status": "available",
                    "implementation": getattr(adapter, "implementation", "configured_external_adapter"),
                    "candidate_count": len(adapted),
                }
            else:
                channel_results[channel] = []
                channel_status[channel] = {"status": "unavailable", "reason": self._unavailable_reason(channel)}

        if fusion_strategy == "rrf":
            fused, fusion_trace = reciprocal_rank_fusion(
                channel_results,
                rank_constant=rrf_rank_constant,
                limit=plan.top_k * 5,
            )
        else:
            selected_channel = next(
                (channel for channel in plan.channels if channel_results.get(channel)),
                None,
            )
            fused = (
                channel_results[selected_channel][: plan.top_k * 5]
                if selected_channel is not None else []
            )
            fusion_trace = {
                "method": "fusion_disabled_first_available_channel",
                "selected_channel": selected_channel,
                "channels_with_results": [
                    channel for channel in plan.channels if channel_results.get(channel)
                ],
                "score_is_probability": False,
            }
        rerank_status: dict[str, object]
        if reranker_enabled and fused:
            if control is not None:
                control.check()
            controlled_rerank = getattr(self.reranker, "rerank_with_control", None)
            reranked = list(
                controlled_rerank(plan, fused, control)
                if control is not None and callable(controlled_rerank)
                else self.reranker(plan, fused)
            )
            if control is not None:
                control.check()
            allowed_ids = {item.evidence_id for item in fused}
            if any(item.evidence_id not in allowed_ids for item in reranked):
                raise ValueError("reranker returned evidence outside the retrieved candidate set")
            fused = reranked[: plan.top_k]
            rerank_status = {
                "status": "available",
                "implementation": getattr(self.reranker, "implementation", "configured_cross_encoder_adapter"),
                "scores_are_probabilities": False,
            }
        else:
            fused = fused[: plan.top_k]
            rerank_status = (
                {"status": "disabled", "reason": "disabled by the active research policy"}
                if self.reranker is not None and not use_reranker
                else {"status": "unavailable", "reason": "no independently versioned cross-encoder reranker configured"}
            )

        status = "available" if fused and all(item["status"] != "unavailable" for item in channel_status.values()) else "partial" if fused else "no_evidence"
        cache_result: dict[str, object] = {"hit": False, "stored": runtime_cacheable}
        if runtime_cacheable:
            cache_result["key_sha256"] = cache_key
        else:
            cache_result["reason"] = "configured_runtime_identity_incomplete"
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
            "continuation_revision": continuation_revision,
            "cache": cache_result,
            "warnings": [
                str(value["reason"]) for value in channel_status.values() if value.get("status") == "unavailable"
            ] + (["retrospective source timing is a diagnostic proxy, not verified historical availability"] if any(row["time_basis"] == "retrospective_proxy" for row in permitted) else []),
        }
        if control is not None:
            control.check()
        if runtime_cacheable:
            self.store.cache_put(cache_key, response)
        if self.store.permission_revision() != permission_revision or self.store.continuation_revision() != continuation_revision:
            raise PermissionError("query state changed before retrieval could return")
        return response

    def search_page(
        self,
        plan: QueryPlan,
        *,
        page_size: int | None = None,
        cursor: str | None = None,
        control: SearchControl | None = None,
    ) -> dict[str, object]:
        """Page through one bounded, cached result set tied to a fixed local release snapshot."""
        if page_size is None:
            page_size = min(20, plan.top_k)
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        if page_size > plan.top_k:
            raise ValueError("page_size cannot exceed the QueryPlan top_k result limit")
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise ReleaseNotReady("no published corpus release is active")
        permission_revision = self.store.permission_revision()
        continuation_revision = self.store.continuation_revision()
        offset = 0
        token: dict[str, object] | None = None
        if cursor is not None:
            token = _decode_search_cursor(cursor, self._cursor_signing_key)
            if token["release_id"] != release_id:
                raise ValueError("search cursor was issued for a different active release")
            if token["permission_revision"] != permission_revision:
                raise ValueError("search cursor was invalidated by a permission change")
            if token["continuation_revision"] != continuation_revision:
                raise ValueError("search cursor was invalidated by a continuation review change")
            if token["page_size"] != page_size:
                raise ValueError("page_size must match the first search page")
            offset = int(token["offset"])

        if control is not None:
            control.check()
        result = self.search(replace(plan, release_id=release_id), control=control)
        if plan.release_id is None and self.store.active_release_id() != release_id:
            raise ValueError("active corpus release changed while the search page was running")
        if (self.store.permission_revision() != result.get("permission_revision")
                or self.store.continuation_revision() != result.get("continuation_revision")):
            raise PermissionError("query state changed while the search page was running")
        result_cache = result.get("cache", {})
        if not isinstance(result_cache, Mapping):
            raise ValueError("search result does not expose a cache snapshot")
        if token is not None and (
            result_cache.get("stored") is not True
            or result_cache.get("key_sha256") != token["cache_key"]
            or result.get("permission_revision") != token["permission_revision"]
            or result.get("continuation_revision") != token["continuation_revision"]
        ):
            raise ValueError("search cursor no longer matches the query snapshot")
        items = result.get("retrieved")
        if not isinstance(items, list):
            raise ValueError("search result does not contain a result list")
        if token is not None and offset >= len(items):
            raise ValueError("search cursor is outside the stored result set")
        end = min(offset + page_size, len(items))
        has_more = end < len(items)
        if has_more and result_cache.get("stored") is not True:
            raise ValueError("stable cursor pagination requires a complete runtime cache identity")
        if control is not None:
            control.check()
        next_cursor = None
        if has_more:
            next_cursor = _encode_search_cursor({
                "schema": "adaptive-mas.financial-rag-search-cursor.v2",
                "release_id": release_id,
                "cache_key": result_cache["key_sha256"],
                "permission_revision": result["permission_revision"],
                "continuation_revision": result["continuation_revision"],
                "offset": end,
                "page_size": page_size,
            }, self._cursor_signing_key)
        return {
            "schema_version": "adaptive-mas.financial-rag-search-page.v1",
            "plan_id": plan.plan_id,
            "query_binding": result.get("query_binding"),
            "status": result.get("status"),
            "retrieved": items[offset:end],
            "channels": result.get("channels", {}),
            "cache": result_cache,
            "pagination": {
                "offset": offset,
                "page_size": page_size,
                "total_items": len(items),
                "result_limit": plan.top_k,
                "result_limit_reached": len(items) == plan.top_k,
                "next_cursor": next_cursor,
            },
            "warnings": result.get("warnings", []),
        }

    def research(
        self,
        plan: QueryPlan,
        *,
        max_searches: int = 8,
        evidence_limit: int = 24,
        table_expansion_limit: int = 12,
        rrf_rank_constant: int = 60,
        use_reranker: bool = True,
        query_rewrite_enabled: bool = True,
        multi_hop_enabled: bool = True,
        fusion_strategy: str = "rrf",
        row_group_chunking: bool = True,
        include_continuation_context: bool = True,
        control: SearchControl | None = None,
    ) -> dict[str, object]:
        """Run a fixed-template financial search plan with explicit coverage and stop reasons."""
        if control is not None:
            control.check()
        if isinstance(evidence_limit, bool) or not 2 <= evidence_limit <= 64:
            raise ValueError("evidence_limit must be between 2 and 64")
        if isinstance(table_expansion_limit, bool) or not 0 <= table_expansion_limit <= 24:
            raise ValueError("table_expansion_limit must be between 0 and 24")
        if isinstance(rrf_rank_constant, bool) or not isinstance(rrf_rank_constant, int) or not 1 <= rrf_rank_constant <= 200:
            raise ValueError("rrf_rank_constant must be an integer within 1..200")
        if not isinstance(use_reranker, bool):
            raise ValueError("use_reranker must be a boolean")
        if not isinstance(query_rewrite_enabled, bool) or not isinstance(multi_hop_enabled, bool):
            raise ValueError("query rewrite and multi-hop switches must be booleans")
        if fusion_strategy not in {"rrf", "first_available_channel"}:
            raise ValueError("fusion_strategy is unsupported")
        if not isinstance(row_group_chunking, bool) or not isinstance(include_continuation_context, bool):
            raise ValueError("research retrieval switches must be booleans")
        if plan.time_mode == "retrospective_diagnostic" and not self.allow_retrospective_diagnostic:
            raise PermissionError("retrospective diagnostic queries are disabled by server policy")
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise ReleaseNotReady("no published corpus release is active")
        # Freeze the release at task start; all subqueries and reopen actions use it.
        pinned_plan = replace(plan, release_id=release_id)
        searches = plan_research(
            pinned_plan,
            max_searches=max_searches,
            query_rewrite_enabled=query_rewrite_enabled,
            multi_hop_enabled=multi_hop_enabled,
        )
        metric_specs = requested_metric_specs(plan.query)
        metrics = tuple(dict.fromkeys(item.metric for item in metric_specs))
        periods = requested_periods(plan.query, plan.report_periods)
        requested_unit_label = requested_unit(plan.query)
        requested_unit_key = unit_signature(requested_unit_label) if requested_unit_label else None
        requested_basis_label = requested_accounting_basis(plan.query, plan.accounting_basis)
        requested_basis_key = accounting_basis_signature(requested_basis_label)
        support: list[dict[str, object]] = []
        counter_candidates: list[dict[str, object]] = []
        step_trace: list[dict[str, object]] = []
        warnings: list[str] = []
        channel_status: dict[str, dict[str, object]] = {}
        seen: dict[tuple[object, ...], dict[str, object]] = {}
        table_expansions = 0
        task_permission_revision = self.store.permission_revision()

        def location_key(item: Mapping[str, object]) -> tuple[object, ...]:
            citation = item.get("citation", {})
            if not isinstance(citation, Mapping):
                citation = {}
            return (
                citation.get("source_id") or item.get("source_id"),
                citation.get("content_sha256") or item.get("content_sha256"),
                citation.get("fact_id") or item.get("fact_id"),
                citation.get("span_id") or item.get("span_id"),
                citation.get("table_id"),
                citation.get("row_start"),
                citation.get("row_end"),
                citation.get("page_number") or item.get("page_number"),
                citation.get("block_id") or item.get("block_id"),
            )

        for index, step in enumerate(searches, start=1):
            if control is not None:
                control.check()
            if self.store.permission_revision() != task_permission_revision:
                raise PermissionError("permissions changed during financial research")
            counter = step.kind == "counter_evidence_candidate"
            channels = ("bm25",) if counter else pinned_plan.channels
            subplan = replace(
                pinned_plan,
                plan_id=f"{plan.plan_id}:s{index}",
                query=step.query,
                task_type="refutation" if counter else plan.task_type,
                report_periods=(
                    (step.report_period,)
                    if step.report_period is not None and plan.report_periods
                    else pinned_plan.report_periods
                ),
                channels=channels,
                top_k=min(pinned_plan.top_k, 10),
            )
            result = self.search(
                subplan,
                rrf_rank_constant=rrf_rank_constant,
                use_reranker=use_reranker,
                fusion_strategy=fusion_strategy,
                row_group_chunking=row_group_chunking,
                control=control,
            )
            if self.store.permission_revision() != task_permission_revision:
                raise PermissionError("permissions changed during financial research")
            for name, status in result.get("channels", {}).items():
                channel_status[f"{step.step_id}:{name}"] = dict(status)
            warnings.extend(str(value) for value in result.get("warnings", []))
            step_trace.append({
                "step_id": step.step_id,
                "kind": step.kind,
                "metric": step.metric,
                "report_period": step.report_period,
                "query_sha256": hashlib.sha256(step.query.encode("utf-8")).hexdigest(),
                "result_status": result.get("status"),
                "candidate_count": len(result.get("retrieved", [])),
                "channels": result.get("channels", {}),
            })
            for raw_item in result.get("retrieved", []):
                item = dict(raw_item)
                item["research_role"] = "counter_evidence_candidate" if counter else "supporting_evidence"
                key = location_key(item)
                if key in seen:
                    if counter:
                        candidate = dict(seen[key])
                        candidate["research_role"] = "counter_evidence_candidate"
                        candidate["duplicate_of_supporting_evidence"] = True
                        candidate_citation = dict(candidate.get("citation", {}))
                        candidate_citation["source_numeric_eligible"] = candidate_citation.get("numeric_eligible") is True
                        candidate_citation["numeric_eligible"] = False
                        candidate["citation"] = candidate_citation
                        candidate["query_numeric_eligible"] = False
                        counter_candidates.append(candidate)
                    continue
                seen[key] = item
                if not counter:
                    citation = item.get("citation", {})
                    if isinstance(citation, Mapping) and citation.get("table_id") is not None:
                        table_id = citation.get("table_id")
                        row_start = citation.get("row_start")
                        row_end = citation.get("row_end")
                        if (
                            table_expansions < table_expansion_limit
                            and isinstance(table_id, str)
                            and isinstance(row_start, int) and not isinstance(row_start, bool)
                            and isinstance(row_end, int) and not isinstance(row_end, bool)
                        ):
                            table_expansions += 1
                            try:
                                if control is not None:
                                    control.check()
                                item["reopened_table_row"] = self.read_table_row(
                                    subplan, table_id, row_start, row_end,
                                    include_continuation_context=include_continuation_context,
                                )
                                if control is not None:
                                    control.check()
                            except (PermissionError, FileNotFoundError, ValueError) as error:
                                item["table_reopen_status"] = str(error)
                    support.append(item)
                else:
                    item["counter_review_status"] = "candidate_requires_source_and_claim_review"
                    citation = dict(item.get("citation", {}))
                    citation["source_numeric_eligible"] = citation.get("numeric_eligible") is True
                    citation["numeric_eligible"] = False
                    item["citation"] = citation
                    item["query_numeric_eligible"] = False
                    counter_candidates.append(item)

        metric_dimensions = [
            (metric, period)
            for metric in (metric_specs or (None,))
            for period in (periods or (None,))
        ]
        observations_by_id: dict[str, list[dict[str, object]]] = {}
        raw_support = support
        support = []
        metric_candidates: list[dict[str, object]] = []
        for item in raw_support:
            evidence_id = str(item.get("evidence_id"))
            observations = _financial_observations(item)
            observations_by_id[evidence_id] = observations
            matched_dimensions: list[str] = []
            matched_observations: list[dict[str, object]] = []
            for metric, period in metric_dimensions:
                if metric is None or metric.match_status != "exact":
                    continue
                requested_period_key = normalize_period(period) if period is not None else None
                for observation in observations:
                    if observation["metric"] != metric.metric or observation["period"] is None:
                        continue
                    if not _period_matches_request(
                        requested_period_key,
                        observation["period"],
                        metric=str(observation["metric"]),
                    ):
                        continue
                    if observation["unit_signature"] is None or observation["basis_signature"] is None:
                        continue
                    if requested_unit_key is not None and observation["unit_signature"] != requested_unit_key:
                        continue
                    if requested_basis_key is not None and observation["basis_signature"] != requested_basis_key:
                        continue
                    dimension_key = f"{metric.metric}|{requested_period_key or '*'}"
                    if dimension_key not in matched_dimensions:
                        matched_dimensions.append(dimension_key)
                    if observation not in matched_observations:
                        matched_observations.append(observation)
            item["query_match_dimensions"] = matched_dimensions
            citation = item.get("citation")
            if not isinstance(citation, Mapping):
                citation = {}
            item["structured_observations"] = [
                {
                    "metric": observation["metric"],
                    "source_metric_label": observation.get("source_metric_label") or citation.get("metric"),
                    "report_period": observation["period"],
                    "unit": observation["unit"],
                    "multiplier": observation["multiplier"],
                    "accounting_basis": observation["basis"],
                    "currency": observation["currency"],
                    "raw_value": observation.get("raw_value", citation.get("raw_value")),
                    "normalized_value": str(observation["value"]),
                    "numeric_claim_eligible": observation["numeric_eligible"],
                    "source_location": {
                        "source_id": item.get("source_id") or citation.get("source_id"),
                        "content_sha256": item.get("content_sha256") or citation.get("content_sha256"),
                        "page_number": observation.get("source_page_number") or item.get("page_number") or citation.get("page_number"),
                        "fact_id": citation.get("fact_id"),
                        "table_id": citation.get("table_id"),
                        "row_start": citation.get("row_start"),
                        "row_end": citation.get("row_end"),
                        "cell_id": observation.get("source_cell_id"),
                    },
                }
                for observation in matched_observations
            ]
            item["query_numeric_eligible"] = any(
                observation["numeric_eligible"] for observation in matched_observations
            )
            item["query_metric_match_status"] = "exact" if matched_dimensions else "candidate_only"
            citation = item.get("citation")
            if isinstance(citation, Mapping):
                citation = dict(citation)
                if not item["query_numeric_eligible"]:
                    citation["source_numeric_eligible"] = citation.get("numeric_eligible") is True
                    citation["numeric_eligible"] = False
                item["citation"] = citation
            if matched_dimensions:
                support.append(item)
            else:
                item["research_role"] = "metric_scope_candidate"
                item["review_status"] = "candidate_requires_exact_metric_period_unit_and_basis_review"
                metric_candidates.append(item)

        supporting_limit = evidence_limit - max(1, min(6, evidence_limit // 4))
        support = support[:supporting_limit]
        counter_reserve = max(1, min(6, evidence_limit // 4))
        counter_candidates = counter_candidates[:counter_reserve]
        metric_candidates = metric_candidates[: max(0, evidence_limit - len(support) - len(counter_candidates))]
        coverage_rows = []
        returned_support_ids = {str(item.get("evidence_id")) for item in support}
        returned_candidate_ids = {str(item.get("evidence_id")) for item in metric_candidates}
        for metric in metric_specs or (None,):
            for period in periods or (None,):
                normalized_period = normalize_period(period) if period is not None else None
                dimension_key = (
                    f"{metric.metric}|{normalized_period or '*'}"
                    if metric is not None else f"unspecified|{normalized_period or '*'}"
                )
                matching = []
                for item in support:
                    if dimension_key in item.get("query_match_dimensions", []):
                        matching.append(item)
                matching_ids = {str(item.get("evidence_id")) for item in matching}
                candidate_ids = sorted(
                    (returned_support_ids | returned_candidate_ids)
                    - matching_ids
                )
                candidate_observations = [
                    observation
                    for evidence_id in candidate_ids
                    for observation in observations_by_id.get(evidence_id, [])
                ]
                structured_observations = [
                    observation
                    for item in matching
                    for observation in item.get("structured_observations", [])
                    if isinstance(observation, Mapping)
                    and (metric is None or observation.get("metric") == metric.metric)
                    and _period_matches_request(
                        normalized_period,
                        str(observation.get("report_period")) if observation.get("report_period") is not None else None,
                        metric=metric.metric if metric is not None else None,
                    )
                ]
                coverage_rows.append({
                    "metric": metric.metric if metric is not None else None,
                    "requested_metric": metric.label if metric is not None else None,
                    "metric_match_status": metric.match_status if metric is not None else "unspecified",
                    "report_period": period,
                    "normalized_report_period": normalized_period,
                    "requested_unit": requested_unit_label,
                    "observed_units": sorted({
                        str(observation["unit"]) for observation in structured_observations
                        if observation.get("unit")
                    }),
                    "requested_accounting_basis": requested_basis_label,
                    "observed_accounting_bases": sorted({
                        str(observation["accounting_basis"]) for observation in structured_observations
                        if observation.get("accounting_basis")
                    }),
                    "observed_report_periods": sorted({
                        str(observation["report_period"]) for observation in structured_observations
                        if observation.get("report_period")
                    }),
                    "observed_metrics": sorted({
                        str(observation["metric"]) for observation in structured_observations
                    }),
                    "numeric_claim_ready": any(
                        observation.get("numeric_claim_eligible") is True
                        for observation in structured_observations
                    ),
                    "claim_status": (
                        "eligible" if any(
                            observation.get("numeric_claim_eligible") is True
                            for observation in structured_observations
                        ) else "review_required" if matching else "missing"
                    ),
                    "observations": structured_observations,
                    "candidate_metrics": sorted({
                        str(observation["metric"]) for observation in candidate_observations
                    }),
                    "status": (
                        "located" if matching else
                        "candidate_only" if candidate_ids else
                        "missing"
                    ),
                    "evidence_ids": [str(item.get("evidence_id")) for item in matching],
                    "candidate_evidence_ids": candidate_ids,
                    "source_locations": [
                        {
                            "evidence_id": item.get("evidence_id"),
                            "source_id": item.get("source_id"),
                            "content_sha256": item.get("content_sha256"),
                            "page_number": item.get("page_number"),
                            "supporting_pages": sorted({
                                int(value)
                                for value in [
                                    *[
                                        span.get("page_number")
                                        for span in ((item.get("citation") or {}).get("supporting_spans", [])
                                                     if isinstance(item.get("citation"), Mapping) else [])
                                        if isinstance(span, Mapping) and span.get("page_number") is not None
                                    ],
                                    *[
                                        cell.get("page_number")
                                        for cell in ((item.get("citation") or {}).get("supporting_cells", [])
                                                     if isinstance(item.get("citation"), Mapping) else [])
                                        if isinstance(cell, Mapping) and cell.get("page_number") is not None
                                    ],
                                    item.get("page_number"),
                                ]
                                if value is not None
                            }),
                            "fact_id": (item.get("citation") or {}).get("fact_id") if isinstance(item.get("citation"), Mapping) else None,
                            "table_id": (item.get("citation") or {}).get("table_id") if isinstance(item.get("citation"), Mapping) else None,
                            "row_start": (item.get("citation") or {}).get("row_start") if isinstance(item.get("citation"), Mapping) else None,
                            "row_end": (item.get("citation") or {}).get("row_end") if isinstance(item.get("citation"), Mapping) else None,
                        }
                        for item in matching
                    ],
                })

        def fact_observation(item: Mapping[str, object]) -> tuple[object, ...] | None:
            citation = item.get("citation", {})
            if not isinstance(citation, Mapping) or citation.get("numeric_eligible") is not True:
                return None
            try:
                value = Decimal(str(citation.get("normalized_value")))
                multiplier = Decimal(str(citation.get("multiplier")))
            except (InvalidOperation, TypeError, ValueError):
                return None
            if not value.is_finite() or not multiplier.is_finite():
                return None
            return (
                str(citation.get("metric") or ""),
                str(citation.get("report_period") or ""),
                str(citation.get("accounting_basis") or ""),
                str(citation.get("unit") or ""),
                str(citation.get("currency") or ""),
                value * multiplier,
            )

        numeric_by_key: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        for item in support:
            observation = fact_observation(item)
            if observation is None:
                continue
            semantic_key = observation[:-1]
            numeric_by_key[semantic_key].append({
                "value": str(observation[-1]),
                "source_id": item.get("source_id"),
                "content_sha256": item.get("content_sha256"),
                "evidence_id": item.get("evidence_id"),
                "citation": item.get("citation"),
            })
        candidate_conflicts = [
            {"metric": key[0], "report_period": key[1], "accounting_basis": key[2], "unit": key[3], "currency": key[4], "observations": rows}
            for key, rows in numeric_by_key.items()
            if len({row["value"] for row in rows}) > 1
        ]
        all_covered = bool(coverage_rows) and all(row["status"] == "located" for row in coverage_rows)
        numeric_claim_ready = (
            all_covered
            and all(row["numeric_claim_ready"] for row in coverage_rows)
            and not candidate_conflicts
        )
        planned_dimensions = max(1, len(metric_specs) * max(1, len(periods)))
        supporting_steps = sum(step["kind"] == "supporting_evidence" for step in step_trace)
        if numeric_claim_ready:
            stop_reason = "requested_metric_period_coverage_located"
            next_action = "answer_with_source_citations_and_review_counter_candidates"
        elif all_covered:
            stop_reason = "coverage_located_but_numeric_claim_requires_review"
            next_action = "review_source_numeric_eligibility_and_citations_before_claim"
        elif not support and metric_candidates:
            stop_reason = "candidate_evidence_requires_metric_scope_review"
            next_action = "review_candidate_metric_period_unit_and_basis_before_numeric_claims"
        elif not support:
            stop_reason = "no_supporting_evidence_after_bounded_search"
            next_action = "abstain_or_clarify_missing_metric_period"
        elif supporting_steps < planned_dimensions:
            stop_reason = "search_budget_reached_before_all_dimensions"
            next_action = "report_coverage_gaps_and_do_not_infer_missing_values"
        else:
            stop_reason = "bounded_search_completed_with_coverage_gaps"
            next_action = "report_coverage_gaps_and_do_not_infer_missing_values"
        if candidate_conflicts:
            next_action = "pause_numeric_claims_until_candidate_conflicts_are_reviewed"
        if self.store.permission_revision() != task_permission_revision:
            raise PermissionError("permissions changed before financial research could return")
        if control is not None:
            control.check()
        return {
            "schema_version": "adaptive-mas.financial-rag-research.v1",
            "plan_id": plan.plan_id,
            "query_binding": {
                "security_code": plan.security_code,
                "as_of": parse_datetime(plan.as_of, "as_of", required=True).isoformat(),
                "time_mode": plan.time_mode,
                "release_id": release_id,
            },
            "planner": {
                "implementation": "fixed_metric_alias_period_templates",
                "max_searches": max_searches,
                "evidence_limit": evidence_limit,
                "rrf_rank_constant": rrf_rank_constant,
                "reranker_profile": "configured" if use_reranker else "disabled",
                "query_rewrite_enabled": query_rewrite_enabled,
                "multi_hop_enabled": multi_hop_enabled,
                "fusion_strategy": fusion_strategy,
                "row_group_chunking": row_group_chunking,
                "include_continuation_context": include_continuation_context,
                "channels": list(plan.channels),
                "returned_supporting_count": len(support),
                "returned_metric_candidate_count": len(metric_candidates),
                "returned_counter_candidate_count": len(counter_candidates),
                "evidence_truncated": max(0, len(seen) - len(support) - len(metric_candidates) - len(counter_candidates)),
                "search_count": len(step_trace),
                "requested_metrics": list(metrics),
                "requested_metric_variants": [
                    {"metric": item.metric, "label": item.label, "match_status": item.match_status}
                    for item in metric_specs
                ],
                "requested_report_periods": list(periods),
                "requested_unit": requested_unit_label,
                "requested_accounting_basis": requested_basis_label,
                "dimensions_truncated": supporting_steps < planned_dimensions,
                "steps": step_trace,
            },
            "status": "available" if numeric_claim_ready else "review_required" if support else "candidate_only" if metric_candidates else "no_evidence",
            "numeric_claim_ready": numeric_claim_ready,
            "claim_status": "eligible" if numeric_claim_ready else "review_required" if support else "missing",
            "coverage": coverage_rows,
            "channels": channel_status,
            "retrieved": [*support, *metric_candidates, *counter_candidates],
            "supporting_evidence": support,
            "candidate_evidence": metric_candidates,
            "counter_evidence": {
                "status": "candidate_search_completed" if step_trace and step_trace[-1]["kind"] == "counter_evidence_candidate" else "not_attempted",
                "backend_scope": "local BM25 search for correction and restatement language; results require source and claim review",
                "candidates": counter_candidates,
            },
            "candidate_numeric_conflicts": candidate_conflicts,
            "channel_status": channel_status,
            "stop_policy": {
                "stop_reason": stop_reason,
                "next_action": next_action,
                "automatic_contradiction_claims": False,
            },
            "warnings": sorted(set(warnings)),
        }

    def list_continuation_candidates(self, plan: QueryPlan) -> dict[str, object]:
        """List source-bound review candidates visible to the server-configured reviewer."""
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise PermissionError("continuation candidates are not currently available")
        as_of = parse_datetime(plan.as_of, "as_of", required=True)
        if as_of > datetime.now(timezone.utc):
            raise ValueError("as_of cannot be in the future")
        permission_before = self.store.permission_revision()
        continuation_before = self.store.continuation_revision()
        versions = self.store.permitted_versions(
            plan.principal_id, plan.security_code, as_of, time_mode=plan.time_mode,
            report_periods=plan.report_periods, accounting_basis=plan.accounting_basis,
            release_id=release_id,
        )
        candidates = self.store.list_table_continuation_candidates(
            [str(row["version_id"]) for row in versions],
        )
        if permission_before != self.store.permission_revision() or continuation_before != self.store.continuation_revision():
            raise PermissionError("review scope changed while listing continuation candidates")
        return {
            "release_id": release_id,
            "permission_revision": permission_before,
            "continuation_revision": continuation_before,
            "candidates": candidates,
        }

    def review_continuation_candidate(
        self,
        plan: QueryPlan,
        child_table_id: str,
        *,
        action: str,
        content_sha256: str,
        note: str = "",
    ) -> dict[str, object]:
        """Apply a durable review decision after checking the reviewer’s live source scope."""
        release_id = plan.release_id or self.store.active_release_id()
        if release_id is None:
            raise PermissionError("continuation candidate is not currently available")
        as_of = parse_datetime(plan.as_of, "as_of", required=True)
        if as_of > datetime.now(timezone.utc):
            raise ValueError("as_of cannot be in the future")
        versions = self.store.permitted_versions(
            plan.principal_id, plan.security_code, as_of, time_mode=plan.time_mode,
            report_periods=plan.report_periods, accounting_basis=plan.accounting_basis,
            release_id=release_id,
        )
        allowed_versions = {str(row["version_id"]) for row in versions}
        candidates = self.store.list_table_continuation_candidates(list(allowed_versions))
        candidate = next((item for item in candidates if item.get("current_table_id") == child_table_id), None)
        if candidate is None or candidate.get("content_sha256") != content_sha256:
            raise PermissionError("continuation candidate is not currently available to this reviewer")
        if str(candidate.get("version_id")) not in allowed_versions:
            raise PermissionError("continuation candidate is not currently available to this reviewer")
        return self.store.review_table_continuation(
            child_table_id,
            action=action,
            reviewer_id=plan.principal_id,
            release_id=release_id,
            expected_content_sha256=content_sha256,
            note=note,
        )

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

    def read_table_row(
        self,
        plan: QueryPlan,
        table_id: str,
        row_start: int,
        row_end: int,
        *,
        include_continuation_context: bool = True,
    ) -> dict[str, object]:
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
        continuation_revision_before = self.store.continuation_revision()
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
        continuation = record.get("continuation")
        table_payload = json.loads(str(table["table_payload_json"]))
        candidate = table_payload.get("continuation_candidate")
        if isinstance(candidate, Mapping) and continuation is None:
            raise PermissionError("table rows remain quarantined until continuation review is confirmed")
        continuation_context = (
            candidate.get("previous_context", {})
            if include_continuation_context and isinstance(candidate, Mapping) else {}
        )
        parent_header_cells = record.get("parent_header_cells", []) if include_continuation_context else []
        header_rows = int(table["header_rows"])
        headers: dict[int, list[str]] = {}
        data: dict[int, list[dict[str, object]]] = {}
        for cell in parent_header_cells:
            column_index = int(cell["column_index"])
            headers.setdefault(column_index, []).append(str(cell["raw_text"]))
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
        used_cells = list(parent_header_cells) + [cell for cell in cells if int(cell["row_index"]) < header_rows]
        data_cells: list[dict[str, object]] = []
        for row_index in sorted(data):
            ordered_cells = sorted(data[row_index], key=lambda cell: int(cell["column_index"]))
            used_cells.extend(ordered_cells)
            data_cells.extend(ordered_cells)
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
                        "unit": cell["unit"] or table["unit_text"] or (continuation_context.get("unit") if isinstance(continuation_context, Mapping) else None),
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
        pages = [int(cell["page_number"]) for cell in data_cells if cell["page_number"] is not None]
        unit = table["unit_text"] or (continuation_context.get("unit") if isinstance(continuation_context, Mapping) else None)
        accounting_basis = table["table_accounting_basis"] or (continuation_context.get("accounting_basis") if isinstance(continuation_context, Mapping) else None)
        report_period = table["table_report_period"] or (continuation_context.get("report_period") if isinstance(continuation_context, Mapping) else None)
        citation = {
            "document_id": table["document_id"], "version_id": table["version_id"],
            "source_id": table["source_id"], "source_url": table["source_url"],
            "published_at": table["published_at"], "first_available_at": table["first_available_at"],
            "captured_at": table["captured_at"], "ingested_at": table["ingested_at"],
            "content_sha256": table["content_sha256"], "time_basis": table["time_basis"],
            "page_number": pages[0] if pages else table["page_start"],
            "table_id": table_id, "row_start": row_start, "row_end": row_end,
            "unit": unit, "accounting_basis": accounting_basis,
            "report_period": report_period, "as_of": as_of.isoformat(),
            "release_id": release_id,
            "supporting_cells": [
                {
                    "cell_id": json.loads(str(cell["payload_json"])).get("cell_id"),
                    "row_index": int(cell["row_index"]),
                    "column_index": int(cell["column_index"]),
                    "page_number": cell["page_number"],
                    "raw_text": str(cell["raw_text"]),
                    "normalized_value": cell["normalized_value"],
                    "value_kind": str(cell["value_kind"]),
                    "unit": cell["unit"] or table["unit_text"] or (continuation_context.get("unit") if isinstance(continuation_context, Mapping) else None),
                    "multiplier": cell["multiplier"],
                    "currency": cell["currency"],
                    "accounting_basis": cell["accounting_basis"] or table["table_accounting_basis"] or (continuation_context.get("accounting_basis") if isinstance(continuation_context, Mapping) else None),
                    "report_period": cell["report_period"] or table["table_report_period"] or (continuation_context.get("report_period") if isinstance(continuation_context, Mapping) else None),
                }
                for cell in used_cells
            ],
        }
        if include_continuation_context and isinstance(continuation, Mapping) and isinstance(candidate, Mapping):
            citation["continuation"] = {
                "candidate_id": candidate.get("candidate_id"),
                "previous_table_id": continuation.get("previous_table_id"),
                "current_table_id": table_id,
                "previous_page": candidate.get("previous_page"),
                "current_page": candidate.get("current_page"),
                "review_status": "confirmed",
                "context": continuation_context,
                "header_labels": candidate.get("header_labels", {}),
            }
            context_span_id = candidate.get("context_span_id")
            if context_span_id:
                context_span = self.store._get_span(str(context_span_id))
                if context_span is not None and str(context_span["version_id"]) == str(table["version_id"]):
                    citation["supporting_spans"] = [{
                        "span_id": context_span["span_id"], "page_number": context_span["page_number"],
                        "char_start": context_span["char_start"], "char_end": context_span["char_end"],
                        "source_id": context_span["source_id"], "content_sha256": context_span["content_sha256"],
                        "version_id": context_span["version_id"], "role": "continuation_context",
                    }]
        text = f"列头：{header_text}\n" + "\n".join(row_texts)
        if self.store.continuation_revision() != continuation_revision_before:
            raise PermissionError("continuation review changed while expanding the table row")
        return {
            "table_id": table_id, "row_start": row_start, "row_end": row_end,
            "title": table["table_title"] or (continuation_context.get("title") if isinstance(continuation_context, Mapping) else None), "header_rows": header_rows,
            "rows": output_rows, "text": text, "citation": citation,
            "permission_revision": self.store.permission_revision(),
            "continuation_revision": self.store.continuation_revision(),
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

    def _fact_results(
        self,
        plan: QueryPlan,
        facts: Sequence[Mapping[str, object]],
        versions: Mapping[str, Mapping[str, object]],
        *,
        control: SearchControl | None = None,
    ) -> list[EvidenceResult]:
        query_tokens = set(financial_tokens(plan.query + " " + " ".join(plan.aliases)))
        rows: list[tuple[float, EvidenceResult]] = []
        referenced_span_ids = [
            str(span_id)
            for fact in facts
            for span_id in json.loads(str(fact["payload_json"])).get("span_ids", [])
        ]
        span_by_id = {str(row["span_id"]): row for row in self.store.read_spans(referenced_span_ids)}
        for fact in facts:
            if control is not None:
                control.check()
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

    def _block_results(
        self,
        plan: QueryPlan,
        blocks: Sequence[Mapping[str, object]],
        versions: Mapping[str, Mapping[str, object]],
        *,
        control: SearchControl | None = None,
    ) -> list[EvidenceResult]:
        if control is not None:
            control.check()
        scoped = [dict(row) | {"text": str(row["text"])} for row in blocks if row["version_id"] in versions]
        broad_ranked = _bm25(plan.query + " " + " ".join(plan.aliases), scoped, limit=plan.top_k * 5, control=control)
        exact_table_rows = [row for row in scoped if _matches_exact_table_row_query(plan, row)]
        focused_ranked: list[tuple[Mapping[str, object], float]] = []
        if exact_table_rows:
            exact_metrics = [
                item.label for item in requested_metric_specs(plan.query)
                if item.match_status == "exact"
            ]
            exact_periods = requested_periods(plan.query, plan.report_periods)
            focused_query = " ".join((*exact_metrics, *exact_periods, *plan.aliases))
            focused_ranked = _bm25(
                focused_query, exact_table_rows, limit=plan.top_k * 5, control=control,
            )
        focused_ids = {str(row.get("block_id")) for row, _ in focused_ranked}
        ranked = focused_ranked + [
            (row, score) for row, score in broad_ranked
            if str(row.get("block_id")) not in focused_ids
        ]
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
