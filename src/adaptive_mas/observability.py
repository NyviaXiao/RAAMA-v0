"""Small, privacy-aware trace records for local harness operations."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Mapping


_SENSITIVE_KEY = re.compile(
    r"(?:api.?key|access.?token|secret|authorization|prompt|message|raw.?text|"
    r"portfolio|account|credential|password|private.?path|(?:^|_)(?:path|file|directory)(?:$|_))",
    re.IGNORECASE,
)
_MAX_TEXT = 240
_OTEL_PROVIDER = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_value(value: object, *, depth: int = 0) -> object:
    if depth > 3:
        return "[nested value omitted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item, depth=depth + 1)
            for key, item in value.items()
            if not _SENSITIVE_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth=depth + 1) for item in value[:32]]
    return str(value)[:_MAX_TEXT]


def safe_attributes(attributes: Mapping[str, object] | None) -> dict[str, object]:
    """Keep compact operational facts while dropping credentials and private content."""
    return {
        str(key): _safe_value(value)
        for key, value in (attributes or {}).items()
        if not _SENSITIVE_KEY.search(str(key))
    }


def span_record(
    *,
    trace_id: str,
    name: str,
    attributes: Mapping[str, object] | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    parent_span_id: str | None = None,
    status: str = "ok",
) -> dict[str, object]:
    end = ended_at or utc_now()
    start = started_at or end
    return {
        "trace_id": trace_id,
        "span_id": uuid.uuid4().hex,
        "parent_span_id": parent_span_id,
        "name": name,
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "duration_ms": max(0.0, (end - start).total_seconds() * 1000),
        "status": status,
        "attributes": safe_attributes(attributes),
    }


def export_span_if_configured(record: Mapping[str, object]) -> str:
    """Optionally export sanitized spans to OTLP; local SQLite remains the default."""
    global _OTEL_PROVIDER
    endpoint = os.environ.get("RAAMA_OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return "disabled"
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        return "configured_but_optional_dependencies_missing"
    try:
        if _OTEL_PROVIDER is None:
            provider = TracerProvider(resource=Resource.create({"service.name": "raama-harness"}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            _OTEL_PROVIDER = provider
            trace.set_tracer_provider(provider)
        tracer = trace.get_tracer("adaptive_mas.harness")
        attributes: dict[str, str | bool | int | float] = {
            "raama.trace_id": str(record.get("trace_id", "")),
            "raama.span_id": str(record.get("span_id", "")),
            "raama.status": str(record.get("status", "ok")),
        }
        if record.get("parent_span_id"):
            attributes["raama.parent_span_id"] = str(record["parent_span_id"])
        for key, value in safe_attributes(record.get("attributes") if isinstance(record.get("attributes"), Mapping) else {}).items():
            if isinstance(value, (bool, int, float, str)):
                attributes[f"raama.{key}"] = value
            elif value is not None:
                attributes[f"raama.{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)[:_MAX_TEXT]
        with tracer.start_as_current_span(str(record.get("name", "raama.operation")), attributes=attributes):
            pass
        return "exported"
    except Exception:
        # Telemetry export must never change a local task/evaluation result.
        return "export_failed"


def decision_summary(result: Mapping[str, object]) -> dict[str, object]:
    """Expose decision/risk/cost/version facts without copying account positions or evidence text."""
    orders = result.get("orders", [])
    predictions = result.get("predictions_by_symbol", {})
    versions = result.get("versions", {})
    evidence = result.get("evidence", {})
    harness = evidence.get("harness", {}) if isinstance(evidence, Mapping) else {}
    cost = harness.get("decision_cost_accounting", {}) if isinstance(harness, Mapping) else {}
    return {
        "schema_version": result.get("schema_version"),
        "as_of_date": result.get("as_of_date"),
        "execution_date": result.get("execution_date"),
        "status": result.get("status"),
        "prediction_status": result.get("prediction_status"),
        "prediction_horizon_sessions": result.get("prediction_horizon_sessions"),
        "prediction_count": len(predictions) if isinstance(predictions, Mapping) else 0,
        "order_count": len(orders) if isinstance(orders, list) else 0,
        "orders": [
            {
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "notional_cny": row.get("notional_cny"),
                "status": row.get("status"),
            }
            for row in orders[:100]
            if isinstance(row, Mapping)
        ] if isinstance(orders, list) else [],
        "current_nav_cny": result.get("current_nav_cny"),
        "annualized_risk": result.get("annualized_risk"),
        "annualized_risk_budget": result.get("annualized_risk_budget"),
        "risk_adjusted_net_incremental_value_cny": result.get(
            "risk_adjusted_net_incremental_value_cny"
        ),
        "expected_transaction_cost_cny": result.get("expected_transaction_cost_cny"),
        "logical_research_cost_cny": cost.get("logical_research_cost_cny") if isinstance(cost, Mapping) else None,
        "new_api_cost_cny": cost.get("new_api_cost_cny") if isinstance(cost, Mapping) else None,
        "versions": {
            str(key): value for key, value in versions.items()
            if str(key).endswith("_sha256") or "version" in str(key)
        } if isinstance(versions, Mapping) else {},
        "risk_checks": harness.get("hard_risk_outcome") if isinstance(harness, Mapping) else None,
    }
