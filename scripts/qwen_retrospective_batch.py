"""Prepare and optionally run a bounded retrospective Qwen event study.

The default command only writes a local outbound-payload preview. Network access
is reachable only through the explicit ``send`` subcommand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "experiments" / "qwen_retrospective_dry_run_20261012"
EVENT_MANIFEST = RESEARCH_DIR / "event_manifest.csv"
OUTBOUND_DIR = RESEARCH_DIR / "qwen_outbound_review_v1"
PAYLOADS_PATH = OUTBOUND_DIR / "outbound_payloads.jsonl"
PREVIEW_SUMMARY_PATH = OUTBOUND_DIR / "outbound_summary.json"
RESPONSE_DIR = OUTBOUND_DIR / "responses"
TRAINING_EFFECT_PATH = OUTBOUND_DIR / "training_effect_summary.json"
TRAINING_MEMORY_PATH = OUTBOUND_DIR / "training_matured_memory.jsonl"

SHANGHAI = ZoneInfo("Asia/Shanghai")
MODEL = "qwen3.7-max-2026-06-08"
PROMPT_VERSION = "retrospective_single_announcement_v4_qwen37max0608_nonthinking"
MAX_TOTAL_CALLS = 118
MAX_TOTAL_ESTIMATED_BUDGET_CNY = 3.99
EXPECTED_SPLIT_COUNTS = {"training": 81, "evaluation": 37}
MAX_INPUT_TOKENS = 2048
MAX_INPUT_MESSAGE_BYTES = 1920
CHAT_TOKEN_OVERHEAD_RESERVE = 128
MAX_OUTPUT_TOKENS = 256
INPUT_CNY_PER_MILLION_TOKENS = 12.0
OUTPUT_CNY_PER_MILLION_TOKENS = 36.0
HORIZON_SESSIONS = 3
TRAIN_END = date(2025, 8, 29)
EVALUATION_START = date(2025, 9, 1)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _num(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _event_id(row: dict[str, str]) -> str:
    key = "\n".join((
        row["data_split"],
        row["prediction_date"],
        row["symbol"],
        row["announcement_source_record_id_citation"],
        row["announcement_sha256"],
        row["financial_indicator_source_record_id"],
        row["financial_indicator_sha256"],
        PROMPT_VERSION,
    ))
    return _sha256(key.encode("utf-8"))


def _citation_ref(kind: str, source_record_id: str) -> str:
    return f"{kind}-{_sha256(source_record_id.encode('utf-8'))[:20]}"


def _build_payload(row: dict[str, str]) -> dict[str, object]:
    event_id = _event_id(row)
    sources = [
        {
            "citation_ref": _citation_ref("announcement", row["announcement_source_record_id_citation"]),
            "sha256": row["announcement_sha256"],
        }
    ]
    financial_source_id = row.get("financial_indicator_source_record_id", "")
    if financial_source_id:
        sources.append({
            "citation_ref": _citation_ref("financial", financial_source_id),
            "sha256": row["financial_indicator_sha256"],
        })
    event = {
        "id": event_id,
        "symbol": row["symbol"],
        "title": row["announcement_title"],
        "financial": {
            "report_period": row.get("financial_indicator_report_period") or None,
            "revenue_yoy_pct": _num(row.get("revenue_yoy_pct")),
            "parent_net_profit_yoy_pct": _num(row.get("parent_net_profit_yoy_pct")),
        },
        "sources": sources,
    }
    system = (
        "只依据输入的一条公告标题与财务同比；不联网、不补外部事实、不臆测正文。"
        "判断该事件对公司未来3个交易日的方向，不估收益；证据不足给0。"
        "财务历史版本未核实，结果为事后合成研究。引用输入的citation_ref，只返回JSON。"
    )
    user = {
        "task": "独立判断这条公告的方向性，不给投资建议。",
        "event": event,
        "output_schema": {
            "event_id": event_id,
            "direction_score": "-1, 0, or 1",
            "evidence_sufficiency": "sufficient, limited, or insufficient",
            "rationale": "brief Chinese explanation, at most 100 Chinese characters",
            "cited_source_record_ids": ["citation_ref from this event"],
        },
    }
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _canonical_json(user)},
        ],
        "temperature": 0,
        "enable_thinking": False,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }


def _load_preview() -> list[dict[str, object]]:
    with PAYLOADS_PATH.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def prepare() -> dict[str, object]:
    with EVENT_MANIFEST.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    split_counts = {
        split: sum(row["data_split"] == split for row in rows)
        for split in EXPECTED_SPLIT_COUNTS
    }
    if len(rows) != MAX_TOTAL_CALLS or split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Frozen event manifest count mismatch: {len(rows)} rows, {split_counts}")

    prepared = []
    input_sizes = []
    for row in rows:
        if row["retrospective_only"].lower() != "true":
            raise ValueError("An event is not marked retrospective_only")
        if row["announcement_vintage_verified"].lower() != "false":
            raise ValueError("An announcement vintage is incorrectly marked verified")
        if row["financial_indicator_vintage_verified"].lower() != "false":
            raise ValueError("A financial vintage is incorrectly marked verified")
        payload = _build_payload(row)
        message_bytes = len(_canonical_json(payload["messages"]).encode("utf-8"))
        if message_bytes > MAX_INPUT_MESSAGE_BYTES:
            raise ValueError(
                f"Input payload exceeds safe byte bound for event {_event_id(row)}: {message_bytes} bytes"
            )
        conservative_tokens = message_bytes + CHAT_TOKEN_OVERHEAD_RESERVE
        if conservative_tokens > MAX_INPUT_TOKENS:
            raise ValueError(f"Input token upper bound exceeded for event {_event_id(row)}")
        serialized = _canonical_json(payload).encode("utf-8")
        prepared.append({
            "event_id": _event_id(row),
            "data_split": row["data_split"],
            "prompt_version": PROMPT_VERSION,
            "request_sha256": _sha256(serialized),
            "message_utf8_bytes": message_bytes,
            "conservative_input_token_upper_bound": conservative_tokens,
            "local_event_metadata": {
                "prediction_date": row["prediction_date"],
                "symbol": row["symbol"],
            },
            "local_source_reference_map": {
                _citation_ref("announcement", row["announcement_source_record_id_citation"]): {
                    "source_record_id": row["announcement_source_record_id_citation"],
                    "url": row["announcement_url"],
                    "sha256": row["announcement_sha256"],
                    "vintage_verified": False,
                },
                **({
                    _citation_ref("financial", row["financial_indicator_source_record_id"]): {
                        "source_record_id": row["financial_indicator_source_record_id"],
                        "raw_response_ref": row["financial_indicator_raw_response_ref"],
                        "sha256": row["financial_indicator_sha256"],
                        "report_period": row["financial_indicator_report_period"],
                        "published_at": row["financial_indicator_published_at"],
                        "vintage_verified": False,
                    }
                } if row.get("financial_indicator_source_record_id") else {}),
            },
            "payload": payload,
        })
        input_sizes.append((message_bytes, conservative_tokens))

    lines = "".join(_canonical_json(row) + "\n" for row in prepared).encode("utf-8")
    payload_hash = _sha256(lines)
    OUTBOUND_DIR.mkdir(parents=True, exist_ok=True)
    if PAYLOADS_PATH.exists():
        if _sha256(PAYLOADS_PATH.read_bytes()) != payload_hash:
            if RESPONSE_DIR.exists() and any(RESPONSE_DIR.glob("*.json")):
                raise FileExistsError("Existing payload preview has call states; refusing to replace it")
            temporary = PAYLOADS_PATH.with_name(PAYLOADS_PATH.name + ".writing")
            temporary.write_bytes(lines)
            os.replace(temporary, PAYLOADS_PATH)
    else:
        temporary = PAYLOADS_PATH.with_name(PAYLOADS_PATH.name + ".writing")
        temporary.write_bytes(lines)
        os.replace(temporary, PAYLOADS_PATH)

    per_call_cap = (
        MAX_INPUT_TOKENS * INPUT_CNY_PER_MILLION_TOKENS
        + MAX_OUTPUT_TOKENS * OUTPUT_CNY_PER_MILLION_TOKENS
    ) / 1_000_000
    summary = {
        "status": "payload_preview_ready_no_network",
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "enable_thinking": False,
        "events": len(prepared),
        "split_counts": split_counts,
        "one_event_per_request": True,
        "input_limit": {
            "max_conservative_input_tokens": MAX_INPUT_TOKENS,
            "max_message_utf8_bytes": MAX_INPUT_MESSAGE_BYTES,
            "chat_overhead_token_reserve": CHAT_TOKEN_OVERHEAD_RESERVE,
            "preflight_rule": "serialized UTF-8 message bytes + 128 framing-token reserve must be <= 2048",
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "payload_message_utf8_bytes": {
            "minimum": min(size for size, _ in input_sizes),
            "maximum": max(size for size, _ in input_sizes),
        },
        "call_hard_limit_total": MAX_TOTAL_CALLS,
        "estimated_cost_cap_cny_at_existing_client_price_formula": {
            "per_call": per_call_cap,
            "training_81_calls": per_call_cap * EXPECTED_SPLIT_COUNTS["training"],
            "evaluation_37_calls": per_call_cap * EXPECTED_SPLIT_COUNTS["evaluation"],
            "all_118_calls": per_call_cap * MAX_TOTAL_CALLS,
        },
        "price_basis": "Aliyun Model Studio official qwen3.7-max-2026-06-08 rate: CNY 12/M input and CNY 36/M output; enable_thinking=false; actual provider bill may differ",
        "price_source": "https://help.aliyun.com/zh/model-studio/model-pricing",
        "training_gate": "evaluation send is refused until training P11 report is candidate_effect_eligible",
        "budget_scope_global_cumulative": True,
        "global_estimated_budget_ceiling_cny": MAX_TOTAL_ESTIMATED_BUDGET_CNY,
        "endpoint_environment_variable": "POPAGENT_LLM_URL (user-configured Beijing chat/completions endpoint; value is not logged)",
        "retry_policy": "submitted or delivery-unknown events are never retried automatically",
        "network_requests": 0,
        "qwen_key_access": 0,
        "test_csv_read": False,
        "payload_manifest_sha256": payload_hash,
        "payload_manifest_path": str(PAYLOADS_PATH),
    }
    _atomic_json(PREVIEW_SUMMARY_PATH, summary)
    return summary


def _state_path(event_id: str) -> Path:
    return RESPONSE_DIR / f"{event_id}.json"


def _load_states() -> dict[str, dict[str, object]]:
    if not RESPONSE_DIR.exists():
        return {}
    states = {}
    for path in RESPONSE_DIR.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        states[str(row["event_id"])] = row
    return states


def _validate_response(content: str, expected_id: str, allowed_sources: set[str]) -> dict[str, object]:
    parsed = json.loads(content)
    if str(parsed["event_id"]) != expected_id:
        raise ValueError("Qwen response event_id mismatch")
    if int(parsed["direction_score"]) not in {-1, 0, 1}:
        raise ValueError("Qwen direction_score must be -1, 0, or 1")
    if parsed["evidence_sufficiency"] not in {"sufficient", "limited", "insufficient"}:
        raise ValueError("Qwen evidence_sufficiency is outside the fixed schema")
    if len(str(parsed["rationale"])) > 500:
        raise ValueError("Qwen rationale exceeds the fixed response bound")
    citations = {str(value) for value in parsed["cited_source_record_ids"]}
    if not citations or not citations <= allowed_sources:
        raise ValueError("Qwen cited a source not present in the request")
    return parsed


def send(split: str, max_calls: int, max_budget_cny: float, expected_preview_sha256: str) -> None:
    if split not in EXPECTED_SPLIT_COUNTS:
        raise ValueError("split must be training or evaluation")
    if max_calls < 1 or max_calls > MAX_TOTAL_CALLS:
        raise ValueError("max-calls must be between 1 and 118")
    if max_budget_cny <= 0:
        raise ValueError("max-budget-cny must be positive")
    if max_budget_cny > MAX_TOTAL_ESTIMATED_BUDGET_CNY:
        raise ValueError("max-budget-cny exceeds the approved global ceiling")
    if not PAYLOADS_PATH.exists() or not PREVIEW_SUMMARY_PATH.exists():
        raise FileNotFoundError("Run the local prepare command first")
    payload_hash = _sha256(PAYLOADS_PATH.read_bytes())
    summary = json.loads(PREVIEW_SUMMARY_PATH.read_text(encoding="utf-8"))
    if payload_hash != expected_preview_sha256 or payload_hash != summary["payload_manifest_sha256"]:
        raise ValueError("Reviewed outbound preview hash does not match current payloads")
    if split == "evaluation":
        if not TRAINING_EFFECT_PATH.exists():
            raise ValueError("Training P11 report is missing; evaluation sends are locked")
        gate = json.loads(TRAINING_EFFECT_PATH.read_text(encoding="utf-8"))
        if gate.get("status") != "candidate_effect_eligible":
            raise ValueError("Training P11 effect did not pass; evaluation sends are locked")

    records = [row for row in _load_preview() if row["data_split"] == split]
    states = _load_states()
    preview_ids = {str(row["event_id"]) for row in _load_preview()}
    if set(states) - preview_ids:
        raise ValueError("Response directory contains event IDs outside the frozen preview")
    selected = []
    reserved = 0.0
    for row in states.values():
        reserved += float(row.get("reserved_cost_cny", 0.0))
    for record in records:
        state = states.get(str(record["event_id"]))
        if state is None:
            selected.append(record)
            continue
        if state.get("request_sha256") != record["request_sha256"]:
            raise ValueError("Existing event state belongs to a different request payload")
        if state.get("status") in {"delivery_unknown", "completed_invalid_response"}:
            continue
        if state.get("status") != "completed":
            raise RuntimeError(
                f"Event {record['event_id']} has status {state.get('status')}; reconcile before proceeding"
            )
    if len(selected) > max_calls:
        raise ValueError(f"Call cap {max_calls} is below {len(selected)} unsent {split} events; no call started")
    if sum(int(row["data_split"] == split) for row in _load_preview()) != EXPECTED_SPLIT_COUNTS[split]:
        raise ValueError("Frozen split event count changed")
    per_call_cap = (
        MAX_INPUT_TOKENS * INPUT_CNY_PER_MILLION_TOKENS
        + MAX_OUTPUT_TOKENS * OUTPUT_CNY_PER_MILLION_TOKENS
    ) / 1_000_000
    if reserved + per_call_cap * len(selected) > max_budget_cny:
        raise ValueError("Worst-case reserved spend exceeds max-budget-cny; no call started")
    attempted_total = len(states)
    if attempted_total + len(selected) > MAX_TOTAL_CALLS:
        raise ValueError("Global 118-call ceiling would be exceeded")

    endpoint = os.environ.get("POPAGENT_LLM_URL")
    configured_model = os.environ.get("POPAGENT_LLM_MODEL", MODEL)
    if not endpoint:
        raise ValueError("Configured Qwen endpoint is missing")
    if configured_model != MODEL:
        raise ValueError("Configured model differs from the reviewed payload model")
    credential = (ROOT / "data" / "raw" / "qwen_key.txt").read_text(encoding="utf-8-sig").strip()
    if not credential:
        raise ValueError("Authorized local Qwen key file is empty")
    direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    for record in selected:
        payload = record["payload"]
        messages_bytes = len(_canonical_json(payload["messages"]).encode("utf-8"))
        if messages_bytes > MAX_INPUT_MESSAGE_BYTES or messages_bytes + CHAT_TOKEN_OVERHEAD_RESERVE > MAX_INPUT_TOKENS:
            raise ValueError("Reviewed input cap failed at send-time; no request sent for this event")
        event_id = str(record["event_id"])
        state_path = _state_path(event_id)
        state = {
            "event_id": event_id,
            "data_split": split,
            "status": "submitted",
            "request_sha256": record["request_sha256"],
            "payload_manifest_sha256": payload_hash,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "attempt_count": 1,
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "reserved_cost_cny": per_call_cap,
            "artifact_class": "retrospective_synthetic",
            "decision_effect": "research_only_until_matured_comparison",
        }
        _atomic_json(state_path, state)
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {credential}",
                "Content-Type": "application/json",
                "Idempotency-Key": event_id,
            },
            method="POST",
        )
        try:
            with direct_opener.open(request, timeout=90) as response:
                body_bytes = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            state.update({
                "status": "delivery_unknown",
                "delivery_error_type": type(error).__name__,
                "delivery_error_reason_type": type(reason).__name__,
                "delivery_error_errno": getattr(reason, "errno", None),
                "delivery_error_winerror": getattr(reason, "winerror", None),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _atomic_json(state_path, state)
            raise RuntimeError("Qwen delivery failed or is unknown; automatic retry is disabled") from None

        body = json.loads(body_bytes.decode("utf-8"))
        usage = body.get("usage")
        if not isinstance(usage, dict) or "prompt_tokens" not in usage or "completion_tokens" not in usage:
            generated_at = datetime.now(timezone.utc).isoformat()
            state.update({
                "status": "completed_usage_missing",
                "raw_response_sha256": _sha256(body_bytes),
                "generated_at_utc": generated_at,
                "finished_at_utc": generated_at,
            })
            _atomic_json(state_path, state)
            raise RuntimeError("Provider usage is missing; reserved cost retained and remaining sends stopped")
        try:
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
        except (TypeError, ValueError):
            generated_at = datetime.now(timezone.utc).isoformat()
            state.update({
                "status": "completed_usage_invalid",
                "raw_response_sha256": _sha256(body_bytes),
                "generated_at_utc": generated_at,
                "finished_at_utc": generated_at,
            })
            _atomic_json(state_path, state)
            raise RuntimeError("Provider usage is invalid; reserved cost retained and remaining sends stopped") from None
        if prompt_tokens < 0 or completion_tokens < 0:
            generated_at = datetime.now(timezone.utc).isoformat()
            state.update({
                "status": "completed_usage_invalid",
                "raw_response_sha256": _sha256(body_bytes),
                "generated_at_utc": generated_at,
                "finished_at_utc": generated_at,
            })
            _atomic_json(state_path, state)
            raise RuntimeError("Provider usage is negative; reserved cost retained and remaining sends stopped")
        actual_cost = (
            prompt_tokens * INPUT_CNY_PER_MILLION_TOKENS
            + completion_tokens * OUTPUT_CNY_PER_MILLION_TOKENS
        ) / 1_000_000
        content = body["choices"][0]["message"]["content"]
        request_json = json.loads(payload["messages"][1]["content"])
        source_ids = {str(row["citation_ref"]) for row in request_json["event"]["sources"]}
        generated_at = datetime.now(timezone.utc).isoformat()
        try:
            analysis = _validate_response(content, event_id, source_ids)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            state.update({
                "status": "completed_invalid_response",
                "response_error_type": type(error).__name__,
                "raw_response_sha256": _sha256(body_bytes),
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                "actual_estimated_cost_cny": actual_cost,
                "generated_at_utc": generated_at,
                "finished_at_utc": generated_at,
            })
            _atomic_json(state_path, state)
            raise RuntimeError("Qwen returned an invalid response; no automatic retry") from None
        state.update({
            "status": "completed",
            "analysis": analysis,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
            },
            "actual_estimated_cost_cny": actual_cost,
            "raw_response_sha256": _sha256(body_bytes),
            "generated_at_utc": generated_at,
            "finished_at_utc": generated_at,
            "retrospective_synthetic": True,
        })
        if prompt_tokens > MAX_INPUT_TOKENS or completion_tokens > MAX_OUTPUT_TOKENS:
            state["status"] = "completed_token_cap_violation"
            _atomic_json(state_path, state)
            raise RuntimeError("Provider-reported usage exceeded a reviewed cap; remaining sends stopped")
        _atomic_json(state_path, state)
        print(f"completed event {event_id} ({split})")


def analyze_training(data_dir: Path, baostock_run_dir: Path) -> dict[str, object]:
    from adaptive_mas.data import load_research_data
    from adaptive_mas.evaluation.expert_memory import (
        candidate_performance_effects,
        update_matured_performance_memory,
    )

    records = [row for row in _load_preview() if row["data_split"] == "training"]
    if len(records) != EXPECTED_SPLIT_COUNTS["training"]:
        raise ValueError("Frozen training payload count mismatch")
    states = _load_states()
    completed_records = []
    response_status_counts = {}
    missing_response_event_ids = []
    for record in records:
        event_id = str(record["event_id"])
        state = states.get(event_id)
        status = str(state.get("status", "missing")) if state else "missing"
        response_status_counts[status] = response_status_counts.get(status, 0) + 1
        if status == "completed":
            completed_records.append(record)
        else:
            missing_response_event_ids.append(event_id)
    research = load_research_data(data_dir, baostock_run_dir)
    date_index = {day: index for index, day in enumerate(research.trading_dates)}
    bars = {(str(row["symbol"]).rsplit(".", 1)[-1], row["trade_date"]): row for row in research.adjusted_bars}
    index_bars = {row["trade_date"]: row for row in research.index_bars}
    forecasts = []
    outcomes = []
    excluded = {"not_market_session": 0, "label_not_mature_before_evaluation": 0, "missing_required_bar": 0}
    for record in completed_records:
        event_id = str(record["event_id"])
        state = states[event_id]
        event = json.loads(record["payload"]["messages"][1]["content"])["event"]
        local_metadata = record["local_event_metadata"]
        decision_day = date.fromisoformat(str(local_metadata["prediction_date"]))
        if decision_day not in date_index:
            excluded["not_market_session"] += 1
            continue
        index = date_index[decision_day]
        exit_index = index + HORIZON_SESSIONS
        if exit_index >= len(research.trading_dates):
            excluded["label_not_mature_before_evaluation"] += 1
            continue
        entry_day = research.trading_dates[index + 1]
        exit_day = research.trading_dates[exit_index]
        if exit_day >= EVALUATION_START:
            excluded["label_not_mature_before_evaluation"] += 1
            continue
        symbol = str(local_metadata["symbol"])
        entry = bars.get((symbol, entry_day))
        exit_bar = bars.get((symbol, exit_day))
        index_entry = index_bars.get(entry_day)
        index_exit = index_bars.get(exit_day)
        if (
            entry is None or exit_bar is None or index_entry is None or index_exit is None
            or any(value is None for value in (entry.get("open"), exit_bar.get("close"), index_entry.get("open"), index_exit.get("close")))
        ):
            excluded["missing_required_bar"] += 1
            continue
        direction = int(state["analysis"]["direction_score"])
        forecasts.append({
            "record_id": event_id,
            "expert": "qwen_retrospective_event_v1",
            "symbol": symbol,
            "prediction_date": decision_day.isoformat(),
            "horizon_sessions": HORIZON_SESSIONS,
            "direction_score": direction,
        })
        stock_return = float(exit_bar["close"]) / float(entry["open"]) - 1
        market_return = float(index_exit["close"]) / float(index_entry["open"]) - 1
        outcomes.append({
            "record_id": event_id,
            "horizon_sessions": HORIZON_SESSIONS,
            "forward_return": stock_return - market_return,
            "label_available_at": datetime.combine(exit_day, time(16), SHANGHAI).isoformat(),
            "outcome_ref": f"baostock:{symbol}:{entry_day.isoformat()}:{exit_day.isoformat()}",
        })
    training_cutoff = datetime.combine(EVALUATION_START, time.min, SHANGHAI)
    memory = update_matured_performance_memory(forecasts, outcomes, training_cutoff)
    effect = candidate_performance_effects(memory, tuple(research.trading_dates)).get(
        f"qwen_retrospective_event_v1:{HORIZON_SESSIONS}s"
    )
    status = "candidate_effect_eligible" if effect and effect["status"] == "candidate_effect_eligible" else "stop_no_eligible_training_effect"
    report = {
        "status": status,
        "artifact_class": "retrospective_synthetic",
        "training_only": True,
        "evaluation_inputs_used": False,
        "training_window_end": TRAIN_END.isoformat(),
        "evaluation_start": EVALUATION_START.isoformat(),
        "training_qwen_responses": len(completed_records),
        "training_qwen_response_expected_count": len(records),
        "training_qwen_response_missing_count": len(missing_response_event_ids),
        "training_qwen_response_status_counts": response_status_counts,
        "missing_training_response_event_ids": missing_response_event_ids,
        "training_matured_forecasts": len(memory),
        "training_nonzero_qwen_scores": sum(int(row["direction_score"]) != 0 for row in forecasts),
        "excluded_events": excluded,
        "candidate_effect": effect,
        "decision": (
            "eligible_for_separately_reviewed_evaluation_stage"
            if status == "candidate_effect_eligible"
            else "stop; do not send evaluation events or run an RL comparison"
        ),
        "research_input": "train.csv plus existing BaoStock cache only; no test.csv",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    TRAINING_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_MEMORY_PATH.open("w", encoding="utf-8", newline="\n") as target:
        for row in memory:
            target.write(_canonical_json(row) + "\n")
    _atomic_json(TRAINING_EFFECT_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="write the local, no-network outbound preview")
    send_parser = subparsers.add_parser("send", help="send one event per explicit Qwen request")
    send_parser.add_argument("--split", choices=("training", "evaluation"), required=True)
    send_parser.add_argument("--max-calls", type=int, required=True)
    send_parser.add_argument("--max-budget-cny", type=float, required=True)
    send_parser.add_argument("--expected-preview-sha256", required=True)
    analyze_parser = subparsers.add_parser("analyze-training", help="compute the training-only P11 gate")
    analyze_parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    analyze_parser.add_argument("--baostock-run-dir", type=Path, default=ROOT / "data" / "raw" / "baostock" / "fetch-20260924T115050Z")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    elif args.command == "send":
        send(args.split, args.max_calls, args.max_budget_cny, args.expected_preview_sha256)
    elif args.command == "analyze-training":
        print(json.dumps(analyze_training(args.training_data_dir, args.baostock_run_dir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
