"""Locally recover only Qwen citations that exactly match supplied public evidence titles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    from prepare_agentic_rl_qwen import (
        INPUT_CNY_PER_MILLION,
        MODEL,
        OUT_DIR,
        OUTPUT_CNY_PER_MILLION,
        PAYLOADS_PATH,
        RAW_RESPONSE_DIR,
        RESPONSE_DIR,
        ROOT,
        _canonical,
        _hash,
        _response_valid,
    )
else:
    from scripts.prepare_agentic_rl_qwen import (
        INPUT_CNY_PER_MILLION,
        MODEL,
        OUT_DIR,
        OUTPUT_CNY_PER_MILLION,
        PAYLOADS_PATH,
        RAW_RESPONSE_DIR,
        RESPONSE_DIR,
        ROOT,
        _canonical,
        _hash,
        _response_valid,
    )


NORMALIZATION_VERSION = "exact_public_title_citation_to_source_type_v1"
NORMALIZED_RESPONSE_DIR = OUT_DIR / "qwen_normalized_responses"
ALLOWED_SOURCE_TYPES = {"public_announcement_title", "structured_financial_facts"}


def _event_input(event: dict[str, object]) -> dict[str, object]:
    messages = event["payload"]["messages"]
    user_messages = [message for message in messages if message.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError("request must contain exactly one user message")
    request = json.loads(user_messages[0]["content"])
    if not isinstance(request, dict) or not isinstance(request.get("public_evidence"), list):
        raise ValueError("request has no public_evidence list")
    return request


def _normalize_analysis(
    event: dict[str, object],
    raw_body: dict[str, object],
) -> tuple[dict[str, object] | None, list[dict[str, str]], str | None]:
    try:
        if raw_body.get("model") not in (None, MODEL):
            raise ValueError("provider model differs from the reviewed model")
        content = raw_body["choices"][0]["message"]["content"]
        analysis = json.loads(content)
        if not isinstance(analysis, dict):
            raise TypeError("analysis JSON root must be an object")
        request = _event_input(event)
        evidence = request["public_evidence"]
        mappings: list[dict[str, str]] = []
        for field in ("supporting_evidence", "contrary_evidence"):
            citations = analysis.get(field)
            if not isinstance(citations, list):
                continue
            normalized = []
            for citation in citations:
                if citation in ALLOWED_SOURCE_TYPES:
                    normalized.append(citation)
                    continue
                matches = [
                    source for source in evidence
                    if isinstance(source, dict)
                    and source.get("source_type") in ALLOWED_SOURCE_TYPES
                    and isinstance(source.get("source_name"), str)
                    and bool(source["source_name"])
                    and isinstance(source.get("title"), str)
                    and source["title"] == citation
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"{field} citation is not an exact unique public title match"
                    )
                source = matches[0]
                source_type = str(source["source_type"])
                normalized.append(source_type)
                mappings.append({
                    "field": field,
                    "original_reference": str(citation),
                    "canonical_source_type": source_type,
                    "input_source_name": str(source["source_name"]),
                    "input_exact_title": str(source["title"]),
                    "basis": "exact_string_match_to_one_request_public_evidence_row",
                })
            analysis[field] = normalized
        validated = _response_valid(json.dumps(analysis, ensure_ascii=False), event)
        return validated, mappings, None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return None, [], f"{type(error).__name__}: {str(error)[:500]}"


def normalize_responses(
    expected_sha256: str,
    output_dir: Path = NORMALIZED_RESPONSE_DIR,
) -> dict[str, object]:
    payload_bytes = PAYLOADS_PATH.read_bytes()
    actual_sha = _hash(payload_bytes)
    if actual_sha != expected_sha256:
        raise ValueError("Current public request manifest differs from the reviewed hash")
    events = {
        str(event["event_id"]): event
        for line in payload_bytes.decode("utf-8").splitlines()
        if line.strip()
        for event in (json.loads(line),)
    }
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite normalized responses: {output_dir}")
    output_dir.mkdir(parents=True)

    result_counts = Counter()
    normalized_by_split = Counter()
    for state_path in sorted(RESPONSE_DIR.glob("*.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "invalid_response":
            continue
        event_id = str(state["event_id"])
        event = events.get(event_id)
        if event is None or state.get("request_sha256") != event.get("request_sha256"):
            raise ValueError(f"Invalid response state does not match the reviewed request: {event_id}")
        if _hash(_canonical(event["payload"]).encode("utf-8")) != event["request_sha256"]:
            raise ValueError(f"Public request payload hash mismatch for {event_id}")
        if state.get("payload_manifest_sha256") != expected_sha256:
            raise ValueError(f"Invalid response state has a different manifest hash: {event_id}")

        record: dict[str, object] = {
            "schema_version": "agentic_rl_normalized_qwen_response.v1",
            "normalization_version": NORMALIZATION_VERSION,
            "event_id": event_id,
            "decision_date": event["prediction_date"],
            "data_split": event["data_split"],
            "request_sha256": event["request_sha256"],
            "payload_manifest_sha256": expected_sha256,
            "original_status": state["status"],
            "original_response_error_type": state.get("response_error_type"),
            "original_response_error_message": state.get("response_error_message"),
            "provider_usage": state.get("usage"),
            "actual_estimated_cost_cny": state.get("actual_estimated_cost_cny"),
            "raw_response_path": state.get("raw_response_path"),
            "raw_response_sha256": state.get("raw_response_sha256"),
            "normalization_basis": [],
            "normalized_analysis": None,
        }
        raw_path = RAW_RESPONSE_DIR / f"{event_id}.response"
        if not raw_path.is_file():
            record.update({
                "normalization_status": "unrecoverable_missing_raw",
                "normalization_error": "raw response body was not retained by the earlier sender",
            })
            result_counts["unrecoverable_missing_raw"] += 1
        else:
            raw = raw_path.read_bytes()
            raw_hash = hashlib.sha256(raw).hexdigest()
            if (
                state.get("raw_response_sha256") != raw_hash
                or state.get("raw_response_path") != raw_path.relative_to(ROOT).as_posix()
            ):
                record.update({
                    "normalization_status": "invalid_unresolved",
                    "normalization_error": "raw response hash does not match saved state",
                })
                result_counts["invalid_unresolved"] += 1
            else:
                try:
                    body = json.loads(raw.decode("utf-8"))
                    if not isinstance(body, dict):
                        raise TypeError("provider response JSON root must be an object")
                    usage = body.get("usage")
                    if not isinstance(usage, dict):
                        raise ValueError("provider usage is missing from raw response")
                    if state.get("usage") != {
                        "prompt_tokens": int(usage["prompt_tokens"]),
                        "completion_tokens": int(usage["completion_tokens"]),
                    }:
                        raise ValueError("provider usage does not match saved invalid state")
                    prompt_tokens = int(usage["prompt_tokens"])
                    completion_tokens = int(usage["completion_tokens"])
                    calculated_cost = (
                        prompt_tokens * INPUT_CNY_PER_MILLION
                        + completion_tokens * OUTPUT_CNY_PER_MILLION
                    ) / 1_000_000
                    if state.get("actual_estimated_cost_cny") is None or abs(
                        float(state["actual_estimated_cost_cny"]) - calculated_cost
                    ) > 1e-12:
                        raise ValueError("actual estimated cost does not match saved usage")
                    content = body["choices"][0]["message"]["content"]
                    analysis_body = json.loads(content)
                    if not isinstance(analysis_body, dict):
                        raise TypeError("analysis JSON root must be an object")
                    normalized, mappings, error = _normalize_analysis(event, body)
                    if error is not None:
                        record.update({
                            "normalization_status": "invalid_unresolved",
                            "normalization_error": error,
                            "normalization_basis": mappings,
                        })
                        result_counts["invalid_unresolved"] += 1
                    else:
                        record.update({
                            "normalization_status": "normalized_for_replay",
                            "normalization_error": None,
                            "normalization_basis": mappings,
                            "normalized_analysis": normalized,
                        })
                        result_counts["normalized_for_replay"] += 1
                        normalized_by_split[str(event["data_split"])] += 1
                except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    record.update({
                        "normalization_status": "invalid_unresolved",
                        "normalization_error": f"{type(error).__name__}: {str(error)[:500]}",
                    })
                    result_counts["invalid_unresolved"] += 1
        target = output_dir / f"{event_id}.json"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    summary = {
        "status": "complete",
        "normalization_version": NORMALIZATION_VERSION,
        "payload_manifest_sha256": expected_sha256,
        "invalid_state_count": sum(result_counts.values()),
        "normalization_status_counts": dict(result_counts),
        "normalized_response_count_by_split": dict(normalized_by_split),
        "output_dir": str(output_dir),
        "raw_bodies_modified": False,
        "original_response_states_modified": False,
        "provider_requests_sent": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=NORMALIZED_RESPONSE_DIR)
    args = parser.parse_args()
    print(json.dumps(normalize_responses(args.expected_sha256, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
