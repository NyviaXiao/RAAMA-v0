"""Public-evidence tool agents for the versioned research harness."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from adaptive_mas.agents.public_prompt_context import build_public_context
from adaptive_mas.agents.quantitative import quantitative_agents
from adaptive_mas.data.research import ResearchData


SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_MODEL = "qwen3.7-max-2026-06-08"
DEFAULT_ENDPOINT = "https://ws-e2j790x1jcafl4cu.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
INPUT_PRICE_CNY_PER_MILLION = 12.0
OUTPUT_PRICE_CNY_PER_MILLION = 36.0
PUBLIC_PRICE_SOURCE_URL = "https://www.alibabacloud.com/help/zh/model-studio/qwen3-7-max"
MAX_CALLS_PER_EXPERIMENT = 160
MAX_INPUT_BYTES_PER_CALL = 24_000
MAX_OUTPUT_TOKENS_PER_CALL = 1_200


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _opaque_id(namespace: str, source_id: str) -> str:
    return hashlib.sha256(f"{namespace}:{source_id}".encode("utf-8")).hexdigest()[:24]


class InvalidResearchOutput(RuntimeError):
    """A model submitted a report that failed the fixed output contract."""

    def __init__(self, role_id: str, category: str) -> None:
        super().__init__(f"{role_id} report failed fixed validation ({category})")
        self.role_id = role_id
        self.category = category
        self.partial_tool_trace: list[dict[str, str]] = []


@dataclass
class CallBudget:
    """One shared, hard call ceiling for runtime, GEPA and its reflection model."""

    max_calls: int = MAX_CALLS_PER_EXPERIMENT
    ledger_path: Path | None = None
    response_cache_dir: Path | None = None
    calls: list[dict[str, object]] = field(default_factory=list)
    logical_calls: list[dict[str, object]] = field(default_factory=list)
    unresolved_call: bool = False
    cache_hits: int = 0
    task_logical_call_start: int = 0
    task_logical_turn_limit: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_calls <= MAX_CALLS_PER_EXPERIMENT:
            raise ValueError(f"max_calls must be within 1..{MAX_CALLS_PER_EXPERIMENT}")
        if self.ledger_path is not None and self.ledger_path.is_file():
            by_index: dict[int, dict[str, object]] = {}
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["event"] == "response_cache_hit":
                    self.cache_hits += 1
                    self.logical_calls.append(self._logical_call(
                        event["usage"], event["role"], event["request_sha256"], "response_cache"
                    ))
                    continue
                index = int(event["call_index"])
                if event["event"] == "request_started":
                    by_index[index] = {key: value for key, value in event.items() if key != "event"}
                elif event["event"] in {
                    "request_completed", "request_recovered_from_cache", "request_failed_before_connect"
                }:
                    by_index[index].update({key: value for key, value in event.items() if key not in {"event", "call_index"}})
            for index in sorted(by_index):
                row = by_index[index]
                if row.get("status") == "started" and self.response_cache_dir is not None:
                    cached = self.response_cache_dir / f"{row['request_sha256']}.json"
                    if cached.is_file():
                        body = json.loads(cached.read_text(encoding="utf-8"))
                        usage = body["usage"]
                        prompt_tokens = int(usage["prompt_tokens"])
                        completion_tokens = int(usage["completion_tokens"])
                        row.update({
                            "status": "completed",
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "actual_estimated_cost_cny": (
                                prompt_tokens * INPUT_PRICE_CNY_PER_MILLION
                                + completion_tokens * OUTPUT_PRICE_CNY_PER_MILLION
                            ) / 1_000_000,
                            "elapsed_ms": None,
                            "recovered_from_exact_request_response_cache": True,
                        })
                        self._append_ledger({
                            "event": "request_recovered_from_cache",
                            **row,
                        })
                self.calls.append(row)
                if row.get("status") == "completed":
                    self.logical_calls.append(self._logical_call(
                        row,
                        str(row.get("role", "unknown")),
                        str(row.get("request_sha256", "")),
                        "provider",
                    ))
            self.unresolved_call = any(row.get("status") == "started" for row in self.calls)

    @staticmethod
    def _logical_call(
        usage: dict[str, object], role: str, request_sha256: str, source: str
    ) -> dict[str, object]:
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        logical_cost = (
            prompt_tokens * INPUT_PRICE_CNY_PER_MILLION
            + completion_tokens * OUTPUT_PRICE_CNY_PER_MILLION
        ) / 1_000_000
        return {
            "role": role,
            "request_sha256": request_sha256,
            "source": source,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "logical_estimated_cost_cny": logical_cost,
            "incremental_api_cost_cny": logical_cost if source == "provider" else 0.0,
        }

    def record_cache_hit(self, usage: dict[str, object], role: str, request_sha256: str) -> None:
        self.assert_task_turn_available()
        self.cache_hits += 1
        logical_call = self._logical_call(usage, role, request_sha256, "response_cache")
        self.logical_calls.append(logical_call)
        self._append_ledger({
            "event": "response_cache_hit",
            "request_sha256": request_sha256,
            "role": role,
            "usage": {
                "prompt_tokens": logical_call["prompt_tokens"],
                "completion_tokens": logical_call["completion_tokens"],
            },
        })

    @property
    def cost_upper_bound_cny(self) -> float:
        per_call = (
            MAX_INPUT_BYTES_PER_CALL * INPUT_PRICE_CNY_PER_MILLION
            + MAX_OUTPUT_TOKENS_PER_CALL * OUTPUT_PRICE_CNY_PER_MILLION
        ) / 1_000_000
        return self.max_calls * per_call

    @property
    def provider_call_count(self) -> int:
        return sum(row.get("status") in {"started", "completed"} for row in self.calls)

    @property
    def remaining_provider_calls(self) -> int:
        return self.max_calls - self.provider_call_count

    def _append_ledger(self, event: dict[str, object]) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as ledger:
            ledger.write(_json(event) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())

    def limit_next_task(self, max_logical_turns: int) -> None:
        if not 1 <= max_logical_turns <= 10:
            raise ValueError("one research task is limited to 1..10 logical model turns")
        self.task_logical_call_start = len(self.logical_calls)
        self.task_logical_turn_limit = max_logical_turns

    @property
    def task_logical_turn_count(self) -> int:
        return len(self.logical_calls) - self.task_logical_call_start

    @property
    def remaining_task_logical_turns(self) -> int:
        if self.task_logical_turn_limit is None:
            return 10
        return self.task_logical_turn_limit - self.task_logical_turn_count

    def assert_task_turn_available(self) -> None:
        if self.task_logical_turn_limit is not None and self.remaining_task_logical_turns <= 0:
            raise RuntimeError(
                f"research task logical model-turn budget exhausted ({self.task_logical_turn_limit})"
            )

    def begin_call(
        self,
        input_bytes: int,
        max_output_tokens: int,
        role: str,
        request_sha256: str,
    ) -> None:
        self.assert_task_turn_available()
        if input_bytes > MAX_INPUT_BYTES_PER_CALL:
            raise ValueError(
                f"serialized request is {input_bytes} bytes; call limit is {MAX_INPUT_BYTES_PER_CALL}"
            )
        if max_output_tokens > MAX_OUTPUT_TOKENS_PER_CALL:
            raise ValueError(
                f"max output tokens exceed the fixed {MAX_OUTPUT_TOKENS_PER_CALL}-token budget"
            )
        if self.provider_call_count >= self.max_calls:
            raise RuntimeError(f"shared model call budget exhausted ({self.max_calls})")
        if self.unresolved_call:
            raise RuntimeError("the prior outbound request has an unknown outcome; inspect its ledger before continuing")
        self.calls.append({
            "call_index": len(self.calls) + 1,
            "role": role,
            "input_bytes": input_bytes,
            "max_output_tokens": max_output_tokens,
            "request_sha256": request_sha256,
            "status": "started",
        })
        self._append_ledger({"event": "request_started", **self.calls[-1]})
        self.unresolved_call = True

    def finish_call(self, call_index: int, usage: dict[str, object], elapsed_ms: float) -> None:
        row = self.calls[call_index - 1]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        row.update({
            "status": "completed",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "actual_estimated_cost_cny": (
                prompt_tokens * INPUT_PRICE_CNY_PER_MILLION
                + completion_tokens * OUTPUT_PRICE_CNY_PER_MILLION
            ) / 1_000_000,
            "elapsed_ms": elapsed_ms,
        })
        self._append_ledger({
            "event": "request_completed",
            "call_index": call_index,
            **{key: value for key, value in row.items() if key != "call_index"},
        })
        self.logical_calls.append(self._logical_call(row, str(row["role"]), str(row["request_sha256"]), "provider"))
        self.unresolved_call = False

    def fail_call_before_connect(self, call_index: int) -> None:
        """Resolve a call only when the local socket permission check proves no connection was made."""
        row = self.calls[call_index - 1]
        if row.get("status") != "started" or self.response_cache_dir is None:
            raise RuntimeError("only a pending call with a response cache can be resolved as pre-connect failure")
        request_sha256 = str(row["request_sha256"])
        cache_path = self.response_cache_dir / f"{request_sha256}.json"
        marker_path = self.response_cache_dir / f"{request_sha256}.started"
        if cache_path.exists() or not marker_path.is_file():
            raise RuntimeError("pre-connect failure resolution requires a marker and no exact response cache")
        marker_path.unlink()
        row.update({
            "status": "failed_before_connect",
            "provider_request_sent": False,
            "failure_reason_code": "local_socket_permission_denied",
        })
        self._append_ledger({
            "event": "request_failed_before_connect",
            "call_index": call_index,
            **{key: value for key, value in row.items() if key != "call_index"},
        })
        self.unresolved_call = any(call.get("status") == "started" for call in self.calls)

    def summary(self) -> dict[str, object]:
        per_call_bound = self.cost_upper_bound_cny / self.max_calls
        unknown_count = sum(row.get("status") == "started" for row in self.calls)
        return {
            "call_count": self.provider_call_count,
            "attempt_count": len(self.calls),
            "maximum_call_count": self.max_calls,
            "response_cache_reuse_count": self.cache_hits,
            "logical_inference_count": len(self.logical_calls),
            "logical_token_usage": {
                "prompt_tokens": sum(int(row["prompt_tokens"]) for row in self.logical_calls),
                "completion_tokens": sum(int(row["completion_tokens"]) for row in self.logical_calls),
            },
            "logical_inference_estimated_cost_cny": sum(
                float(row["logical_estimated_cost_cny"]) for row in self.logical_calls
            ),
            "token_usage": {
                "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in self.calls),
                "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in self.calls),
            },
            "estimated_cost_cny": sum(
                float(row.get("actual_estimated_cost_cny", 0.0)) for row in self.calls
            ),
            "incremental_api_cost_cny": sum(
                float(row.get("actual_estimated_cost_cny", 0.0)) for row in self.calls
            ),
            "unknown_outcome_call_count": unknown_count,
            "unknown_outcome_cost_upper_bound_cny": unknown_count * per_call_bound,
            "cost_status": "unknown_outbound_outcome_present" if unknown_count else "usage_estimated_after_provider_response",
            "cost_upper_bound_cny": self.cost_upper_bound_cny,
            "price_basis": {
                "input_cny_per_million_tokens": INPUT_PRICE_CNY_PER_MILLION,
                "output_cny_per_million_tokens": OUTPUT_PRICE_CNY_PER_MILLION,
                "public_beijing_list_price_usd_per_million": {
                    "input": 1.65,
                    "output": 4.951,
                },
                "source_url": PUBLIC_PRICE_SOURCE_URL,
                "basis": (
                    "rounded CNY planning estimate based on the published Beijing list price; "
                    "custom endpoint deployment may differ; no cache discount applied; invoice not reconciled"
                ),
            },
            "calls": self.calls,
        }


@dataclass(frozen=True)
class QwenClient:
    endpoint: str
    api_key: str
    model: str = DEFAULT_MODEL

    @classmethod
    def from_environment(cls, api_key: str | None = None) -> "QwenClient":
        credential = api_key if api_key is not None else os.environ["POPAGENT_LLM_KEY"]
        endpoint = os.environ.get("POPAGENT_LLM_URL", DEFAULT_ENDPOINT)
        model = os.environ.get("POPAGENT_LLM_MODEL", DEFAULT_MODEL)
        return cls(endpoint.rstrip("/"), credential, model)

    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        *,
        budget: CallBudget,
        role: str,
        max_output_tokens: int = MAX_OUTPUT_TOKENS_PER_CALL,
        tool_choice: str | dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "enable_thinking": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if budget.response_cache_dir is None:
            raise ValueError("real model calls require a persistent exact-request response cache")
        request_sha256 = hashlib.sha256(
            self.endpoint.encode("utf-8") + b"\0" + serialized
        ).hexdigest()
        cache_path = budget.response_cache_dir / f"{request_sha256}.json"
        marker_path = budget.response_cache_dir / f"{request_sha256}.started"
        if cache_path.is_file():
            body = json.loads(cache_path.read_text(encoding="utf-8"))
            marker_path.unlink(missing_ok=True)
            budget.record_cache_hit(body["usage"], role, request_sha256)
            return body
        if marker_path.exists():
            raise RuntimeError("an identical outbound request has an unknown delivery outcome; automatic retry is disabled")
        call_index = len(budget.calls) + 1
        if budget.ledger_path is None:
            raise ValueError("real model calls require a persistent request ledger path")
        budget.begin_call(
            len(serialized), max_output_tokens, role,
            request_sha256,
        )
        budget.response_cache_dir.mkdir(parents=True, exist_ok=True)
        with marker_path.open("x", encoding="utf-8") as marker:
            marker.write(request_sha256)
            marker.flush()
            os.fsync(marker.fileno())
        url = self.endpoint if self.endpoint.endswith("/chat/completions") else f"{self.endpoint}/chat/completions"
        request = Request(
            url,
            data=serialized,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter_ns()
        try:
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, PermissionError) and (
                getattr(reason, "winerror", None) == 10013
                or getattr(reason, "errno", None) == 10013
            ):
                budget.fail_call_before_connect(call_index)
            raise
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        budget.response_cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_cache_path = cache_path.with_suffix(".pending")
        temporary_cache_path.write_text(_json(body), encoding="utf-8")
        temporary_cache_path.replace(cache_path)
        marker_path.unlink(missing_ok=True)
        budget.finish_call(call_index, body["usage"], elapsed_ms)
        return body


def _compact_tool_message(message: dict[str, object]) -> dict[str, object]:
    parsed = json.loads(str(message["content"]))
    if not isinstance(parsed, dict):
        raise TypeError("tool result must be a JSON object")
    keep = {
        "summary", "status", "evidence_refs", "findings", "items", "symbols",
        "evidence_id", "source_type", "title", "report_period", "facts",
        "observations", "count", "selected_source_count", "point_in_time_cutoff",
        "metric", "value", "dataset", "published_at", "available_at", "locator", "signal_group",
        "conclusion", "uncertainties", "evidence_ids", "period_start", "period_end",
    }

    def project(value: object) -> object:
        if isinstance(value, list):
            return [project(item) for item in value]
        if isinstance(value, dict):
            return {key: project(item) for key, item in value.items() if key in keep}
        return value

    return {"tool": message["name"], "result": project(parsed)}


def compact_agent_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compress completed tool steps while retaining identifiers and the latest exchange."""
    serialized_bytes = len(_json(messages).encode("utf-8"))
    if serialized_bytes <= MAX_INPUT_BYTES_PER_CALL - 5_000:
        return messages
    if len(messages) <= 2:
        raise ValueError("initial agent context exceeds the input bound")
    assistant_indices = [
        index for index, message in enumerate(messages[2:], start=2)
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    if not assistant_indices:
        raise ValueError("oversized context has no completed tool exchange to compact")
    last_exchange_start = assistant_indices[-1]
    prior_steps = []
    for message in messages[2:last_exchange_start]:
        if message.get("role") == "tool":
            prior_steps.append(_compact_tool_message(message))
        elif message.get("role") == "user":
            previous = json.loads(str(message["content"]))
            if previous.get("kind") == "completed_steps_summary":
                prior_steps.extend(previous["steps"])
    summary = {
        "kind": "completed_steps_summary",
        "steps": prior_steps,
        "notice": "Full outputs remain in the local task trace; evidence IDs can be reopened with the source tool.",
    }
    compacted = [
        *messages[:2],
        {"role": "user", "content": _json(summary)},
        *[
            {
                **message,
                "content": _json(_compact_tool_message(message)["result"]),
            }
            if message.get("role") == "tool" else message
            for message in messages[last_exchange_start:]
        ],
    ]
    if len(_json(compacted).encode("utf-8")) <= MAX_INPUT_BYTES_PER_CALL - 5_000:
        return compacted
    compacted_summary = json.loads(str(compacted[2]["content"]))
    compacted_summary["steps"] = [
        {
            "tool": step["tool"],
            "result": {
                "evidence_ids": sorted(_evidence_ids(step["result"])),
                "symbols": sorted(_symbol_codes(step["result"])),
            },
        }
        for step in compacted_summary["steps"]
    ]
    compacted[2]["content"] = _json(compacted_summary)
    if len(_json(compacted).encode("utf-8")) > MAX_INPUT_BYTES_PER_CALL - 5_000:
        raise ValueError("tool context remains over budget after reference-only compaction; reopen evidence selectively")
    return compacted


def _evidence_ids(value: object) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_evidence_ids(item) for item in value)) if value else set()
    if isinstance(value, dict):
        found = set()
        for key, item in value.items():
            if key in {"evidence_id", "evidence_ids"}:
                found.update({str(item)} if isinstance(item, str) else {str(part) for part in item})
            else:
                found.update(_evidence_ids(item))
        return found
    return set()


def _symbol_codes(value: object) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_symbol_codes(item) for item in value)) if value else set()
    if isinstance(value, dict):
        found = set()
        for key, item in value.items():
            if key == "symbol" and isinstance(item, str) and len(item) == 6 and item.isdigit():
                found.add(item)
            else:
                found.update(_symbol_codes(item))
        return found
    return set()


def _tool(
    name: str,
    description: str,
    properties: dict[str, object],
    required: list[str] | None = None,
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


SUBMIT_REPORT = _tool(
    "submit_report",
    "Finish the specialist review using only evidence IDs returned by your tools.",
    {
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "pattern": "^[0-9]{6}$"},
                    "conclusion": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "observations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_id": {"type": "string"},
                                "field": {"type": "string"},
                                "value": {},
                                "report_period": {"type": ["string", "null"]},
                            },
                            "required": ["evidence_id", "field", "value", "report_period"],
                            "additionalProperties": False,
                        },
                    },
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["symbol", "conclusion", "evidence_ids", "observations", "uncertainties"],
                "additionalProperties": False,
            },
        },
        "unresolved_items": {"type": "array", "items": {"type": "string"}},
    },
    ["summary", "findings", "unresolved_items"],
)


def _run_tool_agent(
    *,
    role_id: str,
    system_prompt: str,
    task: dict[str, object],
    tools: list[dict[str, object]],
    handlers: dict[str, object],
    client: QwenClient,
    budget: CallBudget,
    max_turns: int,
) -> dict[str, object]:
    final_tool = "finalize_research" if "finalize_research" in handlers else "submit_report"
    if final_tool not in handlers:
        raise ValueError(f"{role_id} has no declared final tool")
    if "submit_report" in handlers:
        system_prompt = "\n\n".join((
            system_prompt,
            "Submission contract: call submit_report with findings as a direct JSON array of objects, "
            "never as a string containing JSON. Provide a complete, readable report and include every "
            "supported exact observation with its source citation and report period. Keep the report "
            "focused; completeness, concision, and tool efficiency are evaluated as soft scores. "
            "Provide unresolved_items as a separate top-level array.",
        ))
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _json(task)},
    ]
    trace: list[dict[str, object]] = []
    context_compression: list[dict[str, object]] = []
    for turn_index in range(max_turns):
        before_bytes = len(_json(messages).encode("utf-8"))
        messages = compact_agent_messages(messages)
        after_bytes = len(_json(messages).encode("utf-8"))
        if after_bytes < before_bytes:
            summary_steps = 0
            reference_only = False
            if len(messages) > 2 and messages[2].get("role") == "user":
                try:
                    steps = json.loads(str(messages[2]["content"])).get("steps", [])
                    summary_steps = len(steps)
                    reference_only = bool(steps) and all(
                        set(step.get("result", {})).issubset({"evidence_ids", "symbols"})
                        for step in steps
                    )
                except (json.JSONDecodeError, TypeError, AttributeError):
                    summary_steps = 0
            context_compression.append({
                "turn": turn_index + 1,
                "strategy": "reference_only" if reference_only else "completed_steps_summary",
                "before_bytes": before_bytes,
                "after_bytes": after_bytes,
                "compressed_tool_steps": summary_steps,
            })
        if turn_index == max_turns - 1:
            messages[0] = {
                **messages[0],
                "content": (
                    str(messages[0]["content"])
                    + f"\n\nFinal permitted turn ({turn_index + 1}/{max_turns}): call {final_tool} now."
                ),
            }
        final_tool_choice = (
            {"type": "function", "function": {"name": final_tool}}
            if turn_index == max_turns - 1 else None
        )
        response = client.complete(
            messages, tools, budget=budget, role=role_id, tool_choice=final_tool_choice
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            raise ValueError(f"{role_id} must finish by calling its submit_report/finalize tool")
        assistant_message = {"role": "assistant", **message}
        messages.append(assistant_message)
        for call in tool_calls:
            function = call["function"]
            name = str(function["name"])
            if name not in handlers:
                raise ValueError(f"{role_id} requested undeclared tool {name!r}")
            arguments = json.loads(function["arguments"])
            try:
                result = handlers[name](arguments)
            except InvalidResearchOutput as exc:
                exc.context_compression = list(context_compression)
                exc.partial_tool_trace = [
                    {"role": role_id, "tool": item["tool"], "status": "completed"}
                    for item in trace
                ] + [{"role": role_id, "tool": name, "status": "rejected"}]
                raise
            trace.append({
                "tool": name,
                "arguments": arguments,
                "result": result,
                "tool_call_id": call["id"],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": _json(result),
            })
            if name in {"submit_report", "finalize_research"}:
                return {
                    "status": "completed",
                    "role_id": role_id,
                    "report": result["accepted_report"],
                    "tool_trace": trace,
                    "context_compression": context_compression,
                    "turn_count": turn_index + 1,
                    "model": client.model,
                }
    raise RuntimeError(f"{role_id} exceeded its {max_turns}-turn tool-calling limit")


def _market_handlers(
    research: ResearchData,
    decision_date: date,
    config: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, dict[str, object]]]:
    snapshot = research.snapshot_at(decision_date)
    signals = quantitative_agents(
        snapshot,
        int(config.get("feature_trend_sessions", 20)),
        int(config.get("feature_reversal_sessions", 5)),
        int(config.get("feature_liquidity_return_sessions", 5)),
        int(config.get("feature_recent_amount_sessions", 5)),
        int(config.get("feature_reference_amount_sessions", 20)),
        horizon_sessions=int(config.get("prediction_horizon_sessions", 3)),
    )
    signal_groups = {
        "trend": ("read_trend_signals", signals["trend"]),
        "reversal": ("read_reversal_signals", signals["reversal"]),
        "volume_price": ("read_volume_price_signals", signals["liquidity_confirmation"]),
    }
    evidence_catalog: dict[str, dict[str, object]] = {}
    handlers = {}

    def read_market_signals(arguments: dict[str, object]) -> dict[str, object]:
        limit = int(arguments.get("limit_per_signal", 6))
        if not 1 <= limit <= 6:
            raise ValueError("market signal limit must be within 1..6 per signal group")
        symbols = arguments.get("symbols", [])
        if not isinstance(symbols, list) or any(
            len(str(symbol)) != 6 or not str(symbol).isdigit() for symbol in symbols
        ):
            raise ValueError("symbols must be a list of six-digit security codes")
        findings = []
        for group, (source_tool, rows) in signal_groups.items():
            eligible = [row for row in rows if not symbols or row.symbol in symbols]
            eligible.sort(key=lambda row: (-abs(float(row.score)), row.symbol))
            for signal in eligible[:limit]:
                evidence_id = _opaque_id(
                    source_tool,
                    f"{signal.symbol}:{decision_date.isoformat()}:{signal.evidence.feature}:{signal.evidence.start_date}:{signal.evidence.end_date}",
                )
                evidence_catalog[evidence_id] = {
                    "evidence_id": evidence_id,
                    "source_record_id": f"market:{evidence_id}",
                    "symbol": signal.symbol,
                    "source_type": "public_market_data",
                    "dataset": "BaoStock_adjusted_market_bars",
                    "published_at": decision_date.isoformat(),
                    "available_at": signal.input_cutoff.isoformat(),
                    "period_start": signal.evidence.start_date.isoformat(),
                    "period_end": signal.evidence.end_date.isoformat(),
                    "metric": signal.evidence.feature,
                    "value": float(signal.score),
                    "source_refs_local_only": list(signal.evidence.source_refs),
                    "vintage_verified": True,
                }
                findings.append({
                    "signal_group": group,
                    "symbol": signal.symbol,
                    "metric": signal.evidence.feature,
                    "value": round(float(signal.score), 10),
                    "period_start": signal.evidence.start_date.isoformat(),
                    "period_end": signal.evidence.end_date.isoformat(),
                    "evidence_id": evidence_id,
                    "dataset": "BaoStock adjusted daily market bars",
                })
        return {"count": len(findings), "findings": findings}

    handlers["read_market_signals"] = read_market_signals
    tools = [_tool(
        "read_market_signals",
        "Read bounded point-in-time trend, reversal, and volume-price evidence for the requested symbols.",
        {
            "symbols": {"type": "array", "items": {"type": "string", "pattern": "^[0-9]{6}$"}, "maxItems": 20},
            "limit_per_signal": {"type": "integer", "minimum": 1, "maximum": 6},
        },
    )]
    return tools, handlers, evidence_catalog


def _fundamental_handlers(
    public_snapshot: dict[str, object] | None,
    decision_time: datetime,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, dict[str, object]]]:
    snapshot = public_snapshot or {}
    source_records: dict[str, dict[str, object]] = {}
    source_ids: dict[str, str] = {}
    public_sources: list[dict[str, object]] = []
    source_rows = list(snapshot.get("p9_outputs", []))
    for record in source_rows:
        if record.get("expert") not in {
            "fundamental_rule_v1", "announcement_title_rule_v1", "cninfo_full_text_llm_v1",
        }:
            continue
        if not record.get("symbol"):
            continue
        symbol = str(record["symbol"]).rsplit(".", 1)[-1]
        evidence = list(record.get("evidence", []))
        for source in evidence:
            raw_source_id = str(source["source_record_id"])
            if raw_source_id in source_ids.values():
                continue
            evidence_id = _opaque_id("public", raw_source_id)
            source_ids[evidence_id] = raw_source_id
            safe_facts = {
                key: source[key]
                for key in (
                    "revenue_yoy_pct", "parent_net_profit_yoy_pct", "roe_pct",
                    "income_statement_revenue_yoy_pct", "income_statement_parent_profit_yoy_pct",
                    "revenue_cny", "parent_net_profit_cny",
                )
                if key in source
            }
            title = str(source.get("title", ""))
            full_text = str(source.get("full_text", ""))
            extracted_text_ref = source.get("extracted_text_ref")
            is_announcement = "announcement" in str(source.get("dataset", ""))
            is_title_only = is_announcement and not full_text and not extracted_text_ref
            public_source = {
                "source_type": (
                    "public_announcement_body" if full_text else
                    "public_announcement_title" if is_announcement else
                    "structured_financial_facts"
                ),
                "symbol": symbol,
                "is_public": True,
                "source_record_id": evidence_id,
                "title": title or None,
                **({"body_text": full_text} if full_text else {}),
                "facts": safe_facts,
                "report_period": source.get("report_period"),
                "published_at": source.get("published_at"),
                "available_at": source.get("available_at"),
                "fetched_at": source.get("fetched_at"),
                "vintage_status": source.get("vintage_status"),
                "retrospective_backfill_proxy": bool(source.get("retrospective_backfill_proxy", False)),
                "vintage_verified": bool(source.get("vintage_verified", False)),
                "content_scope": (
                    "full_text_available" if extracted_text_ref and not full_text else
                    "full_text_excerpt" if full_text else
                    "title_only" if is_title_only else "structured_financial_fields"
                ),
            }
            public_sources.append(public_source)
            source_records[evidence_id] = {
                **public_source,
                "source_record_id": raw_source_id,
                "dataset": source.get("dataset"),
                "available_at": source.get("available_at"),
                "published_at": source.get("published_at"),
                "report_period": source.get("report_period"),
                "facts": safe_facts,
                "title": title,
                "full_text": full_text,
                "extracted_text_ref": extracted_text_ref,
            }

    evidence_catalog = {
        evidence_id: {
            "evidence_id": evidence_id,
            "source_record_id": source_ids[evidence_id],
            "symbol": str(row["symbol"]),
            "source_type": str(row["source_type"]),
            "dataset": str(row.get("dataset", "")),
            "title": str(row.get("title", "")) or None,
            "report_period": row.get("report_period"),
            "published_at": row.get("published_at"),
            "available_at": row.get("available_at"),
            "fetched_at": row.get("fetched_at"),
            "vintage_status": row.get("vintage_status"),
            "retrospective_backfill_proxy": bool(row.get("retrospective_backfill_proxy", False)),
            "facts": row.get("facts", {}),
            "vintage_verified": bool(row.get("vintage_verified", False)),
            "content_scope": row.get("content_scope"),
            "full_text_local_only": bool(row.get("full_text") or row.get("extracted_text_ref")),
        }
        for evidence_id, row in source_records.items()
    }

    def search_sources(arguments: dict[str, object]) -> dict[str, object]:
        question = str(arguments.get("question", "财报与公告的重要证据及其报告期"))
        symbols = arguments.get("symbols", [])
        limit = int(arguments.get("limit", 8))
        if not 1 <= limit <= 4:
            raise ValueError("fundamental search limit must be within 1..4")
        if not isinstance(symbols, list) or any(
            len(str(symbol)) != 6 or not str(symbol).isdigit() for symbol in symbols
        ):
            raise ValueError("symbols must be a list of six-digit security codes")
        selected = [row for row in public_sources if not symbols or row["symbol"] in symbols]
        selected.sort(key=lambda row: (
            str(row.get("available_at", "")),
            str(row.get("title") or ""),
        ), reverse=True)
        contexts = []
        references = []
        for symbol in sorted({str(row["symbol"]) for row in selected})[:limit]:
            context = build_public_context(
                decision_time=decision_time,
                symbol=symbol,
                question=question,
                sources=[row for row in selected if row["symbol"] == symbol],
                role="fundamental",
                max_sources=2,
                max_context_chars=1_200,
                max_context_token_upper_bound=4_000,
                output_token_reserve=400,
            )
            contexts.append(context["messages"][-1])
            for reference in context["local_reference_map"]:
                references.append({
                    "evidence_id": reference["source_record_ids"][0],
                    "source_type": reference["source_type"],
                    "title": reference["exact_title"],
                    "locator": reference["locator"],
                })
        return {
            "context_messages": contexts,
            "evidence_refs": references,
            "selected_source_count": len(references),
            "point_in_time_cutoff": decision_time.isoformat(),
        }

    def read_source(arguments: dict[str, object]) -> dict[str, object]:
        evidence_id = str(arguments["evidence_id"])
        source = source_records[evidence_id]
        available_at = datetime.fromisoformat(str(source["available_at"]))
        if available_at > decision_time:
            raise ValueError("source is unavailable at the requested decision time")
        offset = int(arguments.get("offset", 0))
        length = int(arguments.get("length", 1_200))
        if offset < 0 or not 1 <= length <= 1_600:
            raise ValueError("read offset must be nonnegative and length within 1..1600")
        text = source["full_text"]
        if source.get("extracted_text_ref"):
            text = Path(str(source["extracted_text_ref"])).read_text(encoding="utf-8")
        text = str(text)
        if source.get("content_scope") == "title_only" and not text:
            return {
                "evidence_id": evidence_id,
                "symbol": source["symbol"],
                "title": source.get("title"),
                "report_period": source.get("report_period"),
                "published_at": source.get("published_at"),
                "available_at": source.get("available_at"),
                "fetched_at": source.get("fetched_at"),
                "vintage_status": source.get("vintage_status"),
                "retrospective_backfill_proxy": bool(source.get("retrospective_backfill_proxy", False)),
                "content_scope": "title_only",
                "body_status": "title_only_body_unavailable_no_fetch_performed",
            }
        excerpt = text[offset:offset + length]
        paragraph_start = text.rfind("\n", 0, offset) + 1
        paragraph_end = text.find("\n", offset)
        return {
            "evidence_id": evidence_id,
            "symbol": source["symbol"],
            "source_type": source["source_type"],
            "dataset": source.get("dataset"),
            "title": source.get("title"),
            "report_period": source.get("report_period"),
            "published_at": source.get("published_at"),
            "available_at": source.get("available_at"),
            "fetched_at": source.get("fetched_at"),
            "vintage_status": source.get("vintage_status"),
            "retrospective_backfill_proxy": bool(source.get("retrospective_backfill_proxy", False)),
            "facts": source.get("facts", {}),
            "text_excerpt": excerpt,
            "offset": offset,
            "next_offset": offset + len(excerpt) if offset + len(excerpt) < len(text) else None,
            "location": {"character_offset": offset, "paragraph_start": paragraph_start, "paragraph_end": paragraph_end if paragraph_end >= 0 else len(text)},
            "content_scope": source.get("content_scope"),
        }

    tools = [
        _tool(
            "search_financial_facts_and_announcements",
            "Search only locally cached public financial facts and announcements available by the decision time.",
            {
                "question": {"type": "string", "maxLength": 200},
                "symbols": {"type": "array", "items": {"type": "string", "pattern": "^[0-9]{6}$"}, "maxItems": 12},
                "limit": {"type": "integer", "minimum": 1, "maximum": 4},
            },
        ),
        _tool(
            "read_cached_source",
            "Read a cited cached public financial or announcement source on demand.",
            {
                "evidence_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "length": {"type": "integer", "minimum": 1, "maximum": 1_600},
            },
            ["evidence_id"],
        ),
        SUBMIT_REPORT,
    ]
    return tools, {
        "search_financial_facts_and_announcements": search_sources,
        "read_cached_source": read_source,
    }, evidence_catalog


def _validate_report(
    report: dict[str, object],
    catalog: dict[str, dict[str, object]],
    role_id: str,
) -> dict[str, object]:
    allowed_symbols = {str(row["symbol"]) for row in catalog.values()}
    findings_input = report["findings"]
    if isinstance(findings_input, str):
        findings_input = json.loads(findings_input)
    if not isinstance(findings_input, list):
        raise TypeError(f"{role_id} findings must be a JSON array")
    if not isinstance(report["summary"], str):
        raise TypeError(f"{role_id} summary must be a string")
    unresolved_items = report["unresolved_items"]
    if not isinstance(unresolved_items, list) or any(
        not isinstance(item, str) for item in unresolved_items
    ):
        raise TypeError(f"{role_id} unresolved_items must be an array of strings")
    findings = []
    for finding in findings_input:
        if not isinstance(finding, dict):
            raise TypeError(f"{role_id} finding entries must be JSON objects")
        if not isinstance(finding.get("observations"), list):
            raise TypeError(f"{role_id} finding observations must be an array")
        if not isinstance(finding.get("symbol"), str) or not isinstance(finding.get("conclusion"), str):
            raise TypeError(f"{role_id} finding symbol and conclusion must be strings")
        symbol = finding["symbol"]
        evidence_ids = finding["evidence_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids or any(
            not isinstance(value, str) for value in evidence_ids
        ):
            raise TypeError(f"{role_id} evidence_ids must be a non-empty array of strings")
        uncertainties = finding.get("uncertainties")
        if not isinstance(uncertainties, list) or any(
            not isinstance(value, str) for value in uncertainties
        ):
            raise TypeError(f"{role_id} uncertainties must be an array of strings")
        if symbol not in allowed_symbols:
            raise ValueError(f"{role_id} reported symbol {symbol} without supplied evidence")
        if not evidence_ids or any(
            evidence_id not in catalog or str(catalog[evidence_id]["symbol"]) != symbol
            for evidence_id in evidence_ids
        ):
            raise ValueError(f"{role_id} cited evidence that was not supplied for {symbol}")
        observations = []
        for observation in finding.get("observations", []):
            if not isinstance(observation, dict) or not isinstance(observation.get("evidence_id"), str):
                raise TypeError(f"{role_id} observations must be objects with a string evidence_id")
            evidence_id = observation["evidence_id"]
            if evidence_id not in evidence_ids:
                raise ValueError(f"{role_id} observation is missing its finding citation")
            observations.append(dict(observation))
        findings.append({
            **finding,
            "agent_id": role_id,
            "evidence": [catalog[evidence_id] for evidence_id in evidence_ids],
            "observations": observations,
        })
    return {
        "agent_id": role_id,
        "summary": report["summary"],
        "findings": findings,
        "unresolved_items": unresolved_items,
        "source_record_ids": sorted({
            str(source["source_record_id"])
            for finding in findings for source in finding["evidence"]
        }),
    }


def _accept_report(
    arguments: dict[str, object],
    catalog: dict[str, dict[str, object]],
    role_id: str,
) -> dict[str, object]:
    normalized_fields = ["findings"] if isinstance(arguments.get("findings"), str) else []
    try:
        accepted_report = _validate_report(arguments, catalog, role_id)
    except (TypeError, ValueError) as exc:
        message = str(exc).lower()
        category = "report_field_limit" if "limit" in message or "exceeds" in message else "report_schema_or_evidence"
        raise InvalidResearchOutput(role_id, category) from None
    return {
        "accepted_report": accepted_report,
        "accepted": True,
        "normalized_input_fields": normalized_fields,
    }


def run_research_harness(
    research: ResearchData,
    as_of: date,
    public_snapshot: dict[str, object] | None,
    candidate: dict[str, str],
    client: QwenClient,
    budget: CallBudget,
    config: dict[str, object] | None = None,
    candidate_version_id: str = "seed",
    required_observations: list[dict[str, object]] | None = None,
    procedural_memory: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run market and fundamental specialists, then a planning/query/finalize manager."""
    decision_time = datetime.combine(as_of, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI)
    settings = config or {}
    market_tools, market_handlers, market_catalog = _market_handlers(research, as_of, settings)
    market_finalized: dict[str, object] = {}

    def submit_market(arguments: dict[str, object]) -> dict[str, object]:
        accepted = _accept_report(arguments, market_catalog, "market")
        report = accepted["accepted_report"]
        market_finalized["report"] = report
        return accepted

    market_handlers["submit_report"] = submit_market
    market_tools.append(SUBMIT_REPORT)
    market_task = {
        "as_of": as_of.isoformat(),
        "question": "识别缓存行情中最值得进一步核实的趋势、反转与量价证据。",
        "decision_time": decision_time.isoformat(),
        "allowed_data": "point-in-time adjusted BaoStock market bars and derived market-tool results only",
    }
    market_result = _run_tool_agent(
        role_id="market",
        system_prompt="\n\n".join((candidate["market_prompt"], candidate["context_policy"], candidate["procedural_experience"])),
        task=market_task,
        tools=market_tools,
        handlers=market_handlers,
        client=client,
        budget=budget,
        max_turns=2,
    )
    market_report = market_result["report"]

    fundamental_tools, fundamental_handlers, fundamental_catalog = _fundamental_handlers(
        public_snapshot, decision_time
    )
    fundamental_finalized: dict[str, object] = {}

    def submit_fundamental(arguments: dict[str, object]) -> dict[str, object]:
        accepted = _accept_report(arguments, fundamental_catalog, "fundamental")
        report = accepted["accepted_report"]
        fundamental_finalized["report"] = report
        return accepted

    fundamental_handlers["submit_report"] = submit_fundamental
    fundamental_task = {
        "as_of": as_of.isoformat(),
        "question": "从已归档财报、报告期和公告中找出有来源的基本面证据与未知项。",
        "decision_time": decision_time.isoformat(),
        "public_evidence_mode": (public_snapshot or {}).get("data_mode", "unknown"),
        "mode_limitations": (
            [
                "This is a retrospective source-publication-date proxy. Some cached vendor responses were fetched after as_of.",
                "Each source preserves its actual fetched_at and vintage_status; vendor historical revisions are not verified.",
                "Do not describe this result as strict historical point-in-time evidence.",
            ]
            if (public_snapshot or {}).get("data_mode") == "retrospective_source_publication_proxy" else []
        ),
        "allowed_data": "locally cached public financial statements, indicators, announcement titles and previously extracted text only",
        "required_research": {
            "observations": [
                {key: row[key] for key in ("symbol", "field", "report_period") if key in row}
                for row in (required_observations or [])
            ],
            "instruction": (
                "For each listed symbol and field, report the exact public value and report period when a cited cached source supports it; "
                "if no source supports an item, leave it unresolved instead of guessing."
            ),
        },
    }
    try:
        fundamental_result = _run_tool_agent(
            role_id="fundamental",
            system_prompt="\n\n".join((candidate["fundamental_prompt"], candidate["context_policy"], candidate["procedural_experience"])),
            task=fundamental_task,
            tools=fundamental_tools,
            handlers=fundamental_handlers,
            client=client,
            budget=budget,
            max_turns=4,
        )
    except InvalidResearchOutput as exc:
        exc.partial_tool_trace = [
            {"role": "market", "tool": item["tool"], "status": "completed"}
            for item in market_result["tool_trace"]
        ] + exc.partial_tool_trace
        raise
    fundamental_report = fundamental_result["report"]

    evidence_catalog = {**market_catalog, **fundamental_catalog}
    all_reports = {"market": market_report, "fundamental": fundamental_report}
    manager_state: dict[str, object] = {
        "query_count": 0,
        "query_attempted": False,
        "final": None,
        "directed_queries": [],
    }

    def query_specialist(arguments: dict[str, object]) -> dict[str, object]:
        if manager_state.get("plan") is None:
            raise ValueError("research manager must create a plan before asking a specialist")
        if manager_state["query_attempted"]:
            raise ValueError("research manager may make at most one directed query per task")
        role = str(arguments["role"])
        if role not in {"market", "fundamental"}:
            raise ValueError("directed query role must be market or fundamental")
        symbol = str(arguments["symbol"])
        if len(symbol) != 6 or not symbol.isdigit():
            raise ValueError("directed query symbol must be six digits")
        question = str(arguments["question"])
        if not question.strip() or len(question) > 240:
            raise ValueError("directed query must be 1..240 characters")
        plan = manager_state["plan"]
        if (role, symbol, question) != (
            plan["target_role"], plan["target_symbol"], plan["question"]
        ):
            raise ValueError("directed query must match the research manager's declared plan")
        manager_state["query_attempted"] = True
        remaining_task_turns = budget.remaining_task_logical_turns
        if min(budget.remaining_provider_calls, remaining_task_turns) < 3:
            return {
                "role": role,
                "status": "skipped_budget_reserve",
                "reason": "fewer than two specialist calls plus the manager finalization call remain",
            }
        if role == "market":
            tools = market_tools
            handlers = market_handlers
            catalog = market_catalog
            prompt = candidate["market_prompt"]
        else:
            tools = fundamental_tools
            handlers = fundamental_handlers
            catalog = fundamental_catalog
            prompt = candidate["fundamental_prompt"]
        prior_handoff = {
            "summary": all_reports[role]["summary"],
            "findings": [
                {
                    "symbol": finding["symbol"],
                    "conclusion": finding["conclusion"],
                    "evidence_ids": finding["evidence_ids"],
                    "observations": finding["observations"],
                    "uncertainties": finding["uncertainties"],
                }
                for finding in all_reports[role]["findings"]
                if finding["symbol"] == symbol
            ],
        }

        def read_prior_evidence(evidence_arguments: dict[str, object]) -> dict[str, object]:
            if int(manager_state.get("prior_evidence_read_count", 0)) >= 1:
                raise ValueError("directed specialist may reopen at most one source")
            evidence_id = str(evidence_arguments["evidence_id"])
            handed_off_ids = {
                str(value)
                for finding in prior_handoff["findings"]
                for value in finding["evidence_ids"]
            }
            if evidence_id not in handed_off_ids:
                raise ValueError("directed specialist may reopen only evidence handed off for its planned symbol")
            manager_state["prior_evidence_read_count"] = int(manager_state.get("prior_evidence_read_count", 0)) + 1
            if role == "fundamental":
                return fundamental_handlers["read_cached_source"]({"evidence_id": evidence_id})
            source = catalog[evidence_id]
            return {
                key: source[key]
                for key in (
                    "evidence_id", "symbol", "source_type", "dataset", "published_at",
                    "available_at", "fetched_at", "vintage_status",
                    "retrospective_backfill_proxy", "period_start", "period_end", "metric", "value",
                    "vintage_verified",
                )
                if key in source
            }

        def submit_targeted(arguments: dict[str, object]) -> dict[str, object]:
            return _accept_report(arguments, catalog, f"{role}_directed_query")

        query_result = _run_tool_agent(
            role_id=f"{role}_directed_query",
            system_prompt="\n\n".join((
                prompt,
                candidate["context_policy"],
                candidate["procedural_experience"],
                "Answer the manager's one targeted question from the supplied public handoff and reopened evidence. "
                "If one cited source needs verification, call read_prior_evidence once before submitting. Cite only supplied evidence IDs.",
            )),
            task={
                "as_of": as_of.isoformat(),
                "symbol": symbol,
                "question": question,
                "decision_time": decision_time.isoformat(),
                "prior_public_handoff": prior_handoff,
            },
            tools=[
                _tool(
                    "read_prior_evidence",
                    "Reopen one public source already handed off for the planned symbol.",
                    {"evidence_id": {"type": "string"}},
                    ["evidence_id"],
                ),
                SUBMIT_REPORT,
            ],
            handlers={"read_prior_evidence": read_prior_evidence, "submit_report": submit_targeted},
            client=client,
            budget=budget,
            max_turns=2,
        )
        report = query_result["report"]
        manager_state["directed_queries"].append({
            "role_id": f"{role}_directed_query",
            "tool_trace": [
                {"tool": item["tool"], "status": "completed"}
                for item in query_result["tool_trace"]
            ],
            "context_compression": query_result["context_compression"],
            "turn_count": query_result["turn_count"],
        })
        manager_state["query_count"] = int(manager_state["query_count"]) + 1
        all_reports[f"{role}_directed_query"] = report
        return {
            "role": role,
            "report": {
                "agent_id": report["agent_id"],
                "summary": report["summary"],
                "findings": [
                    {
                        "symbol": finding["symbol"],
                        "conclusion": finding["conclusion"],
                        "evidence_ids": finding["evidence_ids"],
                        "observations": finding["observations"],
                        "uncertainties": finding["uncertainties"],
                    }
                    for finding in report["findings"]
                ],
                "unresolved_items": report["unresolved_items"],
            },
        }

    def plan_research(arguments: dict[str, object]) -> dict[str, object]:
        if manager_state.get("plan") is not None:
            raise ValueError("research plan can be created only once")
        role = str(arguments["target_role"])
        symbol = str(arguments["target_symbol"])
        question = str(arguments["question"])
        if role not in {"market", "fundamental"}:
            raise ValueError("plan target_role must be market or fundamental")
        if len(symbol) != 6 or not symbol.isdigit():
            raise ValueError("plan target_symbol must be six digits")
        if not question.strip() or len(question) > 240:
            raise ValueError("plan question must be 1..240 characters")
        manager_state["plan"] = {
            "target_role": role,
            "target_symbol": symbol,
            "question": question,
        }
        return {"accepted": True, "plan": manager_state["plan"], "query_limit": 1}

    def finalize(arguments: dict[str, object]) -> dict[str, object]:
        if manager_state.get("plan") is None:
            raise ValueError("research manager must create a plan before finalizing")
        no_query_reason = str(arguments["no_query_reason"])
        if manager_state["query_count"] == 0 and len(no_query_reason.strip()) < 8:
            raise ValueError("when no specialist query is made, explain the evidence-based stopping reason")
        all_symbols = {
            str(finding["symbol"])
            for report in all_reports.values()
            for finding in report["findings"]
        }
        seen = set()
        priorities = []
        for row in arguments["priority_symbols"]:
            symbol = str(row["symbol"])
            evidence_ids = [str(value) for value in row["evidence_ids"]]
            if symbol not in all_symbols or symbol in seen:
                raise ValueError("priority symbols must be unique and supported by specialist reports")
            if not evidence_ids or any(
                evidence_id not in evidence_catalog
                or str(evidence_catalog[evidence_id]["symbol"]) != symbol
                for evidence_id in evidence_ids
            ):
                raise ValueError("priority symbol citations must resolve to public evidence for that symbol")
            seen.add(symbol)
            priorities.append({"symbol": symbol, "evidence_ids": evidence_ids})
        if len(priorities) > 12:
            raise ValueError("research manager priority list exceeds the fixed 12-symbol limit")
        manager_state["final"] = {
            "summary": str(arguments["summary"]),
            "priority_symbols": priorities,
            "unresolved_items": list(arguments["unresolved_items"]),
            "no_query_reason": no_query_reason,
            "research_plan": manager_state["plan"],
        }
        return {"accepted_report": manager_state["final"], "accepted": True}

    manager_tools = [
        _tool(
            "plan_research",
            "Plan the research focus before making a directed query or closing the task.",
            {
                "target_role": {"type": "string", "enum": ["market", "fundamental"]},
                "target_symbol": {"type": "string", "pattern": "^[0-9]{6}$"},
                "question": {"type": "string", "maxLength": 240},
            },
            ["target_role", "target_symbol", "question"],
        ),
        _tool(
            "query_specialist",
            "Ask one specialist a targeted question about one cited symbol; one query is allowed.",
            {
                "role": {"type": "string", "enum": ["market", "fundamental"]},
                "symbol": {"type": "string", "pattern": "^[0-9]{6}$"},
                "question": {"type": "string", "maxLength": 240},
            },
            ["role", "symbol", "question"],
        ),
        _tool(
            "finalize_research",
            "Close the research task and rank only symbols supported by cited specialist evidence.",
            {
                "summary": {"type": "string", "maxLength": 700},
            "priority_symbols": {
                    "type": "array", "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "pattern": "^[0-9]{6}$"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
                        },
                        "required": ["symbol", "evidence_ids"],
                        "additionalProperties": False,
                    },
                },
                "unresolved_items": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                "no_query_reason": {"type": "string", "maxLength": 180},
            },
            ["summary", "priority_symbols", "unresolved_items", "no_query_reason"],
        ),
    ]
    manager_handlers = {
        "plan_research": plan_research,
        "query_specialist": query_specialist,
        "finalize_research": finalize,
    }
    manager_task = {
        "as_of": as_of.isoformat(),
        "decision_time": decision_time.isoformat(),
        "public_evidence_mode": (public_snapshot or {}).get("data_mode", "unknown"),
        "specialist_handoffs": {
            role: {
                "summary": report["summary"],
                "findings": [
                    {
                        "symbol": finding["symbol"],
                        "conclusion": finding["conclusion"],
                        "evidence_ids": finding["evidence_ids"],
                        "observations": finding["observations"],
                        "uncertainties": finding["uncertainties"],
                    }
                    for finding in report["findings"]
                ],
                "unresolved_items": report["unresolved_items"],
            }
            for role, report in all_reports.items()
        },
        "constraints": {
            "research_priorities_are_a_candidate_filter_only": True,
            "expected_returns_are_never_generated_by_the_agents": True,
            "account_state_and_internal_forecasts_are_not_in_scope": True,
        },
    }
    try:
        manager_result = _run_tool_agent(
            role_id="research_manager",
            system_prompt="\n\n".join((candidate["manager_prompt"], candidate["handoff_schema"], candidate["procedural_experience"])),
            task=manager_task,
            tools=manager_tools,
            handlers=manager_handlers,
            client=client,
            budget=budget,
            max_turns=3,
        )
    except InvalidResearchOutput as exc:
        exc.partial_tool_trace = [
            {"role": "market", "tool": item["tool"], "status": "completed"}
            for item in market_result["tool_trace"]
        ] + [
            {"role": "fundamental", "tool": item["tool"], "status": "completed"}
            for item in fundamental_result["tool_trace"]
        ] + exc.partial_tool_trace
        raise

    return {
        "schema_version": "adaptive-mas.harness-research.v1",
        "status": "completed",
        "public_evidence_mode": (public_snapshot or {}).get("data_mode", "unknown"),
        "as_of": as_of.isoformat(),
        "decision_time": decision_time.isoformat(),
        "agents": {
            "market": {
                "report": market_report,
                "tool_trace": market_result["tool_trace"],
                "context_compression": market_result["context_compression"],
                "turn_count": market_result["turn_count"],
            },
            "fundamental": {
                "report": fundamental_report,
                "tool_trace": fundamental_result["tool_trace"],
                "context_compression": fundamental_result["context_compression"],
                "turn_count": fundamental_result["turn_count"],
            },
            "research_manager": {
                "report": manager_result["report"],
                "tool_trace": manager_result["tool_trace"],
                "context_compression": manager_result["context_compression"],
                "turn_count": manager_result["turn_count"],
                "directed_queries": manager_state["directed_queries"],
            },
        },
        "evidence_catalog": evidence_catalog,
        "memory": {
            "working_memory": {
                "scope": "single_task",
                "handoff_summary": manager_task["specialist_handoffs"],
                "source_ids": sorted(evidence_catalog),
            },
            "procedural_experience_version": candidate_version_id,
            "procedural_experience_sha256": hashlib.sha256(
                candidate["procedural_experience"].encode("utf-8")
            ).hexdigest(),
            "procedural_memory": procedural_memory or {
                "status": "unversioned_candidate",
                "loaded_from_persistent_store": False,
            },
        },
        "cost": budget.summary(),
        "data_limits": {
            "public_evidence_mode": (public_snapshot or {}).get("data_mode", "unknown"),
            "strict_historical_pit_verified": bool((public_snapshot or {}).get("strict_historical_pit_verified", False)),
            "retrospective_capture_lag_record_count": int((public_snapshot or {}).get("retrospective_capture_lag_record_count", 0)),
            "historical_cases_are_retrospective": True,
            "market_source": "pre-existing BaoStock adjusted market cache",
            "fundamental_source": "pre-existing local financial/announcement cache",
            "announcement_fetching": "disabled; only previously captured text may be read",
            "private_account_or_forecast_fields_sent": False,
        },
    }
