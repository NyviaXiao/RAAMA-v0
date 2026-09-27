"""Small OpenAI-compatible client for the configured Qwen endpoint."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")
QWEN_MODEL = "qwen3.7-plus"


def call_qwen_expert(
    p9_outputs: list[dict[str, object]],
    decision_time: datetime,
    evaluation_horizon_sessions: int,
    api_key: str | None = None,
    max_records: int = 32,
) -> dict[str, object]:
    """Analyze a small, deterministic set of P9 records when a key is present."""
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise ValueError("decision_time 必须带时区")
    if max_records < 1:
        raise ValueError("max_records 必须为正数")
    if evaluation_horizon_sessions < 1:
        raise ValueError("evaluation_horizon_sessions 必须为正整数")
    credential = api_key if api_key is not None else os.environ.get("POPAGENT_LLM_KEY")
    endpoint = os.environ.get("POPAGENT_LLM_URL")
    model = os.environ.get("POPAGENT_LLM_MODEL", QWEN_MODEL)
    if not credential:
        return {
            "status": "not_run_missing_POPAGENT_LLM_KEY",
            "model": model,
            "endpoint": endpoint,
            "outputs": [],
            "usage": None,
            "estimated_token_cost_cny": None,
            "elapsed_ms": 0.0,
            "completed_at": None,
        }

    selected = sorted(
        p9_outputs,
        key=lambda row: (
            row["expert"] == "cninfo_full_text_llm_v1",
            str(row["event_time"]),
            str(row["record_id"]),
        ),
        reverse=True,
    )[:max_records]
    if not selected:
        return {
            "status": "not_run_no_new_records",
            "model": model,
            "endpoint": endpoint,
            "outputs": [],
            "usage": None,
            "estimated_token_cost_cny": 0.0,
            "elapsed_ms": 0.0,
            "completed_at": None,
        }
    if not endpoint:
        raise ValueError("有凭证且存在新分析记录时必须设置 POPAGENT_LLM_URL")
    allowed_by_output = {
        str(row["record_id"]): {
            str(ref["source_record_id"]) for ref in row["evidence"]
        }
        for row in selected
    }
    prompt_rows = []
    for row in selected:
        compact_evidence = []
        for evidence in row["evidence"]:
            compact_evidence.append({
                key: evidence[key]
                for key in (
                    "source_record_id", "dataset", "report_period", "title",
                    "pdf_url", "raw_pdf_ref", "extracted_text_ref", "pdf_sha256",
                    "text_sha256", "page_count", "text_extraction_status",
                    "full_text", "llm_text_truncated",
                    "revenue_yoy_pct", "parent_net_profit_yoy_pct", "roe_pct",
                    "income_statement_revenue_yoy_pct", "income_statement_parent_profit_yoy_pct",
                    "revenue_cny", "parent_net_profit_cny",
                    "published_at", "available_at", "vintage_verified",
                )
                if key in evidence
            })
        prompt_rows.append({
            "record_id": row["record_id"],
            "expert": row["expert"],
            "symbol": row["symbol"],
            "event_type": row["event_type"],
            "event_time": row["event_time"],
            "evaluation_horizon_sessions": evaluation_horizon_sessions,
            "evidence_text_scope": row["evidence_text_scope"],
            "evidence": compact_evidence,
        })
    messages = [
        {
            "role": "system",
            "content": (
                "你是可审计的基本面与公告分析员。只使用输入证据，不补充外部事实；"
                "只有存在公告 PDF 全文时才能称为全文；标题-only 证据必须明确标注。"
                "不得把事件方向说成评价期限内已校准的收益。"
                "每条输出必须引用该条输入中的 source_record_id。若证据不足，direction_score=0。"
                "只返回符合给定 JSON schema 的 JSON 对象，不输出 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "decision_time": decision_time.astimezone(SHANGHAI).isoformat(),
                    "task": (
                        "对每条证据输出 record_id、event_type、direction_score(-1/0/1)、"
                        "event_time_scale、evaluation_horizon_sessions、rationale、cited_source_record_ids。"
                        "event_time_scale 描述事件证据本身的作用时长；evaluation_horizon_sessions "
                        "是统一比较所用的可配置标签期限，两者不能混为一谈。不要声称方向已校准成收益。"
                    ),
                    "records": prompt_rows,
                    "schema": {
                        "analyses": [{
                            "record_id": "input record_id",
                            "event_type": "short event label",
                            "direction_score": -1,
                            "event_time_scale": "short description",
                            "evaluation_horizon_sessions": evaluation_horizon_sessions,
                            "rationale": "brief, evidence-grounded explanation",
                            "cited_source_record_ids": ["input source_record_id"],
                        }]
                    },
                }, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 6000,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter_ns()
    with urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    parsed = json.loads(body["choices"][0]["message"]["content"])
    analyses = parsed["analyses"]
    input_ids = set(allowed_by_output)
    returned_ids = [str(row["record_id"]) for row in analyses]
    if set(returned_ids) != input_ids or len(returned_ids) != len(input_ids):
        raise ValueError("Qwen 输出记录集合与请求输入不一致")
    selected_by_id = {str(row["record_id"]): row for row in selected}
    for row in analyses:
        direction = int(row["direction_score"])
        if direction not in {-1, 0, 1}:
            raise ValueError("Qwen direction_score 必须是 -1、0 或 1")
        citations = {str(value) for value in row["cited_source_record_ids"]}
        if not citations or not citations <= allowed_by_output[str(row["record_id"])]:
            raise ValueError("Qwen 引用了未随该分析提供的原始证据")
        input_record_id = str(row["record_id"])
        source = selected_by_id[input_record_id]
        row["forecast_of_record_id"] = input_record_id
        row["record_id"] = f"{model}:{input_record_id}"
        row["expert"] = "qwen3.7_plus_fundamental_announcement"
        row["model"] = model
        row["endpoint"] = endpoint
        row["symbol"] = source["symbol"]
        row["record_kind"] = source.get("record_kind", "event")
        row["prediction_date"] = decision_time.astimezone(SHANGHAI).date().isoformat()
        row["event_time"] = source["event_time"]
        row["horizon_sessions"] = evaluation_horizon_sessions
        row["horizon_kind"] = "comparison_label_window"
        row["decision_effect"] = "research_only_until_matured_comparison"
        row["evidence_text_scope"] = source["evidence_text_scope"]
        source_evidence = {
            str(evidence["source_record_id"]): evidence for evidence in source["evidence"]
        }
        row["evidence"] = [
            {
                **{key: value for key, value in source_evidence[source_record_id].items() if key != "full_text"},
                "rationale": row["rationale"],
            }
            for source_record_id in sorted(citations)
        ]

    usage = body["usage"]
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    estimated_cost_cny = prompt_tokens * 1.6 / 1_000_000 + completion_tokens * 6.4 / 1_000_000
    completed_at = datetime.now(SHANGHAI).isoformat()
    for row in analyses:
        row["available_at"] = completed_at
    return {
        "status": "completed",
        "model": model,
        "endpoint": endpoint,
        "input_record_count": len(selected),
        "outputs": analyses,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage["total_tokens"]),
        },
        "price_basis": "Aliyun Beijing published Qwen3.7-plus standard-mode promotional rate; provider bill not reconciled",
        "estimated_token_cost_cny": estimated_cost_cny,
        "elapsed_ms": elapsed_ms,
        "completed_at": completed_at,
    }
