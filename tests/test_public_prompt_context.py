import json
from datetime import date

from adaptive_mas.agents.public_prompt_context import build_public_context


def test_public_context_filters_sources_and_preserves_excerpt_units():
    source = {
        "source_type": "public_announcement_body",
        "symbol": "000001",
        "title": "公司合同公告",
        "published_at": "2025-05-01",
        "available_at": "2025-05-02",
        "vintage_verified": False,
        "source_record_id": "local-row-1",
        "sections": [{
            "title": "合同事项",
            "paragraphs": [
                {"paragraph_index": 1, "text": "公司取得合同，合同金额为人民币万元。"},
                {"paragraph_index": 2, "text": "特别说明：合同金额含税，收入须在交付验收后确认。"},
                {
                    "paragraph_index": 3,
                    "kind": "table",
                    "page": 5,
                    "header": "项目 | 金额（万元）",
                    "unit": "人民币万元",
                    "qualifier": "金额含税",
                    "rows": [{"项目": "合同额", "金额": 120}],
                },
            ],
        }],
    }
    sources = [
        source,
        {**source, "source_record_id": "local-row-copy"},
        {**source, "source_record_id": "wrong-symbol", "symbol": "000002"},
        {**source, "source_record_id": "future", "available_at": "2025-05-11"},
        {**source, "source_record_id": "wrong-topic", "topics": ["宏观利率"]},
        {**source, "source_type": "internal_forecast", "source_record_id": "private"},
    ]

    preview = build_public_context(
        decision_time=date(2025, 5, 10),
        symbol="000001",
        question="合同金额收入交付验收",
        sources=sources,
    )

    user_message = next(row["content"] for row in preview["messages"] if row["role"] == "user")
    user = json.loads(user_message)
    assert preview["prompt_version"] == "raama_public_evidence_context_v3_20261022"
    assert preview["selection_audit"]["eligible_unique_source_count"] == 1
    assert preview["selection_audit"]["excluded_counts"] == {
        "not_public_source": 1,
        "question_scope_mismatch": 1,
        "symbol_mismatch": 1,
        "unavailable_at_decision_time": 1,
    }
    assert user["evidence_budgets"] == {
        "supporting_citations_max": 3,
        "contrary_citations_max": 3,
        "review_same_neutral_source_pool_for_both_sides": True,
    }
    assert user["generation_budget"]["max_output_tokens"] == 600
    assert preview["selection_audit"]["counter_evidence_identified"] is False
    assert preview["selection_audit"]["unclassified_evidence_count"] == 1
    input_budget = preview["selection_audit"]["input_budget"]
    assert input_budget["messages_utf8_bytes"] == len(json.dumps(
        preview["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    assert input_budget["messages_utf8_bytes"] + input_budget["framing_reserve_bytes"] == input_budget["conservative_input_token_upper_bound"]
    assert input_budget["combined_input_upper_bound_and_output_reserve"] <= input_budget["max_context_token_upper_bound"]
    assert input_budget["output_token_reserve"] == 600
    evidence = user["public_evidence"][0]
    excerpt = evidence["excerpts"][0]["text"]
    assert evidence["exact_title"] == "公司合同公告"
    assert "合同金额含税" in excerpt
    assert "交付验收后确认" in excerpt
    assert "人民币万元" in excerpt
    assert {"local-row-1", "local-row-copy"} == set(preview["local_reference_map"][0]["source_record_ids"])
    assert "local-row-1" not in user_message
    assert "source_record_id" not in user_message

    table_preview = build_public_context(
        decision_time=date(2025, 5, 10),
        symbol="000001",
        question="人民币万元金额项目含税",
        sources=[source],
    )
    table_user = json.loads(next(row["content"] for row in table_preview["messages"] if row["role"] == "user"))
    table_card = table_user["public_evidence"][0]
    assert table_card["content_scope"] == "table"
    assert "section=合同事项" in table_card["locator"]
    assert "paragraphs=[3]" in table_card["locator"]
    assert "page=5" in table_card["locator"]
    assert table_preview["selection_audit"]["body_text_selected"] is True
    assert "人民币万元" in table_card["excerpts"][0]["text"]

    supporting = {
        **source,
        "source_type": "public_announcement_title",
        "title": "合同收入增长支持公告",
        "evidence_side": "支持",
        "source_record_id": "local-support",
    }
    contrary = {
        **source,
        "source_type": "public_announcement_title",
        "title": "合同履约延期反证公告",
        "evidence_side": "反证",
        "source_record_id": "local-contrary",
    }
    neutral = {
        **source,
        "source_type": "public_announcement_title",
        "title": "合同收入增长详细项目公告",
        "source_record_id": "local-neutral",
    }
    pair_preview = build_public_context(
        decision_time=date(2025, 5, 10),
        symbol="000001",
        question="合同收入增长",
        sources=[supporting, contrary],
        max_sources=2,
        max_context_token_upper_bound=10000,
    )
    pair_context_bound = pair_preview["selection_audit"]["input_budget"]["combined_input_upper_bound_and_output_reserve"]
    tight_preview = build_public_context(
        decision_time=date(2025, 5, 10),
        symbol="000001",
        question="合同收入增长",
        sources=[neutral, supporting, contrary],
        max_sources=3,
        max_context_chars=10000,
        max_context_token_upper_bound=pair_context_bound,
    )
    tight_user_message = next(row["content"] for row in tight_preview["messages"] if row["role"] == "user")
    tight_titles = {row["exact_title"] for row in json.loads(tight_user_message)["public_evidence"]}
    assert tight_titles == {supporting["title"], contrary["title"]}
    assert tight_preview["selection_audit"]["reserved_side_counts"] == {"支持": 1, "反证": 1}
    assert tight_preview["selection_audit"]["dropped_low_priority_sources"] == [{
        "source_type": "public_announcement_title",
        "exact_title": neutral["title"],
    }]
    tight_budget = tight_preview["selection_audit"]["input_budget"]
    assert tight_budget["combined_input_upper_bound_and_output_reserve"] <= tight_budget["max_context_token_upper_bound"]
    assert tight_budget["output_token_reserve"] == 600
    assert "evidence_side" not in tight_user_message

    try:
        build_public_context(
            decision_time=date(2025, 5, 10),
            symbol="000001",
            question="合同收入增长",
            sources=[supporting, contrary],
            max_context_token_upper_bound=1000,
        )
    except ValueError as error:
        assert "reserved output exceed" in str(error)
    else:
        raise AssertionError("oversized reserved evidence should fail instead of being truncated")
