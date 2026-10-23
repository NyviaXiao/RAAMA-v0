from __future__ import annotations

import argparse
import html
import json
import tempfile
from pathlib import Path

from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.harness_evolution import HarnessVersionStore, score_research_result


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20261023104652"
PACKAGE = ROOT / "reports" / "harness_20261023"
TRACE_PATH = ROOT / "data" / "processed" / "harness_versions" / "runs" / RUN_ID / "reuse_demo_research.json"
EVOLUTION_PATH = ROOT / "experiments" / "gepa_harness" / "runs" / RUN_ID / "evolution_result.json"
AUDIT_PATH = PACKAGE / "candidate_audit.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_mechanism_demo() -> dict[str, object]:
    candidate = seed_candidate()
    candidate["procedural_experience"] += " Synthetic fixture for isolated version-store lifecycle validation."
    with tempfile.TemporaryDirectory(prefix="harness-showcase-") as temporary_root:
        store = HarnessVersionStore(temporary_root)
        _, initial_id, _ = store.load_active()
        version_id, _ = store.adopt(
            candidate,
            previous_version_id=initial_id,
            selection={"fixture_only": True},
            reason="Synthetic mechanism demonstration; not an evaluated model improvement.",
            run_id="synthetic-mechanism-demo",
            origin="synthetic_fixture_only",
            procedural_memory_source="synthetic fixture; never produced by GEPA evaluation",
            version_id="synthetic-demo-v1",
        )
        loaded_candidate, loaded_id, loaded_version = store.load_active()
        memory = loaded_version["procedural_memory"]
        memory_reloaded = bool(
            loaded_candidate == candidate
            and loaded_id == version_id
            and memory.get("loaded_from_persistent_store")
            and memory.get("source") == "synthetic fixture; never produced by GEPA evaluation"
        )
        comparison = store.compare("seed", version_id)
        rollback = store.rollback("seed", reason="Return the isolated fixture store to its seed state.")
        _, rollback_id, _ = store.load_active()
        return {
            "label": "synthetic mechanism demonstration only",
            "is_synthetic_fixture": True,
            "model_requests": 0,
            "model_improvement_or_returns_claimed": False,
            "store_is_temporary_and_isolated": True,
            "temporary_store_deleted_after_demo": True,
            "adopted_fixture_version_id": version_id,
            "adopted_origin": "synthetic_fixture_only",
            "changed_components": comparison["changed_components"],
            "procedural_memory_loaded_from_store": memory_reloaded,
            "rollback_target": rollback["to_version_id"],
            "rollback_restored_seed": rollback_id == "seed",
            "live_active_pointer_modified": False,
        }


def _safe_metrics(metrics: dict[str, object]) -> dict[str, object]:
    keys = (
        "evaluator_version", "score", "fact_accuracy", "required_fact_coverage",
        "required_fact_coverage_status", "required_fact_count", "required_facts_satisfied",
        "citation_precision", "report_period_accuracy", "point_in_time_accuracy",
        "strict_capture_time_accuracy", "strict_historical_pit_verified",
        "tool_efficiency", "cost_efficiency", "concision_score", "completion_score",
        "completion_status", "observation_count", "verified_observation_count",
        "research_tool_call_count", "estimated_model_cost_cny", "case_model_turn_count",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _build_showcase_from_local_trace(output_dir: Path) -> dict[str, object]:
    trace = _read_json(TRACE_PATH)
    v2_metrics = score_research_result(trace)
    evolution = _read_json(EVOLUTION_PATH)
    audit = _read_json(AUDIT_PATH)
    old_metrics = evolution["reuse_demo"]["metrics"]
    proposal = audit["proposal"]

    observation_count = 0
    exact_market_metric_observations = 0
    for agent in trace.get("agents", {}).values():
        for finding in agent.get("report", {}).get("findings", []):
            for observation in finding.get("observations", []):
                observation_count += 1
                source = trace.get("evidence_catalog", {}).get(observation.get("evidence_id"), {})
                if observation.get("field") == source.get("metric"):
                    exact_market_metric_observations += 1

    role_traces = []
    for role, agent in trace.get("agents", {}).items():
        tool_rows = agent.get("tool_trace", [])
        role_traces.append({
            "role": role,
            "status": "full saved tool trace",
            "tool_calls": [str(row.get("tool", "unknown")) for row in tool_rows],
            "tool_call_count": len(tool_rows),
            "context_compression": "not recorded in this historical artifact",
            "directed_agent_detail": (
                "not separately recorded in this historical artifact"
                if role == "research_manager" and any(row.get("tool") == "query_specialist" for row in tool_rows)
                else "not present"
            ),
        })

    rescore = {
        "schema_version": "adaptive-mas.harness-rescore.v2",
        "evaluator_version": v2_metrics["evaluator_version"],
        "data_status": "retrospective rescore of one previously saved full trace",
        "run_id": RUN_ID,
        "as_of": trace["as_of"],
        "model_requests": 0,
        "same_saved_report_used_for_both_metric_sets": True,
        "historical_evaluator_metrics": _safe_metrics(old_metrics),
        "evaluator_v2_metrics": _safe_metrics(v2_metrics),
        "metric_contract_correction": {
            "historical_observation_count": observation_count,
            "exact_source_metric_observations": exact_market_metric_observations,
            "description": "The prior scorer looked up market observations in source.facts; the saved market tool contract uses source.metric and source.value.",
            "interpretation": "Corrected scoring of the same saved observations; not a new model run or research improvement.",
        },
        "other_historical_aggregates": "partial trace / not comparable under evaluator v2; no complete per-case report and evidence trace is available.",
    }

    mechanism_demo = _synthetic_mechanism_demo()
    component_hashes = proposal.get("component_sha256", {})
    active_memory = evolution.get("procedural_memory", {}).get("active_version_before") or {}
    reuse_memory = evolution.get("reuse_demo", {}).get("procedural_memory", {})
    showcase = {
        "schema_version": "adaptive-mas.gepa-showcase.v2",
        "title": "RAAMA GEPA multi-agent harness: offline engineering showcase",
        "historical_evidence_notice": "All model-run evidence below is retrospective and comes from saved local artifacts.",
        "model_requests_during_showcase": 0,
        "rescore_file": "evaluator_v2_rescore.json",
        "historical_run": {
            "run_id": RUN_ID,
            "status": evolution.get("status"),
            "active_version_before": evolution.get("active_version_before"),
            "active_version_after": evolution.get("active_version_after"),
            "as_of": trace.get("as_of"),
            "evidence_mode": trace.get("public_evidence_mode"),
            "strict_historical_pit_verified": trace.get("data_limits", {}).get("strict_historical_pit_verified"),
            "old_evaluator_recorded_score": old_metrics.get("score"),
            "saved_role_traces": role_traces,
        },
        "evolution": {
            "optimizer": evolution.get("optimizer"),
            "reflection_components": sorted(component_hashes),
            "reflection_provenance": "Six components verified against persisted reflection responses in candidate_audit.json; includes work before checkpoint resume.",
            "candidate_proposals_generated": evolution.get("candidate_proposals_generated"),
            "candidate_proposals_rejected": evolution.get("candidate_proposals_rejected"),
            "changed_components": proposal.get("changed_components", []),
            "component_sha256": component_hashes,
            "candidate_status": proposal.get("status"),
            "historical_decision": "rejected; the original run used a hard 120-character conclusion cap and did not admit the proposal to independent selection.",
            "decision_scope": "The v2 evaluator is not applied retroactively to adopt this candidate.",
            "archived_diff_file": "rejected-proposal.diff",
        },
        "procedural_memory": {
            "active_version": evolution.get("active_version_after"),
            "active_memory_status": active_memory.get("status"),
            "later_task_version": evolution.get("reuse_demo", {}).get("version_id"),
            "later_task_loaded_persisted_memory": bool(evolution.get("reuse_demo", {}).get("procedural_memory_reloaded")),
            "later_task_memory_status": reuse_memory.get("status"),
            "real_accepted_version_exists": evolution.get("active_version_after") != "seed",
        },
        "synthetic_lifecycle_demo": mechanism_demo,
        "structured_execution_trace": [
            {
                "stage": "saved_multi_agent_execution",
                "evidence_status": "one complete retrospective saved trace",
                "as_of": trace.get("as_of"),
                "candidate_version_id": trace.get("memory", {}).get("procedural_experience_version"),
                "roles_and_tools": role_traces,
                "evaluator_v2_metric_scores": _safe_metrics(v2_metrics),
            },
            {
                "stage": "gepa_reflection_and_proposal",
                "reflection_components": sorted(component_hashes),
            "reflection_provenance": "Six components verified against persisted reflection responses in candidate_audit.json; includes work before checkpoint resume.",
                "candidate_sha256": proposal.get("candidate_sha256"),
                "changed_components": proposal.get("changed_components", []),
                "candidate_status": proposal.get("status"),
            },
            {
                "stage": "independent_selection_and_adoption",
                "status": "not_run_for_proposal",
                "decision": "rejected under the original output contract before candidate-pool admission",
                "active_version_before": evolution.get("active_version_before"),
                "active_version_after": evolution.get("active_version_after"),
            },
            {
                "stage": "later_task_memory_load",
                "version_id": evolution.get("reuse_demo", {}).get("version_id"),
                "persistent_memory_loaded": bool(evolution.get("reuse_demo", {}).get("procedural_memory_reloaded")),
                "memory_status": reuse_memory.get("status"),
            },
        ],
        "partial_trace_policy": "Exploration/frozen summaries without complete reports are labeled partial trace / not comparable; no missing manager output is reconstructed.",
        "artifact_limits": "No raw prompts, source text, evidence IDs, request payloads, account data, credentials, or test.csv contents are embedded.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "evaluator_v2_rescore.json", rescore)
    _write_json(output_dir / "showcase.json", showcase)
    _write_html(output_dir / "showcase.html", showcase, rescore)
    return showcase


def render_showcase(input_dir: Path, output_dir: Path, *, rescore_local: bool) -> dict[str, object]:
    if rescore_local:
        return _build_showcase_from_local_trace(output_dir)
    showcase = _read_json(input_dir / "showcase.json")
    rescore = _read_json(input_dir / "evaluator_v2_rescore.json")
    showcase["synthetic_lifecycle_demo"] = _synthetic_mechanism_demo()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() != input_dir.resolve():
        _write_json(output_dir / "evaluator_v2_rescore.json", rescore)
    _write_json(output_dir / "showcase.json", showcase)
    _write_html(output_dir / "showcase.html", showcase, rescore)
    return showcase


def _write_html(path: Path, showcase: dict[str, object], rescore: dict[str, object]) -> None:
    run = showcase["historical_run"]
    evolution = showcase["evolution"]
    memory = showcase["procedural_memory"]
    demo = showcase["synthetic_lifecycle_demo"]
    old_metrics = rescore["historical_evaluator_metrics"]
    new_metrics = rescore["evaluator_v2_metrics"]
    esc = lambda value: html.escape(str(value), quote=True)

    def percent(value: object) -> str:
        return "N/A" if value is None else f"{float(value) * 100:.1f}%"

    metric_rows = (
        ("综合评分", old_metrics.get("score"), new_metrics.get("score")),
        ("事实准确率", old_metrics.get("fact_accuracy"), new_metrics.get("fact_accuracy")),
        ("必答事实覆盖", old_metrics.get("required_fact_coverage"), new_metrics.get("required_fact_coverage")),
        ("引用准确率", old_metrics.get("citation_precision"), new_metrics.get("citation_precision")),
        ("报告期准确率", old_metrics.get("report_period_accuracy"), new_metrics.get("report_period_accuracy")),
        ("时点准确率", old_metrics.get("point_in_time_accuracy"), new_metrics.get("point_in_time_accuracy")),
        ("工具效率", old_metrics.get("tool_efficiency"), new_metrics.get("tool_efficiency")),
    )
    rows = "\n".join(
        f"<tr><th>{esc(label)}</th><td>{percent(before)}</td><td>{percent(after)}</td></tr>"
        for label, before, after in metric_rows
    )
    roles = "\n".join(
        "<li><strong>{}</strong> · {} tool calls: {}</li>".format(
            esc({"market": "行情专家", "fundamental": "基本面专家", "research_manager": "研究经理"}.get(role["role"], role["role"])),
            esc(role["tool_call_count"]), esc(" → ".join(role["tool_calls"]))
        )
        for role in run["saved_role_traces"]
    )
    components = "\n".join(
        f"<li><code>{esc(component)}</code><small>SHA-256 {esc(evolution['component_sha256'].get(component, '')[:12])}…</small></li>"
        for component in evolution["changed_components"]
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAAMA GEPA 多智能体研究工作台</title>
<style>
:root{{font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#172033;background:#f3f6fb}}
body{{max-width:1080px;margin:36px auto;padding:0 20px}}header,section{{background:#fff;border:1px solid #dbe3ef;border-radius:14px;padding:22px;margin:16px 0;box-shadow:0 4px 20px #182f5010}}
h1{{margin:0 0 8px;font-size:1.8rem}}h2{{font-size:1.15rem;margin:0 0 14px}}p{{margin:8px 0}}.badge{{display:inline-block;background:#edf3ff;color:#244c90;border-radius:99px;padding:4px 10px;font-size:.86rem}}.notice{{border-left:4px solid #d49324;padding-left:12px;color:#5c481f}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{text-align:left;border-bottom:1px solid #e5eaf2;padding:9px}}td{{font-variant-numeric:tabular-nums}}ul{{margin:8px 0;padding-left:20px}}li{{padding:5px 0}}small{{display:block;color:#65728a}}code{{color:#314f84}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}.card{{background:#f8faff;padding:14px;border-radius:10px}}footer{{color:#5f6b7b;font-size:.9rem;padding:12px}}
</style></head><body>
<header><span class="badge">离线 · 历史证据</span><h1>RAAMA GEPA 多智能体研究工作台</h1>
<p>展示已保存的多角色研究轨迹、评估器 v2、候选选择与记忆版本流程。</p>
<p class="notice">重评分使用同一份历史报告，不是新模型运行、投资结果或收益改善证据。</p></header>
<section><h2>已保存运行 · {esc(run['run_id'])}</h2><div class="grid"><div class="card"><strong>研究日期</strong><br>{esc(run['as_of'])}</div><div class="card"><strong>运行结果</strong><br>{esc(run['status'])} · 活动版本保持为 {esc(run['active_version_after'])}</div><div class="card"><strong>严格历史时点验证</strong><br>{esc(run['strict_historical_pit_verified'])}</div></div>
<ul>{roles}</ul><p>这份历史记录没有保存上下文压缩信息，也没有单独保存定向专家的内部工具轨迹。</p></section>
<section><h2>同一报告，由两个评估器评分</h2><table><thead><tr><th>指标</th><th>历史报告记录</th><th>评估器 v2（重算）</th></tr></thead><tbody>{rows}</tbody></table>
<p>评估器 v2 修正了行情来源字段映射（精确匹配 <code>metric</code> 并读取 <code>value</code>），按实际条数计算必答事实，空集合标为不适用，并把简洁度与任务完成度作为软评分。分数变化来自评估逻辑，没有改动模型输出。</p></section>
<section><h2>真实候选提案</h2><p>历史提案改动了六个文本组件，并按当时的结论长度硬限制被拒绝。活动版本仍是 seed；没有真实采纳的版本或学到的程序性记忆。</p><ul>{components}</ul><p>决策记录：{esc(evolution['historical_decision'])}</p><p>原始候选差异文件：<a href="{esc(evolution['archived_diff_file'])}">查看六组件差异</a></p></section>
<section><h2>持久记忆与回滚</h2><div class="grid"><div class="card"><strong>历史运行</strong><br>活动版本：{esc(memory['active_version'])}<br>后续任务加载持久记忆：{esc(memory['later_task_loaded_persisted_memory'])}</div><div class="card"><strong>隔离的合成演示</strong><br>采用 → 读回记忆 → 回滚至 seed：{esc(demo['procedural_memory_loaded_from_store'])} / {esc(demo['rollback_restored_seed'])}<br>是否修改真实活动指针：否</div></div><p class="notice">该流程演示使用临时目录和合成夹具，只验证版本机制，不代表模型效果或收益提升。</p></section>
<footer>生成页面时没有发出模型请求。缺少完整历史报告的分区标记为“轨迹不完整 / 不可比较”，没有补造输出。</footer>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluator-v2 report and harness version utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="Render the published safe summary; optionally rescore local history.")
    render.add_argument("--output-dir", type=Path, default=PACKAGE)
    render.add_argument("--input-dir", type=Path, default=PACKAGE)
    render.add_argument("--rescore-local", action="store_true", help="Recalculate evaluator v2 from the local full trace.")
    compare = subparsers.add_parser("compare", help="Compare two stored harness versions.")
    compare.add_argument("--version-dir", type=Path, required=True)
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    rollback = subparsers.add_parser("rollback", help="Explicitly point a store at an existing harness version.")
    rollback.add_argument("--version-dir", type=Path, required=True)
    rollback.add_argument("--version-id", required=True)
    rollback.add_argument("--reason", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "render":
        output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
        showcase = render_showcase(input_dir, output_dir, rescore_local=args.rescore_local)
        files = [output_dir / name for name in ("evaluator_v2_rescore.json", "showcase.json", "showcase.html")]
        print(json.dumps({"status": "written", "files": [str(path) for path in files], "model_requests": 0}, indent=2))
        return 0
    store = HarnessVersionStore(args.version_dir)
    if args.command == "compare":
        print(json.dumps(store.compare(args.left, args.right), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(store.rollback(args.version_id, reason=args.reason), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
