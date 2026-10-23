"""GEPA search, fixed evaluation, split isolation and active harness versions."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from adaptive_mas.agents.harness import (
    CallBudget,
    INPUT_PRICE_CNY_PER_MILLION,
    InvalidResearchOutput,
    MAX_INPUT_BYTES_PER_CALL,
    MAX_OUTPUT_TOKENS_PER_CALL,
    OUTPUT_PRICE_CNY_PER_MILLION,
    QwenClient,
    run_research_harness,
)
from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.agents.public_experts import build_public_expert_snapshot
from adaptive_mas.data.research import ResearchData


SHANGHAI = ZoneInfo("Asia/Shanghai")
GEPA_METRIC_CALL_LIMIT = 8
GEPA_PROPOSAL_LIMIT = 1
GEPA_REFLECTION_COMPONENTS = 6
GEPA_REFLECTION_REQUESTS_PER_PROPOSAL = GEPA_REFLECTION_COMPONENTS
SELECTION_CASE_LIMIT = 2
FROZEN_CASE_LIMIT = 2
MAX_RESEARCH_CALLS_PER_CASE = 10
MAX_EXPERIMENT_CALLS = 160
RETROSPECTIVE_CASE_DATES = (
    date(2026, 4, 20), date(2026, 4, 24), date(2026, 4, 28), date(2026, 4, 30),
    date(2026, 8, 14), date(2026, 8, 20), date(2026, 8, 28), date(2026, 8, 31),
)
PUBLIC_FACT_FIELDS = (
    "revenue_yoy_pct",
    "parent_net_profit_yoy_pct",
    "roe_pct",
    "income_statement_revenue_yoy_pct",
    "income_statement_parent_profit_yoy_pct",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class HarnessCase:
    as_of: date
    public_snapshot: dict[str, object]
    split: str
    required_observations: list[dict[str, object]] = field(default_factory=list)
    retrospective: bool = True

    def evaluator_input(self) -> dict[str, object]:
        # GEPA retains examples for reflection; keep local evidence, paths and raw IDs out.
        return {"as_of": self.as_of.isoformat()}


class HarnessVersionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.active_path = self.root / "active.json"

    def load_active(self) -> tuple[dict[str, str], str, dict[str, object]]:
        if not self.active_path.is_file():
            return seed_candidate(), "seed", {
                "status": "seed_candidate",
                "procedural_memory": {
                    "status": "seed_defaults",
                    "loaded_from_persistent_store": False,
                    "updated_from_prior_task": False,
                },
            }
        active = json.loads(self.active_path.read_text(encoding="utf-8"))
        version_path = self.root / active["version_file"]
        version = json.loads(version_path.read_text(encoding="utf-8"))
        candidate = dict(version["candidate"])
        memory_file = version.get("procedural_memory_file")
        if memory_file:
            memory_path = self.root / str(memory_file)
            memory_record = json.loads(memory_path.read_text(encoding="utf-8"))
            content = str(memory_record["content"])
            content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content != candidate["procedural_experience"] or content_sha256 != memory_record["content_sha256"]:
                raise ValueError("persisted procedural memory does not match the active candidate version")
            if content_sha256 != version["procedural_experience_sha256"]:
                raise ValueError("active version has a mismatched procedural experience hash")
            version["procedural_memory"] = {
                "status": "loaded_from_persistent_store",
                "loaded_from_persistent_store": True,
                "record_file": str(memory_file).replace("\\", "/"),
                "content_sha256": content_sha256,
                "updated_from_prior_task": bool(memory_record["updated_from_prior_task"]),
                "source_run_id": memory_record["source_run_id"],
                "previous_content_sha256": memory_record["previous_content_sha256"],
            }
        else:
            version["procedural_memory"] = {
                "status": "legacy_version_without_memory_snapshot",
                "loaded_from_persistent_store": False,
                "updated_from_prior_task": False,
            }
        return candidate, str(version["version_id"]), version

    def adopt(
        self,
        candidate: dict[str, str],
        *,
        previous_version_id: str,
        selection: dict[str, object],
        reason: str,
        run_id: str,
    ) -> tuple[str, str]:
        current, current_id, _ = self.load_active()
        if candidate == current:
            raise ValueError("cannot adopt a candidate identical to the active version")
        candidate_sha256 = hashlib.sha256(_json(candidate).encode("utf-8")).hexdigest()
        version_id = f"h{datetime.now(SHANGHAI):%Y%m%d%H%M%S}-{candidate_sha256[:10]}"
        version_file = Path("versions") / f"{version_id}.json"
        memory_file = Path("procedural_memory") / f"{version_id}.json"
        experience = candidate["procedural_experience"]
        experience_sha256 = hashlib.sha256(experience.encode("utf-8")).hexdigest()
        previous_experience_sha256 = hashlib.sha256(
            current["procedural_experience"].encode("utf-8")
        ).hexdigest()
        memory_record = {
            "schema_version": "adaptive-mas.procedural-memory.v1",
            "version_id": version_id,
            "source_run_id": run_id,
            "source": "GEPA reflective mutation accepted by independent selection",
            "content": experience,
            "content_sha256": experience_sha256,
            "previous_content_sha256": previous_experience_sha256,
            "updated_from_prior_task": experience != current["procedural_experience"],
            "persisted_at": datetime.now(SHANGHAI).isoformat(),
        }
        version = {
            "schema_version": "adaptive-mas.harness-version.v1",
            "version_id": version_id,
            "previous_version_id": previous_version_id or current_id,
            "candidate_sha256": candidate_sha256,
            "candidate": candidate,
            "procedural_memory_file": str(memory_file).replace("\\", "/"),
            "procedural_experience_sha256": experience_sha256,
            "procedural_experience_updated": memory_record["updated_from_prior_task"],
            "selection": selection,
            "adoption_reason": reason,
            "origin": "official_gepa_optimize_anything",
            "run_id": run_id,
            "adopted_at": datetime.now(SHANGHAI).isoformat(),
        }
        _write_json(self.root / memory_file, memory_record)
        _write_json(self.root / version_file, version)
        _write_json(self.active_path, {
            "schema_version": "adaptive-mas.active-harness.v1",
            "version_id": version_id,
            "version_file": str(version_file).replace("\\", "/"),
        })
        diff = "".join(
            line
            for key in sorted(current)
            for line in difflib.unified_diff(
                current[key].splitlines(keepends=True),
                candidate[key].splitlines(keepends=True),
                fromfile=f"active/{key}",
                tofile=f"{version_id}/{key}",
            )
        )
        return version_id, diff

    def export_active(self, output_path: str | Path) -> dict[str, object]:
        """Export active prompt text and integrity hashes without task or account artifacts."""
        candidate, version_id, version = self.load_active()
        expected_components = set(seed_candidate())
        if set(candidate) != expected_components or any(not isinstance(value, str) for value in candidate.values()):
            raise ValueError("active harness version does not match the fixed prompt component schema")
        experience_sha256 = hashlib.sha256(candidate["procedural_experience"].encode("utf-8")).hexdigest()
        version_experience_sha256 = version.get("procedural_experience_sha256", experience_sha256)
        if version_experience_sha256 != experience_sha256:
            raise ValueError("active version procedural-memory hash does not match the candidate")
        export = {
            "schema_version": "adaptive-mas.harness-prompt-export.v1",
            "version_id": version_id,
            "candidate_sha256": hashlib.sha256(_json(candidate).encode("utf-8")).hexdigest(),
            "procedural_experience_sha256": experience_sha256,
            "components": candidate,
            "export_scope": "prompt text and integrity metadata only",
        }
        _write_json(Path(output_path), export)
        return export


def _timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHANGHAI)
    return parsed.astimezone(SHANGHAI)


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-10)
    return left == right


def _required_observations(snapshot: dict[str, object]) -> list[dict[str, object]]:
    """Freeze two public fact checks per case when the cache has enough coverage."""
    by_symbol: dict[str, dict[str, object]] = {}
    for record in snapshot.get("p9_outputs", []):
        symbol = str(record.get("symbol", "")).rsplit(".", 1)[-1]
        if len(symbol) != 6 or not symbol.isdigit():
            continue
        for source in record.get("evidence", []):
            if not source.get("report_period"):
                continue
            for field in PUBLIC_FACT_FIELDS:
                value = source.get(field)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    by_symbol.setdefault(symbol, {}).setdefault(field, {
                        "symbol": symbol,
                        "field": field,
                        "report_period": str(source["report_period"]),
                    })
    requirements = []
    for symbol in sorted(by_symbol):
        for field in PUBLIC_FACT_FIELDS:
            if field in by_symbol[symbol]:
                requirements.append(by_symbol[symbol][field])
                break
        if len(requirements) == 2:
            break
    return requirements


def _normalized_report_period(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    quarter_match = re.fullmatch(r"(\d{4})q([1-4])", normalized)
    if quarter_match:
        return f"{quarter_match.group(1)}Q{quarter_match.group(2)}"
    date_match = re.match(r"(\d{4})(\d{2})(\d{2})", normalized)
    if date_match:
        year, month = int(date_match.group(1)), int(date_match.group(2))
        if month in {3, 6, 9, 12}:
            return f"{year}Q{(month + 2) // 3}"
    return normalized.upper()


def _normalized_event_title(value: object) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", str(value)).casefold()
        if character.isalnum()
    )


def _public_event_group_key(
    record: dict[str, object], source: dict[str, object]
) -> tuple[str, str, str] | None:
    symbol = str(source.get("symbol") or record.get("symbol") or "").rsplit(".", 1)[-1]
    dataset = str(source.get("dataset", "")).casefold()
    report_period = source.get("report_period")
    if report_period not in (None, ""):
        entity = symbol or str(source.get("series") or record.get("series") or "macro")
        return "company_report_period", entity, _normalized_report_period(report_period)

    is_announcement = "announcement" in dataset or "announcement" in str(record.get("event_type", "")).casefold()
    if is_announcement:
        parsed_url = urlparse(str(source.get("url", "")))
        query = parse_qs(parsed_url.query)
        event_id = next((
            query[key][0]
            for key in ("announcementId", "announcement_id", "id")
            if query.get(key)
        ), "")
        event_key = event_id or (
            f"{str(source.get('published_at', ''))[:10]}:{_normalized_event_title(source.get('title', ''))}"
        )
        if symbol and event_key.strip(":"):
            return "company_announcement_event", symbol, event_key

    is_news = "news" in dataset
    if is_news:
        event_key = f"{str(source.get('published_at', ''))[:10]}:{_normalized_event_title(source.get('title', ''))}"
        if symbol and event_key.strip(":"):
            return "company_news_event", symbol, event_key

    source_id = source.get("source_record_id")
    if source_id:
        return "source_record", symbol, str(source_id)
    return None


def summarize_retrospective_cases(cases: list[HarnessCase]) -> dict[str, object]:
    """Summarize usable public coverage and event leakage without exposing IDs or source text."""
    rows = []
    groups_by_case: dict[str, set[tuple[str, str, str]]] = {}
    groups_by_split: dict[str, set[tuple[str, str, str]]] = {}
    for case in cases:
        group_keys: set[tuple[str, str, str]] = set()
        source_ids: set[str] = set()
        capture_lag_source_ids: set[str] = set()
        public_record_count = 0
        for collection in ("p9_outputs", "p10_outputs", "macro_outputs", "attention_outputs"):
            for record in case.public_snapshot.get(collection, []):
                if not isinstance(record, dict):
                    continue
                evidence = record.get("evidence", [])
                if not isinstance(evidence, list):
                    continue
                public_record_count += 1
                for source in evidence:
                    if not isinstance(source, dict):
                        continue
                    if source.get("source_record_id"):
                        source_id = str(source["source_record_id"])
                        source_ids.add(source_id)
                        if source.get("retrospective_backfill_proxy") is True:
                            capture_lag_source_ids.add(source_id)
                    group_key = _public_event_group_key(record, source)
                    if group_key is not None:
                        group_keys.add(group_key)
        groups_by_case[case.as_of.isoformat()] = group_keys
        groups_by_split.setdefault(case.split, set()).update(group_keys)
        rows.append({
            "as_of": case.as_of.isoformat(),
            "split": case.split,
            "required_fact_count": len(case.required_observations),
            "public_record_count": public_record_count,
            "unique_cached_source_count": len(source_ids),
            "event_group_count": len(group_keys),
            "retrospective_capture_lag_source_count": len(capture_lag_source_ids),
            "public_evidence_mode": str(case.public_snapshot.get("data_mode", "unknown")),
            "strict_historical_pit_verified": bool(case.public_snapshot.get("strict_historical_pit_verified", False)),
        })

    case_pair_overlaps = []
    all_rows = list(groups_by_case.items())
    for index, (left_date, left_groups) in enumerate(all_rows):
        for right_date, right_groups in all_rows[index + 1:]:
            overlap_count = len(left_groups & right_groups)
            case_pair_overlaps.append({
                "left_as_of": left_date,
                "right_as_of": right_date,
                "overlap_count": overlap_count,
            })
    split_names = ("exploration", "selection", "frozen_evaluation", "reuse_demo")
    split_overlaps = {}
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            split_overlaps[f"{left_name}__{right_name}"] = len(
                groups_by_split.get(left_name, set()) & groups_by_split.get(right_name, set())
            )

    insufficient_cases = [
        row["as_of"] for row in rows
        if int(row["required_fact_count"]) < 2
        or int(row["unique_cached_source_count"]) < 2
    ]
    overlapping_case_pairs = [
        row for row in case_pair_overlaps if int(row["overlap_count"]) > 0
    ]
    eligible = bool(rows) and not insufficient_cases and not overlapping_case_pairs
    return {
        "cases": rows,
        "event_grouping_rules": [
            "company plus normalized report period",
            "company plus announcement identifier, falling back to publication date and normalized title",
            "company plus publication date and normalized news title",
            "source record identity when no richer public event key exists",
        ],
        "event_group_overlap_by_split_pair": split_overlaps,
        "overlapping_case_pairs": overlapping_case_pairs,
        "insufficient_public_evidence_cases": insufficient_cases,
        "minimum_required_facts_per_case": 2,
        "minimum_unique_cached_sources_per_case": 2,
        "source_count_scope": (
            "Unique source records represented in the local cached evidence catalog for each case. "
            "Bounded tool calls page or select subsets and long sources are read on demand; these counts "
            "are catalog coverage, not sources included in a single prompt or sent to the model."
        ),
        "strict_historical_pit_verified": all(
            bool(row["strict_historical_pit_verified"]) for row in rows
        ),
        "retrospective_capture_lag_source_count": sum(
            int(row["retrospective_capture_lag_source_count"]) for row in rows
        ),
        "eligible_for_model_execution": eligible,
        "eligibility_reason": (
            "all eight cases have required facts and distinct public event groups"
            if eligible else
            "every case needs at least two required public facts and two sources, with no company-period or event group repeated across cases"
        ),
    }


def _collect_exposed_evidence_ids(value: object) -> set[str]:
    exposed: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("evidence_id"), str):
            exposed.add(str(value["evidence_id"]))
        if isinstance(value.get("evidence_ids"), list):
            exposed.update(str(item) for item in value["evidence_ids"])
        if isinstance(value.get("source_record_ids"), list):
            exposed.update(str(item) for item in value["source_record_ids"])
        for child in value.values():
            exposed.update(_collect_exposed_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            exposed.update(_collect_exposed_evidence_ids(child))
    return exposed


def score_research_result(result: dict[str, object]) -> dict[str, object]:
    """Fixed source-grounded evaluator; it never scores prose length or direction agreement."""
    catalog = result["evidence_catalog"]
    evidence_catalog: dict[str, dict[str, object]] = catalog
    as_of = date.fromisoformat(str(result["as_of"]))
    decision_time = datetime.combine(as_of, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI)
    exposed_ids: set[str] = set()
    for agent in result["agents"].values():
        for call in agent["tool_trace"]:
            if call["tool"] in {"submit_report", "finalize_research"}:
                continue
            exposed_ids.update(_collect_exposed_evidence_ids(call["result"]))
    cited_ids: list[str] = []
    valid_citations: list[bool] = []
    observations: list[bool] = []
    scored_observations: list[dict[str, object]] = []
    citation_issues: list[dict[str, object]] = []
    period_observations = []
    time_checks = []
    capture_time_checks = []
    capture_lag_source_ids: set[str] = set()
    tool_calls = []
    useful_calls = []
    for role, agent in result["agents"].items():
        role_seen_evidence: set[str] = set()
        report = agent["report"]
        for finding in report.get("findings", []):
            for evidence_id in finding["evidence_ids"]:
                evidence_id = str(evidence_id)
                cited_ids.append(evidence_id)
                source = evidence_catalog.get(evidence_id)
                time_valid = False
                if source is not None and source.get("published_at") and source.get("available_at"):
                    time_valid = (
                        _timestamp(source["published_at"]) <= decision_time
                        and _timestamp(source["available_at"]) <= decision_time
                    )
                time_checks.append(time_valid)
                if source is not None and source.get("fetched_at"):
                    fetched_before_decision = _timestamp(source["fetched_at"]) <= decision_time
                    capture_time_checks.append(fetched_before_decision)
                    if not fetched_before_decision:
                        capture_lag_source_ids.add(evidence_id)
                citation_valid = (
                    source is not None
                    and str(source.get("symbol")) == str(finding["symbol"])
                    and evidence_id in exposed_ids
                    and time_valid
                )
                valid_citations.append(citation_valid)
                if not citation_valid:
                    citation_issues.append({
                        "role": role,
                        "symbol": str(finding["symbol"]),
                        "source_known": source is not None,
                        "source_symbol_matches": source is not None and str(source.get("symbol")) == str(finding["symbol"]),
                        "source_was_exposed": evidence_id in exposed_ids,
                        "visible_by_decision_time": time_valid,
                    })
            for observation in finding.get("observations", []):
                evidence_id = str(observation["evidence_id"])
                source = evidence_catalog.get(evidence_id)
                if source is None:
                    observations.append(False)
                    scored_observations.append({
                        "symbol": str(finding["symbol"]),
                        "field": str(observation["field"]),
                        "report_period": str(observation.get("report_period", "")),
                        "actual_value": observation.get("value"),
                        "expected_value": None,
                        "correct": False,
                    })
                    continue
                if observation["field"] == "value":
                    expected = source.get("value")
                else:
                    expected = source.get("facts", {}).get(str(observation["field"]))
                cited_in_finding = evidence_id in finding["evidence_ids"]
                visible = evidence_id in exposed_ids
                time_valid = bool(
                    source.get("published_at") and source.get("available_at")
                    and _timestamp(source["published_at"]) <= decision_time
                    and _timestamp(source["available_at"]) <= decision_time
                )
                correct = (
                    expected is not None
                    and _same_value(expected, observation["value"])
                    and str(source.get("symbol")) == str(finding["symbol"])
                    and cited_in_finding and visible and time_valid
                )
                observations.append(correct)
                scored_observations.append({
                    "symbol": str(finding["symbol"]),
                    "field": str(observation["field"]),
                    "report_period": str(observation.get("report_period", "")),
                    "actual_value": observation.get("value"),
                    "expected_value": expected,
                    "source_type": source.get("source_type"),
                    "source_title": source.get("title"),
                    "correct": correct,
                })
                if source.get("report_period") is not None:
                    period_observations.append(
                        str(source["report_period"]) == str(observation["report_period"])
                    )
        for call in agent["tool_trace"]:
            if call["tool"] in {"submit_report", "finalize_research", "plan_research"}:
                continue
            tool_calls.append(call)
            returned = call["result"]
            returned_evidence = _collect_exposed_evidence_ids(returned)
            new_evidence = returned_evidence - role_seen_evidence
            role_seen_evidence.update(returned_evidence)
            useful_calls.append(bool(new_evidence))

    citation_precision = sum(valid_citations) / len(valid_citations) if valid_citations else 0.0
    fact_accuracy = sum(observations) / len(observations) if observations else 0.0
    report_period_accuracy = (
        sum(period_observations) / len(period_observations) if period_observations else 0.0
    )
    point_in_time_accuracy = sum(time_checks) / len(time_checks) if time_checks else 0.0
    strict_capture_time_accuracy = (
        sum(capture_time_checks) / len(capture_time_checks) if capture_time_checks else 0.0
    )
    tool_efficiency = sum(useful_calls) / len(useful_calls) if useful_calls else 0.0
    case_usage = result["task_usage"]
    maximum_cost = MAX_RESEARCH_CALLS_PER_CASE * (
        MAX_INPUT_BYTES_PER_CALL * INPUT_PRICE_CNY_PER_MILLION
        + MAX_OUTPUT_TOKENS_PER_CALL * OUTPUT_PRICE_CNY_PER_MILLION
    ) / 1_000_000
    estimated_cost = float(case_usage["logical_estimated_cost_cny"])
    cost_efficiency = max(0.0, 1 - estimated_cost / maximum_cost) if maximum_cost else 0.0
    required = result.get("required_observations", [])
    required_satisfied = [
        any(
            row["correct"]
            and row["symbol"] == str(requirement["symbol"])
            and row["field"] == str(requirement["field"])
            and row["report_period"] == str(requirement["report_period"])
            for row in scored_observations
        )
        for requirement in required
    ]
    required_fact_coverage = (
        sum(required_satisfied) / max(2, len(required_satisfied))
        if required_satisfied else 0.0
    )
    required_diagnostics = []
    for requirement, satisfied in zip(required, required_satisfied, strict=True):
        sources = [
            source for source in evidence_catalog.values()
            if str(source.get("symbol")) == str(requirement["symbol"])
            and str(source.get("report_period")) == str(requirement["report_period"])
            and requirement["field"] in source.get("facts", {})
        ]
        source = sources[0] if sources else None
        corresponding = next((
            row for row in scored_observations
            if row["symbol"] == str(requirement["symbol"])
            and row["field"] == str(requirement["field"])
            and row["report_period"] == str(requirement["report_period"])
        ), None)
        required_diagnostics.append({
            "symbol": str(requirement["symbol"]),
            "field": str(requirement["field"]),
            "report_period": str(requirement["report_period"]),
            "status": "satisfied" if satisfied else "missing_or_inaccurate",
            "expected_value": source.get("facts", {}).get(str(requirement["field"])) if source else None,
            "reported_value": corresponding.get("actual_value") if corresponding else None,
            "source_type": source.get("source_type") if source else None,
            "source_title": source.get("title") if source else None,
            "published_at": source.get("published_at") if source else None,
            "available_at": source.get("available_at") if source else None,
        })
    observation_failures = [
        {
            key: row[key]
            for key in ("symbol", "field", "report_period", "actual_value", "expected_value", "source_type", "source_title")
            if key in row
        }
        for row in scored_observations if not row["correct"]
    ][:6]
    metrics = {
        "fact_accuracy": fact_accuracy,
        "citation_precision": citation_precision,
        "report_period_accuracy": report_period_accuracy,
        "point_in_time_accuracy": point_in_time_accuracy,
        "point_in_time_metric_scope": "source publication and reported availability cutoffs; fetch-time integrity is reported separately",
        "strict_capture_time_accuracy": strict_capture_time_accuracy,
        "strict_historical_pit_verified": (
            result.get("public_evidence_mode") != "retrospective_source_publication_proxy"
            and not capture_lag_source_ids
        ),
        "retrospective_capture_lag_source_count": len(capture_lag_source_ids),
        "public_evidence_mode": str(result.get("public_evidence_mode", "unknown")),
        "required_fact_coverage": required_fact_coverage,
        "required_fact_count": len(required_satisfied),
        "required_facts_satisfied": sum(required_satisfied),
        "tool_efficiency": tool_efficiency,
        "cost_efficiency": cost_efficiency,
        "verified_observation_count": sum(observations),
        "observation_count": len(observations),
        "unique_citation_count": len(set(cited_ids)),
        "duplicate_citation_count": len(cited_ids) - len(set(cited_ids)),
        "valid_citation_count": sum(valid_citations),
        "citation_count": len(cited_ids),
        "research_tool_call_count": len(tool_calls),
        "estimated_model_cost_cny": estimated_cost,
        "incremental_api_cost_cny": float(case_usage["api_incremental_cost_cny"]),
        "case_provider_call_count": int(case_usage["provider_call_count"]),
        "case_response_cache_reuse_count": int(case_usage["response_cache_reuse_count"]),
        "case_model_turn_count": int(case_usage["model_turn_count"]),
        "diagnostics": {
            "required_fact_checks": required_diagnostics,
            "observation_failures": observation_failures,
            "citation_failures": citation_issues[:6],
        },
    }
    metrics["score"] = (
        0.25 * fact_accuracy
        + 0.15 * required_fact_coverage
        + 0.15 * citation_precision
        + 0.15 * report_period_accuracy
        + 0.15 * point_in_time_accuracy
        + 0.10 * tool_efficiency
        + 0.05 * cost_efficiency
    )
    return metrics


def _public_report_projection(report: dict[str, object]) -> dict[str, object]:
    findings = []
    for finding in report.get("findings", []):
        findings.append({
            "symbol": finding.get("symbol"),
            "conclusion": finding.get("conclusion"),
            "evidence_ids": finding.get("evidence_ids", []),
            "observations": [
                {
                    key: observation[key]
                    for key in ("evidence_id", "field", "value", "report_period")
                    if key in observation
                }
                for observation in finding.get("observations", [])
            ],
            "uncertainties": finding.get("uncertainties", []),
        })
    projection = {
        "summary": report.get("summary", ""),
        "findings": findings,
        "unresolved_items": report.get("unresolved_items", []),
    }
    for key in ("priority_symbols", "no_query_reason", "research_plan"):
        if key in report:
            projection[key] = report[key]
    return projection


def _public_tool_projection(tool: str, value: object) -> object:
    """Project each tool result through a public-only field allowlist."""
    if not isinstance(value, dict):
        return None
    if "accepted_report" in value:
        return {
            "accepted": bool(value.get("accepted")),
            "report": _public_report_projection(value["accepted_report"]),
        }
    if tool in {"read_cached_source", "read_evidence", "read_prior_evidence"}:
        projection = {
            key: value[key]
            for key in (
                "evidence_id", "symbol", "source_type", "dataset", "title",
                "report_period", "published_at", "available_at", "facts",
                "text_excerpt", "offset", "next_offset", "location", "content_scope",
                "body_status", "metric", "value", "fetched_at", "vintage_status",
                "retrospective_backfill_proxy", "vintage_verified",
            )
            if key in value
        }
        if isinstance(projection.get("text_excerpt"), str):
            projection["text_excerpt"] = projection["text_excerpt"][:800]
        return projection
    if tool == "search_financial_facts_and_announcements":
        references = [
            {key: ref[key] for key in ("evidence_id", "source_type", "title", "locator") if key in ref}
            for ref in value.get("evidence_refs", [])
        ]
        context_messages = []
        for message in value.get("context_messages", []):
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", ""))
            context_messages.append({
                "role": str(message.get("role", "user")),
                "content": content[:700],
            })
        return {
            "context_messages": context_messages,
            "evidence_refs": references,
            "selected_source_count": value.get("selected_source_count", len(references)),
            "point_in_time_cutoff": value.get("point_in_time_cutoff"),
        }
    if tool in {"read_trend_signals", "read_reversal_signals", "read_volume_price_signals", "read_market_signals"}:
        return {
            "count": value.get("count", 0),
            "findings": [
                {
                    key: finding[key]
                    for key in (
                        "symbol", "metric", "value", "period_start", "period_end",
                        "evidence_id", "dataset", "signal_group",
                    )
                    if key in finding
                }
                for finding in value.get("findings", [])
            ],
        }
    if tool == "plan_research":
        plan = value.get("plan", {})
        return {
            "accepted": bool(value.get("accepted")),
            "plan": {key: plan[key] for key in ("target_role", "target_symbol", "question") if key in plan},
            "query_limit": value.get("query_limit"),
        }
    if tool == "query_specialist":
        if value.get("status") == "skipped_budget_reserve":
            return {
                "role": value.get("role"),
                "status": value.get("status"),
                "reason": value.get("reason"),
            }
        report = value.get("report", {})
        return {"role": value.get("role"), "report": _public_report_projection(report)}
    return {key: value[key] for key in ("accepted", "count", "status") if key in value}


def _public_arguments_projection(tool: str, arguments: dict[str, object]) -> dict[str, object]:
    if tool in {"submit_report", "finalize_research"}:
        return _public_report_projection(arguments)
    allowed = {
        "read_trend_signals": ("symbols", "limit"),
        "read_reversal_signals": ("symbols", "limit"),
        "read_volume_price_signals": ("symbols", "limit"),
        "read_market_signals": ("symbols", "limit_per_signal"),
        "search_financial_facts_and_announcements": ("question", "symbols", "limit"),
        "read_cached_source": ("evidence_id", "offset", "length"),
        "plan_research": ("target_role", "target_symbol", "question"),
        "query_specialist": ("role", "symbol", "question"),
        "read_evidence": ("evidence_id",),
        "read_prior_evidence": ("evidence_id",),
    }.get(tool, ())
    return {key: arguments[key] for key in allowed if key in arguments}


def _reflection_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:120]
    return f"<{type(value).__name__}>"


def _reflection_safe_info(result: dict[str, object], metrics: dict[str, object]) -> dict[str, object]:
    """Give GEPA compact, public, actionable evaluator feedback."""
    report_summaries = {}
    tool_summaries = {}
    for role, agent in result["agents"].items():
        report = agent["report"]
        report_summaries[role] = {
            "summary": str(report.get("summary", ""))[:240],
            "findings": [
                {
                    "symbol": str(finding.get("symbol", "")),
                    "conclusion": str(finding.get("conclusion", ""))[:180],
                    "observations": [
                        {
                            key: _reflection_value(observation[key])
                            for key in ("field", "value", "report_period")
                            if key in observation
                        }
                        for observation in finding.get("observations", [])[:2]
                    ],
                    "uncertainties": [str(item)[:100] for item in finding.get("uncertainties", [])[:2]],
                }
                for finding in report.get("findings", [])[:3]
            ],
            "unresolved_count": len(report.get("unresolved_items", [])),
        }
        counts: dict[str, int] = {}
        evidence_result_count = 0
        for call in agent["tool_trace"]:
            if call["tool"] in {"submit_report", "finalize_research", "plan_research"}:
                continue
            counts[call["tool"]] = counts.get(call["tool"], 0) + 1
            returned = call["result"]
            evidence_result_count += bool(
                returned.get("findings") or returned.get("evidence_refs")
                or returned.get("text_excerpt") or returned.get("evidence_id")
                or returned.get("report", {}).get("findings")
            )
        tool_summaries[role] = {
            "calls_by_tool": counts,
            "calls_returning_public_evidence": evidence_result_count,
        }
    metric_fields = (
        "score", "fact_accuracy", "required_fact_coverage", "citation_precision",
        "report_period_accuracy", "point_in_time_accuracy", "tool_efficiency",
        "point_in_time_metric_scope", "strict_capture_time_accuracy",
        "strict_historical_pit_verified", "retrospective_capture_lag_source_count",
        "public_evidence_mode",
        "cost_efficiency", "verified_observation_count", "observation_count",
        "unique_citation_count", "duplicate_citation_count", "valid_citation_count",
        "citation_count", "research_tool_call_count", "estimated_model_cost_cny",
    )
    diagnostics = metrics["diagnostics"]
    required_fact_checks = [
        {
            **{
                key: item[key]
                for key in ("symbol", "field", "report_period", "status", "source_type", "published_at", "available_at")
                if key in item
            },
            **{
                key: _reflection_value(item[key])
                for key in ("expected_value", "reported_value")
                if key in item
            },
            "source_title": str(item.get("source_title", ""))[:160],
        }
        for item in diagnostics["required_fact_checks"][:2]
    ]
    incorrect_observations = [
        {
            **{
                key: item[key]
                for key in ("symbol", "field", "report_period", "source_type")
                if key in item
            },
            **{
                key: _reflection_value(item[key])
                for key in ("actual_value", "expected_value")
                if key in item
            },
            "source_title": str(item.get("source_title", ""))[:160],
        }
        for item in diagnostics["observation_failures"][:3]
    ]
    return {
        "as_of": result["as_of"],
        "public_evidence_mode": str(result.get("public_evidence_mode", "unknown")),
        "quality_metrics": {key: metrics[key] for key in metric_fields},
        "required_fact_checks": required_fact_checks,
        "incorrect_observations": incorrect_observations,
        "invalid_citation_examples": [
            {key: item[key] for key in (
                "role", "symbol", "source_known", "source_symbol_matches",
                "source_was_exposed", "visible_by_decision_time",
            )}
            for item in diagnostics["citation_failures"][:3]
        ],
        "public_report_summaries": report_summaries,
        "public_tool_summaries": tool_summaries,
        "task_model_usage": result["task_usage"],
        "scope": "retrospective public cached research; no account, private forecast, trade, return or raw source ID",
    }


def evaluate_candidate(
    candidate: dict[str, str],
    case: HarnessCase,
    *,
    research: ResearchData,
    client: QwenClient,
    budget: CallBudget,
    config: dict[str, object],
    candidate_version_id: str = "seed",
    procedural_memory: dict[str, object] | None = None,
    trace_sink: list[dict[str, object]] | None = None,
) -> tuple[float, dict[str, object]]:
    call_start = len(budget.calls)
    provider_call_start = budget.provider_call_count
    logical_start = len(budget.logical_calls)
    budget.limit_next_task(MAX_RESEARCH_CALLS_PER_CASE)
    validation_failure: InvalidResearchOutput | None = None
    try:
        trace = run_research_harness(
            research,
            case.as_of,
            case.public_snapshot,
            candidate,
            client,
            budget,
            config,
            candidate_version_id=candidate_version_id,
            required_observations=case.required_observations,
            procedural_memory=procedural_memory,
        )
    except InvalidResearchOutput as exc:
        trace = None
        validation_failure = exc
    finally:
        budget.task_logical_turn_limit = None
    task_calls = budget.calls[call_start:]
    provider_task_call_count = budget.provider_call_count - provider_call_start
    logical_task_calls = budget.logical_calls[logical_start:]
    task_cost = sum(float(row["logical_estimated_cost_cny"]) for row in logical_task_calls)
    api_incremental_cost = sum(float(row.get("actual_estimated_cost_cny", 0.0)) for row in task_calls)
    cache_reuse_count = sum(row["source"] == "response_cache" for row in logical_task_calls)
    task_turn_count = len(logical_task_calls)
    if task_turn_count > MAX_RESEARCH_CALLS_PER_CASE:
        raise RuntimeError(
            f"research case exceeded its fixed {MAX_RESEARCH_CALLS_PER_CASE}-turn model-call limit"
        )
    task_usage = {
        "provider_call_count": provider_task_call_count,
        "attempt_count": len(task_calls),
        "response_cache_reuse_count": cache_reuse_count,
        "model_turn_count": task_turn_count,
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in logical_task_calls),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in logical_task_calls),
        "logical_estimated_cost_cny": task_cost,
        "api_incremental_cost_cny": api_incremental_cost,
        "unknown_outcome_call_count": sum(row.get("status") == "started" for row in task_calls),
    }
    if validation_failure is not None:
        maximum_cost = MAX_RESEARCH_CALLS_PER_CASE * (
            MAX_INPUT_BYTES_PER_CALL * INPUT_PRICE_CNY_PER_MILLION
            + MAX_OUTPUT_TOKENS_PER_CALL * OUTPUT_PRICE_CNY_PER_MILLION
        ) / 1_000_000
        partial_tool_trace = validation_failure.partial_tool_trace
        quality_metrics = {
            "score": 0.0,
            "fact_accuracy": 0.0,
            "required_fact_coverage": 0.0,
            "citation_precision": 0.0,
            "report_period_accuracy": 0.0,
            "point_in_time_accuracy": 0.0,
            "tool_efficiency": 0.0,
            "point_in_time_metric_scope": "report rejected by fixed output validation; no observations scored",
            "strict_capture_time_accuracy": 0.0,
            "strict_historical_pit_verified": False,
            "retrospective_capture_lag_source_count": 0,
            "public_evidence_mode": str(case.public_snapshot.get("data_mode", "unknown")),
            "cost_efficiency": max(0.0, 1 - task_cost / maximum_cost) if maximum_cost else 0.0,
            "verified_observation_count": 0,
            "observation_count": 0,
            "unique_citation_count": 0,
            "duplicate_citation_count": 0,
            "valid_citation_count": 0,
            "citation_count": 0,
            "research_tool_call_count": sum(
                item["tool"] not in {"submit_report", "finalize_research", "plan_research"}
                for item in partial_tool_trace
            ),
            "estimated_model_cost_cny": task_cost,
            "incremental_api_cost_cny": api_incremental_cost,
            "case_provider_call_count": provider_task_call_count,
            "case_response_cache_reuse_count": cache_reuse_count,
            "case_model_turn_count": task_turn_count,
        }
        side_info = {
            "as_of": case.as_of.isoformat(),
            "public_evidence_mode": quality_metrics["public_evidence_mode"],
            "evaluation_status": "research_failed_output_contract",
            "quality_metrics": quality_metrics,
            "report_validation_failure": {
                "role": validation_failure.role_id,
                "category": validation_failure.category,
            },
            "partial_tool_trace": partial_tool_trace,
            "task_model_usage": task_usage,
            "scope": "retrospective public cached research; invalid report rejected before scoring",
        }
        if trace_sink is not None:
            trace_sink.append({
                "as_of": case.as_of.isoformat(),
                "status": "research_failed_output_contract",
                "partial_tool_trace": partial_tool_trace,
                "task_usage": task_usage,
            })
        return 0.0, side_info
    trace["required_observations"] = case.required_observations
    trace["task_usage"] = task_usage
    trace["cost"] = task_usage
    metrics = score_research_result(trace)
    if trace_sink is not None:
        trace_sink.append(trace)
    return float(metrics["score"]), _reflection_safe_info(trace, metrics)


def build_retrospective_cases(
    research: ResearchData,
    public_cache_dir: str | Path,
    *,
    horizon_sessions: int,
    config: dict[str, object],
    case_count: int = 8,
    case_dates: tuple[date, ...] | list[date] | None = None,
) -> list[HarnessCase]:
    if case_count != 8:
        raise ValueError("the protocol fixes 3/2/2 evaluation cases plus one separate reuse case")
    minimum_history = max(
        int(config.get("feature_trend_sessions", 20)),
        int(config.get("feature_reversal_sessions", 5)),
        int(config.get("feature_liquidity_return_sessions", 5)),
        int(config.get("feature_recent_amount_sessions", 5))
        + int(config.get("feature_reference_amount_sessions", 20)) - 1,
    )
    valid_dates = research.trading_dates[minimum_history:]
    selected_dates = list(case_dates or RETROSPECTIVE_CASE_DATES)
    if len(selected_dates) != case_count:
        raise ValueError("the retrospective protocol requires exactly eight explicit dates")
    if any(as_of not in valid_dates for as_of in selected_dates):
        raise ValueError("every retrospective date must be an eligible cached trading date")
    if selected_dates != sorted(set(selected_dates)):
        raise ValueError("retrospective case dates must be unique and strictly increasing")
    split_names = ["exploration"] * 3 + ["selection"] * 2 + ["frozen_evaluation"] * 2 + ["reuse_demo"]
    source_owner: dict[str, date] = {}
    event_group_owner: dict[tuple[str, str, str], date] = {}
    cases = []
    for as_of, split in zip(selected_dates, split_names, strict=True):
        decision_time = datetime.combine(as_of, datetime.min.time()).replace(hour=16, tzinfo=SHANGHAI)
        snapshot = build_public_expert_snapshot(
            public_cache_dir,
            decision_time,
            horizon_sessions,
            allowed_symbols=set(research.membership_at(as_of)),
            announcement_text_limit=0,
            include_qwen=False,
            retrospective_backfill_proxy=True,
        )
        filtered_snapshot = dict(snapshot)
        for collection in ("p9_outputs", "p10_outputs", "macro_outputs", "attention_outputs"):
            filtered_records = []
            for record in snapshot.get(collection, []):
                if not isinstance(record, dict):
                    continue
                kept_evidence = []
                for source in record.get("evidence", []):
                    if not isinstance(source, dict):
                        continue
                    source_id = str(source.get("source_record_id", ""))
                    group_key = _public_event_group_key(record, source)
                    previous_group_owner = event_group_owner.get(group_key) if group_key is not None else None
                    previous_source_owner = source_owner.get(source_id) if source_id else None
                    if (
                        (previous_group_owner is not None and previous_group_owner != as_of)
                        or (previous_source_owner is not None and previous_source_owner != as_of)
                        or (previous_source_owner == as_of)
                    ):
                        continue
                    kept_evidence.append(source)
                    if group_key is not None:
                        event_group_owner.setdefault(group_key, as_of)
                    if source_id:
                        source_owner.setdefault(source_id, as_of)
                if kept_evidence:
                    filtered_records.append({**record, "evidence": kept_evidence})
            if collection in snapshot or filtered_records:
                filtered_snapshot[collection] = filtered_records
        snapshot = filtered_snapshot
        cases.append(HarnessCase(as_of, snapshot, split, _required_observations(snapshot)))
    return cases


def split_retrospective_cases(cases: list[HarnessCase]) -> dict[str, list[HarnessCase]]:
    groups = {
        "exploration": [case for case in cases if case.split == "exploration"],
        "selection": [case for case in cases if case.split == "selection"],
        "frozen_evaluation": [case for case in cases if case.split == "frozen_evaluation"],
        "reuse_demo": [case for case in cases if case.split == "reuse_demo"],
    }
    if tuple(map(len, (groups["exploration"], groups["selection"], groups["frozen_evaluation"], groups["reuse_demo"]))) != (3, 2, 2, 1):
        raise ValueError("retrospective split must contain 3/2/2 evaluation cases and one distinct reuse case")
    evaluation_dates = {
        case.as_of
        for name in ("exploration", "selection", "frozen_evaluation")
        for case in groups[name]
    }
    if groups["reuse_demo"][0].as_of in evaluation_dates:
        raise ValueError("the reuse task must use a retrospective date outside the evaluation splits")
    return groups


def _selection_summary(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    metrics = [row["metrics"] for row in rows]
    fields = (
        "fact_accuracy", "required_fact_coverage", "citation_precision", "report_period_accuracy",
        "point_in_time_accuracy", "tool_efficiency", "cost_efficiency", "score",
    )
    return {
        "case_count": len(rows),
        "metrics": {
            field: sum(float(row[field]) for row in metrics) / len(metrics)
            for field in fields
        },
        "per_case": [
            {"as_of": row["as_of"], **row["metrics"]}
            for row in rows
        ],
    }


def _quality_gates_hold(candidate: dict[str, object], baseline: dict[str, object]) -> bool:
    for metric in (
        "fact_accuracy", "required_fact_coverage", "citation_precision",
        "report_period_accuracy", "point_in_time_accuracy",
    ):
        if float(candidate[metric]) < float(baseline[metric]):
            return False
    return float(candidate["score"]) > float(baseline["score"])


def run_gepa_evolution(
    cases: dict[str, list[HarnessCase]],
    *,
    research: ResearchData,
    client: QwenClient,
    budget: CallBudget,
    version_store: HarnessVersionStore,
    run_id: str,
    config: dict[str, object],
    reuse_demo: bool = True,
) -> dict[str, object]:
    run_call_start = len(budget.calls)
    run_provider_call_start = sum(
        row.get("status") in {"started", "completed"} for row in budget.calls
    )
    run_logical_start = len(budget.logical_calls)
    active_candidate, active_version_id, active_version = version_store.load_active()
    if len(active_candidate) != GEPA_REFLECTION_COMPONENTS:
        raise ValueError("the GEPA call budget is fixed for exactly six evolvable text components")
    exploration_case_count = len(cases["exploration"])
    if (
        exploration_case_count != 3
        or len(cases["selection"]) != SELECTION_CASE_LIMIT
        or len(cases["frozen_evaluation"]) != FROZEN_CASE_LIMIT
        or (reuse_demo and len(cases["reuse_demo"]) != 1)
    ):
        raise ValueError("the reviewed GEPA protocol requires 3/2/2 exploration, selection and frozen cases")
    case_summary = summarize_retrospective_cases([
        case
        for name in ("exploration", "selection", "frozen_evaluation", "reuse_demo")
        for case in cases[name]
    ])
    if not case_summary["eligible_for_model_execution"]:
        raise ValueError(
            "GEPA evaluation is blocked until every case has two required public facts and two sources with no repeated public event groups"
        )
    unique_exploration_evaluations = exploration_case_count * (1 + GEPA_PROPOSAL_LIMIT)
    planned_research_turns = (
        unique_exploration_evaluations
        + 2 * len(cases["selection"])
        + 2 * len(cases["frozen_evaluation"])
        + (len(cases["reuse_demo"]) if reuse_demo else 0)
    )
    planned_calls = planned_research_turns * MAX_RESEARCH_CALLS_PER_CASE + (
        GEPA_REFLECTION_REQUESTS_PER_PROPOSAL * GEPA_PROPOSAL_LIMIT
    )
    remaining_calls = budget.remaining_provider_calls
    if budget.unresolved_call:
        raise RuntimeError("the shared call ledger has an unknown outbound outcome; resolve it before evolving")
    # Exact cached responses from the interrupted first task may satisfy some
    # planned model requests; CallBudget.begin_call remains the hard provider ceiling.
    run_dir = version_store.root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    reflection_payloads: list[dict[str, object]] = []

    def reflection_lm(prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        payload = {
            "model": client.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": MAX_OUTPUT_TOKENS_PER_CALL,
            "enable_thinking": False,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(serialized) > MAX_INPUT_BYTES_PER_CALL:
            raise ValueError(
                f"rendered GEPA reflection request is {len(serialized)} bytes; call limit is {MAX_INPUT_BYTES_PER_CALL}"
            )
        prompt_text = "\n".join(
            part.get("content", "") if isinstance(part.get("content"), str) else ""
            for message in messages
            for part in ([message] if isinstance(message.get("content"), str) else message.get("content", []))
            if isinstance(part, dict)
        )
        matching_components = [key for key, value in active_candidate.items() if value in prompt_text]
        component = matching_components[0] if len(matching_components) == 1 else "component_identity_unresolved"
        request_sha256 = hashlib.sha256(client.endpoint.encode("utf-8") + b"\0" + serialized).hexdigest()
        reflection_payloads.append({
            "component": component,
            "bytes": len(serialized),
            "request_sha256": request_sha256,
            "request_payload": payload,
        })
        _write_json(run_dir / "reflection_payload_review.json", {
            "schema_version": "adaptive-mas.reflection-payload-review.v1",
            "status": "captured_before_provider_request",
            "contains_api_credential": False,
            "payloads": reflection_payloads,
        })
        response = client.complete(
            messages,
            budget=budget,
            role=f"gepa_reflection:{component}",
            max_output_tokens=MAX_OUTPUT_TOKENS_PER_CALL,
        )
        return str(response["choices"][0]["message"]["content"])

    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    def evaluator(candidate, example):
        case = case_by_date[date.fromisoformat(str(example["as_of"]))]
        key = hashlib.sha256(
            (_json(candidate) + case.as_of.isoformat()).encode("utf-8")
        ).hexdigest()
        if key in evaluation_cache:
            return evaluation_cache[key]
        candidate_memory = (
            active_version.get("procedural_memory")
            if dict(candidate) == active_candidate else None
        )
        score, side_info = evaluate_candidate(
            dict(candidate), case,
            research=research,
            client=client,
            budget=budget,
            config=config,
            candidate_version_id=active_version_id if candidate_memory is not None else "gepa-proposal",
            procedural_memory=candidate_memory if isinstance(candidate_memory, dict) else None,
        )
        evaluation_cache[key] = (score, side_info)
        return score, side_info

    all_cases = [*cases["exploration"], *cases["selection"], *cases["frozen_evaluation"]]
    case_by_date = {case.as_of: case for case in all_cases}
    evaluation_cache: dict[str, tuple[float, dict[str, object]]] = {}
    gepa_result = optimize_anything(
        seed_candidate=active_candidate,
        evaluator=evaluator,
        dataset=[case.evaluator_input() for case in cases["exploration"]],
        objective=(
            "Improve public research factual accuracy, citation precision, report-period correctness, "
            "point-in-time compliance, research-tool efficiency, and measured model cost."
        ),
        background=(
            "Evolve only the supplied market, fundamental, manager, context, handoff and procedural-experience "
            "text components. Tool schemas, evaluator weights, data split, expected-return calibration and portfolio "
            "optimizer are fixed. All cases are retrospective cached data; never put accounts, private forecasts, "
            "trades, internal source paths or portfolio results in reflection feedback."
        ),
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=GEPA_METRIC_CALL_LIMIT,
                max_candidate_proposals=GEPA_PROPOSAL_LIMIT,
                candidate_selection_strategy="current_best",
                acceptance_criterion="strict_improvement",
                parallel=False,
                cache_evaluation=True,
                run_dir=str(run_dir),
                display_progress_bar=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=1,
                module_selector="all",
            ),
        ),
    )
    accepted_proposal_count = max(0, len(gepa_result.candidates) - 1)
    generated_proposal_count = max(
        accepted_proposal_count,
        int(gepa_result.total_metric_calls > len(cases["exploration"])),
    )
    proposed_candidate = dict(gepa_result.candidates[gepa_result.best_idx])
    seed_gepa_score = float(gepa_result.val_aggregate_scores[0])
    proposed_gepa_score = float(gepa_result.val_aggregate_scores[gepa_result.best_idx])
    has_gepa_improvement = (
        gepa_result.best_idx != 0
        and proposed_candidate != active_candidate
        and proposed_gepa_score > seed_gepa_score
    )
    candidate_audit = []
    for index, candidate in enumerate(gepa_result.candidates):
        candidate = dict(candidate)
        candidate_sha256 = hashlib.sha256(_json(candidate).encode("utf-8")).hexdigest()
        changed_components = [
            key for key in active_candidate
            if candidate.get(key) != active_candidate.get(key)
        ]
        candidate_diff_text = "".join(
            line
            for key in sorted(active_candidate)
            if candidate.get(key) != active_candidate.get(key)
            for line in difflib.unified_diff(
                active_candidate[key].splitlines(keepends=True),
                candidate[key].splitlines(keepends=True),
                fromfile=f"active/{key}",
                tofile=f"gepa-candidate-{index}/{key}",
            )
        )
        diff_name = f"candidate-{index}.diff"
        candidate_name = f"candidate-{index}.json"
        (run_dir / diff_name).write_text(candidate_diff_text, encoding="utf-8")
        _write_json(run_dir / candidate_name, candidate)
        candidate_audit.append({
            "candidate_index": index,
            "sha256": candidate_sha256,
            "parent_indices": gepa_result.parents[index],
            "validation_score": float(gepa_result.val_aggregate_scores[index]),
            "per_example_validation_scores": {
                str(as_of): float(score)
                for as_of, score in gepa_result.val_subscores[index].items()
            },
            "discovery_metric_calls": gepa_result.discovery_eval_counts[index],
            "changed_components": changed_components,
            "is_gepa_best": index == gepa_result.best_idx,
            "selection_reason": (
                "best GEPA candidate; eligible for independent selection"
                if index == gepa_result.best_idx and has_gepa_improvement else
                "seed candidate" if index == 0 else
                "GEPA best did not strictly improve the exploration score"
                if index == gepa_result.best_idx else
                "not the selected GEPA best candidate"
            ),
            "candidate_file": candidate_name,
            "diff_file": diff_name,
        })
    _write_json(run_dir / "candidate_audit.json", {
        "schema_version": "adaptive-mas.gepa-candidate-audit.v1",
        "optimizer": "gepa.optimize_anything",
        "component_selector": "all",
        "components_eligible_for_reflection": sorted(active_candidate),
        "candidate_count": len(candidate_audit),
        "candidate_proposals_generated": generated_proposal_count,
        "candidate_proposals_rejected": max(0, generated_proposal_count - accepted_proposal_count),
        "parents": gepa_result.parents,
        "best_index": gepa_result.best_idx,
        "seed_score": seed_gepa_score,
        "best_score": proposed_gepa_score,
        "total_metric_calls": gepa_result.total_metric_calls,
        "candidates": candidate_audit,
    })

    selection_rows = {"active": [], "candidate": []}
    if has_gepa_improvement:
        for case in cases["selection"]:
            for name, candidate, candidate_version, memory in (
                ("active", active_candidate, active_version_id, active_version.get("procedural_memory")),
                ("candidate", proposed_candidate, "gepa-proposal", None),
            ):
                score, info = evaluate_candidate(
                    candidate, case, research=research, client=client, budget=budget, config=config,
                    candidate_version_id=candidate_version,
                    procedural_memory=memory if isinstance(memory, dict) else None,
                )
                selection_rows[name].append({
                    "as_of": case.as_of.isoformat(),
                    "score": score,
                    "metrics": info["quality_metrics"],
                })
        baseline_selection = _selection_summary(selection_rows["active"])
        candidate_selection = _selection_summary(selection_rows["candidate"])
        accepted = _quality_gates_hold(candidate_selection["metrics"], baseline_selection["metrics"])
        reason = (
            "candidate strictly improved the fixed selection score without reducing factuality, citations, "
            "report-period correctness or point-in-time compliance"
            if accepted else
            "candidate rejected by independent fixed selection evaluator; active version retained"
        )
    else:
        baseline_selection = None
        candidate_selection = None
        accepted = False
        reason = (
            "GEPA did not select a distinct candidate with a strict exploration-score gain; "
            "independent selection was not run"
        )
    version_id = active_version_id
    candidate_diff = ""
    if accepted:
        version_id, candidate_diff = version_store.adopt(
            proposed_candidate,
            previous_version_id=active_version_id,
            selection={"active": baseline_selection, "candidate": candidate_selection},
            reason=reason,
            run_id=run_id,
        )
    else:
        current = active_candidate
        candidate_diff = "".join(
            line
            for key in sorted(current)
            for line in difflib.unified_diff(
                current[key].splitlines(keepends=True),
                proposed_candidate[key].splitlines(keepends=True),
                fromfile=f"active/{key}",
                tofile=f"rejected-gepa-candidate/{key}",
            )
        )
    selected_candidate, selected_version_id, selected_version = version_store.load_active()
    selected_memory = selected_version.get("procedural_memory")
    frozen_rows = {"active_baseline": [], "selected_version": []}
    for case in cases["frozen_evaluation"]:
        for label, candidate, candidate_version, memory in (
            ("active_baseline", active_candidate, active_version_id, active_version.get("procedural_memory")),
            ("selected_version", selected_candidate, selected_version_id, selected_memory),
        ):
            score, info = evaluate_candidate(
                candidate, case, research=research, client=client, budget=budget, config=config,
                candidate_version_id=candidate_version,
                procedural_memory=memory if isinstance(memory, dict) else None,
            )
            frozen_rows[label].append({
                "as_of": case.as_of.isoformat(),
                "score": score,
                "version_id": candidate_version,
                "metrics": info["quality_metrics"],
                "task_model_usage": info["task_model_usage"],
                "data_status": "retrospective_frozen_evaluation",
            })
    reuse_demo_result = None
    if reuse_demo:
        fresh_case = cases["reuse_demo"][0]
        if fresh_case.as_of in {case.as_of for case in cases["frozen_evaluation"]}:
            raise ValueError("reuse demonstration must use a distinct date for a genuinely new task")
        reloaded_candidate, reloaded_version_id, reloaded_version = version_store.load_active()
        reloaded_memory = reloaded_version.get("procedural_memory")
        memory_content_loaded = bool(
            isinstance(reloaded_memory, dict)
            and reloaded_memory.get("loaded_from_persistent_store")
            and reloaded_memory.get("content_sha256") == hashlib.sha256(
                reloaded_candidate["procedural_experience"].encode("utf-8")
            ).hexdigest()
        )
        reuse_trace: list[dict[str, object]] = []
        score, info = evaluate_candidate(
            reloaded_candidate, fresh_case,
            research=research, client=client, budget=budget, config=config,
            candidate_version_id=reloaded_version_id,
            procedural_memory=reloaded_memory if isinstance(reloaded_memory, dict) else None,
            trace_sink=reuse_trace,
        )
        reuse_result_file = "reuse_demo_research.json"
        _write_json(run_dir / reuse_result_file, reuse_trace[0])
        reuse_demo_result = {
            "status": "active_version_loaded_for_new_task",
            "version_id": reloaded_version_id,
            "task_id": f"fresh-research-task-{fresh_case.as_of.isoformat()}",
            "as_of": fresh_case.as_of.isoformat(),
            "candidate_sha256": hashlib.sha256(_json(reloaded_candidate).encode("utf-8")).hexdigest(),
            "procedural_memory_reloaded": memory_content_loaded,
            "procedural_memory": reloaded_memory,
            "research_result_file": reuse_result_file,
            "score": score,
            "metrics": info["quality_metrics"],
            "task_model_usage": info["task_model_usage"],
        }

    run_calls = budget.calls[run_call_start:]
    run_logical_calls = budget.logical_calls[run_logical_start:]
    output = {
        "schema_version": "adaptive-mas.gepa-evolution.v1",
        "run_id": run_id,
        "status": "accepted" if accepted else "rejected",
        "optimizer": "gepa.optimize_anything",
        "active_version_before": active_version_id,
        "active_version_after": selected_version_id,
        "candidate_proposals_configured": GEPA_PROPOSAL_LIMIT,
        "candidate_proposals_generated": generated_proposal_count,
        "candidate_proposals_rejected": max(0, generated_proposal_count - accepted_proposal_count),
        "metric_call_limit": GEPA_METRIC_CALL_LIMIT,
        "reflection_component_selector": "all",
        "reflection_component_pool": sorted(active_candidate),
        "reflection_components": sorted({row["component"] for row in reflection_payloads}),
        "reflection_request_count": len(reflection_payloads),
        "reflection_payload_review_file": "reflection_payload_review.json",
        "gepa_search": {
            "candidate_count": len(gepa_result.candidates),
            "best_candidate_index": gepa_result.best_idx,
            "seed_score": seed_gepa_score,
            "best_score": proposed_gepa_score,
            "strict_exploration_improvement": has_gepa_improvement,
            "total_metric_calls": gepa_result.total_metric_calls,
            "discovery_evaluation_counts": gepa_result.discovery_eval_counts,
            "candidate_audit_file": "candidate_audit.json",
        },
        "planned_provider_call_upper_bound": planned_calls,
        "planned_unique_exploration_evaluation_pairs": unique_exploration_evaluations,
        "gepa_metric_call_limit_is_callback_cap": True,
        "planned_external_research_tasks": (
            2 * len(cases["selection"])
            + 2 * len(cases["frozen_evaluation"])
            + (len(cases["reuse_demo"]) if reuse_demo else 0)
        ),
        "provider_calls_before_run": run_provider_call_start,
        "remaining_provider_call_ceiling_before_run": budget.max_calls - run_provider_call_start,
        "provider_calls_this_run": sum(row.get("status") in {"started", "completed"} for row in run_calls),
        "request_attempts_this_run": len(run_calls),
        "logical_inferences_this_run": len(run_logical_calls),
        "planned_research_turn_upper_bound": planned_research_turns * MAX_RESEARCH_CALLS_PER_CASE,
        "max_research_turns_per_case": MAX_RESEARCH_CALLS_PER_CASE,
        "planned_cost_upper_bound_cny": (
            planned_calls
            * (
                MAX_INPUT_BYTES_PER_CALL * INPUT_PRICE_CNY_PER_MILLION
                + MAX_OUTPUT_TOKENS_PER_CALL * OUTPUT_PRICE_CNY_PER_MILLION
            )
            / 1_000_000
        ),
        "hard_provider_call_ceiling": budget.max_calls,
        "remaining_provider_call_ceiling_after_run": budget.remaining_provider_calls,
        "current_run_cost": {
            "logical_estimated_cost_cny": sum(
                float(row["logical_estimated_cost_cny"]) for row in run_logical_calls
            ),
            "actual_incremental_api_cost_cny": sum(
                float(row.get("actual_estimated_cost_cny", 0.0)) for row in run_calls
            ),
        },
        "selection": {
            "active": baseline_selection,
            "candidate": candidate_selection,
            "accepted": accepted,
            "reason": reason,
        },
        "candidate_diff": candidate_diff,
        "candidate_audit_file": "candidate_audit.json",
        "procedural_memory": {
            "active_version_before": active_version.get("procedural_memory"),
            "active_version_after": selected_version.get("procedural_memory"),
            "reuse_demo_loaded_persisted_memory": (
                bool(reuse_demo_result and reuse_demo_result["procedural_memory_reloaded"])
            ),
        },
        "frozen_evaluation": {
            "status": "retrospective_only",
            "active_baseline_version_id": active_version_id,
            "selected_version_id": selected_version_id,
            "paired_cases": frozen_rows,
        },
        "reuse_demo": reuse_demo_result,
        "cost": budget.summary(),
        "data_limits": {
            "all_examples_are_retrospective": True,
            "split_counts": {
                "exploration": len(cases["exploration"]),
                "selection": len(cases["selection"]),
                "frozen_evaluation": len(cases["frozen_evaluation"]),
                "reuse_demo": len(cases["reuse_demo"]),
            },
            "exploration_cases": [case.as_of.isoformat() for case in cases["exploration"]],
            "selection_cases": [case.as_of.isoformat() for case in cases["selection"]],
            "frozen_evaluation_cases": [case.as_of.isoformat() for case in cases["frozen_evaluation"]],
            "reuse_demo_case": cases["reuse_demo"][0].as_of.isoformat(),
            "event_groups_are_deduplicated_from_first_seen_case": True,
            "test_csv_read": False,
        },
    }
    _write_json(run_dir / "evolution_report.json", output)
    (run_dir / "candidate.diff").write_text(candidate_diff, encoding="utf-8")
    return output
