"""Emit an auditable P9/P10 expert snapshot and P11 evidence-memory ledger."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import statistics
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from adaptive_mas.data import load_oos_research_data
from adaptive_mas.agents.public_experts import build_public_expert_snapshot
from adaptive_mas.evaluation.expert_memory import (
    build_evidence_challenges,
    update_matured_performance_memory,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _snapshots_before(root: Path, decision_time: datetime) -> list[tuple[Path, dict[str, object]]]:
    snapshots = []
    for path in root.glob("*/expert_snapshot.json"):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(str(snapshot["decision_time"]))
        if timestamp < decision_time:
            snapshots.append((path, snapshot))
    return sorted(snapshots, key=lambda item: str(item[1]["decision_time"]))


def _snapshot_signals(snapshot: dict[str, object]) -> list[dict[str, object]]:
    return (
        list(snapshot["p9_outputs"])
        + list(snapshot["p10_outputs"])
        + list(snapshot["llm_outputs"]["outputs"])
        + list(snapshot.get("carry_forward_expert_signals", []))
    )


def _last_trading_event_start(
    snapshots: list[tuple[Path, dict[str, object]]],
    trading_dates: tuple[date, ...],
    decision_time: datetime,
) -> datetime:
    trading_date_set = set(trading_dates)
    if snapshots:
        previous = snapshots[-1][1]
        previous_time = datetime.fromisoformat(str(previous["decision_time"]))
        if previous_time.date() in trading_date_set:
            return previous_time
        if previous.get("event_start_time"):
            return datetime.fromisoformat(str(previous["event_start_time"]))
    prior_sessions = [day for day in trading_dates if day < decision_time.date()]
    if prior_sessions:
        return datetime.combine(prior_sessions[-1], time(15, 0), SHANGHAI)
    return decision_time - timedelta(days=3)


def _observed_record_ids(snapshots: list[tuple[Path, dict[str, object]]]) -> set[str]:
    record_ids = set()
    for _, snapshot in snapshots:
        for row in snapshot.get("llm_outputs", {}).get("outputs", []):
            if row.get("forecast_of_record_id"):
                record_ids.add(str(row["forecast_of_record_id"]))
    return record_ids


def _carry_forward_llm_signals(
    snapshots: list[tuple[Path, dict[str, object]]],
    allowed_symbols: set[str],
    event_start: datetime,
    decision_time: datetime,
) -> list[dict[str, object]]:
    carry_by_record_id: dict[str, dict[str, object]] = {}
    for _, previous_snapshot in snapshots:
        for row in previous_snapshot.get("llm_outputs", {}).get("outputs", []):
            available_value = row.get("available_at") or max(
                (evidence.get("available_at") for evidence in row.get("evidence", []) if evidence.get("available_at")),
                default=None,
            )
            if (
                available_value is None
                or datetime.fromisoformat(str(available_value)) <= event_start
                or row.get("symbol") is None
                or str(row["symbol"]).rsplit(".", 1)[-1] not in allowed_symbols
            ):
                continue
            carry_by_record_id[str(row["record_id"])] = {
                **row,
                "carried_from_record_id": str(row["record_id"]),
                "carried_from_decision_time": str(previous_snapshot["decision_time"]),
                "record_id": f"{row['record_id']}:carry:{decision_time.date().isoformat()}",
                "prediction_date": decision_time.date().isoformat(),
                "available_at": str(available_value),
                "decision_effect": "research_only_until_matured_comparison",
            }
    return list(carry_by_record_id.values())


def _matured_outcomes(
    snapshot_rows: list[dict[str, object]], research, cutoff_date: date
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    date_index = {day: index for index, day in enumerate(research.trading_dates)}
    bars = {(str(row["symbol"]), row["trade_date"]): row for row in research.adjusted_bars}
    index_bars = {row["trade_date"]: row for row in research.index_bars}
    forecasts = []
    pending = []
    outcomes = []
    seen = set()
    seen_state_versions = set()
    for snapshot in snapshot_rows:
        prediction_date = datetime.fromisoformat(str(snapshot["decision_time"])).date()
        if prediction_date not in date_index:
            for row in _snapshot_signals(snapshot):
                if row.get("symbol") is not None and row.get("horizon_sessions"):
                    pending.append({"record_id": row["record_id"], "reason": "decision_date_not_a_market_session"})
            continue
        for row in _snapshot_signals(snapshot):
            if row.get("symbol") is None or not row.get("horizon_sessions"):
                continue
            if row.get("record_kind") == "state" and row["expert"] != "fundamental_rule_v1":
                continue
            if row.get("record_kind") == "state":
                if str(row["record_id"]) in seen_state_versions:
                    continue
                seen_state_versions.add(str(row["record_id"]))
            key = (str(row["record_id"]), prediction_date.isoformat())
            if key in seen:
                continue
            seen.add(key)
            normalized_symbol = str(row["symbol"]).rsplit(".", 1)[-1]
            forecast = {
                **row,
                "symbol": normalized_symbol,
                "prediction_date": prediction_date.isoformat(),
            }
            forecasts.append(forecast)
            index = date_index[prediction_date]
            horizon = int(row["horizon_sessions"])
            entry_index = index + 1
            exit_index = index + horizon
            if exit_index >= len(research.trading_dates):
                pending.append({"record_id": row["record_id"], "reason": "forward_label_not_in_market_cache"})
                continue
            entry_date = research.trading_dates[entry_index]
            exit_date = research.trading_dates[exit_index]
            if exit_date >= cutoff_date:
                pending.append({"record_id": row["record_id"], "reason": "forward_label_not_mature_before_decision_date"})
                continue
            symbol = normalized_symbol
            entry = bars.get((symbol, entry_date))
            exit_bar = bars.get((symbol, exit_date))
            index_entry = index_bars.get(entry_date)
            index_exit = index_bars.get(exit_date)
            if (
                entry is None or exit_bar is None or index_entry is None or index_exit is None
                or entry["open"] is None or exit_bar["close"] is None
                or index_entry["open"] is None or index_exit["close"] is None
            ):
                pending.append({"record_id": row["record_id"], "reason": "required_stock_or_index_bar_missing"})
                continue
            stock_return = float(exit_bar["close"]) / float(entry["open"]) - 1
            market_return = float(index_exit["close"]) / float(index_entry["open"]) - 1
            outcome_ref = f"baostock:{symbol}:{entry_date.isoformat()}:{exit_date.isoformat()}"
            forecasts[-1]["planned_exit_date"] = exit_date.isoformat()
            yield_outcome = {
                "record_id": str(row["record_id"]),
                "label_available_at": datetime.combine(exit_date, time(16, 0), SHANGHAI).isoformat(),
                "horizon_sessions": horizon,
                "forward_return": stock_return - market_return,
                "stock_forward_return": stock_return,
                "market_forward_return": market_return,
                "outcome_ref": outcome_ref,
            }
            matured = yield_outcome
            # A label can only be stored when its local availability is before the
            # active decision date; cutoff_date is the observation date.
            if datetime.fromisoformat(matured["label_available_at"]).date() >= cutoff_date:
                pending.append({"record_id": row["record_id"], "reason": "label_not_available_before_cutoff"})
                continue
            outcomes.append(matured)
    return forecasts, pending, outcomes


def _candidate_effects(memory: list[dict[str, object]], market_dates: tuple[date, ...]) -> dict[str, dict[str, object]]:
    index_by_date = {day.isoformat(): index for index, day in enumerate(market_dates)}
    by_expert: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in memory:
        if float(row["directional_return"]) == 0:
            continue
        group = (str(row["expert"]), int(row["horizon_sessions"]))
        by_expert.setdefault(group, {}).setdefault(str(row["prediction_date"]), []).append(
            float(row["directional_return"])
        )
    effects = {}
    for (expert, horizon), values_by_date in sorted(by_expert.items()):
        ordered_dates = sorted(values_by_date)
        nonoverlap = []
        previous_index = None
        for day in ordered_dates:
            current_index = index_by_date[day]
            if previous_index is None or current_index - previous_index >= horizon:
                nonoverlap.append(statistics.fmean(values_by_date[day]))
                previous_index = current_index
        count = len(nonoverlap)
        mean = statistics.fmean(nonoverlap) if nonoverlap else 0.0
        standard_error = statistics.stdev(nonoverlap) / count**0.5 if count > 1 else None
        lower = mean - 1.96 * standard_error if standard_error is not None else None
        effects[f"{expert}:{horizon}s"] = {
            "matured_nonoverlapping_decision_dates": count,
            "mean_directional_active_return": mean if count else None,
            "normal_95pct_lower_bound": lower,
            "minimum_nonoverlapping_decision_dates": 20,
            "candidate_return_adjustment": lower if count >= 20 and lower is not None and lower > 0 else 0.0,
            "status": "candidate_effect_eligible" if count >= 20 and lower is not None and lower > 0 else "pending_or_no_positive_lower_bound",
        }
    return effects


def _read_symbol_lines(path: Path | None) -> set[str]:
    if path is None:
        return set()
    symbols = {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    if any(len(symbol) != 6 or not symbol.isascii() or not symbol.isdigit() for symbol in symbols):
        raise ValueError(f"代码文件须每行一个六位数字代码：{path}")
    return symbols


def _held_symbols_as_of(path: Path, market_date: date) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    available = [row for row in rows if date.fromisoformat(str(row["trade_date"])) <= market_date]
    if not available:
        raise ValueError(f"账户持仓文件没有不晚于 {market_date} 的持仓记录：{path}")
    latest_date = max(date.fromisoformat(str(row["trade_date"])) for row in available)
    symbols = {
        str(row["symbol"])
        for row in available
        if date.fromisoformat(str(row["trade_date"])) == latest_date
        and float(row["market_value_cny"]) > 0
    }
    if any(len(symbol) != 6 or not symbol.isascii() or not symbol.isdigit() for symbol in symbols):
        raise ValueError(f"账户持仓文件含无效证券代码：{path}")
    return symbols


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-cache-dir", type=Path, default=project / "data" / "raw" / "akshare")
    parser.add_argument("--decision-time", type=datetime.fromisoformat, required=True)
    parser.add_argument("--previous-snapshot", type=Path)
    parser.add_argument("--history-root", type=Path)
    parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    parser.add_argument("--baostock-run-dir", type=Path, default=project / "data" / "raw" / "baostock" / "fetch-20260924T150656Z")
    parser.add_argument("--account-positions-csv", type=Path, default=project / "experiments" / "p9_forward_cutover_20260927_v9" / "p9_fixed_market_relative" / "positions.csv")
    parser.add_argument("--candidate-symbols-file", type=Path)
    parser.add_argument("--horizon-sessions", type=int, required=True)
    parser.add_argument("--llm-max-records", type=int, default=32)
    parser.add_argument("--announcement-text-limit", type=int, default=3)
    parser.add_argument("--announcement-id", action="append", help="限定验证指定巨潮公告 ID；可重复传入")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖现有快照目录：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    history_root = (args.history_root or output.parent).resolve()
    previous_candidates = _snapshots_before(history_root, args.decision_time) if args.decision_time.tzinfo else []
    output.mkdir(parents=True)
    status_path = output / "RUN_STATUS.json"
    status = {
        "status": "running",
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    def progress(message: str) -> None:
        status["last_progress"] = message
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(status_path, status)
        with (output / "progress.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{status['updated_at_utc']} {message}\n")
        print(f"[P9-P12 snapshot] {message}", flush=True)

    _write_json(status_path, status)
    try:
        if args.decision_time.tzinfo is None or args.decision_time.utcoffset() is None:
            raise ValueError("decision-time 必须带时区")
        decision_time = args.decision_time.astimezone(SHANGHAI)
        progress("loading the as-of research universe and existing account positions")
        research = load_oos_research_data(args.baostock_run_dir)
        available_market_dates = [day for day in research.trading_dates if day <= decision_time.date()]
        if not available_market_dates:
            raise ValueError("决策时点之前没有可用的研究交易日")
        market_date = available_market_dates[-1]
        if args.previous_snapshot is not None:
            previous = json.loads(args.previous_snapshot.read_text(encoding="utf-8"))
            if all(path.resolve() != args.previous_snapshot.resolve() for path, _ in previous_candidates):
                previous_candidates.append((args.previous_snapshot.resolve(), previous))
            previous_candidates.sort(key=lambda item: str(item[1]["decision_time"]))
        event_start = _last_trading_event_start(
            previous_candidates, research.trading_dates, decision_time
        )
        previously_observed_record_ids = _observed_record_ids(previous_candidates)
        current_members = set(research.membership_at(market_date))
        held_symbols = _held_symbols_as_of(args.account_positions_csv, market_date)
        candidate_symbols = _read_symbol_lines(args.candidate_symbols_file)
        allowed_symbols = current_members | held_symbols | candidate_symbols
        if not allowed_symbols:
            raise ValueError("当日可分析股票池为空")
        progress(
            f"public company scope has {len(current_members)} current members, "
            f"{len(held_symbols)} held positions and {len(candidate_symbols)} supplied candidates"
        )
        progress("loading point-in-time public records and extracting a bounded set of CNINFO PDFs")
        snapshot = build_public_expert_snapshot(
            args.public_cache_dir,
            decision_time,
            args.horizon_sessions,
            event_start_time=event_start,
            llm_max_records=args.llm_max_records,
            allowed_symbols=allowed_symbols,
            announcement_text_limit=args.announcement_text_limit,
            announcement_document_dir=output / "announcement_documents",
            announcement_ids=set(args.announcement_id) if args.announcement_id else None,
            previously_observed_record_ids=previously_observed_record_ids,
        )
        snapshot["carry_forward_expert_signals"] = _carry_forward_llm_signals(
            previous_candidates, allowed_symbols, event_start, decision_time
        )
        snapshot["symbol_scope_inputs"] = {
            "as_of_market_date": market_date.isoformat(),
            "current_index_member_count": len(current_members),
            "existing_account_position_count": len(held_symbols),
            "candidate_file_count": len(candidate_symbols),
            "account_positions_file": str(args.account_positions_csv.resolve()),
            "candidate_symbols_file": str(args.candidate_symbols_file.resolve()) if args.candidate_symbols_file else None,
        }
        _write_json(output / "expert_snapshot.json", snapshot)
        previous_snapshots = [row for _, row in previous_candidates]
        snapshots = previous_snapshots + [snapshot]
        signals = _snapshot_signals(snapshot)
        challenges = build_evidence_challenges(signals)
        progress("maturing only saved expert records whose three-session-or-configured labels have ended")
        forecast_history, pending, outcomes = _matured_outcomes(
            snapshots, research, decision_time.date()
        )
        memory_cutoff = datetime.combine(decision_time.date(), time.min, SHANGHAI)
        performance = update_matured_performance_memory(
            forecast_history, outcomes, memory_cutoff
        )
        candidate_effects = _candidate_effects(performance, research.trading_dates)
        snapshot["p11_challenge_count"] = len(challenges)
        snapshot["p11_matured_performance_count"] = len(performance)
        snapshot["p11_candidate_effects"] = candidate_effects
        _write_json(output / "expert_snapshot.json", snapshot)
        _write_jsonl(output / "evidence_challenges.jsonl", challenges)
        _write_jsonl(output / "performance_memory.jsonl", performance)
        _write_jsonl(output / "matured_outcomes.jsonl", outcomes)
        _write_jsonl(output / "pending_labels.jsonl", pending)
        comparison = {
            "status": "awaiting_forward_maturity" if not performance else "candidate_effects_available_replay_required",
            "captured_forecast_count": len(forecast_history),
            "matured_outcome_count": len(outcomes),
            "usable_performance_memory_count": len(performance),
            "pending_forecast_count": len(pending),
            "candidate_effects_by_expert": candidate_effects,
            "same_account_experiment_status": "not_run_by_snapshot_command",
            "p8_p9_incremental_net_return": None,
            "p8_p9_incremental_cost_cny": None,
            "eligible_oos_count": sum(row["status"] == "candidate_effect_eligible" for row in candidate_effects.values()),
            "live_decision_effect": "research_only",
        }
        _write_json(output / "incremental_evaluation.json", comparison)
        (output / "RESEARCH_REPORT.md").write_text(
            "\n".join([
                "# P9–P12 新增专家与控制器运行记录",
                "",
                f"决策时点：{decision_time.isoformat()}。评价期限：{args.horizon_sessions} 个交易日。",
                "",
                f"P9 结构化基本面状态 {sum(row['expert'] == 'fundamental_rule_v1' for row in snapshot['p9_outputs'])} 条；"
                f"新增公告标题事件 {snapshot['source_coverage']['new_announcements']} 条；"
                f"P10 新新闻提及 {len(snapshot['p10_outputs'])} 条，宏观状态 {len(snapshot['macro_outputs'])} 条。",
                f"Qwen 状态：{snapshot['llm_outputs']['status']}；调用估算成本：{snapshot['llm_outputs']['estimated_token_cost_cny']} 元；"
                f"调用延迟：{snapshot['llm_outputs']['elapsed_ms']:.1f} ms；"
                f"本次新分析 {len(snapshot['llm_outputs']['outputs'])} 条，沿用仍可见周末分析 {len(snapshot['carry_forward_expert_signals'])} 条。",
                f"巨潮公告全文：{snapshot['announcement_full_text']['status']}；"
                f"最多读取 {snapshot['announcement_full_text']['maximum_document_count']} 份，"
                f"本次成功提取 {snapshot['announcement_full_text']['extracted_count']} 份。",
                f"P11 有证据的未解决分歧 {len(challenges)} 条；本次可用成熟绩效记忆 {len(performance)} 条。",
                f"全历史已保存专家预测 {len(forecast_history)} 条，已成熟结果 {len(outcomes)} 条，待成熟/不匹配行情记录 {len(pending)} 条。",
                "",
                "当前财务和公告响应于 2026-09-27 回填。读取严格要求本地抓取不晚于决策时点；因此它们不能提供 2026-09-27 之前的历史点时预测。财务修订版本仍未验证。规则输出只基于标题；本次限定数量的公告 PDF 全文只有在巨潮链接可达且文本提取成功时才交给 LLM，并附 PDF 与文本哈希。新闻仅为 CLS 最近 20 条并要求抓取已发生；宏观只接收有 NBS 正式 PubDate 且完成抓取的值。",
                "",
                "历史 P8/P9 同账户净增量当前不可估：财务/公告回填无历史采集快照，CLS 与宏观也无历史前向 vintage。当前脚本会自动扫描保存的逐日专家快照，用 BaoStock 开盘进场、指定评价期限后的收盘价结算，记录股票相对 CSI 300 的方向性结果，并只把截止当前决策日之前已成熟结果写入 memory。候选调整需至少 20 个不重叠成熟决策日且其 95% 正态下界大于零；当前快照只准备估计，不改动资金决策。run_p9_forward_validation.py 已加入固定 P9 与公共专家 overlay 的同账户重放入口；P12 同账户 RL 与同约束动作基线由 run_p12_rl_control.py 执行。",
                "",
                f"规则计算耗时 {snapshot['cost_and_latency']['rule_engine_elapsed_ms']:.1f} ms；"
                f"快照总耗时 {snapshot['cost_and_latency']['snapshot_elapsed_ms']:.1f} ms。",
            ]) + "\n",
            encoding="utf-8",
        )
        status.update({
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output),
            "llm_status": snapshot["llm_outputs"]["status"],
            "matured_outcomes": len(outcomes),
            "pending_forecasts": len(pending),
        })
        progress("public expert snapshot and evidence ledger complete")
    except Exception as error:
        status.update({
            "status": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
        })
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
