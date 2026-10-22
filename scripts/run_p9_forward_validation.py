"""Replay a P9 shadow account from viewed BaoStock warm-up into genuinely new dates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from adaptive_mas.backtest.simulator import simulate_fixed_risk_market_exposure, simulate_portfolio
from adaptive_mas.data import load_oos_research_data, load_research_data
from adaptive_mas.evaluation.mvp_research import run_mvp_research
from adaptive_mas.evaluation.public_expert_overlay import add_public_expert_overlay
from adaptive_mas.evaluation.uncertainty import paired_block_bootstrap
from run_p9_joint_economics import (
    CALIBRATION_METHODS,
    _adjust_forecast,
    _calibration_for_decision,
    _portfolio_cohorts,
    _read_jsonl_gzip,
    _write_csv,
    _write_json,
)


VIEWED_THROUGH = date(2026, 9, 23)
PREFREEZE_DIAGNOSTIC_THROUGH = date(2026, 9, 24)
FIRST_POST_FREEZE_DATE = date(2026, 9, 28)
FIRST_POST_FREEZE_EXECUTION_DATE = date(2026, 9, 29)


def _partition_forward_dates(trading_dates: list[date] | tuple[date, ...]) -> tuple[list[date], list[date]]:
    """Keep the 9/24 observation diagnostic; score only decisions after the 9/27 freeze."""
    prefreeze = [
        day for day in trading_dates
        if VIEWED_THROUGH < day <= PREFREEZE_DIAGNOSTIC_THROUGH
    ]
    postfreeze = [day for day in trading_dates if day >= FIRST_POST_FREEZE_DATE]
    unclassified = [
        day for day in trading_dates
        if PREFREEZE_DIAGNOSTIC_THROUGH < day < FIRST_POST_FREEZE_DATE
    ]
    if unclassified:
        raise ValueError(f"冻结前出现未分类的交易日：{unclassified}")
    return prefreeze, postfreeze


def _post_freeze_account_decisions(
    forecasts: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep the 9/23 warm-start history and post-freeze decisions, excluding 9/24 actions."""
    def decision_date(row: dict[str, object]) -> date:
        value = row.get("prediction_date", row.get("decision_date"))
        if value is None:
            raise ValueError("账户决策缺少 prediction_date/decision_date")
        return value if isinstance(value, date) else date.fromisoformat(str(value))

    return [
        row for row in forecasts
        if (
            decision_date(row) <= VIEWED_THROUGH
            or decision_date(row) >= FIRST_POST_FREEZE_DATE
        )
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_baostock_responses(run_dir: Path) -> int:
    verified = {}
    with (run_dir / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            relative = str(entry["response_file"])
            expected = str(entry["sha256"])
            if relative in verified:
                if verified[relative] != expected:
                    raise ValueError(f"BaoStock 清单对 {relative} 给出不同哈希")
                continue
            actual = _sha256(run_dir / relative)
            if actual != expected:
                raise ValueError(f"BaoStock 原始响应哈希不符：{relative}")
            verified[relative] = actual
    return len(verified)


def _slice_research(research, end: date):
    return type(research)(
        trading_dates=[day for day in research.trading_dates if day <= end],
        adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= end],
        execution_bars=[row for row in research.execution_bars if row["trade_date"] <= end],
        index_bars=[row for row in research.index_bars if row["trade_date"] <= end],
        membership_snapshots=[row for row in research.membership_snapshots if row.available_at <= end],
    )


def _research_for_as_of(research, as_of_date: date):
    """Slice a verified cache to the requested point-in-time decision boundary."""
    cached_dates = list(research.trading_dates)
    if as_of_date not in cached_dates:
        raise ValueError(f"指定 as-of 日期不在 BaoStock 缓存交易日中：{as_of_date}")
    cache_through = cached_dates[-1]
    excluded_cache_dates = [day for day in cached_dates if day > as_of_date]
    return _slice_research(research, as_of_date), cache_through, excluded_cache_dates


def _paired(left, right, seed: int):
    return paired_block_bootstrap(
        [float(row["daily_return"]) for row in left],
        [float(row["daily_return"]) for row in right],
        block_sessions=20,
        repetitions=2000,
        seed=seed,
    )


def _verify_frozen_manifest(path: Path, project: Path, training_csv: Path) -> dict[str, object]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("account_warm_start_as_of") != VIEWED_THROUGH.isoformat():
        raise ValueError("P9 冻结清单账户暖启动日期不符")
    if frozen.get("pre_freeze_diagnostic_through") != PREFREEZE_DIAGNOSTIC_THROUGH.isoformat():
        raise ValueError("P9 冻结清单冻结前诊断边界不符")
    if frozen.get("first_post_freeze_decision_date") != FIRST_POST_FREEZE_DATE.isoformat():
        raise ValueError("P9 冻结清单首个冻结后决策日不符")
    for relative, expected in frozen["workspace_files_sha256"].items():
        actual = _sha256(project / relative)
        if actual != expected:
            raise ValueError(f"P9 冻结文件已变化：{relative}")
    if _sha256(training_csv) != frozen["training_csv_sha256"]:
        raise ValueError("冻结 train.csv 哈希已变化")
    return frozen


def _account_cutover_state(account: dict[str, object]) -> dict[str, object]:
    last = next(
        row for row in reversed(account["daily_nav"])
        if row["trade_date"] == VIEWED_THROUGH.isoformat()
    )
    positions = {
        row["symbol"]: float(row["market_value_cny"])
        for row in account["positions"]
        if row["trade_date"] == VIEWED_THROUGH.isoformat()
    }
    return {
        "nav_cny": float(last["nav_cny"]),
        "cash_cny": float(last["cash_cny"]),
        "positions_market_value_cny": positions,
    }


def _verify_cutover_account_state(name: str, actual: dict[str, object], frozen: dict[str, object]) -> None:
    if abs(actual["nav_cny"] - frozen["nav_cny"]) > 1e-6:
        raise ValueError(f"{name} 截止日影子账户净值变化")
    if abs(actual["cash_cny"] - frozen["cash_cny"]) > 1e-6:
        raise ValueError(f"{name} 截止日影子账户现金变化")
    if set(actual["positions_market_value_cny"]) != set(frozen["positions_market_value_cny"]):
        raise ValueError(f"{name} 截止日影子账户持仓集合变化")
    for symbol, value in actual["positions_market_value_cny"].items():
        if abs(value - frozen["positions_market_value_cny"][symbol]) > 1e-6:
            raise ValueError(f"{name} 截止日影子账户 {symbol} 持仓金额变化")


def _load_expert_snapshots(root: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    snapshots: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    if not root.exists():
        return snapshots, hashes
    for path in sorted(root.glob("*/expert_snapshot.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        decision_date = datetime.fromisoformat(str(snapshot["decision_time"])).date().isoformat()
        if decision_date in snapshots:
            raise ValueError(f"同一决策日存在多个公共专家快照：{decision_date}")
        snapshots[decision_date] = snapshot
        hashes[decision_date] = _sha256(path)
    return snapshots, hashes


def _new_decision_label_coverage(
    forecasts: list[dict[str, object]],
    settlements: list[dict[str, object]],
    viewed_through: date,
    first_genuinely_new_date: date,
    as_of_date: date,
) -> tuple[list[dict[str, object]], set[str]]:
    settlements_by_date = {}
    for row in settlements:
        settlements_by_date.setdefault(str(row["prediction_date"]), {})[str(row["symbol"])] = row
    coverage = []
    matured_dates = set()
    for forecast in forecasts:
        day = str(forecast["prediction_date"])
        forecast_date = date.fromisoformat(day)
        if forecast_date <= viewed_through or forecast_date < first_genuinely_new_date:
            continue
        needed = {
            symbol
            for method in ("trend", "market_plus_relative_fusion")
            for symbol, weight in forecast["target_weights_by_method"][method].items()
            if float(weight) > 0
        }
        exit_date = forecast["planned_exit_date"]
        rows = settlements_by_date.get(day, {})
        settled = {
            symbol for symbol in needed
            if symbol in rows
            and rows[symbol]["status"] == "settled"
            and rows[symbol]["planned_exit_date"] == exit_date
            and rows[symbol]["available_at"] is not None
            and date.fromisoformat(rows[symbol]["available_at"]) <= as_of_date
            and rows[symbol][f"forward_return_{forecast['horizon_sessions']}s"] is not None
        }
        complete = (
            bool(needed) and exit_date is not None
            and date.fromisoformat(exit_date) <= as_of_date
            and settled == needed
        )
        if complete:
            matured_dates.add(day)
        coverage.append({
            "decision_date": day,
            "planned_exit_date": exit_date,
            "required_basket_symbols": len(needed),
            "settled_required_symbols": len(settled),
            "complete_matured_decision": complete,
        })
    return coverage, matured_dates


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-date", type=date.fromisoformat, required=True)
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument("--development-source-dir", type=Path, default=project / "experiments" / "post_p8_research_20260926" / "three_session_daily")
    parser.add_argument("--development-baostock-dir", type=Path, default=project / "data" / "raw" / "baostock" / "fetch-20260924T115050Z")
    parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    parser.add_argument("--cutover-baostock-dir", type=Path, default=project / "data" / "raw" / "baostock" / "fetch-20260924T150656Z")
    parser.add_argument("--frozen-manifest", type=Path, default=project / "P9_P12_FROZEN_MANIFEST_20260928.json")
    parser.add_argument("--expert-snapshot-root", type=Path, default=project / "experiments" / "p9_p12_forward" / "snapshots")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.as_of_date < VIEWED_THROUGH:
        raise ValueError("P9 前向账户不能在已查看截止日前冻结")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "RUN_STATUS.json"
    status = {
        "status": "running",
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    def progress(message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        status.update({"last_progress": message, "last_progress_at_utc": stamp})
        _write_json(status_path, status)
        with (output / "progress.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{stamp} {message}\n")
        print(f"[P9 forward] {message}", flush=True)

    _write_json(status_path, status)
    try:
        source = args.development_source_dir.resolve()
        config_path = source / "config.toml"
        config_bytes = config_path.read_bytes()
        config = tomllib.loads(config_bytes.decode("utf-8"))
        development = json.loads((source / "study_results.json").read_text(encoding="utf-8"))
        if hashlib.sha256(config_bytes).hexdigest() != development["config_sha256"]:
            raise ValueError("P8 开发配置哈希与冻结研究结果不符")
        seed_path = source / "training_seed.json.gz"
        if _sha256(seed_path) != development["artifacts"]["training_seed_sha256"]:
            raise ValueError("P8 开发训练状态哈希与冻结研究结果不符")
        with gzip.open(seed_path, "rt", encoding="utf-8") as stream:
            development_seed = json.load(stream)

        progress("verifying BaoStock response hashes and loading viewed cutover")
        verified_forward_responses = _verify_baostock_responses(args.baostock_run_dir)
        verified_cutover_responses = _verify_baostock_responses(args.cutover_baostock_dir)
        research, source_cache_through, excluded_cache_dates = _research_for_as_of(
            load_oos_research_data(args.baostock_run_dir.resolve()), args.as_of_date
        )
        if VIEWED_THROUGH not in research.trading_dates:
            raise ValueError("BaoStock 缓存缺少已查看切换日 2026-09-23")
        viewed_research = _slice_research(research, VIEWED_THROUGH)
        frozen_cutover_research = load_oos_research_data(args.cutover_baostock_dir.resolve())
        for field in (
            "trading_dates", "adjusted_bars", "execution_bars", "index_bars",
            "membership_snapshots",
        ):
            current = getattr(viewed_research, field)
            frozen = getattr(frozen_cutover_research, field)
            if field in ("adjusted_bars", "execution_bars", "index_bars"):
                current = [{key: value for key, value in row.items() if key != "source_ref"} for row in current]
                frozen = [{key: value for key, value in row.items() if key != "source_ref"} for row in frozen]
                same = json.dumps(current, sort_keys=True, default=str, allow_nan=True) == json.dumps(
                    frozen, sort_keys=True, default=str, allow_nan=True
                )
            elif field == "membership_snapshots":
                same = [
                    (row.event_time, row.available_at, row.members) for row in current
                ] == [
                    (row.event_time, row.available_at, row.members) for row in frozen
                ]
            else:
                same = current == frozen
            if not same:
                raise ValueError(f"新 BaoStock 缓存改动了已冻结截止日的 {field}")
        frozen_inputs = None
        if args.frozen_manifest.is_file():
            frozen_inputs = _verify_frozen_manifest(
                args.frozen_manifest, project, args.training_data_dir / "train.csv"
            )
        elif args.as_of_date > VIEWED_THROUGH:
            raise ValueError("真正前向运行需要先生成 P9 冻结哈希清单")
        viewed = run_mvp_research(
            viewed_research, config, initial_state=development_seed,
            as_of_date=VIEWED_THROUGH, ensure_as_of_decision=True,
            decision_start_date=viewed_research.trading_dates[0],
            progress_callback=lambda message: progress(f"viewed model warm-up: {message}"),
        )
        viewed_forecasts = viewed["prediction_decisions"]
        if not viewed_forecasts:
            raise ValueError("已查看暖启动期没有可执行预测")
        cutover_seed_bytes = gzip.compress(
            json.dumps(viewed["training_seed"], sort_keys=True, separators=(",", ":")).encode("utf-8"),
            mtime=0,
        )
        if frozen_inputs and hashlib.sha256(cutover_seed_bytes).hexdigest() != frozen_inputs["cutover_model_seed_sha256"]:
            raise ValueError("已查看截止日的模型训练状态与 P9 冻结清单不一致")
        (output / "model_cutover_seed_asof_2026-09-23.json.gz").write_bytes(cutover_seed_bytes)

        prefreeze_diagnostic_dates, fresh_dates = _partition_forward_dates(research.trading_dates)
        post_cutover_dates = prefreeze_diagnostic_dates + fresh_dates
        full = viewed
        if post_cutover_dates:
            progress(
                f"replaying {len(post_cutover_dates)} post-cutover dates, of which "
                f"{len(prefreeze_diagnostic_dates)} precede the 2026-09-27 freeze"
            )
            full = run_mvp_research(
                research, config, initial_state=development_seed,
                as_of_date=args.as_of_date, ensure_as_of_decision=True,
                decision_start_date=research.trading_dates[0],
                progress_callback=lambda message: progress(f"full point-in-time replay: {message}"),
            )
            by_viewed_date = {
                row["prediction_date"]: row for row in viewed_forecasts
            }
            for row in full["prediction_decisions"]:
                earlier = by_viewed_date.get(row["prediction_date"])
                if earlier is None:
                    continue
                for field in (
                    "candidate_symbols_sha256", "scores_by_method",
                    "expected_returns_by_method", "target_weights_by_method",
                ):
                    if row[field] != earlier[field]:
                        raise ValueError(
                            f"{row['prediction_date']} 的已查看预测 {field} 随未来数据改变"
                        )
        all_oos_forecasts = full["prediction_decisions"]
        all_oos_settlements = full["prediction_settlements"]
        expert_snapshots, expert_snapshot_hashes = _load_expert_snapshots(args.expert_snapshot_root)
        post_freeze_expert_snapshots = {
            day: snapshot for day, snapshot in expert_snapshots.items()
            if date.fromisoformat(day) >= FIRST_POST_FREEZE_DATE
        }
        extracted_documents = [
            document
            for snapshot in post_freeze_expert_snapshots.values()
            for document in snapshot.get("announcement_full_text", {}).get("documents", [])
        ]
        full_text_attempts = [
            attempt
            for snapshot in post_freeze_expert_snapshots.values()
            for attempt in snapshot.get("announcement_full_text", {}).get("attempts", [])
        ]
        full_text_llm_outputs = [
            row
            for snapshot in post_freeze_expert_snapshots.values()
            for row in snapshot.get("llm_outputs", {}).get("outputs", [])
            if any(evidence.get("dataset") == "announcement_full_text_pdf" for evidence in row.get("evidence", []))
        ]

        progress("building rolling P9 calibration from mature, non-overlapping cohorts")
        verified_development_responses = _verify_baostock_responses(args.development_baostock_dir)
        development_research = load_research_data(args.training_data_dir, args.development_baostock_dir)
        development_forecasts = _read_jsonl_gzip(source / "prediction_decisions.jsonl.gz")
        development_settlements = _read_jsonl_gzip(source / "prediction_settlements.jsonl.gz")
        combined_forecasts = development_forecasts + all_oos_forecasts
        combined_settlements = development_settlements + all_oos_settlements
        index_bars = development_research.index_bars + research.index_bars
        cohorts = _portfolio_cohorts(
            combined_forecasts, combined_settlements, SimpleNamespace(index_bars=index_bars),
            top_fraction=float(config["portfolio_top_fraction"]),
            invested_weight=float(config["invested_weight"]),
        )
        calibration_rows = []
        calibrated_forecasts = []
        for number, forecast in enumerate(all_oos_forecasts, start=1):
            day = date.fromisoformat(forecast["prediction_date"])
            by_method = {
                method: _calibration_for_decision(cohorts, day, method)
                for method in CALIBRATION_METHODS
            }
            for method, estimate in by_method.items():
                if date.fromisoformat(estimate["window_end_label_date"]) >= day:
                    raise ValueError(f"{day} {method} 使用未成熟校准标签")
                calibration_rows.append({
                    "decision_date": day.isoformat(),
                    "method": method,
                    "window_start_date": estimate["window_start_date"],
                    "window_end_label_date": estimate["window_end_label_date"],
                    "matured_nonoverlapping_cohorts": estimate["matured_nonoverlapping_cohorts"],
                    "total_mean_bias_estimate": estimate["total_mean_bias_estimate"],
                })
            calibrated_forecasts.append(
                _adjust_forecast(forecast, by_method, calibrated=True,
                                 correction_field="total_mean_bias_estimate")
            )
            if number % 20 == 0 or number == len(all_oos_forecasts):
                progress(f"calibrated {number}/{len(all_oos_forecasts)} OOS decisions")
        _write_csv(output / "rolling_calibration.csv", calibration_rows)
        first_tradeable_index = next(
            index for index, forecast in enumerate(all_oos_forecasts)
            if all(forecast["target_weights_by_method"].get(method)
                   for method in ("trend", "market_plus_relative_fusion"))
        )
        if any(
            not all(forecast["target_weights_by_method"].get(method)
                    for method in ("trend", "market_plus_relative_fusion"))
            for forecast in all_oos_forecasts[first_tradeable_index:]
        ):
            raise ValueError("首次可交易日之后存在缺少冻结候选篮子的预测")
        first_decision = date.fromisoformat(
            all_oos_forecasts[first_tradeable_index]["prediction_date"]
        )
        account_forecasts = _post_freeze_account_decisions(
            calibrated_forecasts[first_tradeable_index:]
        )
        overlay_forecasts = []
        overlay_audit = []
        for forecast in account_forecasts:
            decision_date = str(forecast["prediction_date"])
            snapshot = expert_snapshots.get(decision_date) if date.fromisoformat(decision_date) > VIEWED_THROUGH else None
            updated, audit = add_public_expert_overlay(forecast, snapshot)
            overlay_forecasts.append(updated)
            if date.fromisoformat(decision_date) > VIEWED_THROUGH:
                overlay_audit.append({
                    **audit,
                    "expert_snapshot_sha256": expert_snapshot_hashes.get(decision_date),
                })
        if any(
            date.fromisoformat(str(row["decision_date"])) < FIRST_POST_FREEZE_DATE
            for row in overlay_audit
        ):
            raise ValueError("public_expert_overlay audit 包含冻结前决策日")
        with (output / "expert_overlay_decisions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in overlay_audit:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False) + "\n")

        costs = {key: float(config[key]) for key in (
            "commission_per_side_bps", "transfer_fee_per_side_bps",
            "stamp_duty_sell_bps", "slippage_per_side_bps",
        )}
        optimizer_config = {
            **config,
            "optimizer_candidate_scope": "frozen_method_target",
            "risk_aversion": 0.0,
            "forecast_error_multiplier": 0.0,
            "risk_budget_annualized": 0.20,
        }
        accounts = {}
        for name, method, forecasts_for_account in (
            ("p9_daily_selector", "p9_best_trend_or_market_relative", account_forecasts),
            ("p9_fixed_market_relative", "market_plus_relative_fusion", account_forecasts),
            ("p9_public_expert_overlay", "market_plus_relative_fusion", overlay_forecasts),
        ):
            progress(f"replaying viewed account warm-up and {name}")
            account = simulate_portfolio(
                research, [], float(config["initial_capital_cny"]), costs,
                int(config["liquidity_lookback_sessions"]),
                float(config["maximum_daily_participation"]),
                float(config["maximum_one_way_turnover"]),
                forecast_decisions=forecasts_for_account,
                optimizer_config={**optimizer_config, "optimizer_return_method": method},
                evaluation_start_date=first_decision,
                progress_callback=lambda message: progress(f"{name}: {message}"),
            )
            accounts[name] = account

        trend_decisions = _post_freeze_account_decisions(full["decisions_by_method"]["trend"])
        progress("replaying the frozen P8 simple-trend comparator")
        accounts["p8_simple_trend"] = simulate_portfolio(
            research, trend_decisions, float(config["initial_capital_cny"]), costs,
            int(config["liquidity_lookback_sessions"]),
            float(config["maximum_daily_participation"]),
            float(config["maximum_one_way_turnover"]),
            evaluation_start_date=first_decision,
        )
        accounts["csi300_price_proxy"] = simulate_fixed_risk_market_exposure(
            research, first_decision, float(config["initial_capital_cny"]),
            float(config["risk_budget_annualized"]),
            int(config["risk_lookback_sessions"]), costs,
        )
        if frozen_inputs:
            for name, state in frozen_inputs["cutover_account_states"].items():
                _verify_cutover_account_state(
                    name, _account_cutover_state(accounts[name]), state
                )
        if "p9_public_expert_overlay" in accounts:
            _verify_cutover_account_state(
                "p9_public_expert_overlay",
                _account_cutover_state(accounts["p9_public_expert_overlay"]),
                _account_cutover_state(accounts["p9_fixed_market_relative"]),
            )

        for name, account in accounts.items():
            account_dir = output / name
            account_dir.mkdir()
            _write_json(account_dir / "summary_full_warmup.json", account["summary"])
            for field in ("daily_nav", "orders", "positions"):
                _write_csv(account_dir / f"{field}.csv", account[field])
            with (account_dir / "decisions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for row in account["decisions"]:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False) + "\n")

        new_nav = {
            name: [row for row in account["daily_nav"] if row["trade_date"] >= FIRST_POST_FREEZE_DATE.isoformat()]
            for name, account in accounts.items()
        }
        account_action_names = (
            "p9_daily_selector", "p9_fixed_market_relative",
            "p9_public_expert_overlay", "p8_simple_trend",
        )
        prefreeze_warmup_execution_orders = {}
        for name in account_action_names:
            orders = accounts[name]["orders"]
            leaked_diagnostic_orders = [
                row for row in orders
                if row.get("decision_date") == PREFREEZE_DIAGNOSTIC_THROUGH.isoformat()
            ]
            if leaked_diagnostic_orders:
                raise ValueError(f"{name} 将 2026-09-24 冻结前决策带入账户委托")
            warmup_orders = [
                row for row in orders
                if row.get("decision_date") == VIEWED_THROUGH.isoformat()
                and row.get("trade_date") == PREFREEZE_DIAGNOSTIC_THROUGH.isoformat()
            ]
            prefreeze_warmup_execution_orders[name] = {
                "originating_decision_date": VIEWED_THROUGH.isoformat(),
                "execution_date": PREFREEZE_DIAGNOSTIC_THROUGH.isoformat(),
                "order_count": len(warmup_orders),
                "filled_order_count": sum(float(row.get("filled_notional_cny", 0.0)) > 0 for row in warmup_orders),
                "filled_notional_cny": sum(float(row.get("filled_notional_cny", 0.0)) for row in warmup_orders),
                "transaction_cost_cny": sum(float(row.get("cost_cny", 0.0)) for row in warmup_orders),
                "classification": "pre_freeze_warmup_execution_not_a_new_trade_date",
            }
        new_account_results = {}
        for name, account in accounts.items():
            if not new_nav[name]:
                new_account_results[name] = None
                continue
            cutover_row = next(
                row for row in account["daily_nav"]
                if row["trade_date"] == VIEWED_THROUGH.isoformat()
            )
            anchors = [
                row for row in account["daily_nav"]
                if VIEWED_THROUGH.isoformat() <= row["trade_date"] < FIRST_POST_FREEZE_DATE.isoformat()
            ]
            anchor_row = anchors[-1] if anchors else cutover_row
            new_orders = [
                row for row in account["orders"]
                if row["trade_date"] >= FIRST_POST_FREEZE_EXECUTION_DATE.isoformat()
            ]
            new_account_results[name] = {
                "first_date": new_nav[name][0]["trade_date"],
                "last_date": new_nav[name][-1]["trade_date"],
                "trading_days": len(new_nav[name]),
                "cutover_nav_cny": float(cutover_row["nav_cny"]),
                "post_freeze_anchor_date": anchor_row["trade_date"],
                "post_freeze_anchor_nav_cny": float(anchor_row["nav_cny"]),
                "end_nav_cny": float(new_nav[name][-1]["nav_cny"]),
                "net_return_from_prefreeze_anchor_including_first_carry_day": (
                    float(new_nav[name][-1]["nav_cny"]) / float(anchor_row["nav_cny"]) - 1
                ),
                "first_post_freeze_nav_daily_return": float(new_nav[name][0]["daily_return"]),
                "first_post_freeze_nav_return_attribution": (
                    "frozen_pre_freeze_positions_and_any_2026-09-23_warmup_order_filled_on_2026-09-24; "
                    "not caused by the 2026-09-28 close decision, whose earliest execution is 2026-09-29"
                ),
                "net_return_after_first_post_freeze_close": (
                    float(new_nav[name][-1]["nav_cny"]) / float(new_nav[name][0]["nav_cny"]) - 1
                ),
                "new_date_transaction_cost_cny": sum(float(row["cost_cny"]) for row in new_orders),
            }
        new_label_coverage, matured_new_dates = _new_decision_label_coverage(
            all_oos_forecasts, all_oos_settlements, VIEWED_THROUGH,
            FIRST_POST_FREEZE_DATE, args.as_of_date
        )
        paired = {}
        if fresh_dates:
            paired_nav = {
                name: [
                    row for row in nav
                    if row["trade_date"] >= FIRST_POST_FREEZE_EXECUTION_DATE.isoformat()
                ]
                for name, nav in new_nav.items()
            }
            selector = paired_nav["p9_daily_selector"]
            if len(selector) < 20:
                paired = {
                    name: {
                        "status": "not_computed_insufficient_post_freeze_dates",
                        "common_date_count": len(selector),
                        "required_common_dates": 20,
                    }
                    for name in (
                        "p9_fixed_market_relative", "p9_public_expert_overlay",
                        "p8_simple_trend", "csi300_price_proxy", "cash",
                    )
                }
            else:
                for offset, other in enumerate(("p9_fixed_market_relative", "p9_public_expert_overlay", "p8_simple_trend", "csi300_price_proxy"), start=1):
                    if [row["trade_date"] for row in selector] != [row["trade_date"] for row in paired_nav[other]]:
                        raise ValueError("P9 前向账户与基线的交易日期不一致")
                    paired[other] = _paired(selector, paired_nav[other], 20260927 + offset)
                paired["cash"] = paired_block_bootstrap(
                    [float(row["daily_return"]) for row in selector],
                    [0.0] * len(selector), block_sessions=20, repetitions=2000, seed=20260931,
                )
        report = {
            "schema_version": "adaptive-mas.p9-forward.v1",
            "as_of_date": args.as_of_date.isoformat(),
            "source_cache_through": source_cache_through.isoformat(),
            "source_cache_dates_after_as_of_excluded": [day.isoformat() for day in excluded_cache_dates],
            "viewed_through": VIEWED_THROUGH.isoformat(),
            "account_warm_start_as_of": VIEWED_THROUGH.isoformat(),
            "pre_freeze_diagnostic_through": PREFREEZE_DIAGNOSTIC_THROUGH.isoformat(),
            "pre_freeze_diagnostic_dates": [day.isoformat() for day in prefreeze_diagnostic_dates],
            "first_post_freeze_decision_date": FIRST_POST_FREEZE_DATE.isoformat(),
            "first_post_freeze_execution_date": FIRST_POST_FREEZE_EXECUTION_DATE.isoformat(),
            "decision_execution_policy": {
                "decision_formed_after_market_close": True,
                "earliest_execution": "next_trading_session_open",
                "first_post_freeze_decision_date": FIRST_POST_FREEZE_DATE.isoformat(),
                "first_post_freeze_execution_date": FIRST_POST_FREEZE_EXECUTION_DATE.isoformat(),
            },
            "first_shadow_account_decision": first_decision.isoformat(),
            "viewed_warmup_decisions_not_scored": len(viewed_forecasts),
            "pre_freeze_diagnostic_decisions_not_scored": sum(
                date.fromisoformat(str(row["prediction_date"])) in prefreeze_diagnostic_dates
                for row in all_oos_forecasts
            ),
            "pre_freeze_warmup_execution_orders": prefreeze_warmup_execution_orders,
            "new_trading_dates": len(fresh_dates),
            "new_matured_decision_dates": len(matured_new_dates),
            "new_decision_label_coverage": new_label_coverage,
            "required_new_matured_decision_dates": 120,
            "status": (
                "awaiting_new_dates" if not fresh_dates else
                "awaiting_120_new_matured_decision_dates" if len(matured_new_dates) < 120 else
                "ready_for_predeclared_comparison"
            ),
            "claim": "The 2026-09-23 account remains the cutover warm start. The 2026-09-24 market observation is pre-freeze diagnostic: its bars may inform features for the 2026-09-28 close decision, but its forecast is removed from account actions and overlay audit because its next-session order would otherwise execute on 2026-09-28. Any 2026-09-23 warm-up order filled on 2026-09-24 remains warm-up activity. The 2026-09-28 first NAV return is separately attributed to frozen pre-freeze holdings; the 2026-09-28 close signal first executes on 2026-09-29. Maturity counts start with 2026-09-28 decisions, while paired strategy-return comparisons start with 2026-09-29 execution dates.",
            "p9_rule": "rolling mature non-overlapping three-session top-fraction portfolio residual mean; select higher feasible trend/market-relative net objective on actual account state",
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "development_training_seed_sha256": _sha256(seed_path),
            "model_cutover_seed_sha256": hashlib.sha256(cutover_seed_bytes).hexdigest(),
            "development_candidate_forecasts_sha256": _sha256(source / "prediction_decisions.jsonl.gz"),
            "baostock_manifest_sha256": _sha256(args.baostock_run_dir / "manifest.jsonl"),
            "verified_baostock_response_files": {
                "forward_cache": verified_forward_responses,
                "cutover_cache": verified_cutover_responses,
                "development_cache": verified_development_responses,
            },
            "frozen_cutover_baostock_manifest_sha256": _sha256(args.cutover_baostock_dir / "manifest.jsonl"),
            "frozen_manifest_sha256": _sha256(args.frozen_manifest) if args.frozen_manifest.is_file() else None,
            "account_full_span_summaries_including_viewed_warmup": {name: account["summary"] for name, account in accounts.items()},
            "new_date_account_results": new_account_results,
            "new_date_paired_20_session_block_comparisons": paired,
            "public_expert_overlay": {
                "account": "p9_public_expert_overlay",
                "same_simulator_and_starting_cutover_state_as_fixed_p9": True,
                "snapshot_count": len(post_freeze_expert_snapshots),
                "prefreeze_diagnostic_snapshot_dates_excluded": sorted(
                    day for day in expert_snapshots
                    if VIEWED_THROUGH < date.fromisoformat(day) < FIRST_POST_FREEZE_DATE
                ),
                "snapshot_sha256_by_decision_date": {
                    day: expert_snapshot_hashes[day] for day in post_freeze_expert_snapshots
                },
                "decision_adjustments": overlay_audit,
                "current_status": (
                    "no_eligible_matured_expert_effects"
                    if not any(row["status"] == "applied_matured_candidate_effects" for row in overlay_audit)
                    else "research_overlay_replayed"
                ),
                "announcement_full_text_llm": {
                    "status": (
                        "bounded_full_text_llm_verified" if extracted_documents and full_text_llm_outputs else
                        "partial_fetch_or_extraction_failed" if any(row["status"] == "fetch_or_extraction_failed" for row in full_text_attempts) else
                        "pdf_extracted_llm_not_verified_or_not_run" if extracted_documents else
                        "not_completed_no_verified_recent_pdf_text"
                    ),
                    "pdf_document_count": len(extracted_documents),
                    "llm_analyzed_document_count": len(full_text_llm_outputs),
                    "attempts": full_text_attempts,
                    "scope_limit": "bounded recent documents per snapshot; historical 136k announcement archive was not downloaded",
                },
            },
            "market_baseline_limitation": "CSI 300 price proxy is synthetic, not a verified tradable ETF.",
        }
        _write_json(output / "forward_validation_report.json", report)
        status.update({
            "status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "new_trading_dates": len(fresh_dates),
            "new_matured_decision_dates": len(matured_new_dates),
            "pre_freeze_diagnostic_dates": [day.isoformat() for day in prefreeze_diagnostic_dates],
            "first_post_freeze_decision_date": FIRST_POST_FREEZE_DATE.isoformat(),
            "report": str(output / "forward_validation_report.json"),
        })
        progress(f"complete: new trading dates={len(fresh_dates)}, mature decisions={len(matured_new_dates)}")
    except Exception as error:
        status.update({
            "status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
        })
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
