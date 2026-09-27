"""Train a small Q-learning selector and compare it in the frozen P9 simulator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data import load_research_data
from adaptive_mas.evaluation.rl_control import (
    ACTIONS,
    STATES,
    choose_actions,
    market_state_by_date,
    update_q_values,
)
from adaptive_mas.evaluation.uncertainty import paired_block_bootstrap
from run_p9_joint_economics import _actual_marked_risk, _adjust_forecast, _read_jsonl_gzip, _write_csv, _write_json


EPISODES = 5
EPSILON = 0.25
LEARNING_RATE = 0.1
DISCOUNT = 0.9
TRAIN_FRACTION = 0.70
SEED = 20260927


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _slice_research(research, end: date):
    from adaptive_mas.data.research import ResearchData

    return ResearchData(
        trading_dates=[day for day in research.trading_dates if day <= end],
        adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= end],
        execution_bars=[row for row in research.execution_bars if row["trade_date"] <= end],
        index_bars=[row for row in research.index_bars if row["trade_date"] <= end],
        membership_snapshots=[row for row in research.membership_snapshots if row.available_at <= end],
    )


def _one_account_replay(
    research,
    forecasts,
    config,
    costs,
    start_date: date,
    end_date: date,
    progress,
):
    research_slice = _slice_research(research, end_date)
    optimizer_config = dict(config)
    optimizer_config.update({
        "optimizer_return_method": "trend",
        "optimizer_candidate_scope": "frozen_method_target",
        "risk_aversion": 0.0,
        "forecast_error_multiplier": 0.0,
        "risk_budget_annualized": 0.20,
    })
    result = simulate_portfolio(
        research_slice,
        [],
        float(config["initial_capital_cny"]),
        costs,
        int(config["liquidity_lookback_sessions"]),
        float(config["maximum_daily_participation"]),
        float(config["maximum_one_way_turnover"]),
        forecast_decisions=forecasts,
        optimizer_config=optimizer_config,
        progress_callback=progress,
        evaluation_start_date=start_date,
    )
    result["summary"]["maximum_marked_annualized_risk"], result["summary"]["marked_risk_budget_breach_days"], result["summary"]["marked_risk_budget_breach_dates"] = _actual_marked_risk(
        research_slice, result["daily_nav"], result["positions"], optimizer_config
    )
    result["summary"]["net_total_return"] = result["summary"]["total_return"]
    return result


def _with_action(forecasts, action: str):
    return [{**forecast, "controller_action": action} for forecast in forecasts]


def _execution_date(research, decision_date: str) -> date:
    index = research.trading_dates.index(date.fromisoformat(decision_date))
    if index + 1 >= len(research.trading_dates):
        raise ValueError(f"{decision_date} 没有下一交易日执行")
    return research.trading_dates[index + 1]


def _transitions(decision_rows, nav_rows, research, states):
    returns = {str(row["trade_date"]): float(row["daily_return"]) for row in nav_rows}
    rows = []
    for index, row in enumerate(decision_rows):
        decision_date = str(row["decision_date"])
        execution_date = _execution_date(research, decision_date).isoformat()
        next_state = (
            states[str(decision_rows[index + 1]["decision_date"])]
            if index + 1 < len(decision_rows)
            else states[decision_date]
        )
        rows.append({
            "state": states[decision_date],
            "action": str(row["action"]),
            "reward": returns[execution_date],
            "next_state": next_state,
            "terminal": index + 1 == len(decision_rows),
            "execution_date": execution_date,
        })
    return rows


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    parser.add_argument("--baostock-run-dir", type=Path, default=project / "data" / "raw" / "baostock" / "fetch-20260924T115050Z")
    parser.add_argument("--source-study-dir", type=Path, default=project / "experiments" / "post_p8_research_20260926" / "three_session_daily")
    parser.add_argument("--calibration-dir", type=Path, default=project / "experiments" / "p9_joint_economics_20260927_full_exit")
    parser.add_argument("--p9-forward-output-dir", type=Path, default=project / "experiments" / "p9_forward_cutover_20260927_v9")
    parser.add_argument("--decision-limit", type=int, help="仅用于快速实现 smoke；截取最早 N 个已查看决策，不作策略评价")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖现有 P12 输出目录：{output}")
    output.mkdir(parents=True)
    status_path = output / "RUN_STATUS.json"
    status = {
        "status": "running",
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_episodes": EPISODES,
        "train_fraction": TRAIN_FRACTION,
    }

    def progress(message: str) -> None:
        status["last_progress"] = message
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(status_path, status)
        with (output / "progress.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{status['updated_at_utc']} {message}\n")
        print(f"[P12 RL] {message}", flush=True)

    _write_json(status_path, status)
    try:
        source = args.source_study_dir.resolve()
        study = json.loads((source / "study_results.json").read_text(encoding="utf-8"))
        config = tomllib.loads((source / "config.toml").read_text(encoding="utf-8"))
        calibration_rows = _read_jsonl(args.calibration_dir / "rolling_calibration_details.jsonl")
        calibration = {}
        for row in calibration_rows:
            day = str(row["decision_date"])
            for method in ACTIONS:
                if row["by_method"][method]["window_end_label_date"] >= day:
                    raise ValueError(f"{day} {method} 校准使用未成熟结果")
            calibration[day] = row["by_method"]
        frozen = {
            str(row["prediction_date"]): row
            for row in _read_jsonl_gzip(source / "prediction_decisions.jsonl.gz")
            if str(row["prediction_date"]) in calibration
        }
        if set(frozen) != set(calibration):
            raise ValueError("冻结预测日期和成熟校准日期不一致")
        research = load_research_data(args.training_data_dir, args.baostock_run_dir)
        all_frozen_days = sorted(frozen)
        days = [
            day for day in all_frozen_days
            if date.fromisoformat(day) in research.trading_dates
            and research.trading_dates.index(date.fromisoformat(day)) + 1 < len(research.trading_dates)
        ]
        excluded_without_next_execution = len(all_frozen_days) - len(days)
        if args.decision_limit is not None:
            if args.decision_limit < 4:
                raise ValueError("快速 smoke 至少需要四个可执行决策日")
            days = days[:args.decision_limit]
        split = int(len(days) * TRAIN_FRACTION)
        train_days, evaluation_days = days[:split], days[split:]
        if not train_days or not evaluation_days:
            raise ValueError("P12 时间切分缺少训练或评价日期")
        forecasts = {
            day: _adjust_forecast(
                frozen[day], calibration[day], calibrated=True,
                correction_field="total_mean_bias_estimate",
            )
            for day in days
        }
        states = market_state_by_date(research, days)
        costs = {
            key: float(config[key])
            for key in (
                "commission_per_side_bps", "transfer_fee_per_side_bps",
                "stamp_duty_sell_bps", "slippage_per_side_bps",
            )
        }
        train_start = date.fromisoformat(train_days[0])
        train_end = _execution_date(research, train_days[-1])
        q_values = {state: {action: 0.0 for action in ACTIONS} for state in STATES}
        episode_summaries = []
        for episode in range(EPISODES):
            training_forecasts, policy_rows = choose_actions(
                [forecasts[day] for day in train_days], states, q_values,
                epsilon=EPSILON, seed=SEED + episode,
            )
            replay = _one_account_replay(
                research, training_forecasts, config, costs,
                train_start, train_end,
                lambda message, episode=episode: progress(f"train episode {episode + 1}/{EPISODES}: {message}"),
            )
            transitions = _transitions(
                policy_rows, replay["daily_nav"], research, states
            )
            update_q_values(q_values, transitions, LEARNING_RATE, DISCOUNT)
            episode_summaries.append({
                "episode": episode + 1,
                "epsilon": EPSILON,
                "net_total_return": replay["summary"]["net_total_return"],
                "transaction_cost_cny": replay["summary"]["total_transaction_cost_cny"],
                "action_count": {
                    action: sum(row["action"] == action for row in policy_rows)
                    for action in ACTIONS
                },
                "q_updates": len(transitions),
            })
            progress(f"training episode {episode + 1}/{EPISODES} updated {len(transitions)} net-return transitions")

        evaluation_forecasts = [forecasts[day] for day in evaluation_days]
        evaluation_start = date.fromisoformat(evaluation_days[0])
        evaluation_end = _execution_date(research, evaluation_days[-1])
        rl_forecasts, rl_policy_rows = choose_actions(
            evaluation_forecasts, states, q_values, epsilon=0.0, seed=SEED
        )
        strategies = {
            "q_learning_selector": (rl_forecasts, rl_policy_rows),
            "trend_action_baseline": (_with_action(evaluation_forecasts, "trend"), None),
            "market_relative_action_baseline": (_with_action(evaluation_forecasts, "market_plus_relative_fusion"), None),
        }
        evaluation_results = {}
        eval_summary = {}
        for name, (strategy_forecasts, policy_rows) in strategies.items():
            progress(f"evaluating {name} on the locked temporal holdout")
            result = _one_account_replay(
                research, strategy_forecasts, config, costs,
                evaluation_start, evaluation_end,
                lambda message, name=name: progress(f"{name}: {message}"),
            )
            evaluation_results[name] = result
            returns = [float(row["daily_return"]) for row in result["daily_nav"]]
            eval_summary[name] = {
                **result["summary"],
                "sharpe_rf_zero": result["summary"]["sharpe_rf_zero"],
                "transaction_cost_cny": result["summary"]["total_transaction_cost_cny"],
                "risk_budget_annualized": 0.20,
                "evaluation_start_decision_date": evaluation_days[0],
                "evaluation_end_decision_date": evaluation_days[-1],
                "daily_return_count": len(returns),
                "actions": (
                    {action: sum(row["action"] == action for row in policy_rows) for action in ACTIONS}
                    if policy_rows is not None else {
                        "trend" if name == "trend_action_baseline" else "market_plus_relative_fusion": len(evaluation_days)
                    }
                ),
            }
            _write_json(output / f"{name}_summary.json", eval_summary[name])
            _write_csv(output / f"{name}_daily_nav.csv", result["daily_nav"])
            _write_csv(output / f"{name}_orders.csv", result["orders"])
            _write_csv(output / f"{name}_positions.csv", result["positions"])
            with (output / f"{name}_decisions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for row in result["decisions"]:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

        nav_by_name = {
            name: {str(row["trade_date"]): float(row["daily_return"]) for row in result["daily_nav"]}
            for name, result in evaluation_results.items()
        }
        common_dates = sorted(set.intersection(*(set(rows) for rows in nav_by_name.values())))
        if not common_dates:
            raise ValueError("P12 holdout accounts lack common realized dates")
        if len(common_dates) < 20:
            paired_uncertainty = {
                baseline: {
                    "status": "not_computed_insufficient_common_dates",
                    "common_evaluation_dates": len(common_dates),
                    "required_common_dates": 20,
                }
                for baseline in ("trend_action_baseline", "market_relative_action_baseline")
            }
        else:
            paired_uncertainty = {
                baseline: paired_block_bootstrap(
                    [nav_by_name["q_learning_selector"][day] for day in common_dates],
                    [nav_by_name[baseline][day] for day in common_dates],
                    block_sessions=20, repetitions=2000, seed=SEED,
                )
                for baseline in ("trend_action_baseline", "market_relative_action_baseline")
            }
        formal_p9_reference = None
        p9_nav_path = args.p9_forward_output_dir / "p9_fixed_market_relative" / "daily_nav.csv"
        if p9_nav_path.is_file():
            with p9_nav_path.open(encoding="utf-8-sig", newline="") as source:
                p9_rows = list(csv.DictReader(source))
            p9_returns = {
                str(row["trade_date"]): float(row["daily_return"])
                for row in p9_rows
                if str(row["trade_date"]) in nav_by_name["q_learning_selector"]
            }
            shared_p9_dates = sorted(p9_returns)
            if shared_p9_dates:
                formal_p9_reference = {
                    "source": str(p9_nav_path.resolve()),
                    "same_calendar_dates": len(shared_p9_dates),
                    "independent_account_state_and_start_date": True,
                    "paired_daily_return_bootstrap": paired_block_bootstrap(
                        [nav_by_name["q_learning_selector"][day] for day in shared_p9_dates],
                        [p9_returns[day] for day in shared_p9_dates],
                        block_sessions=20, repetitions=2000, seed=SEED + 1,
                    ),
                    "limitation": "descriptive same-date reference; the frozen P9 account carries a different prior account state and is not the same-starting-state controller baseline",
                }
        _write_json(output / "q_values.json", q_values)
        _write_json(output / "training_episodes.json", episode_summaries)
        _write_json(output / "evaluation_summary.json", {
            "status": "already_viewed_temporal_diagnostic_not_new_oos",
            "data_source": str(args.baostock_run_dir.resolve()),
            "input_decision_count": len(days),
            "smoke_limited_input": args.decision_limit is not None,
            "decision_limit": args.decision_limit,
            "excluded_no_next_execution_session_count": excluded_without_next_execution,
            "training_decision_count": len(train_days),
            "evaluation_decision_count": len(evaluation_days),
            "common_evaluation_dates": len(common_dates),
            "controllers": eval_summary,
            "paired_20_session_block_bootstrap": paired_uncertainty,
            "frozen_p9_same_date_reference": formal_p9_reference,
            "same_starting_state_action_baselines": {
                "trend_action_baseline": "trend action on calibrated P9 forecasts, through the same frozen-candidate optimizer and account simulator",
                "market_relative_action_baseline": "market_plus_relative_fusion action on calibrated P9 forecasts, through the same frozen-candidate optimizer and account simulator",
                "these_are_action_baselines_not_the_historical_P8_or_P9_account_ledgers": True,
            },
            "same_simulator_and_constraints": True,
            "training_only_uses_initial_temporal_partition": True,
            "test_csv_read": False,
        })
        protocol = {
            "state": "sign of latest 20-session CSI 300 index return, observable at decision close",
            "actions": list(ACTIONS),
            "reward": "next execution session account daily NAV return after simulator fees and fills",
            "episodes": EPISODES,
            "epsilon": EPSILON,
            "learning_rate": LEARNING_RATE,
            "discount": DISCOUNT,
            "temporal_train_fraction": TRAIN_FRACTION,
            "risk_budget_annualized": 0.20,
            "max_symbol_weight": config["maximum_symbol_weight"],
            "maximum_daily_participation": config["maximum_daily_participation"],
            "maximum_one_way_turnover": config["maximum_one_way_turnover"],
            "costs_bps": costs,
            "initial_capital_cny": config["initial_capital_cny"],
            "all_components_frozen_before_holdout": True,
        }
        _write_json(output / "frozen_protocol.json", protocol)
        report_lines = [
            "# P12 Q-learning 控制器诊断",
            "",
            (
                f"快速实现 smoke 仅截取最早 {len(days)} 个已查看决策日，报告只验证代码路径与账户约束。"
                if args.decision_limit is not None else ""
            ),
            f"训练用前 {len(train_days)} 个时序决策，后 {len(evaluation_days)} 个决策作一次保留窗比较；但所有日期都在已查看开发行情中，因此是时间切分诊断，不是新 OOS。",
            "RL 只在 trend 与 market-relative fusion 两个冻结专家动作间选择。trend_action_baseline 和 market_relative_action_baseline 使用同一校准预测、同一冻结候选篮子、同一 20% 年化风险上限、单票上限、成交参与率、换手、现金、费用和执行模拟，并各自从同一现金起点开始；它们是控制器动作基线，不是 P8/P9 历史正式账户账本。reward 为次日成交后日净值收益，费用已扣。",
            "",
        ]
        if args.decision_limit is None:
            for name, values in eval_summary.items():
                report_lines.append(
                    f"- {name}: 净收益 {values['net_total_return']:.2%}；Sharpe {values['sharpe_rf_zero']:.2f}；"
                    f"最大回撤 {values['maximum_drawdown']:.2%}；费用 ¥{values['transaction_cost_cny']:,.0f}；"
                    f"最大标记年化风险 {values['maximum_marked_annualized_risk']:.2%}。"
                )
        else:
            report_lines.append(
                f"- 实现检查覆盖 {len(days)} 个决策输入、{len(train_days)} 个训练输入与 "
                f"{len(common_dates)} 个保留输出日期；摘要仅用于账户与执行路径审阅，不报告策略收益判断。"
            )
        report_lines.extend([
            "",
            (
                "共同评价日期少于 20 日，未计算配对区间；见 evaluation_summary.json 的显式状态。"
                if len(common_dates) < 20 else
                "配对区间来自 20 交易日区块 bootstrap，见 evaluation_summary.json。"
            ) + "样本短且已被查看，不作盈利或新系统泛化结论。RL 只在一个两态、两动作 Q 表上训练；它不是 FinRL 复刻，也没有进入主资金决策。",
            "",
            "完整重建命令保存在 RUN_STATUS.json；不读取 test.csv。",
        ])
        (output / "RESEARCH_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        status.update({
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_summary": str(output / "evaluation_summary.json"),
        })
        progress("Q-learning training and same-account holdout comparisons complete")
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
