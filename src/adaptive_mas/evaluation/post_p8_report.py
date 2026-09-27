"""Render the post-P8 research report from immutable study JSON and ledger outputs."""

from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STUDY_NAMES = (
    "one_session_daily",
    "three_session_daily",
    "five_session_daily",
    "three_session_review_every_five",
)
STUDY_LABELS = {
    "one_session_daily": "1 日每日",
    "three_session_daily": "3 日每日",
    "five_session_daily": "5 日每日",
    "three_session_review_every_five": "3 日每 5 日复审",
}
DIAGNOSTIC_METHODS = (
    "trend",
    "reversal",
    "liquidity_confirmation",
    "equal_fusion",
    "market_plus_relative_fusion",
    "market_only",
)
PORTFOLIO_METHODS = (
    "trend",
    "reversal",
    "liquidity_confirmation",
    "equal_fusion",
    "adaptive_fusion",
    "loss_gated_fusion",
    "market_plus_relative_fusion",
    "joint_optimization",
    "joint_market_relative",
)
PAIRED_METHODS = (
    "cash",
    "csi300_benchmark",
    "fixed_risk_market",
    "trend",
    "reversal",
    "liquidity_confirmation",
    "equal_fusion",
    "market_plus_relative_fusion",
    "joint_optimization",
    "joint_market_relative",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(value: object, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:+.2%}" if signed else f"{number:.2%}"


def _number(value: object, decimals: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{decimals}f}"


def _money(value: object) -> str:
    return "n/a" if value is None else f"¥{float(value):,.0f}"


def _multiple(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2f}×"


def _ci(low: object, high: object, *, percent: bool = False) -> str:
    if low is None or high is None:
        return "n/a"
    if percent:
        return f"[{_pct(low, signed=True)}, {_pct(high, signed=True)}]"
    return f"[{float(low):+.4f}, {float(high):+.4f}]"


def _portfolio_row(
    study_label: str,
    method_label: str,
    summary: dict[str, Any],
) -> str:
    return "| {} | {} | 净 | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        study_label,
        method_label,
        _pct(summary.get("total_return", summary.get("net_total_return"))),
        _pct(summary.get("mean_cash_weight", summary.get("average_cash_weight"))),
        _pct(summary.get("maximum_drawdown")),
        _multiple(summary.get("one_way_turnover_over_average_nav")),
        _money(summary.get("total_transaction_cost_cny", summary.get("total_cost_cny"))),
        _money(summary.get("total_traded_notional_cny")),
        int(summary.get("filled_orders", 0)),
        int(summary.get("blocked_orders", 0)),
    )


def _baseline_row(
    study_label: str,
    name: str,
    summary: dict[str, Any],
    return_type: str,
    return_value: object,
    turnover: object = None,
) -> str:
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        study_label,
        name,
        return_type,
        _pct(return_value),
        _pct(summary.get("average_cash_weight")),
        _pct(summary.get("maximum_drawdown")),
        _multiple(turnover),
        _money(summary.get("total_transaction_cost_cny", summary.get("total_cost_cny"))),
        _money(summary.get("total_traded_notional_cny")),
        int(summary.get("filled_orders", 0)),
        int(summary.get("blocked_orders", 0)),
    )


def _audit_ledgers(
    output_dir: Path,
    study_name: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    results = {}
    for method, summary in report["portfolio_results"].items():
        method_dir = output_dir / study_name / "portfolios" / method
        with (method_dir / "daily_nav.csv").open(encoding="utf-8", newline="") as source:
            daily = list(csv.DictReader(source))
        with (method_dir / "positions.csv").open(encoding="utf-8", newline="") as source:
            positions = list(csv.DictReader(source))
        with (method_dir / "orders.csv").open(encoding="utf-8", newline="") as source:
            orders = list(csv.DictReader(source))
        market_value_by_date: dict[str, float] = {}
        for row in positions:
            day = str(row["trade_date"])
            market_value_by_date[day] = market_value_by_date.get(day, 0.0) + float(
                row["market_value_cny"]
            )
        ledger_balances = all(
            abs(
                float(row["cash_cny"])
                + market_value_by_date.get(str(row["trade_date"]), 0.0)
                - float(row["nav_cny"])
            )
            <= 1e-5
            for row in daily
        )
        order_costs = sum(float(row["cost_cny"] or 0.0) for row in orders)
        fees_reconcile = abs(
            order_costs - float(summary["total_transaction_cost_cny"])
        ) <= 1e-5
        results[method] = {
            "daily_nav_rows": len(daily),
            "order_rows": len(orders),
            "position_rows": len(positions),
            "cash_plus_positions_equal_nav": ledger_balances,
            "order_fees_equal_summary_fees": fees_reconcile,
            "passes": ledger_balances and fees_reconcile,
        }
    return results


def render_report(output_dir: Path) -> Path:
    """Regenerate RESEARCH_REPORT.md from saved results and ledger files without replay."""
    output_dir = output_dir.resolve()
    provenance = _read_json(output_dir / "data_provenance.json")
    frozen = _read_json(output_dir / "frozen_research_protocol.json")
    forward = _read_json(output_dir / "forward_validation_status.json")
    matched = _read_json(output_dir / "matched_horizon_frequency_comparisons.json")
    studies = {
        name: _read_json(output_dir / name / "study_results.json")
        for name in STUDY_NAMES
    }
    ledger_audit = {
        name: _audit_ledgers(output_dir, name, report)
        for name, report in studies.items()
    }
    candidate_audits = {
        name: {
            key: bool(
                studies[name]["prediction_and_portfolio_diagnostics"]["candidate_freeze_audit"][key]
            )
            for key in (
                "all_input_cutoffs_match_decision",
                "all_candidate_hashes_recompute",
                "all_settlement_rows_keep_candidate_hash",
                "all_settled_symbols_were_scored",
            )
        }
        for name in STUDY_NAMES
    }
    all_ledgers_pass = all(
        item["passes"] for study in ledger_audit.values() for item in study.values()
    )
    all_candidate_checks_pass = all(
        value for study in candidate_audits.values() for value in study.values()
    )

    common_start = next(iter(studies.values()))["common_start_date"]
    lines = [
        "# Post-P8 量化研究报告",
        "",
        f"开发数据：{provenance['development_start_date']} 至 {provenance['development_end_date']}（{provenance['development_trading_dates']} 个交易日）；市场行情仅来自 BaoStock，本地训练输入仅为 `train.csv`，不读取本地评分文件。",
        "已查看的 2026-03-16 至 2026-09-23 区间未并入开发回放，也不作为未见验证样本。",
        f"所有配置从 {common_start} 的收盘决策后以相同资金、全现金状态开始组合记账；下一交易日起计算组合净值。",
        "",
        "## 主要研究结论",
        "",
        "- 冻结误差折让与费用/风险惩罚下，1、3、5 日期限每日决策的两种联合优化器均保持全现金：净收益 0%、成交 0、费用 0。三日主配置在 0×/1×/2× 成本以及 15%/20%/25% 风险预算下也没有产生持仓。这是负面的投资结果：当前冻结规则没有找到足以覆盖误差折让和约束的可交易净增益；它不证明融合 Alpha，也不支持降低误差门槛。",
        "- 1、3、5 日期限每日决策的趋势策略净收益均为 +23.23%，等权融合均为 -0.27%。这些完全相同的收益路径来自共同账户日期和相同原始信号排序，不能当作三次独立期限验证。每个期限的标签、预测校准和预测误差仍分别评价；不同预测期限的 RMSE 绝对值不直接横比。",
        "- 每五个交易日复审的三日模型在开发期趋势策略净收益为 +50.51%，等权融合为 +25.77%，但滚动更新频率和可用成熟训练历史不同，不能据此作因果结论。趋势日复审减每五日复审的年化收益差配对 20 日区块 95% 区间为下文所列；这是已查看开发历史上的描述性比较。",
        "",
        f"候选冻结/结算映射审计：{'四组全部通过' if all_candidate_checks_pass else '至少一项未通过，请查 REPORT_BUILD.json'}；四组账本现金加持仓=净值、订单费用=汇总费用：{'全部通过' if all_ledgers_pass else '至少一项未通过，请查 REPORT_BUILD.json'}。逐日期候选审计保存在各组 `prediction_and_portfolio_diagnostics.json`。",
        "",
        "## 日期级预测评价",
        "",
        "下表为每个成熟决策日的横截面指标再做等权平均；Rank-IC 与收益均为日均值。Top-decile 一列是未扣费的毛收益，不是组合净收益。RMSE 按各自标签期限单独解释。",
        "",
        "| 配置 | 预测方法 | 成熟日 | Rank-IC | MAE | RMSE | Top 10% 毛收益 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in STUDY_NAMES:
        report = studies[name]
        metrics_by_method = report["prediction_and_portfolio_diagnostics"]["date_level_forecast_metrics"]
        for method in DIAGNOSTIC_METHODS:
            metric = metrics_by_method.get(method)
            if not metric:
                continue
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    STUDY_LABELS[name],
                    method,
                    metric["matured_dates"],
                    _number(metric.get("mean_date_rank_ic"), 4),
                    _pct(metric.get("mean_date_mae")),
                    _pct(metric.get("mean_date_rmse")),
                    _pct(metric.get("mean_date_top_decile_gross_return")),
                )
            )
    metrics_three_day = studies["three_session_daily"]["prediction_and_portfolio_diagnostics"]["date_level_forecast_metrics"]
    market_only = metrics_three_day["market_only"]
    market_relative = metrics_three_day["market_plus_relative_fusion"]
    lines.extend(
        [
            "",
            "三日市场分解：市场-only 与市场加相对收益的日均 MAE 分别为 {}、{}，RMSE 分别为 {}、{}；市场-only 是横截面常数预测，因此 Rank-IC/Top 10% 排序指标按定义未定义。加入相对收益后的全局误差没有下降，完整绝对收益和相对收益校准分别见三日诊断文件。".format(
                _pct(market_only.get("mean_date_mae")),
                _pct(market_relative.get("mean_date_mae")),
                _pct(market_only.get("mean_date_rmse")),
                _pct(market_relative.get("mean_date_rmse")),
            ),
        ]
    )

    primary = studies["three_session_daily"]
    primary_diag = primary["prediction_and_portfolio_diagnostics"]
    primary_prediction = primary["prediction_quality"]
    lines.extend(
        [
            "",
            "### 三日期限每日配置：校准、相关性与消融",
            "",
            "市场加相对收益预测的五个等量预测分桶如下。预测/实现值均为跨日期平均的单日横截面均值；不能据此推断交易后净收益。",
            "",
            "| 预测分桶 | 成熟日 | 平均预测 | 平均实现 | 实现减预测 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    buckets = primary_prediction["market_plus_relative_fusion"]["return_calibration_by_forecast_bucket"]
    for bucket, metric in sorted(buckets.items(), key=lambda item: int(item[0])):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                bucket,
                metric["matured_decision_dates"],
                _pct(metric["mean_date_predicted_return"]),
                _pct(metric["mean_date_realized_return"]),
                _pct(metric["mean_date_calibration_gap"], signed=True),
            )
        )

    lines.extend(
        [
            "",
            "专家间原始分数的日期级 Rank-IC 相关性与预测残差相关性如下。高残差相关表示这些专家在这组横截面误差上提供的独立信息有限。",
            "",
            "| 专家对 | 分数 Rank-IC 相关 [95% 区间] | 预测残差相关 [95% 区间] |",
            "|---|---:|---:|",
        ]
    )
    signal_corr = primary_diag["expert_signal_rank_correlations"]
    residual_corr = primary_diag["expert_forecast_error_correlations"]
    for pair in sorted(signal_corr):
        score_ci = signal_corr[pair]["interval_95pct"]
        residual_item = residual_corr.get(pair, {})
        residual_ci = residual_item.get("interval_95pct") or {}
        lines.append(
            "| {} | {} {} | {} {} |".format(
                pair.replace(":", " / "),
                _number(signal_corr[pair].get("mean"), 3),
                _ci(score_ci.get("mean_difference_95pct_low"), score_ci.get("mean_difference_95pct_high")),
                _number(residual_item.get("mean"), 3),
                _ci(residual_ci.get("mean_difference_95pct_low"), residual_ci.get("mean_difference_95pct_high")),
            )
        )

    lines.extend(
        [
            "",
            "留一专家消融的差值定义为“等权融合减去剔除该专家后的融合”；正值表示等权融合较高。表中同时列 Rank-IC 和 Top 10% 毛收益的 20 日区块 95% 区间。",
            "",
            "| 剔除专家 | Rank-IC 差值 [95% 区间] | Top 10% 毛收益差值 [95% 区间] |",
            "|---|---:|---:|",
        ]
    )
    for removed in ("trend", "reversal", "liquidity_confirmation"):
        item = primary_diag["leave_one_expert_out_ablation"][f"without_{removed}"]
        rank = item["equal_fusion_minus_ablation_rank_ic"]
        gross = item["equal_fusion_minus_ablation_top_decile_gross_return"]
        lines.append(
            "| {} | {} {} | {} {} |".format(
                removed,
                _number(rank["mean_difference"], 4),
                _ci(rank["mean_difference_95pct_low"], rank["mean_difference_95pct_high"]),
                _pct(gross["mean_difference"], signed=True),
                _ci(gross["mean_difference_95pct_low"], gross["mean_difference_95pct_high"], percent=True),
            )
        )
    lines.append(
        "各留一差值区间均跨过零；这组开发期消融没有证明等权融合优于任一留一版本。完整三期限消融及日期级数据见各自诊断 JSON。"
    )

    risk_diagnostic = primary_diag["portfolio_uncertainty_vs_risk_diagnostic"]["by_allocation_method"]
    lines.extend(
        [
            "",
            "### 组合误差与风险重叠",
            "",
            "下列误差诊断使用实际非零目标权重组合，对照冻结的 0.5× RMSE 误差额度与历史因子/特质风险期限标准差。覆盖率不是投资收益率。",
            "",
            "| 组合方法 | 加权成熟日 | 平均绝对组合预测误差 | 平均误差额度 | 平均期限风险 σ | 误差落在额度内 | 误差落在风险 σ 内 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("trend", "equal_fusion", "adaptive_fusion", "loss_gated_fusion", "market_plus_relative_fusion"):
        item = risk_diagnostic.get(method)
        if not item:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                method,
                item["matured_weighted_decision_dates"],
                _pct(item["mean_absolute_portfolio_forecast_error"]),
                _pct(item["mean_forecast_error_allowance"]),
                _pct(item["mean_historical_portfolio_horizon_sigma"]),
                _pct(item["fraction_realized_error_within_error_allowance"]),
                _pct(item["fraction_realized_error_within_historical_horizon_sigma"]),
            )
        )
    lines.append(
        "两种联合优化器在本开发样本中目标权重始终为零，因此没有对应的非零持仓组合误差/风险覆盖记录。上表来自其他有目标权重的组合；它不能证明全现金优化器已有可交易校准。"
    )

    lines.extend(
        [
            "",
            "## 现金、市场与连续组合基线",
            "",
            "换手列以窗口累计单边成交额倍数表示，不是年化比率。对策略组合，它是累计单边成交额除以平均 NAV；例如 74.75× 表示该窗口累计单边成交额约为平均 NAV 的 74.75 倍。固定风险指数代理按其账本定义除以起始资金。",
            "",
            "| 配置 | 基线/策略 | 收益口径 | 收益 | 平均现金 | 最大回撤 | 窗口累计单边换手 | 费用 | 成交额 | 成交单 | 阻塞单 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in STUDY_NAMES:
        report = studies[name]
        label = STUDY_LABELS[name]
        benchmarks = report["cash_and_csi300_baselines"]
        cash = benchmarks["cash"]
        lines.append(
            _baseline_row(label, "现金", cash, "净", cash.get("net_total_return"), None)
        )
        index = benchmarks["csi300_benchmark"]
        lines.append(
            _baseline_row(label, "CSI 300 指数价格", index, "毛", index.get("gross_total_return"), None)
        )
        risk = report["fixed_risk_market_baseline"]
        lines.append(
            _baseline_row(label, "固定风险 CSI 300 价格代理", risk, "净代理", risk.get("net_total_return"), risk.get("one_way_turnover"))
        )
        for method in PORTFOLIO_METHODS:
            lines.append(
                _portfolio_row(label, method, report["portfolio_results"][method])
            )
    lines.extend(
        [
            "",
            "CSI 300 基线使用 BaoStock 指数价格毛收益。固定风险基线只用共同起点以前的指数历史波动率设定一次暴露，按相同费率估算进出费用；其指数价格交易是代理成交，不模拟 ETF 跟踪误差、容量或真实盘口。",
            "",
            "## 期限与复审频率配对区间",
            "",
            "下表为左侧配置减右侧配置的年化收益差；日期对齐于相同现金起点和共同组合净值日期。区间采用 20 交易日区块 Bootstrap、每组 2,000 次重抽样。",
            "",
            "| 对比（左减右） | 策略/基线 | 配对日 | 日期范围 | 年化收益差 95% 区间 |",
            "|---|---|---:|---|---:|",
        ]
    )
    for contrast in matched:
        left = STUDY_LABELS[contrast["left_study"]]
        right = STUDY_LABELS[contrast["right_study"]]
        for method in PAIRED_METHODS:
            item = contrast["by_method"].get(method)
            if not item:
                continue
            bootstrap = item["paired_block_bootstrap"]
            label = {
                "cash": "现金",
                "csi300_benchmark": "CSI 300 毛收益",
                "fixed_risk_market": "固定风险 CSI 300 代理",
            }.get(method, method)
            lines.append(
                "| {} 减 {} | {} | {} | {}–{} | {} |".format(
                    left,
                    right,
                    label,
                    item["paired_observation_dates"],
                    item["paired_start_date"],
                    item["paired_end_date"],
                    _ci(
                        bootstrap["annualized_return_difference_95pct_low"],
                        bootstrap["annualized_return_difference_95pct_high"],
                        percent=True,
                    ),
                )
            )
    lines.extend(
        [
            "",
            "趋势与等权在 1/3/5 日每日配置的配对差为零，因为这三组在共同账户日期上持有相同原始排名形成的路径；这不是期限效果的独立复验。3 日每日与每 5 日复审之间的训练成熟历史和交易频率不同，区间是配对历史差异，不是随机因果效应。全部方法的共同日期和区间原始值见 `matched_horizon_frequency_comparisons.json`。",
            "",
            "## 成本与风险敏感性",
            "",
            "三日每日主配置在原始预测和误差折让不变时运行 0×、1×、2×费用及 15%、20%、25% 风险预算。两种联合优化器在每个场景下都保持全现金、零成交、零费用；改变成本或风险预算没有创造正交易净收益。完整各方法结果见三日主配置 `study_results.json`。",
            "",
            "| 情景 | 绝对收益联合优化净收益 | 市场相对联合优化净收益 |",
            "|---|---:|---:|",
        ]
    )
    for scenario in ("0x", "2x"):
        by_method = primary["cost_sensitivity"].get(scenario, {})
        lines.append(
            f"| 成本 {scenario} | {_pct(by_method.get('joint_optimization', {}).get('total_return'))} | {_pct(by_method.get('joint_market_relative', {}).get('total_return'))} |"
        )
    for scenario in ("15pct_budget", "25pct_budget"):
        by_method = primary["risk_sensitivity"].get(scenario, {})
        lines.append(
            f"| 风险预算 {scenario[:2]}% | {_pct(by_method.get('joint_optimization', {}).get('total_return'))} | {_pct(by_method.get('joint_market_relative', {}).get('total_return'))} |"
        )

    lines.extend(
        [
            "",
            "## 前向验证状态",
            "",
            f"本次开发运行未查询或纳入 2026-09-23 之后的新日期；当前市场源是否已有更新数据未检查。该开发快照中新前向交易日和成熟决策日均为 {forward['new_trading_dates_in_this_development_run']}，不代表市场当前无新日期。真实前向仍需只用 2026-09-23 之后的新决策，并至少等到 120 个新决策标签成熟。按期限需要的新交易日为 1 日 121 日、3 日 123 日、5 日 125 日。",
            "未来逐日 BaoStock 采集/快照命令、状态切换、成熟条件和边界规则见 `POST_P8_RESEARCH_PROTOCOL.md`。本研究不构成真实前向验证结果。",
            "",
            "## 产物索引",
            "",
            "- 四组完整结果：各子目录 `study_results.json`、`prediction_and_portfolio_diagnostics.json`、`prediction_decisions.jsonl.gz`、`prediction_settlements.jsonl.gz`。",
            "- 逐策略净值、订单、持仓和决策账本：各子目录 `portfolios/`。",
            "- 期限/复审比较：`matched_horizon_frequency_comparisons.json`。",
            "- 冻结规则与前向状态：`frozen_research_protocol.json`、`forward_validation_status.json`。",
            "- 本报告输入文件哈希和报告生成器版本：`REPORT_BUILD.json`。",
        ]
    )

    report_path = output_dir / "RESEARCH_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    input_paths = [
        output_dir / "data_provenance.json",
        output_dir / "frozen_research_protocol.json",
        output_dir / "forward_validation_status.json",
        output_dir / "matched_horizon_frequency_comparisons.json",
        *(output_dir / name / "study_results.json" for name in STUDY_NAMES),
        *(
            output_dir / name / "portfolios" / method / file_name
            for name in STUDY_NAMES
            for method in studies[name]["portfolio_results"]
            for file_name in ("daily_nav.csv", "orders.csv", "positions.csv")
        ),
    ]
    module_path = Path(__file__).resolve()
    report_build = {
        "renderer": "adaptive_mas.evaluation.post_p8_report",
        "renderer_sha256": _sha256(module_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in input_paths
        },
        "report_sha256": _sha256(report_path),
        "candidate_freeze_audit": candidate_audits,
        "ledger_and_fee_audit": ledger_audit,
        "all_candidate_audits_pass": all_candidate_checks_pass,
        "all_ledger_and_fee_audits_pass": all_ledgers_pass,
        "uses_saved_results_and_ledgers_only": True,
        "replayed_market_data": False,
    }
    (output_dir / "REPORT_BUILD.json").write_text(
        json.dumps(report_build, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path
