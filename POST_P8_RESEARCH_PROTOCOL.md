# Post-P8 研究与前向验证流程

本流程固定 BaoStock 为唯一市场数据源，只从本地 `train.csv` 读取训练样本；`test.csv` 不会被读取。已查看的 2026-03-16 至 2026-09-23 数据只用于新前向运行的状态切换暖启动，不计入前向指标，也不作为未见样本。

## 开发期研究

在项目根目录执行一次四配置研究。输出目录必须不存在；脚本会拒绝覆盖已有结果。

```powershell
.\.venv\Scripts\python.exe scripts\run_post_p8_study.py `
  --training-data-dir "C:\Users\xiao\Desktop\THU-BDC2026\data" `
  --baostock-run-dir data\raw\baostock\fetch-20260924T115050Z `
  --output-dir experiments\post_p8_research_20260926
```

研究比较每日决策的 1、3、5 交易日预测期限，以及每五个交易日复审的 3 日预测。组合都在同一个收盘后日期以相同初始资金和全现金状态开始记账，下一交易日开始记收益；预测期限/复审频率的暖启动历史仍有差异，因此这些历史配对差值只作诊断，不作因果结论。

输出包括完整预测与结算行、候选集哈希审计、专家与融合校准/消融/误差相关性、连续账户净值/成交/持仓账本、现金与 CSI 300 基线、固定风险 CSI 300 价格代理、配对区块区间，以及主配置的成本和风险敏感性。联合优化的预测误差折让固定为 0.5 倍成熟预测 RMSE；风险敏感性不改变该折让。

研究脚本会在目标目录写入 `RUN_STATUS.json` 和 `progress.log`，记录 PID、启动命令、进度时间与阶段。若进程失败，先查看这些记录和具体错误；不要在已有目录上重跑。需要重跑时请使用新的输出目录名。

如果四组结果已存在，只需更新展示报告时，可以从项目根目录执行下面命令。它只读取已保存的结果 JSON 和策略账本，不重新读取行情或重放回测，并会写入报告输入哈希与账本核对结果。

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\render_post_p8_report.py `
  --output-dir experiments\post_p8_research_20260926
```

## 真正的新日期前向验证

只有实际收盘后才把当天日期传给采集命令。每次创建新的 BaoStock 原始缓存目录；样本外采集模式不打开训练或测试 CSV。

```powershell
.\.venv\Scripts\python.exe scripts\acquire_phase2_market_data.py `
  --oos-start-date 2026-03-16 `
  --oos-end-date <实际已完整收盘的日期> `
  --output-root data\raw\baostock
```

采集完成后，使用新生成的 `fetch-...` 目录、真实缓存末日，以及开发研究冻结的结果、种子和配置运行快照。下面示例使用 3 日每日配置；路径应指向本次真实生成的缓存。前向脚本要求缓存末日与 `--as-of-date` 完全一致，并为每个日期建立独立、不可覆盖的输出目录。

```powershell
.\.venv\Scripts\python.exe scripts\run_post_p8_forward_validation.py `
  --baostock-run-dir data\raw\baostock\fetch-<本次新生成的目录> `
  --as-of-date <与缓存末日完全相同的日期> `
  --development-result experiments\post_p8_research_20260926\three_session_daily\study_results.json `
  --training-seed experiments\post_p8_research_20260926\three_session_daily\training_seed.json.gz `
  --config configs\p8_daily_three_session.toml `
  --output-root experiments\post_p8_forward_validation
```

每次快照从冻结的开发种子重新重放 2026-03-16 至 2026-09-23 的已查看区间，得到 2026-09-23 收盘时的校准/可靠性状态；这些暖启动标签不计入验证。随后只对 2026-09-23 之后的决策计分，并按已成熟标签逐步更新状态。这样重复运行新快照不会把同一批标签重复加入状态。快照会保存切换状态和截至当日的在线状态，供审计；本流程不把已查看日期重新命名为新样本。

至少需要 120 个新且已成熟的决策日期才触发预先写定的判断。对单一每日决策模型，考虑开仓和标签成熟所需的新交易日分别为：1 日期限 121 日、3 日期限 123 日、5 日期限 125 日。少于 120 个已成熟日期时只能报告等待；达到后才比较市场相对联合优化与现金、毛收益 CSI 300、固定风险 CSI 300 价格代理、各单体专家、等权融合、P8 自适应融合及损失门控融合。判断同时检查配对 20 交易日区块区间、扣费净贡献和账户/风险约束。该规则不在前向开始后调参。

每次快照会写入 `RUN_STATUS.json`、`progress.log`、新日期预测与结算文件、逐策略账本、诊断及 `forward_validation_report.json`。逐日更新使用完整 BaoStock 缓存，并要求其末日等于运行参数中的截止日，避免意外读入截止日之后的数据。

## 不纳入本流程的工作

强化学习、文本 Agent、生产 API 和 UI 不属于本轮可完成的研究，不会在结果中伪装成已验证功能。任何未来的独立部署或产品决策应另行设计和授权。
