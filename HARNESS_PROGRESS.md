# GEPA Harness 当前进度

## 已完成

- 增加行情、基本面和研究经理三个 function-calling 角色，按本地只读公开工具逐步交接；单案上限为 10 个模型轮次。
- 接入官方 GEPA optimize_anything，演化六个文本组件。固定事实、引用、报告期、点时、效率和成本评价不随候选变化。
- 单日 API 支持复用已完成研究 JSON；复用不创建新的 provider 请求。
- 预测与组合优化继续使用成熟校准模型和现有优化器。旧 contextual-RL 日常入口保留为历史路径。
- 未读取 test.csv、未下载数据。回顾性样本使用来源披露时间代理，strict historical PIT 为 false，供应商历史 vintage 未核验。

## 本次 GEPA 结果

运行编号 20261023104652，状态 rejected。活动版本前后均为 seed。唯一去重提案因研究输出超过固定 120 字结论长度限制，被判无效并记 0 分；候选池仍只有 seed，独立选择没有运行。没有候选获选，也没有投资改善证据。

[可发布安全报告](reports/harness_20261023/RUN_REPORT.md)包含安全汇总、active seed 提示导出、拒绝候选 diff 和必要候选审计。安全包不含原始调用明细、原始 prompt/response、账户字段或本地来源正文。

冻结集两臂均使用 seed，后续对话哈希和调用数不同；分数差异不可归因于候选。细粒度哈希记录只保存在本机，不作为发布链接，路径为 experiments/gepa_harness/runs/20261023104652/frozen_request_hash_audit.json。

## 复用和短窗口模拟摘要

日常 API 演示复用了已完成研究，决策为 hold、无订单、没有新增 provider 请求。短窗口模拟两臂都没有交易；差异仅由逻辑研究成本计入方式造成，不是投资绩效证据。账户和完整模拟细节不包含在安全包中。

## 请求与费用汇总

共享账本累计有 111 次已完成 provider 请求、42 次精确缓存复用；另有两次连接前失败，均未发送且费用为 ¥0。已完成请求为 241,923 输入 token 和 29,921 输出 token。按项目规划单价估算，API 成本 ¥3.980232，缓存对应逻辑成本 ¥1.139268，累计逻辑推理成本 ¥5.1195。它们是 token 用量估值，未与服务账单核对。

本次 run 使用 42 次 provider 请求，API 成本估算 ¥1.506132；共享 160 次请求上限还剩 49 次，未知投递结果为 0。

## 本机审计路径

未发布的 GEPA 运行日志和 reflection review 位于 data/processed/harness_versions/runs/20261023104652/。完整请求账本和响应缓存位于 experiments/gepa_harness/calls.jsonl 与 experiments/gepa_harness/response_cache/。这些原始轨迹未复制到可提交安全包。
