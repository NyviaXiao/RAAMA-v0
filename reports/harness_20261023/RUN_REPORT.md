# GEPA Harness 安全运行报告

## 结果

运行 20261023104652 已完成，GEPA 结果为 rejected。活动版本仍是 seed；唯一去重提案未进入候选池，独立选择集未运行。目前没有候选获选，也没有投资改善证据。

候选提案的结论超过固定 120 字输出限制，被评价器判为无效并记 0.0。两次日志记录对应同一组六组件文本。已从持久提案日志重建真实差异，并与六条精确反思响应逐项核对。

## 可提交产物

- 汇总质量、计数、费用和结论：[run_summary.json](run_summary.json)
- 活动 seed 六组件导出：[active_seed_prompts.json](active_seed_prompts.json)
- 被拒绝提案差异：[rejected-proposal.diff](rejected-proposal.diff)
- 候选计数、哈希和拒绝依据：[candidate_audit.json](candidate_audit.json)

以上链接均指向本包内文件。安全包不包含调用明细、原始 prompt/response、请求账本、完整 GEPA 日志、公共来源正文或账户字段。

## 质量和评价限制

seed 探索均分为 0.7947487923。冻结集两臂均使用 seed；active 和 selected 的平均分分别为 0.7127965580 和 0.6927482790。日志哈希显示部分首轮请求复用，但后续请求序列不同；因此这个分差不能归因于候选修改。

所有样本均为回顾性公开缓存，评价模式使用来源发布时间/可用时间代理。strict historical PIT 为 false，供应商历史版本及当时抓取情况没有独立验证。

## 请求和成本汇总

共享账本累计 111 次已完成 provider 请求、42 次精确缓存复用；另有 2 次连接 provider 前失败，均未发送且费用为 0。累计 provider 输入/输出 token 为 241,923 / 29,921。按项目预算单价估算，新增 API 费用为 ¥3.980232，缓存复用对应逻辑成本为 ¥1.139268，累计逻辑推理成本为 ¥5.1195。这些是 token 用量估值，未按服务账单核对。

## 日常复用摘要

预计算研究复用产生 hold 决策、0 个订单和 0 次新 provider 请求。短窗口演示的两臂均无交易，观测差异只来自逻辑研究成本计入方式；它不构成真实账户表现或投资绩效结论。样例账户金额、持仓和完整模拟数据未包含在本包中。

已发布 diff 只统一输入行的 LF 换行符；六段提示文字内容未改动。本地原始审计及其原始 diff 哈希保持不变。

## 原始审计资料

原始 GEPA 日志、反思请求 review、调用账本和精确响应缓存仅保留在本机，不作为发布链接：data/processed/harness_versions/runs/20261023104652/run_log.txt；data/processed/harness_versions/runs/20261023104652/reflection_payload_review.json；experiments/gepa_harness/calls.jsonl；experiments/gepa_harness/response_cache/。
