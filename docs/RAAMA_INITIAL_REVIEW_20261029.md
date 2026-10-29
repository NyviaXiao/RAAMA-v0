# RAAMA 初版人工审核入口

日期：2026-10-29。当前阶段按用户要求提前收口，提交人工审核；未完成的 A–G 项目留待后续，本文不表示工业级整体验收完成。

## 先看这三处

1. 本文：当前能核验什么、仍缺什么，以及最短复现入口。
2. [逐项实现矩阵](RAAMA_INDUSTRIAL_RAG_TRACEABILITY_20261025.md)：对应源码和证据边界。
3. [Linux CPU 交接](RAAMA_REMOTE_CPU_HANDOFF_20261025.md)：资源限制、依赖/模型、运行顺序和新对话启动词。该文件包含较长历史记录；以本文和其末尾最新记录为当前入口。

## 版本与交付范围

- 本次业务冻结锚点：本地 `c8fdc5dd879c3fb2d2d6e005605defb29597d208`，包含已审核的授权消费门禁。后续封装提交只纳入本文和交接文档。
- 上一个已核备份：本地 `985832a6721eee84148755062ff3df858ac8c901` → `87326710e0573af3cc37dc0224e52fba8325da80`，tree `10926b2ba67ac5d6f76c1dcc12e43453828457e0`。
- 最终源码备份仍在 `NyviaXiao/RAAMA-v0` 的 `backup/raama-implementation-20261024`；交付消息给出精确封装 SHA 与远端快照 SHA。远端提交正文同时记录本地封装 SHA，可用 `git log -1 --format=%B` 核对；不要把本地与远端快照 SHA 当成同一个值。
- Git 包含审核后的源码、配置、测试、锁文件和脱敏文档；本地数据库、原始报告、模型权重、账户资料和原始实验不在交付包中。运行时目录由命令新建，不要求远端具备本机历史 `.tmp`。

## 人工审核优先顺序

| 审核点 | 已有证据 | 初版保留的限制 |
|---|---|---|
| 原金融主线 | 原 `420cc4f` 的 302 项测试验收；BaoStock/AKShare、专家/经理、风控、含费现金和数量账本保留 | 历史数量不代表新环境全部通过；投资经理仅解释汇总 |
| 解析与来源 | 已运行本地 Docling/OCR；CATL 两张已看页面各输出一张表、69/82 个单元格；原生 PDF 另有来源重开证据 | 这次图像运行只有文本/同一输出行匹配检查，未独立核验精确坐标；更多跨页布局和新公司未验 |
| 检索与隔离 | 一次已看 CMB 的真实 Qwen 1024 维查询向量 + Qdrant 应用检索；复用 901 个文档向量，50 个候选、10 项结果；权限/PIT 拒绝证据 | 单公司个案；三项拒绝是直接适配器测试；不能据此推断完整应用负例、生成质量或重排增益 |
| 工具与恢复 | manager 三个离线工具的结果/上下文写入 SQLite，重试复用，参数身份不匹配拒绝，未完成调用要求对账 | 服务端 RAG 禁止这种旧结果重放；真实 Temporal 研究 worker 重启恢复、HA 仍待验 |
| 生成授权入口 | live-agent CLI 强制已有一次性 grant；项目固定 SQLite 库持久消费，不随输出目录重置；预连接失败也消耗 attempt | 当前没有新的有效真实 grant；这不是对所有历史脚本的统一迁移，也不是人工身份的密码学认证 |
| 评价 | 已有固定门禁、轨迹、20-ID 目录；10 个有界运行控制和 2 个参数敏感性契约 | 未完成真实模型驱动的 20 模块效果评价，已看诊断不算盲测 |
| 工作台 | 已有中文 RAG 查询与 Harness 工作台、CLI/API 本地测试 | 初版不宣称完整真实浏览器人工操作已验收 |

**生成报告仍未验收成功。** 已记录的 CMB/Qwen 尝试未通过固定 schema/证据校验；不能将这些失败报告当成通过审核的投资分析。

## 最短离线复现

以下命令在仓库根目录执行，使用已安装项目依赖的 Python。Windows 可将 `python` 换成 `.venv\Scripts\python.exe`；PowerShell 先执行 `$env:PYTHONPATH='src'`，Linux 先执行 `export PYTHONPATH="$PWD/src"`。

```text
python -m adaptive_mas.industrial_rag.cli --help
python -m adaptive_mas.harness_service --help
python -m pytest -q -p no:cacheprovider tests/test_harness_state.py tests/test_harness_unified_workflow.py
```

这些检查使用本地合成输入和 SQLite，不需要真实 provider、Docker、原始年报或模型权重；它们证明契约/流程行为，不能代替真实模型质量验收。首次安装参考 `uv sync --locked --extra dev`，Temporal 回归另需已锁定的 `temporal` extra；安装依赖本身可能需要网络。

本次收口实际检查：

- `c8fdc5d` 精确源码归档中，`test_harness_state.py` 与 `test_financial_rag_live_agent.py` 共 **28 passed，3.46s**。
- 父额外验证两个并发消费者只有一个消费成功；更换 grant ID 复用同一 approval event 被拒绝。全程 0 provider 请求。
- 按已核远端 tree 的 273 个源码文件加本次精确业务差异重建干净快照，逐个 Git blob 哈希核对；未复制旧 `.tmp`、模型、语料或运行数据库。该快照中上述最短复现的两份测试 **21 passed，1.61s**，两条 CLI help 均 exit 0，导入路径确认来自快照本身。
- 干净快照复用本机现有 Python 依赖环境；它不是重新安装依赖的 Git 网络克隆，也不是 Linux 验收。最终封装仅改变审核文档，不重复上述业务测试。

如需人工查看界面，可在新终端启动独立空白 RAG 工作台：

```text
python -m adaptive_mas.industrial_rag.cli serve --host 127.0.0.1 --port 8897 --database .tmp/initial-human-review/rag.sqlite3 --objects .tmp/initial-human-review/objects
```

然后访问 `http://127.0.0.1:8897`。端口若已占用，选择新的空闲端口；不要停止已有服务。此命令新建独立运行库，默认没有本机旧语料，也不加载 Qwen 模型或发起生成请求；空库界面不能展示真实年报检索效果。原 8875/8876 数据库不参与。

## 已知失败与授权记录

- 2026-10-29 编码聊天误将已消费的 10 月 27 日旧答复当成新授权，额外发出了 4 次 `qwen3.8-27b` 请求，报告仍被固定证据校验拒绝；估计成本 CNY 0.290136，非账单核对值。
- 编码聊天和父审核已核原始事件并纠正归属；原始结果、请求账本、失败和成本记录保留，剩余一次不可使用。初版收尾只做本地检查，没有新增生成授权。
- 门禁现在默认拒绝缺失/未知/已消费 grant。真实许可的登记仍须核对新的直接人类授权；迁移或新建目录不能用来重新登记旧答复，本次没有签发任何真实 grant。
- 新取数应优先复用 BaoStock/AKShare→公告→官方 PDF 既有链路。此前 Edge 尝试只发生访问审批超时，未完成比亚迪新年报下载；Edge 不是产品依赖。

## 后续待人工决定

1. 检查上述已实现行为与业务目标是否一致，再决定下一轮优先项。
2. 真实生成报告、更多来源/版面、真实浏览器交互、完整 PG/S3/OpenSearch/Qdrant/Temporal 故障矩阵另行验收。
3. 共享 Linux 全项目合计上限是 20 核、200000000000 字节、无 GPU；必须先由管理员分配并实际核验聚合限制，再运行重模型/服务测试。当前没有远端连接授权。
4. 专家盲标、未来成熟收益、生产规模与科学效果保持 pending。

交付这个初版后停止主动扩展，等待人工审核意见。下一轮从固定版本和明确待办继续，不把历史测试数量相加成新的验收结果。
