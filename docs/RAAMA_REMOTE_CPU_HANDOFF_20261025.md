# RAAMA：Git 迁移与无上下文 Linux CPU 验收交接

日期：2026-10-25。本文是原工业级金融 RAG / Harness A–G 计划的新增交接要求，不替代原计划。

**当前是准备中的交接契约，不是“本机工作全部完成”的证明。** 最终移交前必须更新本文的版本与证据表，提交、独立审核并核对授权备份分支。新对话不能把本文、旧测试数量或接口存在当作验收通过。

## 1. 目标和必须保留的上下文

产品是 A 股金融投资研究与组合分析 Agent。保留 BaoStock/AKShare 数据主线，市场、基本面、新闻、情绪/关注度专家，方案经理、确定性优化器、风控与执行账本。RAG 是共用的证据工具；投资经理只解释与汇总，不获得风险豁免、生产发布或发单权限。

读取顺序：

1. 仓库及上级适用的 `AGENTS.md`。
2. `docs/RAAMA_INDUSTRIAL_RAG_IMPLEMENTATION_PLAN_20261025.md`。
3. `docs/RAAMA_INDUSTRIAL_RAG_HARNESS_REDESIGN_PROPOSAL_20261024.md`。
4. `docs/RAAMA_INDUSTRIAL_RAG_TRACEABILITY_20261025.md`。
5. 本文、`docs/RAAMA_INDUSTRIAL_RAG_OPERATIONS_20261025.md`。
6. 原金融契约 `docs/RAAMA_REQUIREMENTS_TRACEABILITY_20261024.md` 与 `docs/RAAMA_RISK_PROFILE_20261024.md`。

本机聊天记录和 `.tmp` 下监督记录不会随 Git 上传。必须将新对话所需的结论、失败复现、模型/依赖版本和待测项整理进可提交的文档、测试和脱敏报告；不能引用一个远端不存在的 Windows 文件作为唯一证据。

研究代码不得读取 `test.csv`。保留 `experiments/**`、历史冻结记录、用户文件和既有服务。权限、实体、时间点可见性、风险档案、含费现金与账本约束不得因迁移退化。公开文档本次下载时间不等于历史已捕获时间。真实未来收益、专家独立盲标和生产规模效果继续单列待积累。

## 2. 服务器边界

用户提供的是多人共享 CPU 服务器，机器有 128 核、总内存约 515415 MiB；项目授权上限只有 **20 核和 200G 内存**，无 GPU。

保守地把 200G 解释为十进制 **200000000000 字节**，不是 200GiB。所有 RAAMA 进程合计受限：数据库、对象存储、检索、工作流、OCR/模型、解析进程、测试、构建和下载解压均计入。机器显示还有多少空闲内存不扩大授权。

用户提供的 swap 已大量使用是共享机器的观测值，不能据此诊断某个进程，也不能清 swap、关闭主机 swap、重启或杀他人任务。

开始运行前先检查发行版、Python/CPU 架构、磁盘与用户配额、端口、调度器、cgroup 版本和被委派的控制器、容器权限。检查仅针对本用户与项目，不读取其他用户数据。

优先使用管理员已有的调度器/资源分配。例如 Slurm 分配应包括单节点、20 个 CPU，以及不超过上述字节上限的内存申请，并确认站点实际启用内存和 CPU 限制；不能只提交一个资源声明就认定内核已限制。

没有调度器时，使用已获允许的独立项目 cgroup/slice。cgroup v2/systemd 的参考边界是：

| 配置 | 项目总限制 | 含义 |
|---|---:|---|
| CPUQuota | 2000% | 合计最多相当于 20 个逻辑 CPU 的时间配额 |
| MemoryHigh | 180000000000 | 达到此值时提前回收/施压 |
| MemoryMax | 200000000000 | 项目内存硬上限 |
| MemorySwapMax | 0 | 不新增使用共享 swap，须检查内核及控制器支持 |
| CPU affinity / cpuset | 管理员实际分配的 CPU 集合 | 若要求最多使用指定 20 核，还需与 CPUQuota 一起限制 |

上述是待实际验证的资源契约，不是已经部署的配置。CPUQuota 限制 CPU 时间，并不独自保证线程从不在超过 20 个 CPU 上调度。不得擅自选机器的 CPU 0–19 当作自己的独占 CPU。

**必须检查所有服务 PID 的实际 cgroup 和生效值。** 只对终端运行 `systemd-run` 不保证共享 Docker daemon 创建的容器被纳入；每个容器各限 20 核/200G 也会令总量超限。容器与原生进程必须纳入同一受控分配，或使用经核验总和不超限的分配并保留应用/测试预算。无法强制总限制时先处理这一阻塞，再启动重任务。

数值库线程数、解析并发、模型 batch、JVM heap 是第二层保护，不能代替总 cgroup 限制。建议普通测试串行；批处理从少量 worker 开始；单模型 CPU 推理时再使用分配内的线程。禁止 `pytest -n auto` 或根据 `os.cpu_count()==128` 自动满载。多 worker 各自内部 20 线程会造成超订阅。索引/JVM 的堆外内存也要计入。

## 3. Git 传递方式

已授权的仓库和备份分支：

- 仓库：`https://github.com/NyviaXiao/RAAMA-v0.git`
- 分支：`backup/raama-implementation-20261024`

目前该分支是经 GitHub 连接器创建的代码树快照，本地源码提交 SHA 与远端快照 SHA **不同**。验收要记录二者的映射和文件校验，不可要求两个 SHA 相等。不要直接把本机开发历史强推覆盖备份分支。

最终交接需给出一个已核验远端完整 SHA。以下命令在服务器的本人新目录运行，不覆盖现有 checkout：

```bash
git clone --single-branch --branch backup/raama-implementation-20261024 \
  https://github.com/NyviaXiao/RAAMA-v0.git raama-linux-acceptance
cd raama-linux-acceptance
git rev-parse HEAD
git status --short
git switch -c verify/linux-cpu-20261025
```

克隆后把 HEAD 与最终交接的远端 SHA 核对。若用户后来需要回到某次移交，先确认工作区干净，再检出明确 SHA；不使用 `reset --hard` 处理未知改动。私有仓库使用本人 SSH 凭据或 Git 凭据管理器，不把 token 放进 URL、聊天或脚本。

Git 仅传源码、配置模板、依赖锁、schema、测试、允许上传的测试标注与脱敏报告/方案。不会传 Windows `.venv`、原文 PDF、模型权重、本地数据库、账户、密钥或 `.tmp`。依赖在 Linux 重新建环境；公开语料与模型按 URL、明确 revision、hash 和使用边界重新获取。运行状态若需迁移，必须有单独审过的导出/导入与兼容性校验，不把复制 SQLite 文件当作完整状态迁移。

## 4. VS Code 与新 Codex 对话

1. 本地 VS Code 使用 Remote–SSH 连接自己的 Linux 用户。
2. 打开远端 `raama-linux-acceptance` 仓库目录，确认集成终端中的 `pwd`、`hostname` 和 `git rev-parse HEAD` 都是目标服务器及目标版本。
3. 在扩展视图确认 Codex 对当前 SSH 工作区可用；需要安装到远端的扩展按 VS Code 提示安装，使用本人账号登录。不依赖本机原聊天自动同步上下文。
4. 把第 8 节的接手提示词发给新对话。要求它先读文件并复述状态/边界，再执行资源探测与分阶段验收。
5. Web 工作台只绑定服务器 loopback，通过 VS Code Ports/SSH 隧道访问，不公开绑定共享服务器网络。

不要把本机自动恢复许可当成远端新对话的额度授权。此处本机新增两次恢复由父监督计数；远端后续如何持续工作由用户在该对话中明确。

## 5. 当前版本与证据；最终移交前须刷新

| 项目 | 本文初稿时的已核验状态 |
|---|---|
| 原金融/风险输入基线 | 本地 `420cc4f4dc26f875d568095da0fa6ed79e35732a`，父独立 302 项通过 |
| 新 RAG 初始 A/B 检查点 | `d31bcbb5432ce8782799a43eae4b825684a66d43`，父精确归档全 316 项通过，另发现 4 项负例缺陷 |
| 4 项修复 | `c06a102cccecc4f6748eb6b7fbed4f560a599631`，父精确归档相关 82 项通过，4 项负例均复核修复 |
| 最新已核验远端快照 | `861e34bd8b0d7218ea509814f7ad4e24f35ccfb3` / tree `c3a3a70d24f26f426ce0d3dd3c0af5942789142d`，对应本地 `5937d262975f7fdcec66e3a260a3908d4d5fd20a`；8改动blob/ref一致，expectedSHA65b1004c非force。原三回归及配对身份缺陷已复核修复；完整A–G/真实基准与版本物化恢复未完成。前一c462612对应65b1004ca6dad8d294fb44219b038c1f04ff7a33/treefbbdc4ae9aecf6f9050d295a82ac92024daf310e，14blob已核验 |
| 新的桥接与解析增量 | `57beaa3` 父精确归档相关 85 项通过；独立探针确认省略 limit、重复及改写同证据检索修复；另发现 Office 宏包隔离的大小写缺陷 C-P1-01，已要求修复。该备份是注明缺陷的检查点，不是阶段总验收 |
| 宏隔离修复 | `be83128e3e09a56ec151afda9b5749852ac813be`，父精确归档 15 项解析/Office/表格测试通过，原惰性宏标记包独立复现已拒绝 |
| 本机模型与评价切片 | `c3388ca94f25374007bfe275140909f361d4559f` 父精确归档全 355 项通过；真实已看公开财报文本上 CPU Qwen Embedding/Reranker 运行成功，3×1024 向量、单独/批量最大差约 3.15e-7；Docling 对原页 3/4 两页摘录运行返回 2 表、238 格、49 行组。这证明可运行，不证明金融结构与评价全部正确 |
| 3d85e6b 独立复核 | 相关 129 项中 122 通过、7 失败：原统一研究流程因市场派生浮点的舍入被严格 Decimal 比较拒绝，已交修复。G-P1-02 的定位格数值与顶层元数据冲突负例已确认修复。真实 Docling 同源复验仍只分出 2 表，Q1 经营现金流 -1208 被标为年度，C-P1-03 尚未修好 |
| 6a7608a 独立修复验收 | 相关 139 项通过，原 7 项统一流程回归恢复。独立负例确认税前利润不能满足净利润、稀释 EPS 不能满足基本 EPS、主营收入不能满足营业总收入，精确同指标正例通过。真实同源 Docling CPU 复验现为 3 表、238 格、46 行组；Q1 现金流 -1208 保留人民币百万元和 2024 年第一季度，原 C-P1-03 个案已关闭 |
| 缓存与本地接口切片 | 4e58a6e 增加运行时身份后，父发现未知 callable 仍会串用缓存；31d3bf9 修复为身份缺失时跳过持久缓存。父精确归档 RAG/Qwen/实际本机 CLI+HTTP 43 项通过，原 A/B 独立负例确认第二组件得到自身 B。此处 HTTP 流程使用明确标记的合成源，不是浏览器或真实金融决策链验收 |
| D06 分页与控制 | 3618d70 父精确归档相关 43 项通过（9.94s），包含真实本机 HTTP 空结果/默认分页、游标绑定与失效、在途撤权、取消/超时/背压。另独立证实请求槽全满时取消接口也返回 503，无法设置取消事件（D06-P2-01），已交修复；未验证外部索引 PIT、浏览器或目标服务器负载 |
| 8b041f7 主张与取消复核 | 父精确归档全量 386 通过、2 失败（26.52s）：行业风险审查及独立 worker 服务端到端回归失败，已交修复。原 1/3 Decimal 序列化、日期百分比豁免负例与满负荷取消均复核修复；另发现 bp 与百分点被等价化（100 倍差异），须显式单位换算或分别验证。当前并非可直接迁移验收的最终版本 |
| 4463d13 / f75dfbb 修复与语料扩展 | 父精确归档全量 392 项通过（28.10s），行业风险与独立 worker 旧回归恢复，bp/百分点精确换算独立负例通过。公开开发诊断现为 3 个发行人、4 份原文、10 条已看案例；原文 hash 均核验，新增平安/宁德时代页面数值和单位经父视觉核对。宁德时代双层表头的人工行坐标与实际解析 table ID 尚需对齐并实际跑通入库→检索→精确评分；不构成独立盲测或跨页通用质量验收 |
| 832f868 真实公文复现 | 父精确归档相关 81 项通过（6.35s），并独立执行已提交 scripts/run_financial_rag_source_diagnostic.py：核对 CATL 原件 hash、229 页原生文本、前 9 页真实 Docling，人工选定原页 9 表格后 1 表 49 格/13448 块实际入库发布；固定题目 as_of，检索重开 13 个支持格，收入 r2c1 位列第 1，精确评分召回 1/1。只证明一条已看开发诊断和明确人工选页/表范围，不证明全文 OCR、自动选表、盲测或外部后端；原 CATL 双层表头定位个案已关闭 |
| aa3a8aa / 3868a6b 持久增量 | 跨页案例保留实际 0/2 精确证据召回失败基线；通用续表候选/授权持久确认仍实施中。父精确归档相关 80 项通过（1.71s），独立在第 1 次已完成评价后注入优化器异常，recording 状态的 1 条动作记录保留且可跨进程重开。这仅证明评价迹线增量落盘，不是 SIGKILL、任务恢复、不重复调用或完整 GEPA 策略包验收 |
| bd48466 人工复核续表 | 父精确归档全量 398 项通过（26.08s）。独立 CPU 对原 CATL hash 和 229 页原生解析核验；有界原页114/115实际 Docling得到2表151格，实际入库发布/检索/重开，两必需定位均在rank2（Recall@3=1，MRR=0.5）。这是明确人工选页/选表并确认的一条已看开发诊断；保留旧0/2失败，不证明自动续表、全文OCR或盲测效果。真实浏览器验证确认行展开、撤销后重新查询行消失、重确认；界面仍未完整呈现双页上下文，研究覆盖仍缺少精确口径证据，已交修复 |
| 8bee3d3 F02 待修复检查点 | 父精确归档 service 与独立worker端到端20项通过（7.20s），但独立负例发现：attempt1过期隔离unknown，经操作员对账重排后，同worker_id领取attempt2，旧attempt1仍可complete写入结果。需每次claim不可复用generation/token，贯穿续租、阶段、完成、失败和控制检查点；不能仅凭worker_id。复现：在隔离任务中令lease_until=0，claim触发unknown，reconcile(expected_attempt=1)，同ID重新claim，再以旧租约complete应拒绝。此检查点未独立验收；暂停/取消、完整恢复与策略评价仍实施中 |
| 71f6eef / ee807be 恢复与策略切片 | 父精确归档全量408项通过（28.26s）。原F02-P1-01独立复现现被拒：同worker重领后旧token对阶段/续租/完成/失败/控制确认均拒绝，job、事件和worker状态不变，新token正常完成。实际已有CATL原文解析库research返回总资产786658123、2024-12-31、千元、合并及114/115双页，明确review_required/numeric_claim_ready=false；未重复OCR/模型或冒充新盲测。F03为受约束七组件策略包，F04仅同题描述性指标/逻辑费用差值，20模块消融与科学效果待证实 |
| 本切片新增缺陷及复现 | F03-P2-01：策略symbol_limit=1时，indexed工具仍接受两个证券并调用research两次，需schema与服务端同限、定义重复代码处理。C03-P1-04：文档期间2024，CSV列2024-03-31收入100（显式千元/合并元数据），实际入库发布research问2024年度营业收入，coverage误标located，观察期间仍2024-03-31；数值主张仍要求复核。应保守限制年末存量日期桥接，流量期间必须明确匹配。两项均交修复，writer自报143项通过仍待稳定提交独立复验；新UI仍待真实浏览器复核 |
| c65877b 两项独立修复核验 | 父精确归档相关105项通过（12.02s）。证券策略上限1时，超限和重复代码均在0次服务调用前拒绝，单个代码实际调用1次，schema同步maxItems1/uniqueItems。原生CSV→SQLite发布→research独立负例确认3月末/6月末收入不覆盖年度；真实已解析CATL年末总资产仍定位到2024-12-31/千元/合并/114115，并保持数值需复核。两项原发现关闭，不代表新基准科学效果 |
| c65877b 浏览器与F04剩余 | 实际IAB/本机HTTP验证coverage已定位、展开p115原行并单独重开p114上下文；改为无匹配证券返回无证据。初始复核启用占位仍误写未启用，卡片口径未提取与coverage合并不一致，已交修复。F04目前20行catalog及单项参数敏感性运行器不等于真实20模块消融：evaluate_candidate未注入industrial_rag_service/release，部分模块名只改变搜索次数/候选数/RRF常数。须实际RAG接线、锁定release和模型索引版本并实现隔离评价模块干预；不能削弱生产安全门禁或宣称因果效果 |
| 4770c47 队列版本绑定切片 | 父精确归档service/独立worker端到端/runtime/governance62项通过（17.87s）。独立验证未变输入通过，配置文件/缓存目录内容变化拒绝，任务锁定package后active改为seed仍按原ID载入candidate-17，错hash拒绝。只证明兼容性失败关闭及固定包选择，不是历史源码、依赖和数据的自包含恢复。目录摘要静默跳过symlink、逐整文件read_bytes内存峰值问题已交保守拒链/流式hash；Windows非提权造链接探针因1314未运行，未绕权限，不冒充动态缺陷复现。原负例应补入仓库回归，缺失工件、按限额物化及跨进程恢复继续实现 |
| c462612 实际干预与失败检查点 | 父精确归档全量418通过、3失败（40.71s）。真实CSV解析/SQLite发布/检索验证baseline1行组、关闭后0行组，禁用RRF返回first_available_channel轨迹且缓存key不同。F04新增身份负例：撤权使permission_revision1→2但pin身份不变，续表复核版本也未绑定；空cache_identity={}的dense组件仍被接受为pinned。不是ACL绕过，但同条件配对可能漂移；须每次评价前后复核revision，漂移失败并继续遵守当前ACL，组件身份需完整版本配置。尚未完成20模块真实评价或科学效果 |
| c462612 回归与UI结果 | /api/research在途撤权测试旧包装不接fusion/rowgroup新增参数，预期403变400；另market_data/mvp两个旧源码字面禁用评分文件名断言与新runtime_binding拒绝/跳过逻辑冲突，未发现实际读取评分数据。应保留真实禁止访问边界、补动态open负例，勿用字符串拼接掩盖旧测试。三项均交修复。实际IAB确认原两个文字缺陷关闭：复核启用占位正确，CATL资产行卡片口径consolidated与coverage一致；数值仍需复核。新运行包历史物化恢复/F04其余模块仍待实现 |
| 5937d26 独立复核修复 | 父精确归档全量423通过（36.83s），原三失败恢复。独立实际SQLite撤权revision和已看CATL真实续表revoke revision均触发配对pin失败，原确认已恢复；空组件身份拒绝。受控伪评分但不调用RAG的负例，runner返回incomplete_no_rag_execution且0 provider调用。新增回归实际跑本地SQLite配对融合差异，但evaluate_candidate被受控替代，属于工程集成，不能宣称完整真实模型Agent或盲测效果 |
| 后续真实证据与物化 | 需当前捕获的真实多发行人原文、真实本地模型与Agent工具/主张链、各可实现模块同题差异和完整复现。可保守用实际captured_at作首次可用并在捕获之后评价，明确current-capture，不得改原retrospective记录伪装2025历史捕获；未来收益/专家盲标/科学效果仍pending。F05源码/五输入/注册表副本和保存CLI恢复仍WIP，未独立验收；实际依赖版本恢复及缺失工件处理也需证明 |
| 仍待完成的工作 | 扩大真实多发行人/跨页与口径测量、全部非 Docker A–G 实现、真实服务与浏览器验收、完整版本恢复和最终服务器交接仍未完成；上述个案修复不能推出工业级整体质量或独立基准效果 |
| 本机全部无服务工作 | 仍在实施，尚未验收，不应立即把当前 HEAD 当最终移交版 |

测试数量是不同提交上的历史证据，不能相加，也不能替代 Linux 新环境的相关回归。最终还需登记：最终本地源码 SHA、远端快照 SHA/tree、Python/依赖锁 hash、模型 revision/指令 hash、语料 manifest、逐项通过/失败/未执行、失败复现及工件路径。供新对话阅读的证据不能仅存在被忽略的 `.tmp`。

## 6. 服务器验收顺序

下表是任务顺序，不宣称所有对应命令已经实现。新对话按当时提交的 CLI 帮助、操作文档和测试入口执行，不能凭空杜撰成功命令。

1. **版本与资源预检**：Git SHA、工作区状态、资源总限、CPU-only、磁盘/端口/代理与依赖下载可用性。没有隔离资源不得开始模型/索引重任务。
2. **锁定环境和普通回归**：先在获准环境内安装/验证 uv，Python 参考本机 3.12，使用仓库锁文件。初稿实际存在的基础命令：

   ```bash
   uv sync --locked --extra dev
   uv run --locked python -m pytest -p no:cacheprovider -q
   uv run --locked raama-rag --help
   uv run --locked raama-rag status
   ```

   这些是已存在入口的 Linux 待运行命令，本文没有在服务器执行。随后只按实际定义安装 postgres/object-storage/OCR/model 等 extras；禁止 `--all-extras` 盲目装 GPU 运行时。不得为绕过失败临时改锁或吞异常。
3. **真实解析与模型**：按 manifest 获取公开资料；验证原生、扫描中文、多页表、单位/期间/负数和源坐标；实际 CPU Embedding、cross-encoder、视觉模型逐个记录 revision、输出、延迟、内存。没有 GPU 不表示不能运行 CPU 模型，但 CPU 结果不能声称 GPU 性能验证，GPU-only 算子须明确待替代/待验证。
4. **真实存储**：PostgreSQL 迁移/CRUD/并发/隔离/outbox 过期 lease；真实 S3 兼容服务内容校验/并发/恢复。先核查独立服务名、数据卷、loopback 端口，不能碰他人资源。
5. **真实索引与检索**：OpenSearch/Qdrant 实际建索引和物理 receipt；多路融合/重排、权限撤销、时间快照、分页/取消/背压，半建索引不可发布。内存适配或参考 BM25 不能冒充真实后端通过。
6. **真实持久流程**：Temporal 工作流与独立 worker；杀本项目 worker、超时/重复/恢复/对账/死信；版本不匹配恢复拒绝和完整回滚。确认无重复业务副作用。
7. **端到端及浏览器**：真实来源→解析→索引→专家/方案经理→证据/计算校验→风险/账本→评价→有界候选→接受或拒绝→回滚；桌面与窄屏、失败与无答案路径，绑定真实 API。
8. **报告**：按 A01–G05 与 20 模块矩阵登记；七层指标、同题消融、样本量、分组、延迟/内存/故障结果。CPU 单机 20 核结果只代表该环境。未来成熟收益、专家盲标和生产多机规模证据仍保持 pending。

共享机器可能不允许 Docker，服务器上有 Linux 并不等于已有容器权限。优先使用管理员提供的隔离服务或获准 rootless 环境；需要安装系统服务、改主机参数或新增权限时先向用户/管理员交接。不得为了 OpenSearch 等依赖要求直接修改共享主机 sysctl。

## 7. 完成判据和 Git 回传

每项记录代码版本、命令、退出码、输入/模型/索引标识、实际输出和正反例。服务不可用保留 blocked 与脱敏原因；skip 不算 pass。测试发现实现问题时修复并提交，别只修改文档把结果写绿。禁止虚构准确率、收益提升或生产可用性。

服务器在独立验证分支上工作。推回前审查只包含本次源码/配置/测试/锁/脱敏文档；不上传运行库、模型或秘密。用户对现有备份分支的授权不自动扩大成对新分支/目标的上传授权。保留完整回滚点，不触及生产 pointer、真实券商或交易。

## 8. 可直接发给远端新 Codex 的启动提示词

```text
你正在接手 RAAMA 的 Linux CPU 真实集成与最终测试。本对话没有之前上下文；以仓库文件和实际运行证据为准，不要假定计划已经实现。

先读适用 AGENTS.md，再读 docs/RAAMA_REMOTE_CPU_HANDOFF_20261025.md，按它的读取顺序读工业 RAG/Harness A-G 计划、proposal、追踪矩阵、操作说明和原金融/风险契约。检查 git HEAD 与交接表，先报告当前已核验/仍待测/仍未实现项目，不重复把历史数量当本机通过。

这是多人共享 CPU 服务器：机器128核，但本项目所有应用、测试、数据库、索引、模型、解析及容器合计最多20核、内存200000000000字节，无GPU。先探测发行版/调度器/cgroup委派/磁盘配额/端口，并核验总硬资源限制。不要根据全机空闲内存或os.cpu_count扩大使用；不要只设OMP线程就当限制已生效。无法隔离总资源先说明阻塞，继续轻量只读工作。

保留金融主线、权限/PIT/风险档案/含费现金/优化器/数量账本及发布门禁；研究不读test.csv，不修改历史experiments/冻结记录。原文/模型/数据库不在Git，请按锁定manifest获取。只绑定loopback并通过SSH隧道看工作台。不要改共享主机参数/重启/停他人任务/清理他人容器，系统权限问题交接。

按交接文档分阶段实际执行：Linux回归、真实OCR/Embedding/reranker、PG/S3、OpenSearch/Qdrant、Temporal与杀worker恢复、真实API和浏览器、七层评价与消融。测试暴露代码问题则修复，不能用桩/静态页面或文档代替功能。按逐需求ID保存可重放脱敏证据；未执行/失败/通过分开；未来收益、专家独立盲标和生产规模效果诚实pending。每个稳定阶段提交并更新交接和追踪矩阵。上传新分支或目标前确认我的授权范围，不上传秘密/账户/原文/权重/运行库。
```

## 9. 核对过的官方说明

- Codex IDE 扩展：https://learn.chatgpt.com/docs/ide
- Codex AGENTS.md 指令发现：https://learn.chatgpt.com/docs/agent-configuration/agents-md
- VS Code Remote–SSH、扩展位置与端口转发：https://code.visualstudio.com/docs/remote/ssh
- systemd/cgroup 资源控制：https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html

服务器 systemd/内核/扩展版本需现场核对；上述资源配置未经目标服务器验证，不能直接认定可执行或已生效。


### 2026-10-26 F05 本地快照恢复增量核验

本地 `4807a318ed10ab410de7c432b4ad6c19254f0538` 对应授权备份 `28cbf86f0eb93b98d6e421d135393d738c572a90`，tree `97ce5501ef2d19c1c5091c80e5b84493de6d55bd`；8个改动blob/ref匹配，未上传本地工件。相关归档检查48通过、1失败（17.25秒）：Windows较深根路径下长staging目录使创建源码副本失败，POST任务返回422，登记F05-P2-03待修。相同提交短目录下端到端1通过（6.84秒）。独立检查实际排队并停止服务，移走原始config后由新store、新worker及保存CLI完成合成决策；另进程确认加载保存源码；篡改保存config被拒绝。此为合成离线恢复契约，不证明真实模型效果或安装依赖已物化；当前依赖仅验证现有Python/发行包版本身份。后续e285160恢复证据与短staging修复仍需独立审查，不能把此检查点计A–G总验收。最终迁移快照和无上下文克隆自检仍待完成。


### 2026-10-26 F05 路径修复复核

本地 `a5d5dd5dba8f34abf817e27de86cd21410d443ab`（含e285160恢复测试）对应备份 `dc17128393938976ec398e66b5196dea93a0664e`、tree `d71c8831fdd0d9868c8e53c0e32629c054c4b647`；5改动blob/ref匹配，无.tmp。父精确归档4项针对性检查通过（15.17秒），包含原深目录失败的实际服务/worker/CLI端到端流程及新的真实POST路径边界回归，F05-P2-03关闭。新store/原seed缺失/副本篡改及受控Python环境身份差异拒绝通过；此增量未重复无改动全量suite。依赖仅记录/验证现有安装环境身份，尚不是历史依赖物化恢复。真实current-capture多发行人模型/Agent评价及其余A–G和无上下文克隆自检继续待完成。


### 2026-10-26 F04/F03 current-capture 检查结果

新增脚本 `scripts/run_financial_rag_current_capture.py` 在本地 SQLite 发布 `current-capture-20261026T112508Z`，使用三个官方 CNINFO 年报：招商银行 `cninfo-1222895153`（物理页3）、平安银行 `cninfo-1222806505`（物理页16）、宁德时代 `cninfo-1222806982`（物理页9）。全篇文本由 pypdf 提取，每份仅对人工选定并复核的收入表页运行 Docling；源文件 SHA-256、解析器版本、原始数值与证据 ID 见 `docs/RAAMA_FINANCIAL_RAG_CURRENT_CAPTURE_20261026.json`。以记录的本地捕获时间作为 `first_available_at`，按 strict-forward 查询，as-of 为 `2026-10-26T12:03:13.899307+00:00`，未用历史发布时间代理。

招商银行 2024/2023 年收入为 337,488/339,123 百万元人民币，表格标注本集团；平安银行为 146,695/164,699 百万元人民币，但目标表未注明会计口径；宁德时代为 362,012,554/400,917,045 千元，目标表未注明币种，因此没有计算其增长率。真实金融研究处理器重开检索到的来源证据，确定性 Decimal 计算器及报告校验器记录招商银行增长率 -0.48212595429976734105324616731982201148255942534125%、平安银行 -10.931456778729682633167171628243037298344252241969%；校验状态为 `calculation_recomputed_decimal_50_half_even_semantics_unverified`。宁德时代两项数值通过来源值校验，状态为 `source_observations_validated_semantics_unverified`。该工具协议由确定性脚本驱动，没有调用语言模型或 provider（`provider_calls=0`）。同时修复工业 RAG 来源引用缺少 `source_record_id` 导致共享校验器取值失败的问题。

独立补充检索采用 structured-facts 与 BM25，Qwen、dense retrieval 和 reranker 均关闭；包括6次基线期间查询及60次模块处理器查询（10个干预 × 3家公司 × 2个年度），每个处理器最多进行2次研究搜索。每家公司记录了20个模块状态，实际运行干预模块2、3、7、8、9、11、13、14、15、16。模块3仅招商银行/宁德时代、模块9和11三家公司、模块13招商银行/宁德时代、模块14三家公司观察到证据顺序变化；模块14的平安银行状态为 `candidate_only`，另两家为 `available`。模块2、7、8、15、16在查看的查询中未改变证据顺序。模块1、4–6、10、12、17–20未做消融；本基线未启用 dense/reranker，因此关闭它们会是空操作，其余模块保留重建或安全/校验行为。这些是单次查询观察，不是因果消融、独立验收或质量评分。

此前另有一次本地 Qwen CPU 检索尝试，载入了固定版本的 embedding/reranker，但脚本驱动未取得两年度来源事实，流程在写入检索轨迹前停止；没有留存准确候选项、排序或配对 Qwen 结果。它与本次关闭 Qwen 的补充检索分开记录，不得把补充结果描述为 Qwen 成果。本轮为三个开发样例和人工选页检查，不是盲测或独立基准；完整报告未全部运行 Docling。A–G其余验收及无上下文克隆自检仍未完成，数值校验也不代表语义蕴含、投资质量或科学效果。


### 2026-10-26 current-capture 父独立复核与未闭缺陷

本地 `9a22f781494a24ef37d4c9e602889e312a2b772d` 对应注明缺陷的备份 `9790ad9e45c344996172cb1e01a7be8ee93cfece`、tree `3fada973921ac62ba9ff46b032ab48e1bce3c511`，5个改动blob/ref一致，无.tmp。父精确归档相关测试6通过（1.98秒）；在父独立SQLite副本复用已发布证据，经实际基本面处理器、证据重开、Decimal和报告校验器复现招商银行已知本集团口径计算，source_record_id与citation.source_id对应。

仍有C05-P1-01：平安银行明确口径未注明却以not_stated_in_target_table字符串当有效口径，允许增长率计算并通过数值校验，违反未知口径隔离要求，待修。G-P1-01：恢复只按case_id复用旧进度；父在不做模型推理的受控运行时下，用真实已完成BM25进度执行默认Qwen恢复，全部queries保持旧结果，但顶层标为Qwen启用、channels含dense；必须绑定并核验运行配置身份、拒绝跨模式混用。上述两问题不视为验收通过。

本轮成功记录只证明三份已看开发原件、单页人工事实和确定性脚本驱动的真实工具协议；补充检索为BM25/结构化，先前Qwen失败缺精确轨迹，不是成功模型对照。20模块状态和单查询干预不是完整配对消融或科学效果。完整LLM多专家/方案经理Agent、真实固定模型检索、全部A–G及Git无上下文克隆自检仍待完成。


### 2026-10-26 口径与恢复门禁增量审核

本地 `b1e0977838bf23d388fab1ea31416ea3cd679777`（含9ff9109）对应备份 `630a187cddd3f86d748962a627dab33cfbf23d7a`、tree `a6e19e40b7409b0232a004378945529d8f14af69`；10改动blob/ref一致，无.tmp。父精确归档全量442通过（40.62秒）。父真实SQLite副本证实招商银行已知口径完整计算链保留，旧平安银行未知口径的事实numeric_eligible均为false且计算被拒；旧记录的脚本来源观察未取齐事实而失败关闭，未计通过。旧错误结果另有C05 correction记录，不抹去历史。模式/策略/代码/来源等恢复身份回归通过。

仍有G-P1-02：父实际撤销招商银行访问权，permission_revision从3到4，capture run identity未变，已完成工具链进度经identity_check后仍直接复用旧结果；续表复核revision同样未纳入身份。此为capture恢复边界，非普通search绕权结论，待加入现行principal/权限及续表revision的前后校验，不能冻结旧授权。后续106b328时序修正及该权限修复尚待稳定归档独立复核。上述备份仍非完整A–G验收。


### 2026-10-26 capture恢复现行权限复核

本地 `68002cff5069f59c9ffe16f994056e08273e2a9e`（含106b328、0bbc38e）对应备份 `480d34d0b977a73ec2a0819f0fd0830b55ba1a9e`、tree `1ea4a468b3326ba9aa55174933cc58a09d20246f`；4改动blob/ref一致，无.tmp。父精确归档7项针对性检查通过（2.33秒）。在父真实SQLite副本中，权限revision相同时已完成CMB工具链可复用；实际set_access撤权后旧completed结果被RunIdentityError(access_scope)拒绝，G-P1-02独立关闭。受控修改父DB续表revision也被拒绝，此项仅验证版本漂移，不冒充真实人工复核动作。未重复未变442项全量与模型推理。106b328使decision_time在release之后设置as_of后生成，已审源码，新的完整模型运行仍待结果。其余真实模型、多专家LLM Agent、完整模块比较及A–G和干净Git克隆自检继续pending。


### Parent checkpoint 1a96efc (2026-10-26)

- Exact local source `1a96efce71bac8e77f999e9dc2e8a1df19a80377` maps to backup `9d2ec3b2506c6ecf32bc34f09481a210e1f85e59`, tree `711c7adfd9bbf2322bd5d5536a732e1011a52ee8`; nine changed blobs match, no `.tmp` paths. Prior parent handoff draft is included in that snapshot.
- Parent exact archive suite: **445 passed, 2 failed, 49.92s**. Failures are the strict policy options assertion and live API in-flight revocation shim receiving the added `trace_callback` parameter. Preserve the actual revoke-denial assertion while repairing compatibility.
- Open **G-P2-03**: after deleting complete trailing journal events and retaining a valid header, completed case progress still returns its cached result although it references missing sequence 1. Verify all progress event sequence/hash references against the journal before any completed-result reuse, including overall case and module progress. This is a controlled integrity test, not a real model result.
- Open **D-P2-04**: `cache_capacity=1` with two persistent hits leaves two in-memory vectors. Bound persistent-hit insertions with the same LRU policy as newly embedded vectors.
- Controlled positive/negative evidence: identical runtime/source reuses persisted vectors; coherent-checksum NaN and wrong-dimension payloads recompute; real synthetic SQLite retrieval emits structured candidates before a controlled dense-channel failure and preserves its failure event. No new Qwen inference, full LLM Agent, blind study, or complete 20-module acceptance is implied.
- Fresh clone reproduction after installing the documented dependencies: `python -m pytest -p no:cacheprovider -q`; relevant committed suites are `tests/test_qwen_models.py`, `tests/test_financial_rag_current_capture.py`, `tests/test_harness_evolution.py`, and `tests/test_industrial_rag_api_e2e.py`. Private parent probes are diagnostic only and are not required by the handoff. A-G remains incomplete.


### Parent checkpoint 8261e26 (2026-10-26)

- Local `8261e262db925a36130cf5f1a67c554849dbda5e` maps to backup `161373a246201ddbcad21c0c1bca458c36139a7f`, tree `a89c174636e021cd92cf10e9f83356c3c42cdf67`; eight changed blobs match, no `.tmp` paths.
- Parent exact archive affected suites: **123 passed in 22.95s**. Both prior optional-trace compatibility failures are closed; actual live API in-flight revocation still returns 403. Full unchanged suites were not rerun.
- **G-P2-03 closed for the reproduced boundaries**: intact journal cached-case reuse succeeds; deleting complete event suffixes makes cached case and actual module progress reject missing sequence/hash references. Overall case progress verifies nonempty references before reuse. **D-P2-04 closed**: two persistent hits with memory capacity one retain one vector. Controlled fake embedding/progress probes are engineering evidence, not real model or scientific results.
- Single-case and module-matrix scope are bound to run identity; changed scope rejects before model construction. Relevant committed regression suites: `tests/test_financial_rag_current_capture.py`, `tests/test_qwen_models.py`, `tests/test_harness_evolution.py`, `tests/test_industrial_rag_api_e2e.py`, `tests/test_industrial_rag.py`.
- New actual local model diagnostic, real online generation-driven financial Agent, complete paired module evaluation, external services and final clean-clone handoff self-check remain pending. Preserve all prior failures and the distinction between scripted tools and generation-driven decisions.


### Reboot recovery and actual local model evidence (2026-10-26)

- After the user reboot, a read-only Docker engine check succeeded (server 29.9.2); the initial restricted process could not read Docker config/connect to the pipe. No agreement was accepted and no service was started by the parent. Actual PostgreSQL/S3/OpenSearch/Qdrant/Temporal integration remains to be exercised.
- Viewed CMB current-capture run `current-capture-qwen-cmb-traced-20261026`: the pinned local Qwen embedding/reranker baseline completed two-year retrieval, two original source reads on page 3, Decimal growth calculation `-0.48212595429976734105324616731982201148255942534125` percent, and mathematical report validation (semantic entailment remains unverified). Driver is explicitly `offline-scripted-tool-protocol-v1`, with generation model called=false. The reranker-disabled paired attempt failed because both annual facts were not returned; do not present that pair as a completed comparison.
- Parent verified 365 journal events and all 337 baseline/28 failed-pair references and progress hashes. Journal SHA256 `1e82b33c129bbd6d3897969fe4b0e1c35e92da0ba6bcaa3dc5f62199917f8512`. All eight recorded source hashes match exact original run bytes; normalized text matches Git checkpoint 8261e26, but mixed LF/CRLF patterns differ from Git archive bytes. Preserve the old identity; a new Linux run has a new byte identity and must not silently reuse an incompatible historical progress record. Exact raw source copies and full private diagnostic journal remain local evidence, not Git inputs.
- Continue with generation-driven financial Agent and isolated real-service checks. The restart continuation is ordinary work and consumes none of the two additional authorized future Plus five-hour recoveries.


### Parent checkpoint 4cf50bc (2026-10-27 Asia/Shanghai)

- Source `4cf50bc50e242221caf7dfa459bf68b55565c512` maps to backup `bb26dc5d4e84280d4a0f324589448b923809566a`, tree `4adf89aa43650f84be684dbf1d3e3af268c7f561`; seven changed blobs match, no `.tmp` paths. This is a known-finding checkpoint.
- Parent exact archive provider/live-agent/state/evolution suites: 64 passed in 2.73s; unified workflow/runtime: 27 passed in 1.71s. No actual provider call was made by these tests. The committed sanitized local-Qwen summary matches independently retained run/runtime/journal/progress hashes and success/failure labels.
- Open **G-P2-04**: the live runner accepts edited top-level baseline source hash, principal, and security while retaining the old case hash; the registered case still names CMB. Validate the complete persisted source/case/runtime/run identity and current release/access revisions before live requests; this is a provenance-integrity negative, not a demonstrated service search authorization bypass. Keep the intact positive.
- Actual online generation-driven Agent and physical backend integration remain pending; scripted tools, offline fallback tests, and summaries do not establish that acceptance. Preserve raw request/response artifacts locally and upload only reviewed sanitized summaries.


### Parent checkpoint 4452c76 (2026-10-27 Asia/Shanghai)

- Source `4452c76c6008b6f8b641ba1546da99a0d18fb2fb` maps to backup `e554026852508e7bc7cdc1b4f064a4658be9e08c`, tree `0ede050ba82ab5e468c8a9a70d259b5d3148aaa0`; four changed blobs match and no `.tmp` paths.
- Parent exact archive live/capture suites: **23 passed in 2.19s**. In a copy of actual CMB SQLite/CAS/release and historical completed progress, the intact positive verifies; changes to source hash, principal or security and removal of the case hash reject; actual pre-start `set_access` revocation also rejects. **G-P2-04 closed for these reproduced boundaries**. No model inference or provider request was performed by the parent.
- Historical manifest and code hashes bind exact original newline bytes; preserve old identity. The three literal source/run/runtime/case pins in this diagnostic runner support only the existing viewed CMB run. Portable new-run bundle inputs, new issuers, clean-clone Linux reproduction, actual generation-driven multi-expert workflow and real physical services remain pending; do not claim those from this fixed diagnostic.


### Parent checkpoint c533dcd (2026-10-27 Asia/Shanghai)

- Source `c533dcdae0976fc974d953a51658e8c20facfda4` maps to backup `b2bcec8b1d763a8d7e0478cb0e0d8483fd91928e`, tree `e538bbd4580bd93c37b4c0c1e3537ed197e70e7e`; two documentation blobs match, no `.tmp` paths. No business code changed and no previously passed tests or model inference were repeated.
- Parent independently matched all five local image IDs/registry digests to the committed report and all four compose/initial/corrected/result artifact hashes. Earlier loopback container observation and later absent containers match the reported cleanup; the ephemeral credentials file is absent. This reviews the physical smoke record, not a repeated independent application run.
- Keep the report distinction: real PostgreSQL/S3 application smoke, direct OpenSearch/Qdrant API checks, and Temporal health only. Full PostgreSQL failure recovery, release-bound external index adapters and durable Temporal application recovery remain pending. No aggregate or container resource caps were declared/proven; this does not establish the shared Linux project limit.
- The smoke commands still reference ignored probe/config artifacts. Before final handoff, commit sanitized reproducible scripts/templates/synthetic inputs using operator-provided ephemeral credentials; do not upload secrets or rely on the private `.tmp` directory. Online data transmission awaits the specific human authorization already requested; all other engineering continues.


### 7097642 parent independent checkpoint (2026-10-27)
- Local `70976429f227a4c96e9a4846b6d16a8eb52223ab` maps to backup `7da428b0c30012db5f181769d945a5d66ea65c3b`, tree `67e23a236ca0d903ca917d25a00dc5f961609945`; expected previous `b2bcec8b1d763a8d7e0478cb0e0d8483fd91928e`, nonforce, 13 changed blobs and ref verified, 0 .tmp, main unchanged. Known-finding checkpoint, not A-G acceptance.
- Exact archive 26 related tests passed in 2.58s. Parent copied actual CMB SQLite/CAS/progress: relative versioned bundle relocation accepted; altered bundle bytes and raw manifest drift rejected before inference/provider. Probe archive `7097642-7_mk085s`.
- D-P2-05 pending: fresh per-output dense cache abandons verified source vectors. Parent controlled embedding over actual copied 901 blocks: fresh cache 901 document embeddings +1 query; verified existing cache 0 document embeddings +1 query. This is engineering control, not new real inference. Preserve strict source/block/model/instruction/dimension identity when reusing cache. Do not rerun old 901-block inference.
- Bundle export currently depends on a later paired path whose historical reranker-disabled attempt failed. Preserve the failed label and permit independently verified baseline export without rerunning or relabelling that pair.
- Versioned backend smoke inputs, synthetic fixture and schema now in Git. Recorded result SHA256 `3d91ee9cf82720d95dde0c189a837f61332e0ef23d6c4428f8bdb1533d6b1491` verified and private.env absent. Four documented input hashes match archive bytes with explicit CRLF-to-LF representation; old captured identity is not rewritten. PG/S3 adapter smoke and PG outage/restart durable recovery only; OS/Qdrant direct APIs and Temporal health are not application adapters/workflow acceptance. Later transaction rollback/outbox/release-negative WIP is not included in this checkpoint review.
- Full multi-expert generation, complete module evaluation, final clean-clone handoff and shared Linux aggregate 20 CPU / 200000000000-byte enforcement remain pending. External Qwen data-transmission approval remains unanswered; no provider request authorized by this checkpoint. Quota 7/8, new grant used1/2 remaining1; current ordinary active work does not consume recovery.


### 8ea15c8 bounded PostgreSQL fault probe review (2026-10-27)
- Local `8ea15c856346c3a6ef57428bd7ab002fe999e558` maps to backup `11210fa752eb1425497b5ce86d0bf2b0f2f93ec7`, tree `a4b68bc96193b4b7e6c95cd1cd8617ed7d26b26d`; expected previous `7da428b0c30012db5f181769d945a5d66ea65c3b`, nonforce, 3 changed blobs/ref verified, 0 .tmp, main unchanged.
- Exact committed probe and saved physical result reviewed in `parent_reviews/8ea15c8-oo7vr69l/parent_backend_boundary_review.json`. Result SHA256 `56dddfb80d94868c17b4a2c4cf120811dedc157c9c321493f08e4c92d3ccd8ad`; four source input hashes match explicit archive CRLF-to-LF representation; private.env absent. No unchanged service/model/test rerun by parent.
- Bounded actual PG/S3 evidence covers late invalid-principal transaction abort, failed outbox handler with no receipt then lease-expiry retry, missing-backend publish rejection, and durable state after PG outage/reconnect. Scope is narrower than some labels: rollback explicitly counts document/version/outbox, not child tables; no second consume is not an explicit duplicate-ack call; failed publication starts with an empty active pointer, not an established active release. Strengthen checks or narrow labels in the next probe revision.
- Full PG contract suite, application external-index receipts/Temporal workflow, D-P2-05 cache reuse, full multi-expert generation, full module evaluation, clean-clone handoff and shared Linux aggregate resource enforcement remain pending. External provider transfer approval still awaits the human answer. Quota remains 7/8, new grant used1/2 remaining1; ordinary active work does not consume recovery.


### e8ffc06 cache and PostgreSQL follow-up review (2026-10-27)
- Local `e8ffc06d28ff5edbedeaa66d5a5ef2a8c4617fa9` maps to backup `529a3ac034111a80250bcd7887eede4092848754`, tree `f4a501a0044d28662dd39921f4ac380ab8b30a1b`; expected previous `11210fa752eb1425497b5ce86d0bf2b0f2f93ec7`, nonforce, 9 changed blobs/ref verified, 0 .tmp, main unchanged.
- Exact archive 34 capture/live/Qwen tests passed in 4.11s. D-P2-05 original boundary independently closed: actual copied CMB 901 source rows and persisted real vectors through `_load_service` yield 0 document embeddings and 1 controlled query; missing one vector and empty cache reject (1/901 missing) without document inference. Probe: `e8ffc06-yhwwcl_f/parent_cache_fixed_result.json`. Controlled embedding only, no new real model/provider request.
- Baseline bundle export now precedes paired reranker diagnostic by source review, and live preflight permits absence of paired results. Full fresh-capture export runtime remains pending; historical failed pair is retained.
- Strengthened bounded PG evidence matches saved result SHA256 `6aee849130a051d256c9efd778fc8689b013d593c84008f62d556b94c229c563` and archived probe explicit LF byte SHA256 `27aa574f45966975727a48e5ffa1bba73c4cbe4c2a19c24bfa1500b60540f14c`; original private env absent. Six document/version/span/block/fact/outbox counts are zero after rollback; actual duplicate `finish_outbox` returns False without changing receipt; later failed publication preserves a nonempty active release. Scope gaps from 8ea15c8 addressed in this synthetic probe, full PG contract/HA still pending.
- OpenSearch application adapter and service run are subsequent WIP, not accepted here. Qdrant direct checks and Temporal health remain below application/workflow acceptance. Full A-G, actual generation, complete module evaluation, final clean clone and shared Linux aggregate resource enforcement remain pending. External provider transfer consent unanswered. Quota remains 7/8, latest grant used1/2 remaining1; ordinary work does not consume recovery.


### 9491dd7 OpenSearch application slice review (2026-10-27)
- Local `9491dd73036c1bc09c8ae554ec2cca072f957a5b` maps to backup `63ae9449ded3cd0ac76ef6e7f9399268b49adf0b`, tree `c7f19aea04bfc31fe1a8182e574f83a0b07b91ab`; expected previous `529a3ac034111a80250bcd7887eede4092848754`, nonforce, 6 changed blobs/ref verified, 0 .tmp, main unchanged. Known D-P1-04 checkpoint, not A-G acceptance.
- Exact archive RAG/API/planning 66 tests passed in 10.69s. Saved actual PG/S3/OpenSearch text-only integration result SHA256 `de5bff6f143bc1abf49ec499a1729f5b1e245df0fac09ee5469f8b774a398fc4`, probe LF byte SHA256 `6b2b01de3ede999a59d980608e39e5490bd23f82bfdecc8fd15b86733286e53c`, adapter LF SHA256 `5dc4ed68ca9fb157a2b6ce4ef39dd0a4003e0fd7c1b036ef8cd1998bf8543393` all match; private env absent. Text-only release receipt/publication/service BM25 positive and pre-search nonactive/revoked/PIT negatives recorded. No service/model repeat by parent.
- New D-P1-04: adapter build receipt always sets embedding_dimension=None but verifier returns release spec dimension. Parent actual SQLite release/build/verify with controlled REST responses accepts None text-only release and rejects Qwen1024 hybrid release at mark_backend_ready with ImmutableConflict. Probe `9491dd7-auvjo76n/parent_opensearch_dimension_result.json`; no network/model/provider. Writer fixing; physical-record/source validation must be preserved.
- Full PG contract/HA, hybrid OpenSearch readiness, Qdrant application adapter/Temporal workflow, full multi-expert generation/module evaluation, clean-clone final handoff and aggregate Linux resource enforcement remain pending. External provider transfer approval unanswered; quota7/8 newgrant1/2 remaining1. High usage with ordinaryAllowed is not a stop or recovery trigger.


### OpenSearch hybrid receipt and Qdrant application follow-up (2026-10-27)
- OpenSearch build and verification now both carry the immutable release embedding dimension. A local SQLite release configured with Qwen3 dimension 1024 plus controlled synthetic REST responses successfully produced and recorded the receipt; no Qwen inference ran. Focused regression: `tests/test_industrial_rag_opensearch_adapter.py`.
- Added a release-bound Qdrant vector adapter. It uses verified cached source vectors, binds model/revision/dimension and point/vector hashes into the release manifest and collection metadata, verifies persisted records, and routes `dense` through the shared retrieval service. ACL, release, security-code, and point-in-time checks happen before the Qdrant query.
- After narrowing vector-cache checks to indexed block keys, isolated Qdrant v1.19.1 application probe `raama-qdrant-smoke-95f37dd5`, run `ad09c3e8d680`, passed synthetic release receipt/publication and dense source retrieval; nonactive release, revoked principal and pre-availability negatives issued zero Qdrant search requests. Sanitized result SHA256 `4b00775fa68e429d0443b53304c7d8bfd20ceb19b400bf7bc018d82bbfb70041`; probe SHA256 `bd42b5e62bf525775a701cf35432d0759b7e60cd86391a44db243b3bc4457db7`; adapter SHA256 `2dac544b9044ef266cf3d5799763d2e509b0db40a59565fb9ac015b2400a1ce6`. All isolated Qdrant resources and generated private env were removed.
- Final combined synthetic PostgreSQL/RustFS/OpenSearch 3.9.0/Qdrant 1.19.1/Temporal run `raama-smoke-hybrid-81cb79ba`, run `2ea65d6e95ee`, passed release receipt publication, BM25 and dense service retrieval, transaction/outbox/release checks, PostgreSQL outage recovery, and Temporal health. Result artifact `.tmp/industrial_rag_backend_integration_20261027/application-hybrid-raama-smoke-hybrid-81cb79ba.json` SHA256 `3b4c2f7b5c4e3b5de51a28429db76ec24422bd0fd169d56d2f1d8585f29c2dfb`; final probe SHA256 `0b08595dfa3bdab88ab25193c10a188b412e09d4e237b9fa98908c8e99287504`. All containers, three volumes, network and private env were removed. The release used synthetic 2D vectors; no source corpus, model inference, or provider request was used.
- Focused RAG/API/planning/Qwen/capture/live-agent suite: 102 passed in 13.14s, including selected cached-block coverage and OpenSearch 1024-dimension receipt readiness. Changed Python files compile. Full PostgreSQL contract suite, Temporal application workflow, 1024-dimension physical Qdrant run, and remaining A-G acceptance remain pending. Parent's `9491dd7` review notes above are preserved verbatim.


### Authorized recovery 8 (2026-10-27)
Parent verified turn `01a17923-738d-7273-b5a8-17beecc505f4` ended with an actual Plus usage-limit error (retry local 05:13). Natural restoration was then verified: ordinary usage allowed, primary 2%, weekly 32%, primary reset epoch 1793067558. The last directly authorized additional recovery was atomically reserved as `RAAMA-INDUSTRIAL-RAG-ADDITIONAL-PLUS-RESUME-20261027-8` and sent once to the sole writer. New turn `01a17a16-1ede-7fc0-b043-901a3542aa04` confirmed active, cursor 970ce7de:31. Cumulative 8/8, latest two-recovery grant used2/2 remaining0; historical grants and ordinary reboot continuation retained. No reset card, purchase, account or billing change. Next new confirmed usage-limit failure requires stopping and pausing automation for the human. Normal stage continuation is still authorized. Stable reviewed source/backup remains 9491dd7/63ae9449; OpenSearch dimension fix and Qdrant/combined smoke are interrupted WIP awaiting stable review. External generation transfer consent remains unanswered.


### 8cf6d55 Qdrant application slice review (2026-10-27)
- Local `8cf6d55bd0a9ac6d29c103b8c4abbd2e4f7bf7d7` maps to backup `5eaaa1ea07c5f9d2bf6d2c0a3673088777b619c7`, tree `c3cfe520d9588d44fdb028ee747421bbb145345e`; expected previous `63ae9449ded3cd0ac76ef6e7f9399268b49adf0b`, nonforce, 8 changed blobs/ref verified, 0 .tmp, main unchanged. Known D-P1-05 checkpoint, not A-G acceptance.
- Exact archive OpenSearch/Qwen/RAG/API/planning 75 tests passed in 12.33s. D-P1-04 independently closed using original actual SQLite/controlled REST probe: None and hybrid Qwen1024 receipt/verifier dimensions agree and both mark ready. No model/network/provider.
- New D-P1-05: parent actual SQLite immutable release plus controlled REST and synthetic cached2D vectors fully build/markready/query positive. After changing only returned point vector [1,0] to [0,1] and coherently updating vector_sha256/record_sha256, query still returns1 hit while SQLite vector manifest and physical collection metadata/count remain unchanged. Query compares vector hash to the returned payload itself instead of the authoritative cached/frozen point manifest. Probe `8cf6d55-vz5ek51_/parent_qdrant_tamper_result.json`; physical integrity boundary, not an ACL bypass claim. Fix requested with positive and coherent-tamper negative.
- Actual combined service record SHA256 `3b4c2f7b5c4e3b5de51a28429db76ec24422bd0fd169d56d2f1d8585f29c2dfb` and probe explicit LF bytes `0b08595dfa3bdab88ab25193c10a188b412e09d4e237b9fa98908c8e99287504` verified, private env absent. Service vectors/source are synthetic2D; BM25/dense receipt/publication paths are exercised, Temporal is still health-only. Few existing real1024 cached source blocks must complete a bounded physical roundtrip without document re-inference; full PG contract/HA, Temporal application workflow, full Agent/module evaluation and final clone/resource handoff remain pending.
- Authorized recovery8 remains active `01a17a16-1ede-7fc0-b043-901a3542aa04`, cumulative8/8/latestgrant2/2 remaining0. Next new confirmed usage-limit failure must stop/pause; normal active work continues. External provider transfer consent unanswered.


### D-P1-05 integrity repair, bounded Qwen vector roundtrip, and Temporal ingest recovery (2026-10-27)
- Qdrant dense search now rebuilds the expected point hashes from verified persistent local vectors and requires the resulting point-manifest digest to match the immutable release. Each returned vector and payload is compared with that local expected point record. The full synthetic app probe first retrieved valid BM25+dense evidence, then changed a Qdrant vector from `[1,0]` to `[0,1]` and recomputed both payload hashes; the normal application query rejected it. This closes the coherent-tamper finding without changing ACL conclusions.
- A separate physical Qdrant v1.19.1 probe used three exact source blocks and 1024-dimensional vectors from the existing verified local Qwen cache. It wrote, scrolled, numerically compared and queried the vectors; maximum absolute round-trip error was 0, cosine similarity cleared 0.999999, no document inference or provider request occurred, and the ephemeral collection was deleted. Result `.tmp/industrial_rag_backend_integration_20261027/qdrant-cached-1024-20261027.json`, SHA256 `ed0c5d46af0aa5a44239d9ad056f11e6f254a4c97e23f0c8a9c204d7e80e20a3`; probe script SHA256 `83c379144d5afda0334aabee80d2d8e927841adc6e35b3ade813250296c830e1`. This is a bounded physical storage/query roundtrip, not a full 1024-dimension release receipt.
- Added optional pinned `temporalio==1.34.0`, `industrial-rag-ingest-v1`, worker/submit/query entry points, and durable task/artifact checkpoints. The workflow freezes capture/ingest timestamps before scheduling the Activity. A live Temporal server probe killed worker 1 after the ingest side effect and before Activity acknowledgement; worker 2 retried, reused the immutable result, and completed with exactly one persisted completion event and zero provider calls.
- Final isolated combined service run `raama-smoke-temporal-7d117c54cb`, run `394f87258015`: PostgreSQL/RustFS transaction, outbox, release publication and outage-reconnect checks passed; OpenSearch 3.9 and Qdrant 1.19.1 release receipts and BM25+dense retrieval passed; coherent Qdrant tamper was rejected; Temporal UI/RPC health and app-workflow recovery passed. Result `.tmp/industrial_rag_backend_integration_20261027/application-temporal-qdrant-tamper-20261027.json`, SHA256 `2eee618487b41db034cd62c0fabd28cd2c486b62a06297908f63521ba9de8270`; probe script SHA256 `12884ed90a10964c412c39d34fe299ee4e199ab3ecf5ac5131254830a5d4b576`; Qdrant adapter SHA256 `c9012b907ab70c466863e4fdd71569c8662272a85dd348f2e65dda6b8e422384`. Containers, three named volumes, network, private credentials, debug state and copied cache/database were removed.
- Focused RAG/API/planning/Qwen/capture/live-agent suite: 102 passed in 13.11s. `py_compile`, `git diff --check`, and offline `uv lock --check` passed. Existing 901-vector cache contents were not re-inferred or modified. Frozen manifests, `experiments/**`, and `test.csv` were not read or changed.
- Scope remains deliberately partial: the live Temporal workflow currently covers local ingest only; index/research/evaluate workflows and pause/cancel/resume/dead-letter contracts remain open. The physical Qwen 1024-vector smoke does not establish a full 1024-dimension release receipt. Full PostgreSQL contract/HA, other A-G evaluation and the independent review sequence remain pending. No external provider request was made; its approval boundary remains unchanged.

### da63618 Temporal and vector follow-up review (2026-10-27)
- Local `da636184bc1717442037fcd58bef6bf962c50504` maps to backup `45a803acf7367157a76b305651557a06565cf8a3`, tree `015b41bb25ba0e8ae7cf13f6ef34db5822c0a3d0`; expected previous `5eaaa1ea07c5f9d2bf6d2c0a3673088777b619c7`, nonforce, 10 changed blobs/ref verified, 0 .tmp, main unchanged. Known F06-P1-01 checkpoint, not A-G acceptance.
- Exact archive 102 related tests passed in 13.50s. Original D-P1-05 coherent vector/payload mutation now rejected; intact query positive1hit. Controlled transport/synthetic2D only; no parent service or model repeat.
- New F06-P1-01: request freezes source/metadata paths, not original bytes. Parent actual activity -> CLI -> isolated text parser -> SQLite accepts source modified after request creation (original SHA256 `fba56e2ba4c0091bb27de98690f570fa752cf88fffd8a271a93e0dbc0a4be55b`, changed/ingested SHA256 `7234f40e356b0667787d267b7aa3a98f1db8671f3bee996f1084dd3f7146459a`) while captured_at/ingested_at both remain the earlier scheduled timestamp. Probe `da63618-mazqpi4u/parent_temporal_input_result.json`; heartbeat context controlled, no Temporal network/provider. Byte/hash anchored immutable source+metadata and honest first durable ingest timestamp required, preserving lost-ack replay identity.
- Saved actual combined result SHA256 `2eee618487b41db034cd62c0fabd28cd2c486b62a06297908f63521ba9de8270`: synthetic2D service query/coherent-tamper rejection and Temporal local SQLite ingest worker crash after persistence/before ack, second attempt reuses artifact,4events,0provider. Separate actual three existing real Qwen1024 vector write/scroll/query record SHA256 `ed0c5d46af0aa5a44239d9ad056f11e6f254a4c97e23f0c8a9c204d7e80e20a3`, max roundtrip error0,0document inference,0provider. This direct bounded1024 collection probe does not prove full release-bound application retrieval at1024; Temporal local-ingest/lost-ack does not establish all workflow lifecycle operations. Reviewed source LF hashes saved separately; historical identities are not rewritten.
- Full PG contract/HA, remaining Temporal operations, complete multi-expert Agent/module evaluation, fresh capture export and final clean clone/aggregate Linux resource verification remain pending. New PG continuation probes and Temporal input fixes are WIP. Quota8/8/latestgrant2/2 remaining0, active normal work continues; external provider approval unanswered.

### PostgreSQL continuation and Temporal capture follow-up (2026-10-27)
- Fixed the PostgreSQL continuation-review query by translating its SQLite `json_extract` expression to PostgreSQL JSONB path extraction. The isolated PostgreSQL application smoke confirmed and revoked a synthetic continuation candidate and observed two revision increments.
- Temporal submit now captures source and metadata bytes before scheduling, carries both SHA-256 values and the capture timestamp in the workflow input, verifies the exact captured bytes before parsing, and persists `ingested_at` after parsing as a retry-stable task artifact. The live recovery probe changed both original paths after capture, verified the ingest retained the original source and metadata, rejected tampered captured source and metadata, and recovered after a worker crash before acknowledgement.
- Final combined synthetic run: project `raama-audit-1027-03`, run `21be972ffc44`, result `.tmp/industrial_rag_backend_integration_20261027/application-temporal-capture-postgres-03-20261027.json`, SHA-256 `373eafe7342cf6e970bbd0a3c0918ec0efe658eb7df449b53a19ed7f21dc85b7`. PostgreSQL ingest rollback, outbox retry/duplicate acknowledgement, publication atomicity, outage/reconnect, continuation review, OpenSearch/Qdrant receipts and service search, coherent Qdrant tamper rejection, and Temporal worker recovery passed. All services, volumes, network, and generated credentials were removed; no provider calls were made.
- Focused RAG/API/planning/Qwen/capture/live-agent/Temporal regression suite: 103 passed in 13.90s. `py_compile`, `git diff --check`, and offline `uv lock --check` passed. Run-input SHA-256s: backend probe `16d890455a05bc3948f3f7d4ada7a7dda2fef8dd99c7c1e9044270cafce759c1`; Temporal probe `e5f5bb0938695fba8d8989d0b2d16727f1634f86c39a421661c51892aab4702a`; workflow `9cb47a196c7366e774c9cfc8e2410243f99b20eddddcc31cff78b03d795b4ad1`; ingest CLI `ea7be1404c1bde088abe05cde820bd1c3e57625e9a9998132256fdb951b8f3cf`; PostgreSQL adapter `8b76b6070bc288eadc7839ba795bf1b7c13a05cef7afcde47a5dfdf15be009d8`; Temporal regression test `a0718db99a1a1ed774d21d2aa5372e9117544470cba00aec98a91ccd5021a711`; Compose `dc42e0dc0d6de518f3e549ddce05987e87f68f96cbfd31ef419bc5b4aee39ab7`; fixture `c3d1e33addd625932dad47ca9352d203dfbefc36cff1b6b5ae68f307780e4407`; `pyproject.toml` `c9e28f6ebf630e1f21578b11a62779e4e870c29a52c8d4f6304802df6cbd1285`; `uv.lock` `60422202eac55c7d22109e397687de908f1588b5237aba87a0ec2a0cfa2b3f5a`.
- This closes the observed PostgreSQL JSON path and Temporal mutable-input defects only. Full PostgreSQL contract/HA, Temporal operations and lifecycle controls, release-bound Qwen1024 application retrieval, and remaining A-G acceptance remain pending. No external provider request was made; exact human approval remains pending.

### PostgreSQL outbox concurrency follow-up (2026-10-27)
- Added a real PostgreSQL probe for two simultaneous outbox consumers and lease-generation fencing. Two concurrent claims received distinct pending events; after the first lease expired and was reclaimed, its stale generation could not persist a receipt, while the current generation acknowledged successfully.
- Isolated run `raama-pgoutbox-1027-04`, run `71b349d84ab8`, passed with PostgreSQL 17.11 and the previous continuation, transaction, release, OpenSearch/Qdrant, and Temporal drift/recovery checks. Result `.tmp/industrial_rag_backend_integration_20261027/postgres-outbox-concurrency-04-20261027.json`, SHA-256 `3637360260cb3a4f8e772dd3acadc61cb53ca416266acab461a134e2fe69091b`; updated backend probe SHA-256 `09af95475473ce879ff2753c0e50d7cb2008cb55f1d921dd69d2fc98d663c49c`. Other input/config/lock hashes match the preceding follow-up entry. Stack containers, volumes, network, and synthetic credentials were removed.
- Focused regression suite passed again: 103 tests in 13.90s; compile, diff, and offline lock checks passed. This adds concurrency and stale-lease coverage, not a claim that the full PostgreSQL HA/contract suite is complete.


### b1ffa51 input pin follow-up review (2026-10-27)
- Local `b1ffa510632d6e7f80dbebb3dd9de400d15397b7` maps to backup `6afaecc46d2a8b6264cfeda59b57c0888aa7481b`, tree `37331b190056b8f88eb13ca875798a4c42b02aba`; expected previous `45a803acf7367157a76b305651557a06565cf8a3`, nonforce, 8 changed blobs/ref verified, 0 .tmp, main unchanged. F06-P1-01 partially fixed, not full A-G acceptance.
- Exact archive Temporal/RAG/API/planning 67 affected tests passed in 11.17s. Parent actual activity/CLI/isolated parser/SQLite confirms original source+metadata path drift is ignored, captured source/metadata byte changes reject, and completed retry returns same result. Byte-pin sub-boundaries independently closed.
- F06-P1-01 timing remains open: the task-state identity freezes ingested_at before the RAG transaction succeeds. Parent injected one store precommit failure, verified zero version rows, then delayed the controlled retry clock by2h. The first successful RAG version still records the failed-attempt time. Probe `b1ffa51-05of4myl/parent_temporal_pin_followup_result.json`; real local parser/store, controlled heartbeat/time/precommit fault, no Temporal network/model/provider. Repository transaction must own first actual ingest time; a previously committed version must retain its true timestamp during lost-ack recovery.
- Saved physical result SHA256 `373eafe7342cf6e970bbd0a3c0918ec0efe658eb7df449b53a19ed7f21dc85b7` verified: actual synthetic PG continuation confirm/revoke with2revision increments and existing PG fault/search paths; Temporal captured-input/lost-ack recovery. All10 documented run-input hashes match explicit archived LF bytes. This evidence closes the specific JSON path/captured-byte defects, and does not close the timestamp precommit/retry gap.
- Writer is fixing transaction timestamps and extending actual PG concurrent claim/lease fencing; those WIP changes await stable review. Full PG/HA/Temporal lifecycle/complete Agent/modules/final clone/resource enforcement remain pending. Quota8/8/latestgrant2/2 remaining0; normal work continues, external provider transmission approval still unanswered.


### 59ffddf PostgreSQL outbox probe review (2026-10-27)
- Local `59ffddfd59db83bb3244b45e26a871dee7f031a9` maps to backup `0bbfadbfbfbf95cf17a96fe45d565a1fb8b6422f`, tree `75cba4f954454d9f36b62f100548ad0cc304a02a`; expected previous `6afaecc46d2a8b6264cfeda59b57c0888aa7481b`, nonforce, 2 changed blobs/ref verified, 0 .tmp, main unchanged.
- This checkpoint changes only the versioned backend probe and5 documentation lines. Parent inspected the exact committed probe and saved real PostgreSQL record SHA256 `3637360260cb3a4f8e772dd3acadc61cb53ca416266acab461a134e2fe69091b`, matching probe explicit LF SHA256 `09af95475473ce879ff2753c0e50d7cb2008cb55f1d921dd69d2fc98d663c49c`. No unchanged tests/models/services repeated.
- Two simultaneous consumers received distinct events; after lease expiry/reclaim the stale generation could not save a receipt, while the current generation committed and completed events could not be reclaimed. Bounded synthetic real-PG evidence, not full PostgreSQL contract/HA.
- Transaction-owned ingest timestamp and latest reported live-run fixes are separate subsequent work; F06-P1-01 remains unaccepted until exact stable source and original fault probe pass. Quota8/8 remaining0, normal active work continues; external provider consent remains pending.


### 316efc5 transaction timestamp acceptance boundary (2026-10-27)
- Local `316efc5a8e76757ed654651b690a198865be364a` maps to backup `95f6f6d9570468be2dfdc6292b254e6706652b3f`, tree `164eac0dad8bfe1a39ff3e59ba7865c19cc37db8`; expected previous `0bbfadbfbfbf95cf17a96fe45d565a1fb8b6422f`, nonforce, 10 changed blobs/ref verified, 0 .tmp, main unchanged. Stage repair, not A-G completion.
- Exact archive Temporal/RAG/API/planning 68 affected tests passed in11.80s. F06-P1-01 independently closed through actual activity/CLI/isolated parser/SQLite: fault after SQL version insertion rolls back to0versions;2h delayed first successful transaction uses the later time; separate failure after RAG commit/before task-result save then2h replay reuses the committed time. Source/metadata byte capture, original-path drift immunity and captured-byte rejection retained. Heartbeat/clock/faults controlled; no parent Temporal network/model/provider rerun. Probe `316efc5-y86q7r7k/parent_transaction_timestamp_fixed_result.json`.
- Saved real-service results `ba6ba9161d2bc95a1cfebda8907429d6a557104ba315efdc4ae0e8dce4cddd0a` and `e8cf7d6110481f0be4893793b29b8d5b470bfcd02a9996041824a7ca05fc693d` plus6 documented current source LF hashes match. They add READ COMMITTED/SKIP LOCKED claim concurrency, cache upsert/readback, no-op versus revoke/restore permission revisions and continuation idempotency; full PG contract flag remains false.
- Parent independently read Docker exact project labels and verified `raama-pgcontract-1027-07` and `raama-temporal-txn-1027-06` have no containers, named volumes or networks. Initial parent guessed ProgramFiles executable path was absent; resolved installed per-user executable via Get-Command and corrected the same read-only check successfully. Initial empty arrays from failed invocation were not accepted as cleanup evidence.
- Previous writer turn completed normally. Ordinary stage continuation `RAAMA-INDUSTRIAL-RAG-CONTINUE-316EFC5-20261027` was reserved/sent once; new turn `01a17a5e-cce4-7001-b231-6ef02656a97a` confirmed active. This consumes no quota recovery: count8/8/latestgrant2/2 remaining0. Subsequent Temporal indexing/offline evaluation/control paths are WIP. Full PG/HA/Temporal lifecycle/full Agent/modules/release1024/final clone/aggregate Linux cap remain pending; external provider transfer approval unanswered.


### dc83332 Temporal operation and synthetic1024 checkpoint (2026-10-27)
- Local `dc8333202274670964a9cf3f9fb6a938f6ce1f5d` (including ae52187 and713151d) maps to backup `98ca43f4237dd7ea4eaa449f5e3cb789e555ede5`, tree `02300034c364428dc1e36c3e88194d6b9a109d37`; expected previous95f6f6d, nonforce10changedblobs/ref verified,0.tmp,main unchanged. Known-defect stage checkpoint, not A-G acceptance.
- Parent exact ae52187 archive Temporal/state/RAG/unified-workflow78passed10.58s; exact dc83332 archive Qdrant/API/planning16passed3.48s. Synthetic1024 source-cache/REST release receipt/publication/retrieval positive and unauthorized backend-query count unchanged verified; this does not prove real Qwen inference/live1024 Qdrant application retrieval. Offline-only Qwen client cache miss refuses before network. Mocked workflow controls and direct activities do not establish real Temporal restart/replay lifecycle.
- F06-P2-02 OPEN: actual local build-index activity persisted published release/result artifact; injected task completion-store failure, then durable failed and operator dead_letter succeeded. initialize_temporal_task_activity loaded result artifact before terminal fence and silently changed dead_letter to completed (events created/index_started/index_persisted/failed/dead_letter/completed). Probe ae52187-pg9cmdu7/parent_temporal_dead_letter_result.json; only completion failure/heartbeat controlled, no Temporal service/model/provider. Require terminal-state protection plus explicit side-effect reconciliation, preserving normal committed lost-ack replay.
- Previous stage completed normally. Ordinary continuation RAAMA-INDUSTRIAL-RAG-CONTINUE-DC83332-20261027 sent once and new turn01a17a85-acda-7b20-a45d-357f1adde5fb confirmed active,cursor51; no quota recovery consumed (8/8,remaining0). Full PG/HA/realTemporalops/fullAgent/modules/freshcapture/release1024/finalclone/Linuxaggregatecap remain pending; external Qwen transfer consent still unanswered.
