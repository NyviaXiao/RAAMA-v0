# 系统架构边界与实施顺序

这是新项目的**设计契约**，并标明当前实现边界。P3–P8 首个数值研究 MVP 已贯通点时市场数据、三个量价专家、Ridge 基线、等权/静态/自适应/状态条件融合、Top-K 与逆波动率组合，以及带费用、停牌和成交额参与率约束的日线名义账户模拟。它不是逐股股数/券商成交回放，也不是完整协方差或 CVaR 优化。P9 以后的文本 LLM、基本面/新闻专家、辩论、记忆和强化学习没有实现。代码与真实数据实验见 [MVP_RESEARCH.md](MVP_RESEARCH.md)；早期信号诊断见 [AGENT_RESEARCH.md](AGENT_RESEARCH.md)。未实现模块只保留设计，不创建空 API，也不沿用旧选股项目代码。

## 决策链

```mermaid
flowchart TD
    SRC[外部数据源] --> RAW[原始记录与来源证据]
    RAW --> PIT[时间点数据视图]
    PIT --> MS[决策时刻 MarketState]
    MS --> TA[技术/趋势 Agent]
    MS --> FA[横截面因子 Agent]
    MS --> LA[流动性 Agent]
    MS --> RG[市场状态识别]
    MS -. P9/P10，具备点时资料后 .-> TXT[基本面/公告/新闻/宏观 Agent]
    TA --> SIG[统一 AgentSignal[]]
    FA --> SIG
    LA --> SIG
    TXT --> SIG
    SIG --> CAL[信号校准]
    CAL --> REL[已成熟结果驱动的可靠性]
    REL --> GATE[自适应门控]
    RG --> GATE
    GATE --> ALPHA[合成预期收益与不确定性]
    ALPHA --> OPT[风险与交易约束组合优化器]
    OPT --> SIM[执行模拟与已实现结果]
    SIM --> EVAL[评价]
    SIM --> MATURE[五日标签成熟]
    MATURE --> REL
```

P9/P10 的 Fundamental、Announcement、News、Macro 专家在对应数据源完成点时验证后，接入同一个 `AgentSignal[]` 边界；它们不会绕过校准、可靠性和组合风险约束。

核心原则是分开**预测**、**融合**和**组合决策**。多个 Agent 可以提供不同证据，但不能各自下单或决定最终仓位；组合优化器也不直接解释文本。这样才能在相同数据和成本条件下，把 Agent、门控与优化器分别同简单基线比较。

这套系统采用**专长 Agent 并行产出、确定性流程集中编排、风险优化器统一决策**的形态。Agent 之间不通过自由文本互相传递指令，也不让 LLM 充当不透明的总交易员；必要的辩论仅在 P11 作为可消融的实验功能。

`reliability` 中的自适应门控解决“当前该相信哪些预测源”；`portfolio` 中的优化器解决“在风险、资金和交易约束下持有哪些资产”。两种权重不是同一个问题，不能合并成 Agent 投票后直接下单。组合侧另设可审计的硬约束检查；不满足条件时拒绝提案并记录触发的约束，成交模拟器再决定目标权重是否可实际成交。

## 模块责任与输入输出

| 边界 | 责任 | 不能做什么 | 实施阶段 |
| --- | --- | --- | --- |
| `data` | 验证外部文件，记录来源、事件时间和可用时间；按决策时点给出可见数据与历史股票池 | 将 `test.csv` 暴露给研究代码；臆造发布时间；静默换数据源 | P1 训练入口；P2 时间点平台 |
| `features` / `targets` | 当前研究流水线从当时可见行情构造三类量价信号，并只在成熟历史样本上训练五日收益模型 | 跨验证边界拟合标准化；把未来收益放入决策输入 | P3 原型已用于评估脚本；独立 API 未扩展 |
| `backtest` / `evaluation` | 统一策略的目标权重、次日开盘名义成交、停牌/流动性限制、费用、持仓和净值评价 | 将名义金额账本称作逐股成交回放；为策略采用不同执行规则 | P4 最小模拟已实现 |
| `agents` | 在同一股票池与预测期限输出结构化候选信号及依据 | 自行读取外部文件、访问未来记录或直接下单 | P5 量化；P9/P10 文本与宏观 |
| `reliability` | 对已到期预测的 Rank IC 估算可靠性并形成融合权重 | 在五日结果尚未成熟时更新历史成绩；将排序分数宣称为校准概率 | P6 基础版本已实现；概率校准未实现 |
| `regime` | 用决策时点可见的沪深300过去20日价格变化提供两态分类，并按成熟样本构造条件权重 | 用未来全样本划分历史状态；样本不足仍拟合状态权重 | P7 最小版本已实现 |
| `portfolio` | 生成 Top-K 目标；以波动率倒数分配风险组合权重，并设个股、现金、换手和成交额限制 | 从新闻文本直接指定仓位；将当前方法称作协方差/CVaR优化器 | P8 最小风险分配已实现 |

## 代码包与计划中的 Agent 角色

| 代码包 | 稳定职责 | 计划中的内容与阶段 |
| --- | --- | --- |
| `adaptive_mas.data` | 外部数据边界、规范记录和按决策时点形成可见数据 | P1 最小 CSV 加载；P2 时间点存储和历史成分 |
| `adaptive_mas.features`、`adaptive_mas.targets` | 从可见数据构造模型输入与五交易日监督目标 | 当前 MVP 在研究运行器内计算；将来按复用需求再拆出独立包 |
| `adaptive_mas.agents` | 专家预测器，只返回信号，不管理持仓 | P5 已实现趋势、反转、量价确认打分；P9 基本面/公告；P10 新闻/宏观。后续角色依据增量实验保留 |
| `adaptive_mas.reliability` | 专家可靠性估计及融合权重 | 当前 MVP 在评价流程中实现成熟 Rank IC 门控；概率校准仍待研究 |
| `adaptive_mas.regime` | 依据时点可见市场数据产出状态信息 | 当前有两态、样本不足回退全局权重的原型；更多状态须另做消融 |
| `adaptive_mas.portfolio` | 将融合后的数值观点映射为满足约束的目标权重 | 当前有 Top-K、逆波动率、集中度/现金/换手上限；完整优化器与 CVaR 未实现 |
| `adaptive_mas.backtest`、`adaptive_mas.evaluation` | 驱动历史决策流程并评价组合表现 | 当前有一致规则的日线名义金额账本、成本情景、持仓/订单/净值轨迹和区块 Bootstrap；不等同逐股/券商执行重放 |
| `adaptive_mas.orchestration` | 按时间顺序调用数据视图、专家、融合、优化和执行组件，保存可重放轨迹 | 只有 P5 真正需要多 Agent 调度时再创建；Phase 1 不建空包 |

量化专家共享同一时点可见的行情视图，但拥有各自明确的信息/方法范围；由于多个 OHLCV 信号可能高度相关，必须先测量误差相关性和增量贡献再保留。P9/P10 的文本型 Agent 只在对应阶段读取已通过点时审查的资料，负责抽取可核查依据并转成统一信号，不直接修改组合。

### Agent 与融合器之间的契约

当前 `AgentSignal` 标明：Agent 与版本、股票、决策日、五交易日预测期限、原始分数、输入截止日和证据引用。分数用于横截面排序，融合前按当日横截面转为百分位分数；当前不把原始分数伪装成预期收益或自报置信度。未来若组合优化需要收益单位，由只使用已成熟历史预测的校准层估计，并单独记录估计不确定性。

`MarketState` 必须由时间点数据层生成，Agent 只能接收其授权视图。融合器产生统一的预期收益与不确定性，组合层独立负责权重。每一步保存输入引用、参数版本和输出，便于重放一笔决定及其事后评价。

## 代码依赖方向

数据边界是唯一接触外部文件和网络响应的生产模块。后续特征、Agent、可靠性、状态和组合模块只依赖规范数据或明确的上游结果；它们不直接依赖特定 CSV 文件名或供应商。回测负责调度各模块在历史时间线上运行；评价读取回测记录，不向上游反写未来绩效。`test.csv` 只允许由未来隔离的最终本地评分器访问，绝不进入这条依赖链。

## 面向实现的目标代码布局

以下是**模块真正进入开发阶段后的目标布局**，不是现在要创建的空文件：

```text
src/adaptive_mas/
├── data/            # 数据边界、点时视图、历史成分
├── domain.py        # 跨模块记录契约；首次需要时再创建
├── features/        # 按时点构造特征
├── targets/         # 到期结果与监督目标
├── models/          # 非 Agent 量化基线（P4）
├── agents/          # 专家预测器与后续文本分析器
├── reliability/     # 校准、评分卡、门控权重
├── regime/          # 按时点判断市场状态
├── portfolio/       # 约束、风险估计、组合优化
├── backtest/        # 时间顺序运行、成交与成本模拟
├── evaluation/      # 滚动验证、指标和比较
└── orchestration/   # 确定性决策流水线（P5 需要时创建）
```

`configs/`保存显式研究协议和实验参数；`experiments/`保存配置快照、数据清单、版本和指标；`data/raw/`只放可追溯的原始外部响应，`data/processed/`存派生数据。源 CSV 继续留在给定外部目录，不复制进仓库。研究阶段先用简单文件与 JSONL 记录，不先引入服务、消息队列或向量库。

### 跨模块记录契约（计划定义，不在 Phase 1 建类）

| 记录 | 关键字段 | 产生方 → 消费方 |
| --- | --- | --- |
| `MarketSnapshot` | `decision_time, visible_bars, visible_membership, source_refs` | 时间点数据层 → 特征和 Agent |
| `AgentSignal` | `agent_id, version, decision_time, horizon, symbol, expected_return, uncertainty, evidence_refs, input_cutoff` | 专家 Agent → 校准/可靠性 |
| `ReliabilitySnapshot` | `as_of, matured_through, agent_scores, calibrated_weights, sample_counts` | 可靠性层 → 门控 |
| `RegimeAssessment` | `as_of, regime_probabilities, evidence_refs` | 状态层 → 门控 |
| `AllocationProposal` | `decision_time, target_weights, cash_weight, risk_summary, binding_constraints` | 组合优化器 → 回测执行模拟器 |
| `DecisionRecord` | 上述记录引用、配置/代码版本、输出权重与交易结果引用 | 编排与回测 → 评价和审计 |

数值 Agent 与 LLM Agent 共用同一个 `AgentSignal` 边界；LLM 原文和证据引用额外留档。信号契约统一五交易日预测期限，但行情的确切可用时间和成交约定仍须数据/研究协议证明。每个记录保留 `decision_time` 和其输入截止时间。五日标签完全成熟后才可写入相应的 `ReliabilitySnapshot`。组合输出是权重提案，回测执行器应用涨跌停、停牌、流动性、成本与成交规则后才产生实际持仓；不能把目标权重当成已成交结果。

`domain.py` 只在 P2/P3/P5 的字段经验证后创建；稳定记录可先用带类型注解的标准库数据类表达。外部 CSV/API/LLM 响应和用户配置在入口处验证，内部模块通过类型明确的契约协作，不在每个函数重复校验同一内部不变量。

P4/P5 初期由普通 Python 流程按交易日明确调用组件，确保研究运行可重放。TradingAgents 使用 LangGraph 编排 LLM 多角色流程，这可作为后续文本专家的参考；只有出现需要检查点恢复或复杂条件路由的真实需求时才引入图编排，不让数值研究主链依赖一个 LLM Agent 框架。此决定符合本项目先确定性研究、后可选 LLM 工作流的阶段安排。

## 依据来源的架构取舍

| 依据 | 对本项目有用的部分 | 本项目的取舍 |
| --- | --- | --- |
| [Microsoft Qlib 数据层](https://github.com/microsoft/qlib/blob/main/docs/component/data.rst)、[组合策略接口](https://github.com/microsoft/qlib/blob/main/docs/component/strategy.rst) | 数据读取/处理、预测模型、组合策略、回测松耦合；可学习处理参数的 fit/transform 边界有利于只在训练窗口拟合标准化。 | 采用职责分离和实验可重放思想；不在当前小样本阶段引入 Qlib 专用的二进制数据存储或全套配置框架，先依赖可审计的明文数据契约。 |
| [FinRL 论文](https://arxiv.org/abs/2011.09607)、[FinRL-Meta 论文](https://arxiv.org/abs/2211.03107) | 将市场环境、决策算法和金融应用分层；论文明确指出低信噪比、幸存者偏差和回测过拟合会损害结论，并强调成本、流动性及可复现实验。 | 先建立数据、基线和时间序列回测，再研究 RL；把股票池、成本、可成交性和时间切分视为核心架构输入，不把 RL 当作系统起点。 |
| [TradingAgents 官方项目](https://github.com/TauricResearch/TradingAgents)、[论文](https://arxiv.org/abs/2412.20138) | 专长分析、研究分工、结构化决策和集中组合审核可借鉴；项目把自身定位为研究框架，并说明表现依赖模型、温度、时期和数据。官方发布记录持续加入过日期路径的点时修复。 | 仅借鉴职责拆分与审计轨迹；量化模型先于 LLM，辩论延迟到 P11 做消融。点时访问在 `data` 边界拒绝未来数据，不能依赖 prompt 提醒 Agent“不要看未来”。 |
| [FinCon 论文](https://arxiv.org/abs/2407.06567) | Manager-Analyst 分工、不同资料由专门分析器处理、结果成熟后总结经验的研究启发。 | 论文实验使用其特定多模态数据和任务；不将其收益结论外推到 A 股。自由文本记忆不写入核心量化权重；CVaR 由可复现的数值优化模块单独实现，再独立做消融。 |
| [FinRL-X 官方项目](https://github.com/AI4Finance-Foundation/FinRL-Trading) | 以目标权重作为策略与执行接口的做法能让选股/择时/风险覆盖可替换、回测和执行接口一致。 | 借鉴权重作为组合边界；其文档中的多供应商自动选择不符合本项目“失败显式、禁止静默切源”规则，因此数据 Provider 必须显式配置且失败时停止。 |

上述资料提供的是架构模式和风险提示，不证明这些组件能在当前沪深300样本上盈利。最终保留哪个 Agent、融合器和优化器由同一滚动样本外协议的增量实验决定；重点对比单一专家、等权、静态权重和自适应权重，并完整记录无增益结果。

当前项目按 [执行计划](EXECUTION_PLAN.md) 逐阶段增量实现。Phase 1 不创建 `MarketState`、`AgentSignal`、Provider、优化器或编排器的空类；在 P2/P5/P8 的真实输入与验收条件明确后再落地相应接口。
