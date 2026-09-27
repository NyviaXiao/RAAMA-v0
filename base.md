# Adaptive-MAS
## Risk-Aware Adaptive Multi-Agent Portfolio Decision System for CSI 300

---

# 0. 项目核心理念

本项目不是一个“LLM 股票聊天系统”，也不是简单的“多个 Agent 投票选股”。

系统需要解决的核心问题是：

> 多个拥有不同信息、不同分析方法和不同历史可靠性的专业金融 Agent，在同一时间对同一批资产形成独立判断；系统根据 Agent 在当前市场状态下的历史表现、置信度、不确定性以及观点冲突程度，动态学习每个 Agent 当前应该获得多大的决策权，并最终在风险、交易成本、换手率和仓位约束下生成多资产投资组合。

核心决策链必须始终保持：

```text
Financial Information
        ↓
Independent Specialized Agents
        ↓
Structured Alpha Signals
        ↓
Debate / Challenge
        ↓
Confidence Calibration
        ↓
Historical Agent Reliability
        ↓
Market Regime
        ↓
Adaptive Gating
        ↓
Combined Expected Return
        ↓
Risk-Aware Portfolio Optimization
        ↓
Target Portfolio
        ↓
Execution / Backtest
        ↓
Outcome
        ↓
Performance Memory
        └────────────→ Reliability Update
```

其中：

**Data Layer 是 Agent 的信息来源。**

**Agent Layer 是 Alpha 产生层。**

**Reliability/Gating Layer 是本项目核心智能决策层。**

**Portfolio Layer 是最终金融决策层。**

---

# 1. 与现有开源项目的关系

本项目不是复制任何一个现有框架，而是组合并改造其最有价值的部分。

## TradingAgents

借鉴：

```text
Analyst Team
Research / Debate Team
Risk / Portfolio hierarchy
LangGraph orchestration
shared typed state
```

TradingAgents 的核心流程大致是：

```text
Fundamental Analyst
News Analyst
Market Analyst
Sentiment Analyst
        ↓
Bull Researcher
vs
Bear Researcher
        ↓
Research Manager
        ↓
Trader
        ↓
Aggressive / Neutral / Conservative Risk Analysts
        ↓
Portfolio Manager
```

我们的关键改造是：

```text
TradingAgents:
Research → LLM Trader → BUY/SELL

Adaptive-MAS:
Research → Structured Alpha
         → Reliability/Gating
         → Mathematical Portfolio Optimizer
```

即：

**不让 LLM Trader 成为最终决策者。**

---

## FinCON

借鉴：

```text
Manager–Analyst hierarchy
cross-functional financial agents
CVaR
performance feedback
episodic/procedural memory
```

改造：

FinCON 的 verbal reinforcement 在本项目里主要转换为：

```text
Prediction
↓
Realized Outcome
↓
Quantitative Attribution
↓
Agent Performance Memory
↓
Reliability Update
```

文字 reflection 可以存在，但不能替代数值 reliability。

---

## QuantHarness / QuantAgent

借鉴：

```text
Indicator Agent
Trend Agent
Pattern Agent
Decision State
Tool-generated quantitative evidence
```

核心思想：

> LLM 不应该自己计算 RSI、MACD、波动率或统计指标。

Python/tool 负责：

```text
calculation
statistics
features
signals
```

Agent 负责：

```text
interpretation
reasoning
context
```

我们的 Quant Agents 会进一步扩展为：

```text
Technical Agent
Factor Agent
Liquidity Agent
```

---

## FinAgent

借鉴：

```text
Market Intelligence
Reflection
Memory
Portfolio feedback
```

但 Memory 不放在第一决策层。

本项目 Memory 分为：

```text
Performance Memory    ← 核心
Episodic Memory       ← 后续增强
Semantic Memory       ← RAG/知识增强
```

Performance Memory 服务于 adaptive gating。

---

## Qlib

借鉴最关键的金融工程原则：

```text
Forecast Signal
       ≠
Portfolio Strategy
       ≠
Execution
       ≠
Evaluation
```

也就是说：

```text
Agent 预测 Alpha

不是

Agent 决定仓位
```

---

## StockBench

借鉴：

```text
continuous trading evaluation
standardized backtest
consistent agent evaluation
real portfolio metrics
```

系统必须评价：

```text
Agent prediction quality
+
Portfolio performance
```

不能只看最终收益。

---

## FinRL

只用于：

```text
RL portfolio baseline
```

不是项目主体。

最终用于回答：

> 为什么 Adaptive Gating + Optimization 不直接换成 PPO/SAC？

---

# 2. 系统总体架构

```text
┌───────────────────────────────────────────────────────────────┐
│                    INFORMATION LAYER                          │
│                                                               │
│ Price │ Volume │ Fundamental │ Announcement │ News │ Macro   │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │   MarketContext     │
                    │      Builder        │
                    └──────────┬──────────┘
                               │
             Same point-in-time MarketState
                               │
        ┌──────────────────────┼─────────────────────┐
        │                      │                     │
        ▼                      ▼                     ▼
 Quantitative Desk        Fundamental Desk       Event Desk
        │                      │                     │
 Technical Agent          Fundamental Agent      Event Agent
 Factor Agent                                     News Agent
 Liquidity Agent                                  Macro Agent
        │                      │                     │
        └──────────────────────┼─────────────────────┘
                               │
                               ▼
                    Independent AgentSignals
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Research Challenge  │
                    │                     │
                    │ Bull Researcher     │
                    │ Bear Researcher     │
                    │ Critic              │
                    └──────────┬──────────┘
                               │
                      Challenge Features
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Confidence Calibration   │
                 └─────────────┬────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  Reliability Engine    │
                  │                        │
                  │ IC / Accuracy          │
                  │ Brier / Error          │
                  │ Historical performance │
                  └────────────┬───────────┘
                               │
                               │
                   ┌───────────┴───────────┐
                   │                       │
                   ▼                       ▼
           Market Regime            Performance Memory
                   │
                   └───────────┬───────────┘
                               ▼
                    ┌────────────────────┐
                    │ Adaptive Gating    │
                    │    Meta Allocator  │
                    └─────────┬──────────┘
                              │
                              ▼
                    Combined Alpha Vector
                              │
                              ▼
             ┌────────────────────────────────┐
             │ Risk-Aware Portfolio Optimizer │
             │                                │
             │ Expected Return                │
             │ CVaR                           │
             │ Covariance                     │
             │ Transaction Cost               │
             │ Turnover                       │
             │ Uncertainty                    │
             │ Position Constraints           │
             └──────────────┬─────────────────┘
                            │
                            ▼
                    TargetPortfolio
                            │
                            ▼
                     Execution Layer
                            │
                            ▼
                         Outcome
                            │
                  ┌─────────┴──────────┐
                  ▼                    ▼
           Portfolio Metrics    Agent Attribution
                                       │
                                       ▼
                               Performance Memory
```

这张图是项目真正的主架构。

---

# 3. 系统核心 State

参考 TradingAgents 的 AgentState 思路，但我们的 State 更偏量化。

核心对象：

```python
DecisionState
```

概念结构：

```python
DecisionState:
    decision_time

    universe
    market_context
    portfolio_context

    agent_signals

    debate_result

    calibrated_signals

    agent_reliability

    market_regime

    gating_weights

    combined_alpha

    target_portfolio

    execution_result
```

LangGraph 或其他 orchestrator 只负责：

```text
节点调度
state propagation
parallel execution
trace
```

不要把金融逻辑写进 Graph 本身。

---

# 4. MarketContext

所有 Agent 在某个 decision time 必须看到同一个：

```python
MarketContext
```

它代表：

> 截至当前决策时点真实可获得的信息快照。

例如：

```python
MarketContext:
    decision_time

    universe

    price_history
    volume_history

    technical_features
    factor_features

    fundamentals

    announcements
    news

    macro

    current_portfolio
```

Agent 不允许：

```text
自己访问 CSV
自己调用网络
自己查询未来数据
```

而是：

```text
Agent
  ↓
MarketContext / tools
```

---

# 5. Research Agent Team

整个 Research Team 分为两大类。

---

# 6. Quantitative Desk

## Technical Agent

负责研究：

```text
trend
momentum
mean reversion
breakout
volatility
technical structure
```

底层 Python tools 计算：

```text
RSI
MACD
ATR

MA
EMA

return_5
return_10
return_20
return_60

realized volatility

price breakout
```

Agent 不手工计算指标。

Agent 输出：

```python
AgentSignal(
    agent_id="technical",
    symbol="600000",

    expected_return=...,
    probability_up=...,

    raw_confidence=...,
    uncertainty=...,

    evidence=[...],
)
```

---

# 7. Factor Agent

这是横截面 Alpha Agent。

关注：

```text
cross-sectional momentum
short-term reversal
volatility
turnover
liquidity
volume surprise
relative strength
```

它和 Technical Agent 最大区别：

```text
Technical:
单股票时间序列视角

Factor:
整个股票池横截面视角
```

可以使用：

```text
rule factor
linear model
LightGBM
```

生成 structured alpha。

---

# 8. Liquidity Agent

充分利用已有：

```text
成交量
成交额
换手率
```

分析：

```text
volume anomaly
turnover anomaly
liquidity expansion
price-volume divergence
volume confirmation
crowding proxy
```

它解决：

> 某个 Alpha 在当前流动性环境下是否值得相信？

---

# 9. Fundamental Desk

## Fundamental Agent

读取：

```text
revenue
profit
EPS
ROE
margin
cash flow
debt
growth
valuation
financial statements
```

Python 先计算所有数值指标。

LLM 只负责：

```text
解释
异常识别
跨指标关系
企业经营变化
```

输出同样的：

```text
expected_return
probability_up
confidence
uncertainty
```

---

# 10. Event Desk

## Announcement/Event Agent

优先于 News Agent 实现。

关注：

```text
业绩预告
重大合同
回购
增持
减持
并购
重组
监管
诉讼
高管变化
分红
风险公告
```

LLM 负责：

```text
event_type

materiality

direction

impact_horizon

novelty

evidence
```

最后转换成标准：

```text
AgentSignal
```

---

# 11. News Agent

与 Announcement Agent 不同：

```text
Announcement = company disclosure

News = external information
```

分析：

```text
company news
industry news
policy
sentiment
unexpected events
```

必须保留：

```text
published_at
```

---

# 12. Macro Agent

Macro Agent 的主要目标不是预测某个公司的 EPS。

它判断：

```text
risk-on
risk-off

liquidity environment

market trend

policy environment

systematic pressure
```

可以同时产生：

```text
market-level signal
```

供股票 Agent 和 Regime 模块使用。

---

# 13. Agent 独立性

所有第一阶段 Research Agents：

```text
Technical
Factor
Liquidity
Fundamental
Event
News
Macro
```

必须先：

> **独立产生判断。**

即：

```text
Technical 看不到 News 的观点

News 看不到 Factor 的预测

Fundamental 看不到 Technical 的结果
```

否则不同 Agent 会发生 opinion contamination。

只有完成独立 AgentSignals 之后才能进入：

```text
Research Challenge Layer
```

---

# 14. AgentSignal Contract

所有 Agent 必须使用完全相同的数据契约：

```python
class AgentSignal:
    agent_id: str
    symbol: str

    decision_time: datetime
    horizon_sessions: int

    expected_return: float
    probability_up: float

    raw_confidence: float
    uncertainty: float

    evidence: list[EvidenceRef]

    metadata: dict
```

统一 horizon：

```text
5 trading sessions
```

这是整个 Multi-Agent Fusion 能成立的前提。

---

# 15. Evidence Contract

不要只保存：

```text
reasoning = "I think..."
```

每个结论都需要：

```python
EvidenceRef:
    evidence_type
    source_id
    timestamp
```

例如：

```text
TECHNICAL_FEATURE
FUNDAMENTAL_RECORD
ANNOUNCEMENT
NEWS
MACRO
```

这样以后：

```text
为什么买这只股票？
```

能够向下追踪。

---

# 16. Research Challenge Layer

借鉴 TradingAgents 的：

```text
Bull Researcher
Bear Researcher
```

但改变目的。

输入：

```text
所有独立 AgentSignal
+
对应 evidence
```

---

# 17. Bull Researcher

任务：

```text
寻找支持当前 bullish alpha 的证据
发现可能被低估的正面信息
攻击 bearish assumptions
```

---

# 18. Bear Researcher

任务：

```text
寻找负面证据
识别风险
攻击 bullish assumptions
寻找 Agent 忽略的信息
```

---

# 19. Critic Agent

Critic 不决定：

```text
BUY
SELL
```

Critic 输出：

```python
ChallengeResult:
    symbol

    evidence_conflict
    unsupported_claims
    missing_information

    bullish_support
    bearish_support

    confidence_adjustment_features
```

关键原则：

> Debate 只改变“我们有多相信这个预测”，不直接改变 portfolio。

---

# 20. 为什么 Debate 放在这里

TradingAgents 的 debate：

```text
观点
↓
Trader
```

我们的：

```text
观点
↓
Debate
↓
Calibration
↓
Reliability
↓
Gating
```

所以 Debate 是：

> **uncertainty discovery mechanism**

而不是最终交易机制。

---

# 21. Confidence Calibration

Agent 自报：

```text
confidence = 0.85
```

不能直接相信。

因此：

```text
raw confidence
      ↓
historical calibration
      ↓
calibrated confidence
```

例如：

```text
过去该 Agent 声称 0.8 confidence 的预测

实际成功率只有 0.58
```

则：

```text
raw 0.80
→ calibrated ≈ 0.58
```

第一版：

```text
rolling empirical calibration
```

后续：

```text
isotonic
Platt-like calibration
```

---

# 22. Prediction Lifecycle

一个 AgentSignal 不是当天就能知道正确还是错误。

例如：

```text
t = 2025-05-01
```

产生：

```text
5-day expected return
```

必须等 label mature。

因此：

```text
SignalCreated
     ↓
PendingPrediction
     ↓
5 sessions pass
     ↓
ResolvedPrediction
     ↓
Performance attribution
```

这是 Reliability Engine 非常重要的部分。

---

# 23. Performance Memory

每个 resolved prediction 保存：

```python
AgentPerformanceRecord:
    agent_id
    symbol

    prediction_time
    resolution_time

    predicted_return
    realized_return

    probability_up

    error
    direction_correct
    brier_score

    regime

    confidence

    challenge_features
```

它是整个系统 adaptive learning 的基础。

---

# 24. Reliability Engine

对于每个 Agent，计算：

```text
IC

Rank IC

Directional Accuracy

MAE

Brier Score

Calibration Error

recent contribution
```

并维护：

```text
rolling 20 sessions
rolling 60 sessions
EWMA
```

形成：

\[
R_{i,t}
\]

---

# 25. 为什么 Reliability 是项目核心

普通 MAS：

```text
Technical = 一票
News = 一票
Fundamental = 一票
```

我们的系统：

```text
Technical:
最近在当前环境准确率低
→ 权重下降

Event:
最近在当前环境表现好
→ 权重上升
```

因此系统学习的不是：

> 哪只股票涨。

还学习：

> **当前环境下，谁更会判断股票涨跌。**

这是项目的核心区别。

---

# 26. Market Regime

独立：

```python
MarketRegimeDetector
```

不要用 LLM 自由判断。

输入：

```text
market return
market momentum
breadth
realized volatility
cross-sectional volatility
aggregate turnover
```

第一版：

```text
Bull / Bear
×
HighVol / LowVol
```

得到：

```text
Bull-LowVol
Bull-HighVol
Bear-LowVol
Bear-HighVol
```

---

# 27. Regime-specific Reliability

维护：

\[
R(agent, regime)
\]

例如：

```text
Technical:

Bull-LowVol   0.76
Bull-HighVol  0.61
Bear-LowVol   0.53
Bear-HighVol  0.37
```

Event Agent 可能刚好相反。

---

# 28. Meta Allocator / Adaptive Gating

这是系统真正的“总决策 Agent”。

但它第一版不应该是 LLM。

它是：

```python
AdaptiveGatingEngine
```

输入：

```text
Agent Signals

Calibrated Confidence

Uncertainty

Historical Reliability

Current Regime

Debate Challenge Features
```

---

# 29. 第一版 Gating

定义：

\[
S_{i,t}
=
\alpha R_{i,t,r}
+
\beta C_{i,t}
-
\gamma U_{i,t}
+
\delta D_{i,t}
\]

其中：

```text
R = regime-specific reliability

C = calibrated confidence

U = uncertainty

D = debate/challenge support
```

然后：

\[
w_{i,t}
=
Softmax(S_{i,t})
\]

---

# 30. Combined Alpha

每只股票：

\[
\hat\mu_{j,t}
=
\sum_i
w_{i,t}
\hat\mu_{i,j,t}
\]

输出：

```python
CombinedAlpha:
    symbol
    expected_return
    uncertainty
    agent_weights
    contributing_signals
```

---

# 31. 第一版权重粒度

不要一开始学习：

```text
每支股票 × 每个Agent × 每一天
```

过于复杂。

第一版：

```text
Agent × Regime × Time
```

即：

当前 Regime 下：

```text
Technical  0.28
Factor     0.31
Liquidity  0.12
Event      0.18
...
```

股票仍然有自己的 `expected_return` 和 confidence。

后续才考虑 asset-specific gating。

---

# 32. Portfolio Optimization

Meta Allocator 输出 Alpha。

Portfolio Optimizer 输入：

```text
Combined Alpha Vector
Current Holdings
Covariance
Historical Scenario Returns
Uncertainty
Transaction Costs
```

---

# 33. Optimization Objective

逐步实现，最终目标：

\[
\max_{\mathbf w}
\quad
\mathbf w^\top\mu
-
\lambda_{risk}Risk(w)
-
\lambda_{CVaR}CVaR(w)
-
\lambda_{turn}Turnover(w)
-
\lambda_{cost}Cost(w)
-
\lambda_{unc}U(w)
\]

---

# 34. Portfolio Constraints

第一版：

```text
long only

sum(weights) <= 1

weight_i <= max_position

cash >= min_cash

turnover <= threshold
```

后续才增加：

```text
sector exposure
industry constraints
```

---

# 35. 为什么 Portfolio Optimizer 不是 Agent

因为：

```text
仓位
风险
成本
约束
```

是明确数学问题。

没有必要：

```text
让LLM猜应该买8%还是12%
```

所以：

```text
LLM / models → Alpha

Optimization → Allocation
```

---

# 36. Risk Layer

不复制 TradingAgents 的三个 LLM Risk debator 作为主要风险控制。

我们采用：

```text
Quantitative Risk Engine
```

负责：

```text
volatility
covariance
VaR
CVaR
drawdown exposure
concentration
turnover
liquidity
```

可以增加一个：

```text
RiskCritic
```

负责解释异常风险。

但数学 constraint 有最终权力。

---

# 37. PortfolioManager

项目中的 PortfolioManager 不应该重新“凭感觉”推翻 Optimizer。

它负责：

```text
validate optimizer output
produce decision trace
attach risk summary
```

而不是第二次自由决策。

---

# 38. Execution Layer

输入：

```text
TargetPortfolio
```

输出：

```text
Orders
Fills
NewPortfolioState
```

统一执行规则。

例如：

```text
t close 做决策
t+1 open 执行
```

具体 convention 后续根据数据确定。

---

# 39. Feedback Loop

真正的 adaptive loop：

```text
Decision at t
      ↓
Prediction pending
      ↓
Future outcome observed
      ↓
Attribution
      ↓
Agent metrics updated
      ↓
Regime-specific reliability updated
      ↓
Next decision weights change
```

这比：

```text
LLM写一段“今天亏了我下次注意”
```

更重要。

---

# 40. Reflection Agent

Reflection 是辅助。

输入：

```text
large prediction error

large portfolio loss

major agent disagreement

unexpected event
```

输出：

```text
structured post-mortem
```

可以保存到 Episodic Memory。

但：

> Reflection 不直接修改模型参数。

真正 adaptive 的更新仍然来自 Performance Memory。

---

# 41. Memory Architecture

```text
Memory
│
├── PerformanceMemory
│     └── structured quantitative history
│
├── EpisodicMemory
│     └── notable decisions / failures / surprises
│
└── SemanticMemory
      └── company/events/research RAG
```

优先级：

```text
Performance
>
Episodic
>
Semantic
```

---

# 42. LangGraph Orchestration

建议使用 LangGraph，但只做 orchestration。

核心 Graph：

```text
START
  │
  ▼
BuildMarketContext
  │
  ├──────────── parallel ──────────────┐
  │                                   │
  ▼                                   ▼
Technical                         Fundamental
Factor                            Event
Liquidity                        News
                                 Macro
  │                                   │
  └───────────────┬───────────────────┘
                  ▼
          CollectAgentSignals
                  │
                  ▼
             BullResearch
                  │
                  ▼
             BearResearch
                  │
                  ▼
                Critic
                  │
                  ▼
         ConfidenceCalibration
                  │
                  ▼
          LoadReliability
                  │
             + Regime
                  │
                  ▼
           AdaptiveGating
                  │
                  ▼
          BuildCombinedAlpha
                  │
                  ▼
          PortfolioOptimizer
                  │
                  ▼
              RiskCheck
                  │
                  ▼
            TargetPortfolio
                  │
                  ▼
               END
```

Backtest engine 在 Graph 外推进时间。

不要让 LangGraph 自己管理市场时间。

---

# 43. Daily Backtest Loop

外部：

```python
for decision_time in trading_calendar:

    resolve_matured_predictions(decision_time)

    update_performance_memory()

    state = build_decision_state(decision_time)

    result = agent_graph.invoke(state)

    execute(result.target_portfolio)

    register_pending_predictions()
```

这是非常重要的架构边界：

```text
LangGraph
=
单次 decision reasoning

Backtest Engine
=
时间推进
```

---

# 44. Data Layer 在整个项目里的正确位置

Data Layer 不再是项目主角。

它只需要提供：

```python
MarketDataTool

FundamentalTool

AnnouncementTool

NewsTool

MacroTool

UniverseTool
```

所有工具支持：

```text
as_of
```

就够。

Agent 不关心：

```text
CSV在哪
API是什么
AKShare还是Tushare
```

---

# 45. Data Sources

现有：

```text
train.csv
stock_data.csv
hs300_stock_list.csv
```

先用于：

```text
Quant Agents
Backtest
Market Regime
```

未来需要再补：

```text
Fundamental
Announcement
News
Macro
Historical membership
```

但这些是 Agent 的输入资源，不是架构中心。

---

# 46. Evaluation Architecture

Evaluation 分四级。

## Level 1：Agent

```text
IC
Rank IC
Accuracy
MAE
Brier
Calibration
```

---

## Level 2：Gating

评价：

```text
Equal Weight
Static Weight
Adaptive Weight
Regime Adaptive
```

---

## Level 3：Portfolio

```text
Return
Sharpe
Sortino
MDD
Calmar
CVaR
Turnover
Cost
```

---

## Level 4：System

```text
LLM calls
Token cost
Latency
Decision trace completeness
```

---

# 47. 必须存在的 Baselines

```text
Traditional Quant
│
├── Equal Weight
├── Momentum
├── Factor
└── LightGBM

Single Agent
│
├── Technical
├── Fundamental
└── Event

Multi-Agent
│
├── Equal Weight
├── Static Vote
├── Debate
├── Adaptive
├── Regime Adaptive
└── Full Risk-Aware Adaptive MAS

RL
│
└── PPO / SAC Portfolio
```

---

# 48. 必须存在的 Ablation

```text
Full System

- Regime
- Reliability
- Confidence Calibration
- Debate
- Uncertainty
- CVaR
- Transaction Cost
- Memory
```

核心问题：

```text
Adaptive 是否比 Equal Weight 好？

Regime 是否增加价值？

Calibration 是否改善 Gating？

Debate 是否真的降低错误 confidence？

Risk optimization 是否提高 risk-adjusted performance？
```

---

# 49. 推荐代码架构

```text
adaptive-mas/
│
├── configs/
│
├── src/adaptive_mas/
│
│   ├── core/
│   │   ├── state.py
│   │   ├── signals.py
│   │   └── portfolio.py
│   │
│   ├── data/
│   │   ├── context.py
│   │   ├── market.py
│   │   ├── fundamentals.py
│   │   ├── events.py
│   │   ├── news.py
│   │   ├── macro.py
│   │   └── universe.py
│   │
│   ├── tools/
│   │   ├── technical.py
│   │   ├── factors.py
│   │   ├── liquidity.py
│   │   └── risk.py
│   │
│   ├── agents/
│   │   ├── base.py
│   │
│   │   ├── research/
│   │   │   ├── technical.py
│   │   │   ├── factor.py
│   │   │   ├── liquidity.py
│   │   │   ├── fundamental.py
│   │   │   ├── event.py
│   │   │   ├── news.py
│   │   │   └── macro.py
│   │   │
│   │   └── debate/
│   │       ├── bull.py
│   │       ├── bear.py
│   │       └── critic.py
│   │
│   ├── intelligence/
│   │   ├── calibration.py
│   │   ├── reliability.py
│   │   ├── regime.py
│   │   └── gating.py
│   │
│   ├── portfolio/
│   │   ├── optimizer.py
│   │   ├── risk.py
│   │   ├── constraints.py
│   │   └── costs.py
│   │
│   ├── memory/
│   │   ├── performance.py
│   │   ├── episodic.py
│   │   └── semantic.py
│   │
│   ├── orchestration/
│   │   ├── graph.py
│   │   └── nodes.py
│   │
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── execution.py
│   │   └── accounting.py
│   │
│   └── evaluation/
│       ├── agent_metrics.py
│       ├── portfolio_metrics.py
│       ├── benchmark.py
│       └── ablation.py
│
├── experiments/
├── tests/
├── scripts/
├── AGENTS.md
└── pyproject.toml
```

这不是要求一次生成所有空文件。

它是模块边界。

只有实现对应能力时再创建具体模块。

---

# 50. 开源项目源码阅读映射

Codex 在实现对应模块前，可以参考以下源码设计。

## TradingAgents

重点研究：

```text
tradingagents/agents/analysts/
tradingagents/agents/researchers/
tradingagents/agents/risk_mgmt/
tradingagents/agents/managers/
tradingagents/agents/trader/
tradingagents/graph/
```

重点学习：

```text
AgentState
parallel analysts
Bull/Bear debate
Research Manager
risk discussion
LangGraph conditional flow
structured output
```

不要复制其最终 LLM transaction decision。

---

## FinCON

重点：

```text
fincon/agents/manager_agent.py
fincon/agents/risk_control_agent.py

fincon/agents/analyst_agents/
    data_analyst.py
    news_analyst.py
    filing_analyst.py
    ecc_analyst.py
    stock_selection_agent.py

fincon/memory/

fincon/risk_control/
    within_episode.py
    over_episode.py

fincon/evaluation/
```

重点学习：

```text
Manager–Analyst organization

CVaR risk control

episodic/procedural memory

cross-episode feedback
```

---

## QuantHarness

重点：

```text
indicator_agent.py
trend_agent.py
pattern_agent.py
decision_agent.py

agent_state.py
graph_setup.py
trading_graph.py

benchmark/
```

重点学习：

```text
tool-derived quantitative evidence

specialized technical agents

state-based orchestration
```

但不要复制：

```text
LONG / SHORT final LLM decision
```

---

## FinAgent

重点：

```text
finagent/data/fetcher.py

finagent/memory/store.py

finagent/modules/
    market_intelligence.py
    low_level_reflection.py
    high_level_reflection.py
    decision_making.py

finagent/portfolio/portfolio.py
```

主要学习：

```text
memory separation

retrieval

reflection loop

portfolio feedback
```

---

## Qlib

重点理解：

```text
Forecast Model
→ Signal

Portfolio Strategy
→ Position

Executor
→ Trade

Recorder
→ Evaluation
```

重点参考：

```text
qlib.contrib.strategy
qlib.backtest
Recorder workflow
portfolio analysis
```

我们也保持：

```text
AgentSignal
≠
Portfolio
≠
Execution
```

---

## StockBench

重点：

```text
stockbench/agents/
stockbench/backtest/
stockbench/adapters/
stockbench/apps/
```

主要学习：

```text
standardized agent execution

continuous trading benchmark

uniform evaluation
```

---

## FinRL

后期参考：

```text
finrl/meta/env_portfolio_allocation/
finrl/meta/env_stock_trading/
finrl/agents/stablebaseline3/
finrl/applications/portfolio_allocation/
```

用于：

```text
PPO
SAC
A2C
```

baseline。

---

# 51. 开发顺序必须体现系统精髓

不再采用：

```text
Phase 1:
Data

Phase 2:
Data

Phase 3:
Features

...
Phase 9:
终于开始Agent
```

而采用：

# Milestone 1 — Core Multi-Agent Vertical Slice

第一版本直接打通：

```text
Local Market Data
      ↓
MarketContext
      ↓
Technical Agent
Factor Agent
Liquidity Agent
      ↓
AgentSignal
      ↓
Reliability
      ↓
Adaptive Gating
      ↓
Combined Alpha
      ↓
Portfolio Optimizer
      ↓
Backtest
```

这一步就必须出现：

```text
Agent
Reliability
Gating
Portfolio
```

让工程从第一天就体现项目主题。

数据检查只是其中一个 supporting task。

---

# 52. Milestone 1 最低实现

实现三个真实 Agent：

```text
Technical
Factor
Liquidity
```

不做假 stub。

同时实现：

```text
AgentSignal

PerformanceRecord

ReliabilityTracker

EqualWeightGating

AdaptiveGating

CombinedAlpha

simple PortfolioOptimizer

Backtest Loop
```

第一批实验：

```text
Technical only

Factor only

Liquidity only

Equal-weight MAS

Adaptive MAS
```

做到这里，Codex 就应该已经完全理解：

> 项目研究的是 Multi-Agent aggregation，而不是 CSV。

---

# 53. Milestone 2 — Regime + Risk

增加：

```text
MarketRegime

Regime-specific Reliability

Regime-aware Gating

CVaR

Cost

Turnover
```

形成真正 MVP：

```text
Risk-Aware Adaptive Multi-Agent Portfolio System
```

---

# 54. Milestone 3 — Heterogeneous LLM Agents

再补数据并增加：

```text
Fundamental Agent

Announcement/Event Agent
```

使系统从：

```text
Quant Multi-Agent
```

升级为：

```text
Heterogeneous Multi-Agent
```

---

# 55. Milestone 4 — Debate

增加：

```text
Bull
Bear
Critic
```

测试：

```text
Debate
→ calibration 是否改善
```

而不是：

```text
Debate
→ 文字是不是更精彩
```

---

# 56. Milestone 5 — News / Macro / Memory

增加：

```text
News Agent

Macro Agent

Episodic Memory

Semantic Memory
```

Performance Memory 从早期就存在。

---

# 57. Milestone 6 — Full Evaluation

加入：

```text
Traditional quant baselines

Single Agent baselines

Static MAS

Adaptive MAS

Regime MAS

Debate MAS

Risk-Aware MAS

RL baseline

Ablation
```

---

# 58. Codex 不应该做的事

不要把项目实现成：

```text
data/
features/
models/
backtest/
```

然后把 Agent 当最后一个附加模块。

也不要实现：

```text
TechnicalAgent:
    return "BUY"
```

然后：

```text
vote(BUY, BUY, SELL)
```

更不要：

```text
LLM PortfolioManager
```

直接决定全部仓位。

---

# 59. Codex 应该理解的核心对象关系

```text
MarketContext
     │
     ├──── Agent ───→ AgentSignal
     ├──── Agent ───→ AgentSignal
     ├──── Agent ───→ AgentSignal
     │
     ▼
ChallengeResult
     │
     ▼
CalibratedSignal
     │
     ▼
ReliabilitySnapshot
     │
     +
MarketRegime
     │
     ▼
GatingWeights
     │
     ▼
CombinedAlpha
     │
     ▼
TargetPortfolio
     │
     ▼
ExecutionResult
     │
     ▼
AgentPerformanceRecord
```

这就是整个项目的数据模型主链。

---

# 60. 项目真正的反馈闭环

最后必须存在：

```text
         ┌───────────────────────┐
         │                       │
         ▼                       │
Agent Prediction                 │
         │                       │
         ▼                       │
Adaptive Fusion                  │
         │                       │
         ▼                       │
Portfolio                        │
         │                       │
         ▼                       │
Realized Market Outcome          │
         │                       │
         ▼                       │
Attribution                      │
         │                       │
         ▼                       │
Agent Reliability ───────────────┘
```

没有这个闭环：

> 就不是 Adaptive Multi-Agent。

---

# 61. 项目最重要的三个实验假设

## H1

```text
Specialized MAS
>
Single Agent
```

---

## H2

```text
Adaptive Reliability Gating
>
Equal Weight / Static Vote
```

---

## H3

```text
Regime-Aware Risk MAS
>
Non-regime Adaptive MAS
```

整个项目围绕这三个问题设计。

---

# 62. 项目一句话架构总结

最终架构不是：

```text
Data
→ LLM
→ BUY
```

而是：

```text
Point-in-Time Financial Context

        ↓

Independent Specialized Alpha Agents

        ↓

Debate-driven Uncertainty Discovery

        ↓

Historical Performance Calibration

        ↓

Regime-Aware Agent Reliability

        ↓

Adaptive Mixture-of-Experts Gating

        ↓

Expected Return Vector

        ↓

CVaR / Cost / Turnover Constrained
Portfolio Optimization

        ↓

Continuous Backtest

        ↓

Outcome Attribution

        ↓

Performance Memory

        ↓

Next-round Adaptive Weighting
```

---

# 63. 最终系统的精髓

用一句工程语言描述：

> **这是一个金融 Multi-Agent Mixture-of-Experts 系统：多个异构专业 Agent 独立产生 Alpha，系统不是让 LLM 投票，而是通过市场状态、历史预测质量、置信度校准与不确定性学习 Agent 的时变 gating 权重，再把融合后的 Alpha 交给独立的风险约束 Portfolio Optimizer；真实结果继续反馈到 Agent Performance Memory，使下一轮 gating 自动变化。**

这必须成为 Codex 对整个项目的第一理解。

如果它实现的代码不能体现：

```text
Independent Agents

Performance Feedback

Dynamic Reliability

Adaptive Gating

Portfolio Separation
```

那么即使有十个 Agent，也没有实现本项目。