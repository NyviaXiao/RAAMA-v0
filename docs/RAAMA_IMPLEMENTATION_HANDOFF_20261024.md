# Luna 编码与父审核协作契约

用户已明确授权：新建 GPT-6 Luna、极高推理强度编码线程，落实 RAAMA_HARNESS_FINAL_PLAN_20261024.md 的全部阶段和内容；父聊天独立审核 diff/输出、指导修正并介入模拟器测试问题。仅在五小时额度耗尽且恢复后允许自动续跑一次，不使用额度重置卡、不购买额度、不切 API 计费规避额度。

## 工作区和职责

- 实际 Git 仓库：C:\Users\xiao\Desktop\RAAMA\adaptive-mas；父目录 RAAMA 不是 Git 仓库。
- 起点分支 harness/gepa-evolution-20261022，基线 772b4fd20a4cc27342434eb16aec6515c2cad419。父已新增方案和本交接文档。读取 AGENTS.md 与适用更深层规则。
- Luna 是业务代码唯一写入者。父读取/测试/审核，不与 Luna 同时改业务代码。父独占 .tmp/harness_supervision_20261024/supervision_state.json 和 review.md；Luna 独占 implementation_status.json 和 implementation_notes.md，避免覆盖。
- 现存大量未跟踪 experiments/ 和 manifest 是历史/用户文件；不要 git add .、清理、移动、覆盖、上传它们。逐个暂存你新增/修改的文件；父新增的两份 docs 可一起提交。
- 按阶段形成有意义的可审核 commit。遵守每版 GitHub 备份，只推本项目授权改动，不 force、不改远端历史；网络受限交父处理，独立工作继续。
- 用户要求全部落地，不停在方案、脚手架、模拟 UI 或模块孤岛。维护 docs/RAAMA_REQUIREMENTS_TRACEABILITY_20261024.md，逐条列需求、代码入口、验证、实际连接、限制和状态。
- 状态区分 implemented/tested/reviewed；真实未来数据成熟与投资效果单列 pending_forward/scientific_not_proven。已实现可用适配器不等于拥有缺失数据，且缺数据也不免除完成可实现工程。
- 不创建额外编码线程。需要方案例外先陈述具体差异、证据与影响，父有授权指导常规实施选择。

## 实施和验证

首先读取最终方案全部内容，核对 baseline，运行必要基线测试。审阅时两项失败位于 public_expert_overlay：旧测试预期改基础收益，当前实现分离 component 并排除 Qwen direction；先按设计契约判断实现与测试，禁止无依据放宽断言。

每阶段汇报 commit、变更入口、完成需求、真实命令/退出码/时长、失败和下一步，随后继续下一阶段，不为普通里程碑反复停下等确认。父可并行审核已提交阶段。最终全量相关检查一次，新增改动和新失败才重复适当检查。

不得读取 test.csv；CSV 只读，研究仅 train.csv。不得读取/输出/提交密钥与账户私有信息；合成数据测试明确 fixture，不能冒充真实历史实验或 UI 后端。真实模型/provider 调用不突破既有共享预算与出站审查；先以已允许缓存、契约测试和离线运行验证，不启动新付费大实验或真实交易。额外权限/外部资料真正必要时保存具体可审阅请求，由父处理。

## 模拟器/浏览器/应用测试介入机制

本方案目标为本地量化 API/worker/研究工作台，优先真实本地运行与浏览器验证；未要求原生移动端，不要为了“模拟器”安装 Android/iOS 或引入移动框架。

测试先检查依赖、端口/服务与可用控制接口。可用即由 Luna 执行。遇到基础设施不能启动/浏览器连接失败/进程权限问题，最多一次初试和一次有证据的修复后重试，或累计 5 分钟没有新增证据即停止该环境排障，禁止反复盲重试。大型金融回放不是环境失败：先 smoke，实际必要批任务记录进度，使用有超时的后台进程，保留 heartbeat；不要重复启动同一任务。

写 .tmp/harness_supervision_20261024/implementation_status.json：status=needs_environment_help，记录 command、working_directory、exit_code、error_excerpt（脱敏）、log_paths、attempts、elapsed_seconds、expected_behavior、next_independent_work。父将读取并介入；Luna 继续不依赖环境的实现。故障如实际为业务 bug，由 Luna 修复。

父在 review.md 写环境处理结论并发跟进提示。环境恢复后 Luna 继续真实测试。不能把服务 health200 等同全部交互验收；不能隐瞒未跑的 UI 测试。浏览器工具不可用时保存可复现测试和错误，等待父检查，不能用伪造截图替代。

## 额度与恢复

五小时额度与 provider API 预算分开。不要操作重置卡、购买 credits、改计费方式或换账号。正常工作按持久状态持续，不用人为 sleep 五小时。

父独占 supervision_state.json 的 quota_resume_count（初始0，上限1）。额度确实耗尽后记录阻断类型/重置时间；恢复后父先原子记一次恢复，再向本线程发送恢复提示。首次恢复后若再次因额度耗尽而中断，停止自动续跑并等待用户。普通代码评审跟进不是额度恢复。不得另建自动续跑或第二个监控任务。

在可写时持续保存完成阶段/commit/待办/测试进程与下一条动作，使额度中断不用重做已完成工作。父通过 wait_threads、共享状态和 diff 审核；Luna 无需跨线程回信，共享文件即可传递进展。

## 状态文件最小字段

implementation_status.json 建议字段：schema_version、updated_at、status、current_stage、completed_requirements、latest_commit、tests（command/exit_code/elapsed/log）、environment_issue、remaining_requirements、next_action。仅更新自己的状态。needs_review 用于已完成的阶段/最终候选，普通阶段提交后仍继续后续工作；最终候选 ready_for_final_review。

阶段验收完整执行最终方案第12节0–6，各需求有实际入口。父的通过只覆盖已审查范围，不因测试全绿就将所有需求标完成。
