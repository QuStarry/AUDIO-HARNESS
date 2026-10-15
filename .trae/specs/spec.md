# AUDIO-HARNESS 最小原型验证 - 产品需求文档

## Overview
- **Summary**: 实现「无损音频记忆 + 主动回放」Harness 框架，在长音频推理场景下验证核心假设：无需微调基础音频模型，仅通过外部 Harness 层提供无损原始音频记忆与模型自主的主动回放工具，即可显著提升长音频推理准确率。
- **Purpose**: 用最小成本验证 Harness 范式在听觉领域是否成立，为后续研究投入提供决策依据。对标视觉领域 VISTA 框架。
- **Target Users**: 音频大模型研究工程师，通过本地 Mock 调试 + 远程 A800 GPU 全量实验的方式完成验证。

## Goals
1. 实现完整的 Harness 核心组件：无损音频记忆库、inspect 主动回放工具、双笔记状态机制、端到端运行流水线。
2. 实现三组实验配置（B1 原生基线、B2 分段摘要基线、H1 Harness 实验组），唯一变量为历史信息承载形式。
3. 在 LongAudioSpan 基准上完成全量评测，输出分长度、分任务、分位置的性能对比与效率开销分析。
4. 本地可调试（Mock 模型 + 真实音频），远程可执行（AF3 全量推理），代码模块化可复制粘贴部署。

## Non-Goals
1. 不引入慢放（slow_down）、分轨（separate_track）、读采样值（read_samples）等扩展工具。
2. 不进行模型微调或权重修改，所有能力完全通过外部 Harness 层实现。
3. 不实现知识蒸馏等后续扩展方向。
4. 不构建 ChronosAudio 数据集（未公开发布），使用 LongAudioSpan 替代。

## Background & Context
- **基础模型**: Audio Flamingo 3 (AF3)，HuggingFace ID `nvidia/audio-flamingo-3-hf`，8.3B 参数，10 分钟音频硬限制（30s × 20 windows），32.8K 文本上下文。通过 `AudioFlamingo3ForConditionalGeneration` + `AutoProcessor.apply_chat_template` 调用，支持多轮对话中嵌入音频。
- **测试基准**: LongAudioSpan（HuggingFace: `holvan/LongAudioSpan`），音频时长 10 分钟~2 小时，3240 道题，3 个认知层级（perception / understanding / reasoning）。替代未公开发布的 ChronosAudio。
- **远程环境**: 8 卡 A800 GPU，仅通过内网 web VS Code 复制粘贴代码运行，无 SSH/rsync。代码需模块化、可单文件复制。
- **本地调试**: 无 GPU，用 Mock 模型模拟 AF3 行为，真实测试分段/记忆/工具调用/上下文管理等核心逻辑。
- **对标**: VISTA (Han et al., 2026, MIT) 视觉 Harness 框架的听觉领域迁移。

## Functional Requirements

### 核心组件
- **FR-1 音频分段器**: 将输入音频按 30 秒分段，每段导出为 FLAC 无损格式，记录分段 ID 与起止毫秒时间戳。
- **FR-2 无损音频记忆库**: 基于分段 ID + 毫秒级时间戳的双层索引，支持任意时间区间精确寻址（跨分段自动拼接）；任务级持久化，不调用时不占用上下文。
- **FR-3 inspect 主动回放工具**: 向模型暴露 `inspect(start_time: float, end_time: float) -> None` 接口；内部从记忆库提取片段 → 重新输入 AF3 编码器 → 编码结果追加到当前对话上下文。
- **FR-4 双笔记状态机制**: `WORKING.md` 记录关键信息/待确认问题/推理进度；上下文接近 28K token 时触发交接摘要写入并重置上下文；记忆库不受重置影响。
- **FR-5 端到端流水线**: 分段输入 → 模型推理 → 按需 inspect 回放 → 中间/最终答案 → 上下文超限则交接重置。

### 实验配置
- **FR-6 B1 原生基线**: ≤10min 音频完整直接输入；>10min 截断前 10min。无工具、无记忆库。
- **FR-7 B2 分段摘要基线**: 音频按 30s 分段，每段生成文本摘要，用摘要串联推理。无 inspect 工具。
- **FR-8 H1 Harness 实验组**: 30s 分段输入 + 无损记忆库 + inspect 工具。模型自主决定是否回放。

### 评测与分析
- **FR-9 核心评测指标**: 客观题 Accuracy / F1；摘要题 ROUGE-L / BERTScore。
- **FR-10 分层分析**: 分长度（短/中/长）、分任务类型、分答案位置（前/中/后）统计。
- **FR-11 效率开销**: 平均每题 token 数、推理时间、工具调用次数与分布。

### 工程化
- **FR-12 Mock 模型后端**: 模拟 AF3 的对话接口，返回可控的预设回复，用于本地无 GPU 调试流水线逻辑。
- **FR-13 基准适配器**: LongAudioSpan 数据加载与格式转换，提供统一的题目/音频/答案接口。
- **FR-14 远程部署支持**: 代码模块化、单文件可复制粘贴，附带远程运行指南（环境搭建、模型下载、执行命令）。

## Non-Functional Requirements
- **NFR-1 零微调约束**: 所有能力通过外部 Harness 层实现，禁止修改模型权重或插入内部模块。
- **NFR-2 变量控制**: 三组实验使用同一模型、同一基准、同一提示词模板、同一 token 预算与时间上限。
- **NFR-3 本地可调试**: 无需 GPU 即可跑通完整流水线（Mock 模型 + 真实短音频）。
- **NFR-4 远程可执行**: 代码可通过 web VS Code 复制粘贴运行，不依赖 git/rsync 等同步工具。
- **NFR-5 可复现性**: 固定随机种子，实验日志完整记录每题输入/输出/工具调用。
- **NFR-6 内存效率**: 记忆库按需加载，未调用的分段不驻留内存；及时释放不再需要的张量。

## Constraints
- **Technical**: AF3 10 分钟音频硬限制；32.8K 文本上下文窗口；本地无 GPU；远程仅 web VS Code 访问。
- **Business**: 严格不微调；仅验证核心假设，不扩展能力。
- **Dependencies**: transformers>=4.45, librosa, soundfile, pydub, datasets, rouge-score, bert-score; AF3 模型权重需在远程下载。

## Assumptions
1. LongAudioSpan 数据集可通过 HuggingFace `datasets` 库直接加载。
2. AF3 模型 `nvidia/audio-flamingo-3-hf` 可在远程机器通过 HuggingFace 下载。
3. 远程机器已安装 Python 3.10+、CUDA、PyTorch。
4. AF3 的 `apply_chat_template` 支持在多轮对话的任意轮次中插入音频（用于实现 inspect 回放）。
5. 用户可在远程 web VS Code 中手动执行命令并将输出反馈给我。

## Acceptance Criteria

### AC-1: 音频分段与无损存储
- **Type**: `rule`
- **Given**: 一段任意时长的音频文件
- **When**: 运行音频分段器
- **Then**: 音频被切分为 30 秒片段（最后一段不足 30s 保留剩余），每段保存为 FLAC 格式，且可通过分段 ID 和毫秒时间戳精确检索任意时间区间
- **Pass Condition**: 分段后所有片段拼接还原的音频与原始音频波形完全一致（FLAC 无损），时间戳索引误差 < 10ms
- **Evidence**: 单元测试断言 + 波形对比脚本输出

### AC-2: inspect 工具回放与上下文追加
- **Type**: `rule`
- **Given**: 模型调用 `inspect(start_time, end_time)` 且记忆库中存在对应音频
- **When**: 工具执行
- **Then**: 对应时间段的原始音频被提取、重新编码，并作为新的用户消息追加到当前对话上下文
- **Pass Condition**: 回放的音频内容与记忆库中存储的原始片段完全一致；对话历史中新增一条含音频的消息
- **Evidence**: Mock 模型对话历史断言 + 音频片段哈希校验

### AC-3: 双笔记状态交接
- **Type**: `rule`
- **Given**: 当前对话上下文 token 数接近 28K
- **When**: 触发上下文重置
- **Then**: 模型生成交接摘要写入 WORKING.md，上下文重置后从 WORKING.md 恢复关键状态继续推理
- **Pass Condition**: 重置后 WORKING.md 非空且包含推理进度信息；后续推理可基于交接内容继续
- **Evidence**: 模拟上下文超限场景的集成测试输出

### AC-4: 三组实验配置可独立运行
- **Type**: `rule`
- **Given**: 同一测试题目
- **When**: 分别运行 B1、B2、H1 配置
- **Then**: 三种配置均能完成推理并输出答案，且行为符合各自定义（B1 无回放、B2 用摘要、H1 可调用 inspect）
- **Pass Condition**: B1 不调用 inspect；B2 输出包含分段摘要；H1 至少在信息不足时调用 inspect
- **Evidence**: 三组配置的运行日志对比

### AC-5: 评测指标计算正确
- **Type**: `rule`
- **Given**: 一组模型预测与参考答案
- **When**: 运行评测模块
- **Then**: 正确计算 Accuracy、F1、ROUGE-L、BERTScore
- **Pass Condition**: 与手动计算的预期值一致（单元测试覆盖）
- **Evidence**: 单元测试断言输出

### AC-6: 本地 Mock 调试跑通全流程
- **Type**: `rule`
- **Given**: 本地无 GPU 环境 + 一段短音频 + Mock 模型
- **When**: 运行 H1 配置端到端流水线
- **Then**: 完整执行分段→记忆→推理→inspect 回放→答案输出的全流程，无报错
- **Pass Condition**: 流水线无异常退出，输出结构化结果（含工具调用记录）
- **Evidence**: 本地执行成功的命令输出

### AC-7: 远程 AF3 推理可执行
- **Type**: `rule`
- **Given**: 远程 A800 机器已下载 AF3 模型与 LongAudioSpan 数据
- **When**: 用户按指南复制代码并执行
- **Then**: AF3 模型可正常加载并完成至少一条长音频题目的推理
- **Pass Condition**: 模型推理无 OOM、无报错，输出合理答案
- **Evidence**: 远程执行成功的命令输出（由用户提供）

### AC-8: 分层分析输出完整
- **Type**: `rule`
- **Given**: 三组配置的全量实验结果
- **When**: 运行分析模块
- **Then**: 生成分长度、分任务、分位置的性能对比表 + 效率开销统计
- **Pass Condition**: 输出包含所有要求的分析维度，数据可追溯到原始日志
- **Evidence**: 分析报告 CSV/Markdown 输出

### AC-9: 核心假设验证结论明确
- **Type**: `rubric`
- **Dimension**: 实验结论的明确性与数据支撑充分性
- **Scale**: 1-5
- **Anchors**: 1 = 数据缺失无法得出结论；3 = 有部分数据但结论不明确；5 = 全量数据完整，明确给出假设成立/不成立结论
- **Pass Threshold**: >= 4
- **Evidence**: 实验报告中的增益对比表与结论段落

### AC-10: 代码模块化与可复制部署性
- **Type**: `rubric`
- **Dimension**: 代码的模块化程度与远程复制粘贴部署的便利性
- **Scale**: 1-5
- **Anchors**: 1 = 代码耦合严重无法拆分复制；3 = 基本可拆分但依赖复杂；5 = 模块清晰、单文件可复制、附带完整远程运行指南
- **Pass Threshold**: >= 4
- **Evidence**: 代码目录结构 + 远程运行指南文档

## Open Questions
- [ ] LongAudioSpan 数据集的具体字段结构（题目格式、答案类型）需在加载后确认，适配器需据此调整。
- [ ] 远程机器的具体 CUDA/PyTorch 版本，可能影响 AF3 的 `device_map` 配置。
- [ ] AF3 对多轮对话中多段音频的 10 分钟限制是单轮还是全局，需在远程验证。
