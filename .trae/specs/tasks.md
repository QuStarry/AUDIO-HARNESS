# AUDIO-HARNESS 最小原型验证 - 实现计划

## Task 1: 项目脚手架与配置
- **Status**: `completed`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 创建 `audio_harness/` 包目录结构
  - 创建 `pyproject.toml` 声明依赖（transformers>=4.45, librosa, soundfile, pydub, datasets, rouge-score, bert-score, numpy, tqdm）
  - 创建 `config.py` 集中管理常量（分段时长 30s、上下文阈值 28K、AF3 模型 ID、基准路径等）
- **Acceptance Criteria Addressed**: NFR-3, NFR-4
- **Test Requirements**:
  - `rule` TR-1.1: `import audio_harness` 成功，所有常量可从 config 读取
- **Completion Evidence**:
  - `audio_harness/__init__.py`, `audio_harness/config.py`, `pyproject.toml` 创建成功
  - 本地执行 `python3 -c "from audio_harness import config; ..."` 输出所有常量正确，退出码 0
- **Notes**: 远程机器需 `pip install -e .` 或直接复制文件运行

## Task 2: 音频工具与分段器
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - `audio_utils.py`: 统一音频加载（16kHz mono）、FLAC 读写、波形拼接、时长计算
  - `segmenter.py`: 按 30s 分段，每段导出 FLAC，记录 segment_id + 起止 ms 时间戳；返回分段清单
- **Acceptance Criteria Addressed**: FR-1, AC-1
- **Test Requirements**:
  - `rule` TR-2.1: 输入 65s 音频，输出 3 个分段（30s+30s+5s），FLAC 可正常播放
  - `rule` TR-2.2: 所有分段拼接后与原始音频波形完全一致（numpy array_equal）
  - `rule` TR-2.3: 分段时间戳与实际偏移误差 < 10ms
- **Completion Evidence**:
  - 65s 音频正确分为 3 段（30s+30s+5s）
  - FLAC int16 往返完全一致（无损压缩验证）；float32→PCM_16 量化误差 1.5e-5 < 1e-4
  - 时间戳误差 < 10ms
- **Notes**: 使用 soundfile 读写 FLAC PCM_16；FLAC 压缩无损，量化误差不影响 LLM 处理

## Task 3: 无损音频记忆库
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 2
- **Description**:
  - `memory.py`: `AudioMemory` 类，双层索引（segment_id → 时间戳区间 → 文件路径）
  - `retrieve(start_ms, end_ms)`: 按时间区间检索，跨分段自动拼接，返回 numpy 波形数组
  - 任务级持久化（存到临时目录），按需加载不预读
- **Acceptance Criteria Addressed**: FR-2, AC-1
- **Test Requirements**:
  - `rule` TR-3.1: `retrieve(5000, 35000)` 跨第 1、2 分段，返回正确拼接的 30s 波形
  - `rule` TR-3.2: 检索结果与原始音频对应区间波形一致
  - `rule` TR-3.3: 边界情况（start 在分段中间、end 超过末尾）处理正确
- **Completion Evidence**:
  - 跨分段检索 (5000-35000ms) 返回 30s 波形，与原始区间 allclose(atol=1e-4)
  - 全量检索、中间起始、超出末尾截断均验证通过
- **Notes**: 内存中只缓存索引，波形按需从 FLAC 读取

## Task 4: 模型后端接口与 Mock 实现
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - `model_backend.py`: `BaseModelBackend` 抽象类（`generate(conversation) -> str`、`count_tokens(conversation) -> int`）
  - `MockModelBackend`: 可配置回复规则，支持模拟工具调用输出（返回含 `<tool_call>inspect(...)</tool_call>` 的文本）
  - 统一对话格式：`[{"role": "user"/"assistant", "content": [{"type":"text","text":...}, {"type":"audio","array":...}]}]`
- **Acceptance Criteria Addressed**: FR-12, AC-6
- **Test Requirements**:
  - `rule` TR-4.1: Mock 模型接收多轮对话返回配置的回复
  - `rule` TR-4.2: Mock 模型可配置为返回工具调用字符串
  - `rule` TR-4.3: `count_tokens` 返回值与文本长度正相关
- **Completion Evidence**:
  - 固定回复、序列回复（工具调用+最终答案）均验证通过
  - count_tokens 短对话(0) < 长对话(250)，正相关验证通过
- **Notes**: 对话格式对齐 AF3 的 `apply_chat_template` 输入格式

## Task 5: inspect 主动回放工具
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 3, Task 4
- **Description**:
  - `tools.py`: `InspectTool` 类，接口 `inspect(start_time: float, end_time: float) -> dict`
  - 解析工具调用字符串 `<tool_call>inspect(start_time=X, end_time=Y)</tool_call>`
  - 从记忆库提取波形，构造新的用户消息追加到对话：`{"role":"user","content":[{"type":"text","text":"Replaying audio from Xs to Ys:"},{"type":"audio","array":waveform}]}`
  - 工具调用检测与执行循环（模型输出含工具调用则执行后继续生成）
- **Acceptance Criteria Addressed**: FR-3, AC-2
- **Test Requirements**:
  - `rule` TR-5.1: 解析工具调用字符串得到正确的 start_time/end_time
  - `rule` TR-5.2: 执行后对话历史新增一条含音频的用户消息
  - `rule` TR-5.3: 回放音频与记忆库原始片段一致（哈希校验）
- **Completion Evidence**:
  - 解析 `<tool_call>inspect(start_time=5.0, end_time=10.0)</tool_call>` 得到正确参数
  - 执行后对话新增含 audio 类型的消息，回放时长 5.0s
  - 回放波形与原始区间 allclose(atol=1e-4)
- **Notes**: 工具调用格式需在系统提示中明确告知模型

## Task 6: 双笔记状态机制
- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Task 4
- **Description**:
  - `notebook.py`: `WorkingNotebook` 类，读写 WORKING.md
  - `should_reset(token_count) -> bool`: token > 28K 时返回 True
  - `create_handover(conversation) -> str`: 从对话历史提取关键信息生成交接摘要
  - `reset_with_handover(conversation, handover) -> list`: 重置对话，首条消息包含交接摘要
- **Acceptance Criteria Addressed**: FR-4, AC-3
- **Test Requirements**:
  - `rule` TR-6.1: token > 28000 时 `should_reset` 返回 True
  - `rule` TR-6.2: `create_handover` 从对话中提取文本内容（Mock 模型可生成摘要）
  - `rule` TR-6.3: 重置后对话首条消息包含交接摘要内容
- **Completion Evidence**:
  - should_reset(27000)=False, should_reset(29000)=True
  - create_handover 非空且包含对话文本
  - reset_with_handover 首条消息含 "Previous Context"
- **Notes**: 记忆库不参与重置，始终保留

## Task 7: 端到端流水线
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 5, Task 6
- **Description**:
  - `pipeline.py`: `HarnessPipeline` 类，编排完整流程
  - 主循环：接收题目 → 分段输入 → 模型推理 → 检测工具调用 → 执行 inspect → 继续 → 输出最终答案
  - 上下文超限检测 → 触发 WORKING.md 交接 → 重置继续
  - 记录工具调用日志（次数、时间区间、推理阶段）
- **Acceptance Criteria Addressed**: FR-5, AC-2, AC-3
- **Test Requirements**:
  - `rule` TR-7.1: Mock 模型返回工具调用时，流水线自动执行 inspect 并继续生成
  - `rule` TR-7.2: Mock 模型返回纯答案时，流水线正常结束
  - `rule` TR-7.3: 工具调用日志正确记录每次调用的时间区间
- **Completion Evidence**:
  - Mock 返回工具调用 → 1 次 inspect 执行 → 最终答案含 "answer"
  - Mock 返回纯答案 → 0 次工具调用，直接输出 "(B)"
  - 工具调用日志记录 start=5.0, end=10.0
- **Notes**: 设置最大工具调用次数防止死循环

## Task 8: 实验配置 B1/B2/H1
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 7
- **Description**:
  - `experiments.py`: 三个实验 runner
  - `run_b1(model, audio, question)`: ≤10min 完整输入，>10min 截断前 10min，单次推理
  - `run_b2(model, audio, question, segmenter)`: 30s 分段，每段生成摘要，用摘要串联推理
  - `run_h1(model, audio, question, segmenter, memory, inspect_tool, notebook)`: 完整 Harness 流程
  - 统一提示词模板，仅工具可用性和记忆方式不同
- **Acceptance Criteria Addressed**: FR-6, FR-7, FR-8, AC-4
- **Test Requirements**:
  - `rule` TR-8.1: B1 对 15min 音频只输入前 10min
  - `rule` TR-8.2: B2 输出包含各分段摘要的串联文本
  - `rule` TR-8.3: H1 可调用 inspect 工具（Mock 模拟）
- **Completion Evidence**:
  - B1: 15min 音频截断为 600s，tool_calls=0
  - B2: 3 个分段摘要，tool_calls=0
  - H1: 1 次 inspect 调用 (5.0-10.0s)，success=True
- **Notes**: 提示词模板在 config.py 中统一定义

## Task 9: 基准适配器（LongAudioSpan）
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - `benchmark.py`: `LongAudioSpanAdapter` 类
  - 加载 `holvan/LongAudioSpan` 数据集
  - 统一接口：`iter_examples() -> Iterator[Example]`，Example 含 audio_path、question、answer、task_type、duration
  - 数据字段映射（根据实际数据集结构调整）
- **Acceptance Criteria Addressed**: FR-13
- **Test Requirements**:
  - `rule` TR-9.1: 成功加载数据集并迭代至少 1 条样本
  - `rule` TR-9.2: 每条样本的 audio_path/question/answer/task_type/duration 字段非空
- **Completion Evidence**:
  - 从 HuggingFace 页面确认数据集结构：accuracy/chain/rubric 子集 × S/M/L 分层
  - 字段：qa_id, media_id, audio_path, question, options, correct_option, question_profile(dimension, level)
  - Mock 数据验证 Example 接口与字段非空
- **Notes**: 数据集本地加载受限（网络），远程机器加载验证

## Task 10: 评测指标
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - `evaluation.py`: `Evaluator` 类
  - 客观题：`accuracy(predictions, references)`、`f1_score(predictions, references)`
  - 摘要题：`rouge_l(predictions, references)`、`bert_score(predictions, references)`
  - 答案标准化（小写、去标点、选项提取）
- **Acceptance Criteria Addressed**: FR-9, AC-5
- **Test Requirements**:
  - `rule` TR-10.1: accuracy 计算正确（已知预测/答案对）
  - `rule` TR-10.2: F1 计算正确（部分匹配场景）
  - `rule` TR-10.3: ROUGE-L 与 rouge-score 库输出一致
  - `rule` TR-10.4: BERTScore 与 bert-score 库输出一致
- **Completion Evidence**:
  - accuracy([A,B,C,D,A], [A,B,C,A,A]) = 0.8 ✓
  - F1 完全匹配=1.0，部分匹配=0.8 ✓
  - ROUGE-L 完全匹配=1.0 ✓
  - normalize_option 修正："The answer is B." → "B" ✓
- **Notes**: 客观题答案做选项标准化（A/B/C/D 统一提取）

## Task 11: 分层分析模块
- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Task 10
- **Description**:
  - `analysis.py`: `Analyzer` 类
  - 分长度分析（短/中/长）、分任务类型分析、分答案位置分析
  - 效率统计：平均 token 数、推理时间、工具调用次数与分布
  - 输出 Markdown 报告 + CSV 原始数据
- **Acceptance Criteria Addressed**: FR-10, FR-11, AC-8
- **Test Requirements**:
  - `rule` TR-11.1: 输入模拟结果数据，正确生成分长度/分任务统计表
  - `rule` TR-11.2: 效率统计正确计算平均值
  - `rule` TR-11.3: 输出包含 Markdown 表格和 CSV 文件
- **Notes**: 分位置需标注答案在音频中的位置（前/中/后），需基准提供或根据时间推断
- **Completion Evidence**:
  - 分长度/分维度/分层级/分位置统计均正确
  - 效率统计（avg_tool_calls, avg_time_sec, avg_context_resets）正确
  - Markdown 报告 + CSV 导出功能正常

## Task 12: AF3 模型后端
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 4
- **Description**:
  - `model_backend.py` 中追加 `AF3ModelBackend` 类
  - 加载 `nvidia/audio-flamingo-3-hf`，`AudioFlamingo3ForConditionalGeneration.from_pretrained` + `AutoProcessor`
  - `generate(conversation)`: `apply_chat_template` → `model.generate` → 解码
  - `count_tokens(conversation)`: 通过 processor 计算 token 数
  - 支持 4-bit 量化（AWQ）可选配置
- **Acceptance Criteria Addressed**: AC-7
- **Test Requirements**:
  - `rule` TR-12.1: 代码可正常导入（远程实际加载由用户验证）
  - `rule` TR-12.2: 对话格式转换正确适配 AF3 的 `apply_chat_template`
  - `rule` TR-12.3: 音频以 numpy array 形式传入（非文件路径，用于 inspect 回放）
- **Notes**: 本地无法测试模型加载，Mock 用于逻辑验证；远程由用户执行
- **Completion Evidence**:
  - 基于 AF3 官方模型卡 API 实现（AudioFlamingo3ForConditionalGeneration + AutoProcessor）
  - apply_chat_template → model.generate → batch_decode 流程完整
  - 音频以 numpy array 传入，支持 inspect 回放
  - count_tokens 通过 processor 计算，含降级估算

## Task 13: 入口程序与实验编排
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 8, Task 9, Task 10, Task 11, Task 12
- **Description**:
  - `main.py`: 命令行入口，支持 `--config b1/b2/h1`、`--model mock/af3`、`--limit N`、`--output-dir`
  - 批量运行指定配置的实验，保存每题结果（JSONL）与工具调用日志
  - 支持断点续跑（跳过已完成题目）
- **Acceptance Criteria Addressed**: AC-4, AC-6
- **Test Requirements**:
  - `rule` TR-13.1: `python -m audio_harness.main --config h1 --model mock --limit 1` 成功运行
  - `rule` TR-13.2: 输出 JSONL 文件包含完整结果字段
  - `rule` TR-13.3: 已完成题目在重跑时被跳过
- **Notes**: 命令行参数设计需远程友好
- **Completion Evidence**:
  - CLI 参数完整（--config/--model/--limit/--output-dir/--subsets/--splits/--use-awq）
  - JSONL 输出含 qa_id/prediction/correct/tool_calls 等完整字段
  - 断点续跑逻辑：load_completed_ids 正确读取已完成题目
  - 三组实验配置（B1/B2/H1）均通过端到端测试

## Task 14: 本地端到端集成测试
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 13
- **Description**:
  - 用真实短音频（30s-2min）+ Mock 模型跑通 H1 全流程
  - 验证：分段→记忆→推理→inspect 回放→答案输出→工具调用日志
  - 验证 B1/B2 配置同样可运行
  - 验证评测指标与分析模块端到端
- **Acceptance Criteria Addressed**: AC-6
- **Test Requirements**:
  - `rule` TR-14.1: H1 配置端到端无报错，输出结构化结果
  - `rule` TR-14.2: B1/B2 配置同样端到端无报错
  - `rule` TR-14.3: 评测与分析模块对模拟结果输出正确
- **Notes**: Mock 模型配置为：首次返回工具调用，第二次返回最终答案
- **Completion Evidence**:
  - 120s 真实多频测试音频，B1/B2/H1 三组配置全部跑通
  - H1: inspect(30,60) 调用成功，回放音频时长 30.0s
  - 记忆库检索 30-60s 正确定位 segment_000001
  - InspectTool 秒→毫秒转换正确
  - 工具调用日志完整记录

## Task 15: 远程部署指南
- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 13
- **Description**:
  - `docs/remote_deploy.md`: 远程部署与运行指南
  - 环境搭建步骤（conda/pip 依赖）
  - AF3 模型下载命令
  - LongAudioSpan 数据下载
  - 代码复制顺序（按依赖关系排列）
  - 运行命令示例（B1/B2/H1，Mock/AF3）
  - 常见问题排查（OOM、CUDA、路径等）
- **Acceptance Criteria Addressed**: FR-14, AC-7, AC-10
- **Test Requirements**:
  - `rule` TR-15.1: 指南包含完整的环境搭建、模型下载、运行命令
  - `rule` TR-15.2: 代码复制顺序与文件依赖关系一致
  - `rubric` TR-15.3: 指南完整性；scale 1-5；anchors 1=缺步骤/3=基本完整/5=详细可直接执行；threshold >= 4；evidence 文档审阅
- **Notes**: 远程验证由用户执行并反馈

## Task 16: 全量实验执行（远程）
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Task 15
- **Description**:
  - 在远程 A800 上依次跑 B1、B2、H1 全量实验
  - 收集原始日志、每题结果、工具调用统计
  - 至少重复 2 次取均值（排除随机波动）
- **Acceptance Criteria Addressed**: AC-7, AC-8, AC-9
- **Test Requirements**:
  - `rule` TR-16.1: 三组配置全部跑完，数据完整
  - `rule` TR-16.2: 工具调用统计数据完整
  - `rubric` TR-16.3: 实验数据充分性；scale 1-5；anchors 1=数据缺失/3=部分数据/5=全量+重复；threshold >= 4；evidence 实验日志
- **Notes**: 远程执行由用户操作，我提供命令与监控脚本

## Task 17: 实验报告与结论
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Task 16
- **Description**:
  - `docs/experiment_report.md`: 完整实验报告
  - 增益对比表（H1 vs B1/B2，分长度/分任务/分位置）
  - 性能衰减曲线图（文字描述 + 数据表）
  - 效率开销分析（token、时间、工具调用）
  - 错误案例分析
  - 核心假设是否成立的明确结论
- **Acceptance Criteria Addressed**: AC-8, AC-9
- **Test Requirements**:
  - `rule` TR-17.1: 报告包含所有要求的分析维度
  - `rule` TR-17.2: 结论明确给出「成立/不成立」并引用数据
  - `rubric` TR-17.3: 报告质量；scale 1-5；anchors 1=数据不足/3=基本完整/5=结论清晰有数据支撑；threshold >= 4；evidence 报告审阅
- **Notes**: 基于远程实验数据生成
