# AUDIO-HARNESS 远程部署与运行指南

本文档面向远程 GPU 机器（8×A800）的部署与实验执行。
由于远程机器仅支持内网 web VS Code 复制粘贴，所有代码需按依赖顺序逐文件复制。

---

## 一、环境搭建

### 1.1 Python 环境

```bash
# 推荐使用 conda
conda create -n audio-harness python=3.10 -y
conda activate audio-harness

# 或使用 venv
python3.10 -m venv ~/venvs/audio-harness
source ~/venvs/audio-harness/bin/activate
```

### 1.2 安装依赖

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.45 accelerate librosa soundfile pydub datasets rouge-score bert-score numpy tqdm

# 可选：4-bit 量化（显存受限时启用）
pip install autoawq

# 可选：Flash Attention 2（加速推理）
pip install flash-attn --no-build-isolation
```

### 1.3 验证环境

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "from transformers import AudioFlamingo3ForConditionalGeneration; print('AF3 import OK')"
```

---

## 二、模型与数据下载

### 2.1 下载 AF3 模型

```bash
# 下载到本地缓存（约 16GB，bfloat16）
huggingface-cli download nvidia/audio-flamingo-3-hf

# 验证模型文件
ls ~/.cache/huggingface/hub/models--nvidia--audio-flamingo-3-hf/
```

### 2.2 下载 LongAudioSpan 数据集

```bash
# 下载数据集
huggingface-cli download holvan/LongAudioSpan --repo-type dataset

# 验证数据集
python -c "from datasets import load_dataset; ds = load_dataset('holvan/LongAudioSpan', 'accuracy', split='S'); print(f'Samples: {len(ds)}'); print(ds[0].keys())"
```

### 2.3 模型快速加载测试

```python
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
import torch

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
print(f"Model loaded. Device: {model.device}")
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
```

---

## 三、代码复制顺序

**重要**：必须按以下顺序复制，因为后续文件依赖前面的模块。

### 步骤 1：创建项目目录

```bash
mkdir -p ~/AUDIO-HARNESS/audio_harness
mkdir -p ~/AUDIO-HARNESS/data
mkdir -p ~/AUDIO-HARNESS/outputs
cd ~/AUDIO-HARNESS
```

### 步骤 2：复制代码文件（按依赖顺序）

| 序号 | 文件 | 依赖 | 说明 |
|------|------|------|------|
| 1 | `audio_harness/__init__.py` | 无 | 包初始化 |
| 2 | `audio_harness/config.py` | 无 | 全局配置常量 |
| 3 | `audio_harness/audio_utils.py` | config | 音频加载/FLAC 读写 |
| 4 | `audio_harness/segmenter.py` | audio_utils | 30s 分段器 |
| 5 | `audio_harness/memory.py` | segmenter | 无损记忆库 |
| 6 | `audio_harness/model_backend.py` | config | 模型后端（Mock + AF3） |
| 7 | `audio_harness/tools.py` | memory | inspect 回放工具 |
| 8 | `audio_harness/notebook.py` | 无 | 双笔记状态机制 |
| 9 | `audio_harness/pipeline.py` | tools, notebook | 端到端流水线 |
| 10 | `audio_harness/experiments.py` | pipeline | B1/B2/H1 实验配置 |
| 11 | `audio_harness/benchmark.py` | 无 | LongAudioSpan 适配器 |
| 12 | `audio_harness/evaluation.py` | 无 | 评测指标 |
| 13 | `audio_harness/analysis.py` | 无 | 分层分析 |
| 14 | `audio_harness/main.py` | 全部 | CLI 入口 |

### 步骤 3：创建 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "audio-harness"
version = "0.1.0"
description = "Lossless audio memory + active replay harness"
requires-python = ">=3.10"
dependencies = [
    "transformers>=4.45", "librosa", "soundfile", "pydub",
    "datasets", "rouge-score", "bert-score", "numpy", "tqdm",
]

[project.optional-dependencies]
quantize = ["autoawq"]

[tool.setuptools.packages.find]
include = ["audio_harness*"]
```

### 步骤 4：验证导入

```bash
cd ~/AUDIO-HARNESS
python -c "from audio_harness import config, audio_utils, segmenter, memory, model_backend, tools, notebook, pipeline, experiments, benchmark, evaluation, analysis, main; print('All modules imported OK')"
```

---

## 四、运行实验

### 4.1 先用 Mock 模型验证流程

```bash
cd ~/AUDIO-HARNESS

# 验证 B1
python -m audio_harness.main --config b1 --model mock --limit 3 --output-dir outputs/test

# 验证 B2
python -m audio_harness.main --config b2 --model mock --limit 3 --output-dir outputs/test

# 验证 H1
python -m audio_harness.main --config h1 --model mock --limit 3 --output-dir outputs/test
```

### 4.2 正式运行（AF3 模型）

```bash
cd ~/AUDIO-HARNESS

# B1 原生基线
python -m audio_harness.main --config b1 --model af3 --output-dir outputs/final

# B2 分段摘要基线
python -m audio_harness.main --config b2 --model af3 --output-dir outputs/final

# H1 Harness 实验组
python -m audio_harness.main --config h1 --model af3 --output-dir outputs/final
```

### 4.3 显存受限时使用 AWQ 量化

```bash
python -m audio_harness.main --config h1 --model af3 --use-awq --output-dir outputs/final
```

### 4.4 指定子集/分层

```bash
# 只跑 accuracy 子集的 S 分层（快速测试）
python -m audio_harness.main --config h1 --model af3 --subsets accuracy --splits S --output-dir outputs/test

# 跑全部子集和分层
python -m audio_harness.main --config h1 --model af3 --subsets accuracy chain rubric --splits S M L --output-dir outputs/final
```

### 4.5 断点续跑

程序自动支持断点续跑。如果运行中断，重新执行相同命令即可跳过已完成的题目。

```bash
# 检查已完成题目数
wc -l outputs/final/results_h1.jsonl

# 重新运行（自动跳过已完成）
python -m audio_harness.main --config h1 --model af3 --output-dir outputs/final
```

---

## 五、输出文件说明

```
outputs/final/
├── results_b1.jsonl      # B1 每题结果
├── results_b2.jsonl      # B2 每题结果
├── results_h1.jsonl      # H1 每题结果
├── results_b1.csv        # B1 CSV 格式
├── results_b2.csv        # B2 CSV 格式
├── results_h1.csv        # H1 CSV 格式
├── report_b1.md          # B1 分析报告
├── report_b2.md          # B2 分析报告
└── report_h1.md          # H1 分析报告
```

### JSONL 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| qa_id | str | 题目 ID |
| config | str | 配置（b1/b2/h1） |
| prediction | str | 模型原始预测文本 |
| predicted_option | str | 提取的选项（A/B/C/D） |
| reference | str | 参考答案选项 |
| correct | bool | 是否正确 |
| duration_band | str | 长度分层（short/mid/long） |
| dimension | str | 任务维度 |
| level | str | 认知层级 |
| tool_calls | int | 工具调用次数 |
| total_time_sec | float | 总推理时间（秒） |
| context_resets | int | 上下文重置次数 |

---

## 六、常见问题排查

### 6.1 CUDA Out of Memory (OOM)

```bash
# 检查 GPU 显存
nvidia-smi

# 解决方案 1：使用 AWQ 4-bit 量化
python -m audio_harness.main --config h1 --model af3 --use-awq

# 解决方案 2：减少并发（单卡运行）
CUDA_VISIBLE_DEVICES=0 python -m audio_harness.main --config h1 --model af3

# 解决方案 3：使用 float16（比 bfloat16 略省显存）
# 修改 model_backend.py 中 dtype="float16"
```

### 6.2 模型加载失败

```bash
# 检查模型缓存
ls ~/.cache/huggingface/hub/ | grep audio-flamingo

# 重新下载
huggingface-cli download nvidia/audio-flamingo-3-hf --resume-download

# 检查磁盘空间
df -h ~/.cache/huggingface/
```

### 6.3 数据集加载失败

```bash
# 检查网络连接
python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('holvan/LongAudioSpan'))"

# 手动下载
huggingface-cli download holvan/LongAudioSpan --repo-type dataset --resume-download
```

### 6.4 音频加载失败

```bash
# 检查音频文件路径
python -c "from datasets import load_dataset; ds = load_dataset('holvan/LongAudioSpan', 'accuracy', split='S'); print(ds[0]['audio_path'])"

# 确保 ffmpeg 已安装
ffmpeg -version
```

### 6.5 显存估算

| 配置 | 显存占用（估算） |
|------|-----------------|
| AF3 bfloat16 | ~17 GB |
| AF3 float16 | ~17 GB |
| AF3 AWQ 4-bit | ~5 GB |

A800 单卡 80GB 可轻松运行 AF3。如需数据并行，可使用多卡。

---

## 七、实验执行检查清单

- [ ] Python 环境创建并激活
- [ ] 所有依赖安装完成
- [ ] CUDA 可用，GPU 数量正确
- [ ] AF3 模型下载完成
- [ ] LongAudioSpan 数据集下载完成
- [ ] 所有代码文件按顺序复制
- [ ] `python -c "from audio_harness import ..."` 导入测试通过
- [ ] Mock 模型三组配置运行通过
- [ ] AF3 单题测试通过（--limit 1）
- [ ] 全量实验 B1/B2/H1 依次运行
- [ ] 输出文件完整（JSONL + CSV + MD）
