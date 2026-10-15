"""
AUDIO-HARNESS 全局配置常量

集中管理所有实验参数，确保三组配置（B1/B2/H1）共享同一套超参数，
唯一变量为「历史信息承载形式」，严格遵循变量控制原则。
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
MEMORY_DIR: Path = PROJECT_ROOT / ".memory_cache"
WORKING_NOTE_NAME: str = "WORKING.md"

for _d in (DATA_DIR, OUTPUT_DIR, MEMORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 音频分段参数
# ============================================================
# 每段音频时长（秒），对齐 AF3 的 30s 窗口
SEGMENT_DURATION_SEC: float = 30.0
# 音频采样率（Hz），AF3 原生 16kHz mono
SAMPLE_RATE: int = 16000
# 音频通道数
NUM_CHANNELS: int = 1

# ============================================================
# 模型参数
# ============================================================
AF3_MODEL_ID: str = "nvidia/audio-flamingo-3-hf"
# AF3 原生最大音频长度（秒）
AF3_MAX_AUDIO_SEC: float = 600.0  # 10 minutes
# AF3 文本上下文窗口（tokens）
AF3_CONTEXT_WINDOW: int = 32768
# 触发上下文重置的 token 阈值（留余量给生成）
CONTEXT_RESET_THRESHOLD: int = 28000

# ============================================================
# 推理参数
# ============================================================
# 最大生成 token 数
MAX_NEW_TOKENS: int = 1024
# 生成温度
TEMPERATURE: float = 0.0
# 单题最大工具调用次数（防死循环）
MAX_TOOL_CALLS: int = 10
# 单题最大推理时间（秒）
MAX_INFERENCE_TIME_SEC: float = 600.0

# ============================================================
# 基准参数
# ============================================================
LONG_AUDIO_SPAN_ID: str = "holvan/LongAudioSpan"
# 长度分层阈值（秒）
SHORT_AUDIO_MAX_SEC: float = 300.0   # < 5 min
MID_AUDIO_MAX_SEC: float = 600.0     # 5 ~ 10 min
# > 10 min 为长音频

# ============================================================
# 工具调用协议
# ============================================================
# 工具调用的 XML 标签格式
TOOL_CALL_OPEN_TAG: str = "<tool_call>"
TOOL_CALL_CLOSE_TAG: str = "</tool_call>"
# inspect 工具的参数名
INSPECT_PARAM_START: str = "start_time"
INSPECT_PARAM_END: str = "end_time"

# ============================================================
# 提示词模板（三组配置共享，仅工具可用性不同）
# ============================================================
SYSTEM_PROMPT_BASE: str = """You are an audio reasoning assistant. You will be given an audio recording and a question about it.

When you are uncertain about specific details, you can use the inspect tool to replay any segment of the original audio and listen again carefully.

Tool usage format: <tool_call>inspect(start_time=X, end_time=Y)</tool_call>
- start_time: replay start time in seconds
- end_time: replay end time in seconds

After replaying, continue your reasoning. When you are confident, provide your final answer."""

# B1/B2 无工具可用时的提示词（移除工具说明）
SYSTEM_PROMPT_NO_TOOL: str = """You are an audio reasoning assistant. You will be given an audio recording and a question about it.

Answer the question based on the audio content. When you are confident, provide your final answer."""

# 分段摘要的指令
SEGMENT_SUMMARY_PROMPT: str = "Summarize the key information in this 30-second audio segment in one concise sentence."

# 最终答案提取提示
FINAL_ANSWER_PROMPT: str = "Based on all the information above, provide your final answer to the question. Output only the answer, no explanation."


def get_env_bool(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔值配置。"""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")
