"""
实验配置：B1 原生基线 / B2 分段摘要基线 / H1 Harness 实验组

三组配置使用同一基础模型、同一提示词模板、同一 token 预算，
唯一变量为「历史信息承载形式」：
  B1: 原生压缩编码（完整输入或截断）
  B2: 文本摘要记忆
  H1: 无损原始波形 + 主动回放
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import audio_utils
from .config import (
    AF3_MAX_AUDIO_SEC,
    FINAL_ANSWER_PROMPT,
    MAX_NEW_TOKENS,
    SEGMENT_DURATION_SEC,
    SEGMENT_SUMMARY_PROMPT,
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_NO_TOOL,
)
from .memory import AudioMemory
from .model_backend import BaseModelBackend, Conversation, Message
from .notebook import WorkingNotebook
from .pipeline import HarnessPipeline, PipelineResult
from .segmenter import AudioSegmenter, SegmentationResult
from .tools import InspectTool


@dataclass
class ExperimentOutput:
    """单题实验输出。"""

    config: str  # "b1" | "b2" | "h1"
    question: str
    prediction: str
    tool_calls: int = 0
    total_time_sec: float = 0.0
    context_resets: int = 0
    extra: dict = None


def _build_system_prompt(use_tools: bool) -> str:
    """根据是否使用工具选择系统提示词。"""
    return SYSTEM_PROMPT_BASE if use_tools else SYSTEM_PROMPT_NO_TOOL


def run_b1(
    model: BaseModelBackend,
    waveform: np.ndarray,
    question: str,
    max_audio_sec: float = AF3_MAX_AUDIO_SEC,
) -> ExperimentOutput:
    """B1 原生基线：≤10min 完整输入，>10min 截断前 10min。

    代表模型原生能力上限，无工具、无记忆库。
    """
    import time

    start = time.time()

    # 截断到 max_audio_sec
    max_samples = int(max_audio_sec * audio_utils.SAMPLE_RATE)
    truncated = waveform[:max_samples]

    # 单次推理
    conversation: Conversation = [
        {"role": "system", "content": [{"type": "text", "text": _build_system_prompt(False)}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "array": truncated},
                {"type": "text", "text": f"\n\nQuestion: {question}\n\n{FINAL_ANSWER_PROMPT}"},
            ],
        },
    ]

    prediction = model.generate(conversation, max_new_tokens=MAX_NEW_TOKENS)

    return ExperimentOutput(
        config="b1",
        question=question,
        prediction=prediction.strip(),
        tool_calls=0,
        total_time_sec=time.time() - start,
        context_resets=0,
        extra={"audio_duration_sec": len(truncated) / audio_utils.SAMPLE_RATE},
    )


def run_b2(
    model: BaseModelBackend,
    waveform: np.ndarray,
    question: str,
    segmenter: Optional[AudioSegmenter] = None,
) -> ExperimentOutput:
    """B2 分段摘要基线：30s 分段 → 每段生成摘要 → 摘要串联推理。

    代表业界主流长音频处理方案，无 inspect 工具。
    """
    import time

    start = time.time()

    if segmenter is None:
        segmenter = AudioSegmenter()

    task_id = f"b2_{int(time.time() * 1000)}"
    seg_result = segmenter.segment(waveform, task_id)

    # 逐段生成摘要
    summaries: list[str] = []
    for seg in seg_result.segments:
        seg_waveform = audio_utils.load_flac(seg.flac_path)
        summary_conv: Conversation = [
            {"role": "system", "content": [{"type": "text", "text": _build_system_prompt(False)}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "array": seg_waveform},
                    {"type": "text", "text": f"\n\n{SEGMENT_SUMMARY_PROMPT}"},
                ],
            },
        ]
        summary = model.generate(summary_conv, max_new_tokens=256).strip()
        summaries.append(f"[Segment {seg.segment_id} ({seg.start_ms/1000:.0f}s-{seg.end_ms/1000:.0f}s)]: {summary}")

    # 用摘要串联推理
    full_summary = "\n".join(summaries)
    reasoning_conv: Conversation = [
        {"role": "system", "content": [{"type": "text", "text": _build_system_prompt(False)}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here are summaries of audio segments:\n\n{full_summary}\n\nQuestion: {question}\n\n{FINAL_ANSWER_PROMPT}",
                }
            ],
        },
    ]

    prediction = model.generate(reasoning_conv, max_new_tokens=MAX_NEW_TOKENS)

    return ExperimentOutput(
        config="b2",
        question=question,
        prediction=prediction.strip(),
        tool_calls=0,
        total_time_sec=time.time() - start,
        context_resets=0,
        extra={"num_segments": len(seg_result.segments), "summaries": summaries},
    )


def run_h1(
    model: BaseModelBackend,
    waveform: np.ndarray,
    question: str,
    segmenter: Optional[AudioSegmenter] = None,
) -> ExperimentOutput:
    """H1 Harness 实验组：30s 分段 + 无损记忆库 + inspect 工具。

    模型自主决定是否回放，代表本研究提出的 Harness 方案。
    """
    import time

    start = time.time()

    if segmenter is None:
        segmenter = AudioSegmenter()

    task_id = f"h1_{int(time.time() * 1000)}"
    seg_result = segmenter.segment(waveform, task_id)

    # 构建记忆库与工具
    memory = AudioMemory(seg_result)
    inspect_tool = InspectTool(memory)
    notebook = WorkingNotebook(segmenter.output_dir / task_id)

    # 加载分段波形
    audio_segments = [audio_utils.load_flac(seg.flac_path) for seg in seg_result.segments]

    # 运行 Harness 流水线
    pipeline = HarnessPipeline(
        model=model,
        memory=memory,
        inspect_tool=inspect_tool,
        notebook=notebook,
    )

    result: PipelineResult = pipeline.run(
        question=question,
        audio_segments=audio_segments,
        system_prompt=_build_system_prompt(True),
    )

    return ExperimentOutput(
        config="h1",
        question=question,
        prediction=result.final_answer,
        tool_calls=result.num_tool_calls,
        total_time_sec=result.total_time_sec,
        context_resets=result.context_resets,
        extra={
            "num_segments": len(seg_result.segments),
            "tool_call_logs": [
                {"start": c.start_time, "end": c.end_time, "success": c.success}
                for c in result.tool_calls
            ],
        },
    )
