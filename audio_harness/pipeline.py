"""
端到端运行流水线

编排完整的 Harness 推理流程：
    音频分段输入 → 模型接收当前片段 → 更新推理状态
           ↓
    信息不足？→ 调用 inspect 工具 → 提取原始音频重编码 → 修正推理
           ↓
    生成中间答案 / 最终答案
           ↓
    上下文接近上限 → 写入 WORKING.md → 重置上下文 → 继续
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import (
    FINAL_ANSWER_PROMPT,
    MAX_INFERENCE_TIME_SEC,
    MAX_NEW_TOKENS,
    MAX_TOOL_CALLS,
    SYSTEM_PROMPT_BASE,
)
from .memory import AudioMemory
from .model_backend import BaseModelBackend, Conversation, Message
from .notebook import WorkingNotebook
from .tools import InspectTool, has_tool_call, parse_tool_calls


@dataclass
class ToolCallLog:
    """单次工具调用的日志记录。"""

    call_index: int
    start_time: float
    end_time: float
    success: bool
    duration_sec: float
    stage: str  # 推理阶段标识


@dataclass
class PipelineResult:
    """流水线执行结果。"""

    final_answer: str
    conversation: Conversation
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    total_time_sec: float = 0.0
    num_tool_calls: int = 0
    context_resets: int = 0


class HarnessPipeline:
    """Harness 端到端流水线。

    协调模型推理、工具调用、上下文重置的完整流程。
    """

    def __init__(
        self,
        model: BaseModelBackend,
        memory: AudioMemory,
        inspect_tool: InspectTool,
        notebook: WorkingNotebook,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_new_tokens: int = MAX_NEW_TOKENS,
        max_time_sec: float = MAX_INFERENCE_TIME_SEC,
    ):
        self._model = model
        self._memory = memory
        self._inspect_tool = inspect_tool
        self._notebook = notebook
        self._max_tool_calls = max_tool_calls
        self._max_new_tokens = max_new_tokens
        self._max_time_sec = max_time_sec

    def run(
        self,
        question: str,
        audio_segments: list[np.ndarray],
        system_prompt: str = SYSTEM_PROMPT_BASE,
    ) -> PipelineResult:
        """执行完整的 Harness 推理流程。

        Args:
            question: 问题文本
            audio_segments: 音频分段波形列表
            system_prompt: 系统提示词

        Returns:
            PipelineResult 包含最终答案与执行日志
        """
        start_time = time.time()
        tool_calls: list[ToolCallLog] = []
        context_resets = 0

        # 构建初始对话：系统提示 + 所有音频分段 + 问题
        conversation: Conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        ]

        # 将所有音频分段作为用户消息加入（模拟分段输入）
        audio_content = []
        for i, seg_waveform in enumerate(audio_segments):
            audio_content.append(
                {
                    "type": "text",
                    "text": f"[Audio segment {i + 1}/{len(audio_segments)}]:",
                }
            )
            audio_content.append({"type": "audio", "array": seg_waveform})

        audio_content.append({"type": "text", "text": f"\n\nQuestion: {question}"})
        conversation.append({"role": "user", "content": audio_content})

        # 主推理循环
        final_answer = ""
        call_count = 0

        while call_count < self._max_tool_calls:
            # 检查上下文是否需要重置
            token_count = self._model.count_tokens(conversation)
            if self._notebook.should_reset(token_count):
                handover = self._notebook.create_handover(conversation, model=self._model)
                conversation = self._notebook.reset_with_handover(
                    conversation, handover, question
                )
                context_resets += 1

            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > self._max_time_sec:
                final_answer = "[TIMEOUT] Inference exceeded time limit."
                break

            # 模型生成
            response = self._model.generate(conversation, max_new_tokens=self._max_new_tokens)

            # 将模型回复加入对话
            conversation.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )

            # 检测工具调用
            if has_tool_call(response):
                calls = parse_tool_calls(response)
                for call in calls:
                    if call.tool_name != "inspect":
                        continue

                    start_t = call.params.get("start_time", 0.0)
                    end_t = call.params.get("end_time", 0.0)

                    # 执行 inspect 并追加到对话
                    result = self._inspect_tool.execute_and_append(
                        conversation, start_t, end_t
                    )

                    call_count += 1
                    tool_calls.append(
                        ToolCallLog(
                            call_index=call_count,
                            start_time=start_t,
                            end_time=end_t,
                            success=result.success,
                            duration_sec=result.duration_sec,
                            stage="reasoning",
                        )
                    )

                # 继续循环，让模型基于回放音频继续推理
                continue
            else:
                # 无工具调用，提取最终答案
                final_answer = self._extract_final_answer(response)
                break

        # 如果达到最大工具调用次数仍未得到答案
        if not final_answer and call_count >= self._max_tool_calls:
            final_answer = "[MAX_TOOL_CALLS] Reached maximum tool call limit."

        total_time = time.time() - start_time

        return PipelineResult(
            final_answer=final_answer,
            conversation=conversation,
            tool_calls=tool_calls,
            total_time_sec=total_time,
            num_tool_calls=call_count,
            context_resets=context_resets,
        )

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        """从模型回复中提取最终答案。

        如果回复包含工具调用则返回空字符串（继续循环），
        否则直接返回回复作为最终答案。
        """
        if has_tool_call(response):
            return ""
        return response.strip()
