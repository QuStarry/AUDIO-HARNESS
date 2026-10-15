"""
inspect 主动回放工具

核心工具：模型调用 inspect(start_time, end_time) 回放指定时间段的原始音频，
重新编码送入当前对话上下文。

实现机制：
1. 解析模型输出中的 <tool_call>inspect(...)</tool_call> 标签
2. 从记忆库按时间戳提取音频片段（跨分段自动拼接）
3. 将回放音频作为新的用户消息追加到对话上下文
4. 利用 AF3 多轮对话支持任意轮次插入音频的特性实现回放

工具调用格式（向模型暴露）：
    <tool_call>inspect(start_time=X, end_time=Y)</tool_call>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import (
    INSPECT_PARAM_END,
    INSPECT_PARAM_START,
    TOOL_CALL_CLOSE_TAG,
    TOOL_CALL_OPEN_TAG,
)
from .memory import AudioMemory
from .model_backend import Conversation, Message


@dataclass
class ToolCall:
    """解析后的工具调用。"""

    tool_name: str
    params: dict[str, float]
    raw_text: str


@dataclass
class InspectResult:
    """inspect 工具执行结果。"""

    success: bool
    start_time: float
    end_time: float
    waveform: Optional[np.ndarray]
    error: Optional[str] = None

    @property
    def duration_sec(self) -> float:
        if self.waveform is None:
            return 0.0
        return len(self.waveform) / 16000


def parse_tool_calls(text: str) -> list[ToolCall]:
    """从模型输出文本中解析工具调用。

    支持格式：<tool_call>inspect(start_time=X, end_time=Y)</tool_call>

    Args:
        text: 模型输出文本

    Returns:
        解析到的工具调用列表
    """
    calls: list[ToolCall] = []
    # 匹配 <tool_call>...</tool_call>
    pattern = re.escape(TOOL_CALL_OPEN_TAG) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE_TAG)
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        match = match.strip()
        # 解析工具名和参数: tool_name(param1=val1, param2=val2)
        call_match = re.match(r"(\w+)\s*\((.*)\)", match)
        if not call_match:
            continue

        tool_name = call_match.group(1)
        params_str = call_match.group(2)

        # 解析参数
        params: dict[str, float] = {}
        for param in re.finditer(r"(\w+)\s*=\s*([0-9.]+)", params_str):
            params[param.group(1)] = float(param.group(2))

        calls.append(
            ToolCall(
                tool_name=tool_name,
                params=params,
                raw_text=TOOL_CALL_OPEN_TAG + match + TOOL_CALL_CLOSE_TAG,
            )
        )

    return calls


def has_tool_call(text: str) -> bool:
    """检查文本中是否包含工具调用。"""
    return TOOL_CALL_OPEN_TAG in text and TOOL_CALL_CLOSE_TAG in text


class InspectTool:
    """inspect 主动回放工具。

    从记忆库提取指定时间段的原始音频，追加到对话上下文供模型重听。
    """

    def __init__(self, memory: AudioMemory):
        self._memory = memory

    def inspect(self, start_time: float, end_time: float) -> InspectResult:
        """回放指定时间段的原始音频。

        Args:
            start_time: 起始时间（秒）
            end_time: 结束时间（秒）

        Returns:
            InspectResult 包含回放的波形
        """
        if end_time <= start_time:
            return InspectResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                waveform=None,
                error=f"end_time ({end_time}) must be > start_time ({start_time})",
            )

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        try:
            retrieved = self._memory.retrieve(start_ms, end_ms)
        except ValueError as e:
            return InspectResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                waveform=None,
                error=str(e),
            )

        return InspectResult(
            success=True,
            start_time=start_time,
            end_time=end_time,
            waveform=retrieved.waveform,
        )

    def append_replay_to_conversation(
        self,
        conversation: Conversation,
        result: InspectResult,
    ) -> Conversation:
        """将回放音频作为新用户消息追加到对话上下文。

        利用 AF3 多轮对话支持任意轮次插入音频的特性，
        回放片段作为新的用户消息，模型可在上下文中"重听"该片段。
        """
        if not result.success or result.waveform is None:
            # 回放失败时，追加错误提示
            err_msg: Message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[inspect failed: {result.error}]. Please try a different time range or answer based on what you have.",
                    }
                ],
            }
            conversation.append(err_msg)
            return conversation

        # 回放成功，追加含音频的用户消息
        replay_msg: Message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Replaying audio from {result.start_time:.1f}s to {result.end_time:.1f}s "
                    f"(duration: {result.duration_sec:.1f}s). Listen carefully:",
                },
                {"type": "audio", "array": result.waveform},
            ],
        }
        conversation.append(replay_msg)
        return conversation

    def execute_and_append(
        self,
        conversation: Conversation,
        start_time: float,
        end_time: float,
    ) -> InspectResult:
        """执行 inspect 并将结果追加到对话。

        便捷方法：执行回放 + 追加到上下文。
        """
        result = self.inspect(start_time, end_time)
        self.append_replay_to_conversation(conversation, result)
        return result
