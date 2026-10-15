"""
双笔记状态机制

对标 VISTA 的双文件设计，用于上下文重置时的状态延续。

WORKING.md：记录当前已提取的关键信息、待确认问题、推理进度。
上下文接近 28K token 时，生成精简交接摘要写入 WORKING.md，
重置上下文后继续推理。原始音频记忆库全程保留，不受上下文重置影响。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .config import CONTEXT_RESET_THRESHOLD, WORKING_NOTE_NAME
from .model_backend import Conversation, extract_text_from_conversation


class WorkingNotebook:
    """工作笔记管理器。

    管理 WORKING.md 的读写与上下文交接逻辑。
    """

    def __init__(self, task_dir: Union[str, Path]):
        self._task_dir = Path(task_dir)
        self._task_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._task_dir / WORKING_NOTE_NAME

    @property
    def path(self) -> Path:
        return self._path

    def should_reset(self, token_count: int) -> bool:
        """判断是否需要触发上下文重置。

        当 token 数超过阈值时返回 True。
        """
        return token_count > CONTEXT_RESET_THRESHOLD

    def read(self) -> str:
        """读取 WORKING.md 内容。"""
        if self._path.exists():
            return self._path.read_text(encoding="utf-8")
        return ""

    def write(self, content: str) -> None:
        """写入 WORKING.md。"""
        self._path.write_text(content, encoding="utf-8")

    def append(self, content: str) -> None:
        """追加内容到 WORKING.md。"""
        existing = self.read()
        if existing:
            content = existing + "\n\n" + content
        self.write(content)

    def create_handover(self, conversation: Conversation, model=None) -> str:
        """从对话历史提取关键信息生成交接摘要。

        如果提供了 model，则用模型生成摘要；否则简单拼接对话文本。

        Args:
            conversation: 当前对话历史
            model: 可选的模型后端，用于生成摘要

        Returns:
            交接摘要文本
        """
        if model is not None:
            # 构造摘要生成请求
            summary_conv: Conversation = conversation + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Summarize the key information, pending questions, and reasoning progress from the conversation above in a concise handover note.",
                        }
                    ],
                }
            ]
            handover = model.generate(summary_conv, max_new_tokens=512)
        else:
            # 无模型时，拼接所有文本作为简单交接
            text = extract_text_from_conversation(conversation)
            # 截断到合理长度
            if len(text) > 2000:
                text = text[:2000] + "..."
            handover = f"## Handover Summary\n\nKey information extracted from conversation:\n" + text

        # 写入 WORKING.md
        self.write(handover)
        return handover

    def reset_with_handover(self, conversation: Conversation, handover: str, question: str) -> Conversation:
        """重置对话，首条消息包含交接摘要。

        Args:
            conversation: 原对话（将被替换）
            handover: 交接摘要
            question: 原始问题（保留在新对话中

        Returns:
            重置后的新对话
        """
        new_conversation: Conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"## Previous Context (from WORKING.md)\n\n{handover}\n\n---\n\nPlease continue answering the question: {question}",
                    }
                ],
            }
        ]
        return new_conversation
