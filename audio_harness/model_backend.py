"""
模型后端接口

定义统一的模型后端抽象，支持 Mock（本地调试）和 AF3（远程推理）两种实现。
对话格式对齐 AF3 的 `apply_chat_template` 输入格式：
    [{"role": "user"/"assistant", "content": [{"type":"text","text":...}, {"type":"audio","array":...}]}]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .config import AF3_MODEL_ID, MAX_NEW_TOKENS


# 统一的内容项类型
ContentItem = dict  # {"type": "text"|"audio", "text": str, "array": np.ndarray}
Message = dict  # {"role": str, "content": list[ContentItem]}
Conversation = list[Message]


class BaseModelBackend(ABC):
    """模型后端抽象基类。

    所有后端（Mock、AF3）必须实现 generate 和 count_tokens 接口，
    确保流水线与具体模型解耦。
    """

    @abstractmethod
    def generate(self, conversation: Conversation, max_new_tokens: int = 1024) -> str:
        """根据对话历史生成回复文本。

        Args:
            conversation: 对话历史列表
            max_new_tokens: 最大生成 token 数

        Returns:
            模型生成的文本回复
        """
        ...

    @abstractmethod
    def count_tokens(self, conversation: Conversation) -> int:
        """计算对话的 token 数。

        Args:
            conversation: 对话历史列表

        Returns:
            token 数量
        """
        ...


def extract_text_from_conversation(conversation: Conversation) -> str:
    """从对话中提取所有文本内容（用于 Mock 的 token 估算）。"""
    texts = []
    for msg in conversation:
        for item in msg.get("content", []):
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
    return " ".join(texts)


class MockModelBackend(BaseModelBackend):
    """Mock 模型后端，用于本地无 GPU 调试。

    支持配置回复规则：
    - 固定回复：每次返回相同文本
    - 序列回复：按顺序返回预设回复列表（模拟工具调用循环）
    - 函数回复：根据对话内容动态生成回复
    """

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        fixed_response: Optional[str] = None,
        response_fn=None,
        tokens_per_char: float = 0.25,
    ):
        """初始化 Mock 模型。

        Args:
            responses: 预设回复序列，按调用顺序返回（循环使用）
            fixed_response: 固定回复（优先级低于 responses）
            response_fn: 自定义回复函数 fn(conversation) -> str
            tokens_per_char: token 估算系数（每字符约 0.25 token）
        """
        self._responses = responses or []
        self._fixed_response = fixed_response
        self._response_fn = response_fn
        self._tokens_per_char = tokens_per_char
        self._call_count = 0

    def generate(self, conversation: Conversation, max_new_tokens: int = 1024) -> str:
        """生成 Mock 回复。"""
        self._call_count += 1

        # 优先级：response_fn > responses > fixed_response > 默认
        if self._response_fn is not None:
            return self._response_fn(conversation)

        if self._responses:
            idx = (self._call_count - 1) % len(self._responses)
            return self._responses[idx]

        if self._fixed_response is not None:
            return self._fixed_response

        return "This is a mock response."

    def count_tokens(self, conversation: Conversation) -> int:
        """估算对话 token 数。"""
        text = extract_text_from_conversation(conversation)
        # 音频内容按固定 token 估算（30s 音频约 500 tokens）
        audio_token_estimate = 0
        for msg in conversation:
            for item in msg.get("content", []):
                if item.get("type") == "audio":
                    arr = item.get("array")
                    if arr is not None and len(arr) > 0:
                        duration_sec = len(arr) / 16000
                        audio_token_estimate += int(duration_sec / 30 * 500)

        text_tokens = int(len(text) * self._tokens_per_char)
        return text_tokens + audio_token_estimate


def create_mock_with_tool_call(
    first_response: str = "<tool_call>inspect(start_time=5.0, end_time=10.0)</tool_call>",
    final_response: str = "Based on the replayed audio, the answer is (A).",
) -> MockModelBackend:
    """创建一个模拟工具调用循环的 Mock 模型。

    首次调用返回工具调用，第二次返回最终答案。
    用于测试 Harness 流水线的 inspect 执行逻辑。
    """
    return MockModelBackend(responses=[first_response, final_response])


# ============================================================
# AF3 真实模型后端（远程 GPU 运行）
# ============================================================

class AF3ModelBackend(BaseModelBackend):
    """Audio Flamingo 3 模型后端。

    封装 AF3 的加载、对话模板应用、生成与解码。
    仅在远程 GPU 机器上使用，本地无 GPU 时使用 Mock。

    红线约束：不修改模型权重，不插入内部模块。
    """

    def __init__(
        self,
        model_id: str = AF3_MODEL_ID,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        use_awq: bool = False,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        """加载 AF3 模型与 Processor。

        Args:
            model_id: HuggingFace 模型 ID
            dtype: 推理精度（bfloat16/float16）
            device_map: 设备映射策略
            use_awq: 是否启用 4-bit AWQ 量化（需 autoawq）
            max_new_tokens: 默认最大生成 token 数
        """
        import torch
        from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration

        self._max_new_tokens = max_new_tokens
        self._processor = AutoProcessor.from_pretrained(model_id)

        torch_dtype = getattr(torch, dtype)

        if use_awq:
            # 4-bit AWQ 量化，显存约 8.3B * 4bit ≈ 4.2GB
            from awq import AutoAWQForCausalLM

            # AWQ 模型需要预量化版本，此处加载后量化
            model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            self._model = model
        else:
            self._model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

        self._model.eval()

    def generate(self, conversation: Conversation, max_new_tokens: Optional[int] = None) -> str:
        """根据对话历史生成回复。

        Args:
            conversation: 对话历史（含音频 numpy array）
            max_new_tokens: 最大生成 token 数

        Returns:
            生成的文本回复
        """
        import torch

        max_tokens = max_new_tokens or self._max_new_tokens

        # 应用聊天模板，获取模型输入
        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self._model.device)

        # 生成
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
            )

        # 解码生成部分（去除 prompt）
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return decoded[0] if decoded else ""

    def count_tokens(self, conversation: Conversation) -> int:
        """计算对话的 token 数（含音频编码后的 token）。"""
        try:
            inputs = self._processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )
            return inputs["input_ids"].shape[-1]
        except Exception:
            # 降级：用文本长度估算
            text = extract_text_from_conversation(conversation)
            return int(len(text) * 0.25)
