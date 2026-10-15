"""
基准适配器：LongAudioSpan

加载 LongAudioSpan 数据集，提供统一的题目/音频/答案接口。
数据集结构：
  - 子集: accuracy / chain / rubric
  - 分层: S (短) / M (中) / L (长)
  - 字段: qa_id, media_id, audio_path, question, options, correct_option, question_profile
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .config import LONG_AUDIO_SPAN_ID, MID_AUDIO_MAX_SEC, SHORT_AUDIO_MAX_SEC


@dataclass
class Example:
    """统一的测试样本格式。"""

    qa_id: str
    media_id: str
    audio_path: str
    question: str
    options: list[str]
    correct_option: str  # "A" / "B" / "C" / "D"
    dimension: str  # localization, causal, counting, recognition, tracking, interpretation, logical, relational, summarization
    level: str  # perception, understanding, reasoning
    split: str  # S / M / L
    duration_band: str  # short / mid / long

    @property
    def full_question(self) -> str:
        """将问题和选项拼接为完整问题文本。"""
        options_str = "\n".join(self.options)
        return f"{self.question}\n\nOptions:\n{options_str}"

    @property
    def correct_answer_text(self) -> str:
        """获取正确选项的完整文本。"""
        for opt in self.options:
            if opt.startswith(f"{self.correct_option}."):
                return opt
        return self.correct_option


def _get_duration_band(split: str) -> str:
    """根据 split 推断长度分层。"""
    if split == "S":
        return "short"
    elif split == "M":
        return "mid"
    else:
        return "long"


class LongAudioSpanAdapter:
    """LongAudioSpan 基准适配器。

    加载数据集并提供统一的迭代接口。
    """

    def __init__(
        self,
        dataset_name: str = LONG_AUDIO_SPAN_ID,
        subsets: Optional[list[str]] = None,
        splits: Optional[list[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        self._dataset_name = dataset_name
        self._subsets = subsets or ["accuracy", "chain", "rubric"]
        self._splits = splits or ["S", "M", "L"]
        self._cache_dir = cache_dir
        self._dataset = None

    def _load(self):
        """懒加载数据集。"""
        if self._dataset is not None:
            return

        from datasets import load_dataset

        # LongAudioSpan 使用子集+分层结构
        self._dataset = {}
        for subset in self._subsets:
            for split in self._splits:
                config_name = subset  # 子集作为 config
                try:
                    ds = load_dataset(
                        self._dataset_name,
                        config_name,
                        split=split,
                        cache_dir=self._cache_dir,
                    )
                    self._dataset[(subset, split)] = ds
                except Exception as e:
                    print(f"Warning: Failed to load {subset}/{split}: {e}")

    def iter_examples(self) -> Iterator[Example]:
        """迭代所有测试样本。"""
        self._load()

        for (subset, split), ds in self._dataset.items():
            for item in ds:
                profile = item.get("question_profile", {}) or {}
                yield Example(
                    qa_id=item.get("qa_id", ""),
                    media_id=item.get("media_id", ""),
                    audio_path=item.get("audio_path", ""),
                    question=item.get("question", ""),
                    options=list(item.get("options", [])),
                    correct_option=item.get("correct_option", ""),
                    dimension=profile.get("dimension", ""),
                    level=profile.get("level", ""),
                    split=split,
                    duration_band=_get_duration_band(split),
                )

    def get_audio_path(self, example: Example, dataset_root: Optional[str] = None) -> Path:
        """获取音频文件的绝对路径。

        如果提供了 dataset_root，则从该目录查找；否则返回相对路径。
        """
        if dataset_root:
            return Path(dataset_root) / example.audio_path
        return Path(example.audio_path)

    def __len__(self) -> int:
        """返回总样本数。"""
        self._load()
        return sum(len(ds) for ds in self._dataset.values())
