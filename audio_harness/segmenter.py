"""
音频分段器

将输入音频按固定时长（默认 30 秒）分段，每段导出为 FLAC 无损格式，
记录分段 ID 与起止毫秒时间戳。分段结果供记忆库建立索引。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from . import audio_utils
from .config import MEMORY_DIR, SEGMENT_DURATION_SEC


@dataclass
class Segment:
    """单个音频分段的元数据。"""

    segment_id: int
    start_ms: int
    end_ms: int
    flac_path: Path
    duration_sec: float

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class SegmentationResult:
    """分段结果，包含所有分段的元数据。"""

    segments: list[Segment]
    total_duration_sec: float
    total_duration_ms: int

    def __len__(self) -> int:
        return len(self.segments)


class AudioSegmenter:
    """音频分段器。

    将音频按固定时长分段并存储为 FLAC 无损格式。
    分段时长对齐 AF3 的 30 秒窗口，确保每段可被 AF3 原生处理。
    """

    def __init__(
        self,
        segment_duration_sec: float = SEGMENT_DURATION_SEC,
        output_dir: Union[str, Path, None] = None,
    ):
        self.segment_duration_sec = segment_duration_sec
        self.output_dir = Path(output_dir) if output_dir else MEMORY_DIR

    def segment(
        self,
        waveform: np.ndarray,
        task_id: str,
    ) -> SegmentationResult:
        """对波形进行分段。

        Args:
            waveform: 输入音频波形（16kHz mono float32）
            task_id: 任务 ID，用于创建独立的存储目录

        Returns:
            SegmentationResult 包含所有分段元数据
        """
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        total_samples = len(waveform)
        segment_samples = int(self.segment_duration_sec * audio_utils.SAMPLE_RATE)
        total_duration_sec = audio_utils.get_duration_sec(waveform)
        total_duration_ms = audio_utils.get_duration_ms(waveform)

        segments: list[Segment] = []
        segment_id = 0
        offset = 0

        while offset < total_samples:
            end = min(offset + segment_samples, total_samples)
            seg_waveform = waveform[offset:end]

            start_ms = int(offset / audio_utils.SAMPLE_RATE * 1000)
            end_ms = int(end / audio_utils.SAMPLE_RATE * 1000)

            flac_path = task_dir / f"segment_{segment_id:06d}.flac"
            audio_utils.save_flac(seg_waveform, flac_path)

            segments.append(
                Segment(
                    segment_id=segment_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    flac_path=flac_path,
                    duration_sec=audio_utils.get_duration_sec(seg_waveform),
                )
            )

            offset = end
            segment_id += 1

        return SegmentationResult(
            segments=segments,
            total_duration_sec=total_duration_sec,
            total_duration_ms=total_duration_ms,
        )

    def segment_file(
        self,
        audio_path: Union[str, Path],
        task_id: str,
    ) -> SegmentationResult:
        """从音频文件分段。"""
        waveform = audio_utils.load_audio(audio_path)
        return self.segment(waveform, task_id)
