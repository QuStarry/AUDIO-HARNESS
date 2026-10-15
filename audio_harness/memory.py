"""
无损音频记忆库

基于分段 ID + 毫秒级时间戳的双层索引，支持任意时间区间精确寻址。
跨分段自动拼接，任务级持久化，按需加载不预读。

设计原则：
- 只存不读：模型不调用时不占用上下文空间
- 调用时才提取编码送入上下文
- 内存中只缓存索引结构，波形按需从 FLAC 读取
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import audio_utils
from .segmenter import SegmentationResult, Segment


@dataclass
class RetrievedSegment:
    """检索结果，包含波形与时间信息。"""

    waveform: np.ndarray
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class AudioMemory:
    """无损音频记忆库。

    双层索引：
    - 第一层：segment_id → Segment 元数据
    - 第二层：时间戳 → 定位到具体分段 + 分段内偏移

    检索时根据时间区间定位涉及的分段，按需加载 FLAC 并拼接。
    """

    def __init__(self, segmentation_result: SegmentationResult):
        self._segments: list[Segment] = segmentation_result.segments
        self._total_duration_ms: int = segmentation_result.total_duration_ms
        # segment_id → Segment 的快速索引
        self._segment_by_id: dict[int, Segment] = {
            seg.segment_id: seg for seg in self._segments
        }

    @property
    def total_duration_ms(self) -> int:
        return self._total_duration_ms

    @property
    def num_segments(self) -> int:
        return len(self._segments)

    def _find_segment_by_time(self, time_ms: int) -> Optional[Segment]:
        """根据时间戳定位所属分段。

        使用线性查找（分段数量通常 ≤ 40，无需二分）。
        """
        for seg in self._segments:
            if seg.start_ms <= time_ms < seg.end_ms:
                return seg
        # 处理边界：time_ms 恰好等于总时长
        if time_ms == self._total_duration_ms and self._segments:
            return self._segments[-1]
        return None

    def retrieve(self, start_ms: int, end_ms: int) -> RetrievedSegment:
        """按时间区间检索音频，跨分段自动拼接。

        Args:
            start_ms: 起始时间（毫秒）
            end_ms: 结束时间（毫秒）

        Returns:
            RetrievedSegment 包含拼接后的波形与实际时间区间

        Raises:
            ValueError: 时间区间无效或完全超出音频范围
        """
        if end_ms <= start_ms:
            raise ValueError(f"end_ms ({end_ms}) must be > start_ms ({start_ms})")
        if start_ms < 0:
            raise ValueError(f"start_ms ({start_ms}) must be >= 0")
        if start_ms >= self._total_duration_ms:
            raise ValueError(
                f"start_ms ({start_ms}) exceeds audio duration ({self._total_duration_ms}ms)"
            )

        # 限制 end_ms 不超过音频总时长
        end_ms = min(end_ms, self._total_duration_ms)

        # 定位起始和结束分段
        start_seg = self._find_segment_by_time(start_ms)
        end_seg = self._find_segment_by_time(end_ms - 1)  # end_ms-1 确保命中最后一个采样所在分段

        if start_seg is None or end_seg is None:
            raise ValueError(f"Cannot locate segments for time range [{start_ms}, {end_ms})")

        # 收集涉及的所有分段
        involved = [
            seg
            for seg in self._segments
            if start_seg.segment_id <= seg.segment_id <= end_seg.segment_id
        ]

        # 逐段加载并截取
        parts: list[np.ndarray] = []
        for seg in involved:
            seg_waveform = audio_utils.load_flac(seg.flac_path)
            # 计算该分段内需截取的样本范围
            seg_start_ms = max(start_ms, seg.start_ms)
            seg_end_ms = min(end_ms, seg.end_ms)

            offset_start = int((seg_start_ms - seg.start_ms) / 1000 * audio_utils.SAMPLE_RATE)
            offset_end = int((seg_end_ms - seg.start_ms) / 1000 * audio_utils.SAMPLE_RATE)
            offset_end = min(offset_end, len(seg_waveform))

            if offset_end > offset_start:
                parts.append(seg_waveform[offset_start:offset_end])

        if not parts:
            return RetrievedSegment(
                waveform=np.array([], dtype=np.float32),
                start_ms=start_ms,
                end_ms=start_ms,
            )

        concatenated = np.concatenate(parts).astype(np.float32)
        actual_end_ms = start_ms + int(len(concatenated) / audio_utils.SAMPLE_RATE * 1000)

        return RetrievedSegment(
            waveform=concatenated,
            start_ms=start_ms,
            end_ms=actual_end_ms,
        )

    def get_segment(self, segment_id: int) -> Optional[Segment]:
        """根据分段 ID 获取分段元数据。"""
        return self._segment_by_id.get(segment_id)
