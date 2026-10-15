"""
音频工具模块

统一音频加载、FLAC 无损读写、波形拼接、时长计算。
所有音频统一为 16kHz mono，对齐 AF3 原生输入规格。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

from .config import NUM_CHANNELS, SAMPLE_RATE


def load_audio(path: Union[str, Path]) -> np.ndarray:
    """加载音频为 16kHz mono numpy 数组（float32, range [-1, 1]）。

    使用 librosa 统一重采样到 16kHz、转 mono，确保与 AF3 输入规格一致。
    """
    import librosa

    waveform, _ = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=(NUM_CHANNELS == 1),
    )
    return waveform.astype(np.float32)


def save_flac(waveform: np.ndarray, path: Union[str, Path]) -> None:
    """将波形保存为 FLAC 无损格式。

    FLAC 为无损压缩，信息零损失，是记忆库的存储格式。
    """
    sf.write(str(path), waveform, SAMPLE_RATE, format="FLAC", subtype="PCM_16")


def load_flac(path: Union[str, Path]) -> np.ndarray:
    """从 FLAC 文件加载波形（返回 float32 numpy 数组）。"""
    waveform, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
    return waveform.astype(np.float32)


def get_duration_sec(waveform: np.ndarray) -> float:
    """计算波形时长（秒）。"""
    return len(waveform) / SAMPLE_RATE


def get_duration_ms(waveform: np.ndarray) -> int:
    """计算波形时长（毫秒，向下取整）。"""
    return int(len(waveform) / SAMPLE_RATE * 1000)


def concat_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
    """拼接多个波形数组。"""
    if not waveforms:
        return np.array([], dtype=np.float32)
    return np.concatenate(waveforms).astype(np.float32)


def slice_waveform(waveform: np.ndarray, start_ms: int, end_ms: int) -> np.ndarray:
    """按毫秒时间戳截取波形片段。

    Args:
        waveform: 原始波形
        start_ms: 起始时间（毫秒）
        end_ms: 结束时间（毫秒）

    Returns:
        截取的波形片段（float32）
    """
    start_sample = int(start_ms / 1000 * SAMPLE_RATE)
    end_sample = int(end_ms / 1000 * SAMPLE_RATE)
    start_sample = max(0, start_sample)
    end_sample = min(len(waveform), end_sample)
    if end_sample <= start_sample:
        return np.array([], dtype=np.float32)
    return waveform[start_sample:end_sample].astype(np.float32)


def waveform_hash(waveform: np.ndarray) -> str:
    """计算波形的 SHA256 哈希，用于无损校验。"""
    import hashlib

    return hashlib.sha256(waveform.tobytes()).hexdigest()[:16]
