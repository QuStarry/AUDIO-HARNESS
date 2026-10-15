"""
分层分析模块

对实验结果进行多维度分析：
  1. 分长度分析（短/中/长）
  2. 分任务类型分析（dimension）
  3. 分答案位置分析（前/中/后，需题目提供位置信息）
  4. 效率统计（token 数、推理时间、工具调用次数）

输出 Markdown 报告 + CSV 原始数据。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ResultRecord:
    """单题实验结果记录。"""

    qa_id: str
    config: str  # b1 / b2 / h1
    prediction: str
    reference: str
    correct: bool
    duration_band: str  # short / mid / long
    dimension: str  # task dimension
    level: str  # perception / understanding / reasoning
    position: str = "unknown"  # front / middle / back
    tool_calls: int = 0
    total_time_sec: float = 0.0
    context_resets: int = 0


@dataclass
class AnalysisResult:
    """分析结果汇总。"""

    overall: dict = field(default_factory=dict)
    by_duration: dict = field(default_factory=dict)
    by_dimension: dict = field(default_factory=dict)
    by_level: dict = field(default_factory=dict)
    by_position: dict = field(default_factory=dict)
    efficiency: dict = field(default_factory=dict)


class Analyzer:
    """分层分析器。"""

    def __init__(self):
        self._records: list[ResultRecord] = []

    def add_record(self, record: ResultRecord) -> None:
        """添加一条结果记录。"""
        self._records.append(record)

    def add_records(self, records: list[ResultRecord]) -> None:
        """批量添加结果记录。"""
        self._records.extend(records)

    def analyze(self) -> AnalysisResult:
        """执行全维度分析。"""
        result = AnalysisResult()

        # 总体
        result.overall = self._compute_metrics(self._records)

        # 分长度
        result.by_duration = self._group_and_compute(self._records, "duration_band")

        # 分任务维度
        result.by_dimension = self._group_and_compute(self._records, "dimension")

        # 分认知层级
        result.by_level = self._group_and_compute(self._records, "level")

        # 分位置
        result.by_position = self._group_and_compute(self._records, "position")

        # 效率统计
        result.efficiency = self._compute_efficiency(self._records)

        return result

    @staticmethod
    def _compute_metrics(records: list[ResultRecord]) -> dict:
        """计算一组记录的性能指标。"""
        if not records:
            return {"count": 0, "accuracy": 0.0}
        correct = sum(1 for r in records if r.correct)
        return {
            "count": len(records),
            "accuracy": correct / len(records),
            "correct": correct,
        }

    @staticmethod
    def _group_and_compute(records: list[ResultRecord], key: str) -> dict:
        """按指定字段分组并计算指标。"""
        groups: dict[str, list[ResultRecord]] = defaultdict(list)
        for r in records:
            groups[getattr(r, key)].append(r)
        return {k: Analyzer._compute_metrics(v) for k, v in groups.items()}

    @staticmethod
    def _compute_efficiency(records: list[ResultRecord]) -> dict:
        """计算效率指标。"""
        if not records:
            return {}
        tool_calls = [r.tool_calls for r in records]
        times = [r.total_time_sec for r in records]
        resets = [r.context_resets for r in records]
        return {
            "avg_tool_calls": float(np.mean(tool_calls)),
            "max_tool_calls": int(max(tool_calls)),
            "avg_time_sec": float(np.mean(times)),
            "max_time_sec": float(max(times)),
            "avg_context_resets": float(np.mean(resets)),
        }

    def generate_markdown_report(
        self,
        analysis: AnalysisResult,
        output_path: Optional[str] = None,
    ) -> str:
        """生成 Markdown 格式的分析报告。"""
        lines = []
        lines.append("# AUDIO-HARNESS 实验分析报告\n")

        # 总体
        lines.append("## 总体性能\n")
        lines.append(f"- 样本数: **{analysis.overall['count']}**")
        lines.append(f"- 准确率: **{analysis.overall['accuracy']:.4f}**\n")

        # 分长度
        lines.append("## 分长度分析\n")
        lines.append("| 长度分层 | 样本数 | 准确率 |")
        lines.append("|---------|-------|-------|")
        for band, metrics in sorted(analysis.by_duration.items()):
            lines.append(f"| {band} | {metrics['count']} | {metrics['accuracy']:.4f} |")
        lines.append("")

        # 分任务维度
        lines.append("## 分任务维度分析\n")
        lines.append("| 任务维度 | 样本数 | 准确率 |")
        lines.append("|---------|-------|-------|")
        for dim, metrics in sorted(analysis.by_dimension.items()):
            lines.append(f"| {dim} | {metrics['count']} | {metrics['accuracy']:.4f} |")
        lines.append("")

        # 分认知层级
        lines.append("## 分认知层级分析\n")
        lines.append("| 认知层级 | 样本数 | 准确率 |")
        lines.append("|---------|-------|-------|")
        for level, metrics in sorted(analysis.by_level.items()):
            lines.append(f"| {level} | {metrics['count']} | {metrics['accuracy']:.4f} |")
        lines.append("")

        # 效率
        lines.append("## 效率开销\n")
        eff = analysis.efficiency
        if eff:
            lines.append(f"- 平均工具调用次数: **{eff['avg_tool_calls']:.2f}**")
            lines.append(f"- 最大工具调用次数: {eff['max_tool_calls']}")
            lines.append(f"- 平均推理时间: **{eff['avg_time_sec']:.2f}s**")
            lines.append(f"- 最大推理时间: {eff['max_time_sec']:.2f}s")
            lines.append(f"- 平均上下文重置次数: {eff['avg_context_resets']:.2f}")
        lines.append("")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")

        return report

    def export_csv(self, output_path: str) -> None:
        """导出原始结果为 CSV。"""
        if not self._records:
            return

        fieldnames = [
            "qa_id", "config", "prediction", "reference", "correct",
            "duration_band", "dimension", "level", "position",
            "tool_calls", "total_time_sec", "context_resets",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records:
                writer.writerow({k: getattr(r, k) for k in fieldnames})
