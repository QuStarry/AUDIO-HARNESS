"""
AUDIO-HARNESS 命令行入口

支持三组实验配置（B1/B2/H1）的批量运行与结果持久化。

用法：
    python -m audio_harness.main --config h1 --model mock --limit 10 --output-dir outputs
    python -m audio_harness.main --config b1 --model af3 --subsets accuracy --splits S
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import audio_utils
from .analysis import Analyzer, ResultRecord
from .benchmark import LongAudioSpanAdapter
from .config import OUTPUT_DIR
from .evaluation import Evaluator, extract_option_from_prediction, normalize_option
from .experiments import ExperimentOutput, run_b1, run_b2, run_h1
from .model_backend import AF3ModelBackend, MockModelBackend


def load_model(model_type: str, **kwargs):
    """加载模型后端。"""
    if model_type == "mock":
        return MockModelBackend(fixed_response="(A) Option A")
    elif model_type == "af3":
        return AF3ModelBackend(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(
    config: str,
    model,
    waveform: np.ndarray,
    question: str,
) -> ExperimentOutput:
    """根据配置运行对应实验。"""
    if config == "b1":
        return run_b1(model, waveform, question)
    elif config == "b2":
        return run_b2(model, waveform, question)
    elif config == "h1":
        return run_h1(model, waveform, question)
    else:
        raise ValueError(f"Unknown config: {config}")


def load_completed_ids(output_path: Path) -> set[str]:
    """加载已完成的题目 ID（用于断点续跑）。"""
    completed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed.add(record["qa_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def main():
    parser = argparse.ArgumentParser(description="AUDIO-HARNESS 实验运行器")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["b1", "b2", "h1"],
        help="实验配置：b1（原生基线）/ b2（分段摘要）/ h1（Harness 实验组）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        choices=["mock", "af3"],
        help="模型后端：mock（本地调试）/ af3（远程 GPU）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制运行题目数量（用于快速测试）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="结果输出目录",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["accuracy", "chain", "rubric"],
        help="LongAudioSpan 子集",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["S", "M", "L"],
        help="LongAudioSpan 分层",
    )
    parser.add_argument(
        "--use-awq",
        action="store_true",
        help="AF3 使用 4-bit AWQ 量化（显存受限时）",
    )

    args = parser.parse_args()

    # 输出路径
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"results_{args.config}.jsonl"

    # 加载模型
    print(f"[INFO] 加载模型: {args.model}")
    model_kwargs = {}
    if args.model == "af3" and args.use_awq:
        model_kwargs["use_awq"] = True
    model = load_model(args.model, **model_kwargs)

    # 加载数据集
    print(f"[INFO] 加载数据集: subsets={args.subsets}, splits={args.splits}")
    adapter = LongAudioSpanAdapter(subsets=args.subsets, splits=args.splits)

    # 断点续跑
    completed_ids = load_completed_ids(output_file)
    if completed_ids:
        print(f"[INFO] 断点续跑：已完成 {len(completed_ids)} 题，将跳过")

    # 运行实验
    analyzer = Analyzer()
    evaluator = Evaluator()
    count = 0
    predictions, references, options_list = [], [], []

    for example in adapter.iter_examples():
        if args.limit and count >= args.limit:
            break

        if example.qa_id in completed_ids:
            continue

        count += 1
        print(f"[{count}] {example.qa_id} (duration={example.duration_band}, dim={example.dimension})")

        # 加载音频
        try:
            waveform = audio_utils.load_audio(example.audio_path)
        except Exception as e:
            print(f"  [WARN] 音频加载失败: {e}")
            continue

        # 运行实验
        try:
            result = run_experiment(args.config, model, waveform, example.full_question)
        except Exception as e:
            print(f"  [ERROR] 实验运行失败: {e}")
            continue

        # 评估
        pred_option = extract_option_from_prediction(result.prediction, example.options)
        ref_option = normalize_option(example.correct_option)
        correct = pred_option == ref_option

        predictions.append(result.prediction)
        references.append(example.correct_option)
        options_list.append(example.options)

        # 记录结果
        record = ResultRecord(
            qa_id=example.qa_id,
            config=args.config,
            prediction=pred_option,
            reference=ref_option,
            correct=correct,
            duration_band=example.duration_band,
            dimension=example.dimension,
            level=example.level,
            tool_calls=result.tool_calls,
            total_time_sec=result.total_time_sec,
            context_resets=result.context_resets,
        )
        analyzer.add_record(record)

        # 写入 JSONL
        result_dict = {
            "qa_id": example.qa_id,
            "config": args.config,
            "prediction": result.prediction,
            "predicted_option": pred_option,
            "reference": example.correct_option,
            "correct": correct,
            "duration_band": example.duration_band,
            "dimension": example.dimension,
            "level": example.level,
            "tool_calls": result.tool_calls,
            "total_time_sec": round(result.total_time_sec, 2),
            "context_resets": result.context_resets,
            "options": example.options,
        }
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

        print(f"  -> pred={pred_option}, ref={ref_option}, correct={correct}, "
              f"tools={result.tool_calls}, time={result.total_time_sec:.1f}s")

    # 生成汇总报告
    if count > 0:
        analysis = analyzer.analyze()
        report_path = output_dir / f"report_{args.config}.md"
        analyzer.generate_markdown_report(analysis, str(report_path))
        analyzer.export_csv(str(output_dir / f"results_{args.config}.csv"))

        # 总体准确率
        eval_result = evaluator.evaluate_mcq(predictions, references, options_list)
        print(f"\n[RESULT] 配置={args.config}, 运行={count} 题")
        print(f"  Accuracy: {eval_result['accuracy']:.4f}")
        print(f"  F1: {eval_result['f1']:.4f}")
        print(f"  报告: {report_path}")
    else:
        print("[INFO] 没有运行任何题目（可能全部已完成或数据集为空）")


if __name__ == "__main__":
    main()
