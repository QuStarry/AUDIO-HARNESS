"""
评测指标

核心性能指标：
  - 客观题：Accuracy、F1
  - 摘要题：ROUGE-L、BERTScore

LongAudioSpan 为多选题（A/B/C/D），主要使用 Accuracy。
对于需要文本匹配的场景，使用 F1 / ROUGE-L / BERTScore。
"""

from __future__ import annotations

import re
from typing import Union

import numpy as np


# ============================================================
# 答案标准化
# ============================================================

def normalize_option(text: str) -> str:
    """标准化选项答案，提取选项字母。

    支持多种格式："A"、"(A)"、"A. xxx"、"option A"、"The answer is A"
    优先级：显式标注 > 括号/点号格式 > 裸字母
    """
    text = text.strip().upper()
    # 直接匹配单个字母
    if re.match(r"^[A-D]$", text):
        return text
    # 1. 优先匹配 "answer is A" / "option A" / "correct: A" 等显式标注
    match = re.search(r"(?:ANSWER|OPTION|CORRECT|CHOICE)\s*(?:IS|:)?\s*\(?([A-D])\)?", text)
    if match:
        return match.group(1)
    # 2. 匹配行首的 "(A)" 或 "A." 格式（答案通常在开头）
    match = re.match(r"^\(?([A-D])\)?\s*[\.\)]\s*", text)
    if match:
        return match.group(1)
    # 3. 匹配独立的 "(A)" 格式
    match = re.search(r"\(([A-D])\)", text)
    if match:
        return match.group(1)
    # 4. 最后兜底：匹配行首的裸字母
    match = re.match(r"^([A-D])\b", text)
    if match:
        return match.group(1)
    return text


def extract_option_from_prediction(prediction: str, options: list[str]) -> str:
    """从模型预测中提取选项字母。

    优先匹配预测中出现的选项文本，其次匹配字母。
    """
    prediction_upper = prediction.upper()

    # 1. 尝试匹配完整选项文本
    for opt in options:
        opt_letter = opt.split(".")[0].strip().upper()
        if opt.upper() in prediction_upper or f"({opt_letter})" in prediction_upper:
            return opt_letter

    # 2. 尝试匹配选项字母
    match = re.search(r"\(?([A-D])\)?", prediction_upper)
    if match:
        return match.group(1)

    return prediction_upper[:1] if prediction_upper else ""


def normalize_text(text: str) -> str:
    """标准化文本：小写、去标点、去多余空格。"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# 客观题指标
# ============================================================

def accuracy(predictions: list[str], references: list[str]) -> float:
    """计算准确率。

    Args:
        predictions: 预测选项列表（标准化后的字母）
        references: 参考答案选项列表

    Returns:
        准确率 (0-1)
    """
    if not predictions:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


def f1_score(predictions: list[str], references: list[str]) -> float:
    """计算 token 级 F1 分数（宏平均）。

    对多选题，将选项文本进行 token 级匹配。
    """
    if not predictions:
        return 0.0

    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(normalize_text(pred).split())
        ref_tokens = set(normalize_text(ref).split())

        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue

        common = pred_tokens & ref_tokens
        if not common:
            f1_scores.append(0.0)
            continue

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


# ============================================================
# 摘要题指标
# ============================================================

def rouge_l(predictions: list[str], references: list[str]) -> float:
    """计算 ROUGE-L 分数（平均）。

    使用 rouge-score 库。
    """
    from rouge_score import rouge_scorer

    if not predictions:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)

    return float(np.mean(scores))


def bert_score(predictions: list[str], references: list[str]) -> float:
    """计算 BERTScore F1（平均）。

    使用 bert-score 库。注意：首次运行会下载模型。
    """
    if not predictions:
        return 0.0

    from bert_score import score

    # 过滤空字符串
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not valid_pairs:
        return 0.0

    preds = [p for p, _ in valid_pairs]
    refs = [r for _, r in valid_pairs]

    try:
        _, _, f1 = score(preds, refs, lang="en", verbose=False)
        return float(f1.mean())
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}")
        return 0.0


# ============================================================
# 统一评测器
# ============================================================

class Evaluator:
    """统一评测器，支持多选题和开放题。"""

    def __init__(self):
        self._results: list[dict] = []

    def evaluate_mcq(
        self,
        predictions: list[str],
        references: list[str],
        options_list: list[list[str]],
    ) -> dict:
        """评测多选题。

        Args:
            predictions: 模型原始预测文本列表
            references: 参考答案选项字母列表
            options_list: 每题的选项列表

        Returns:
            包含 accuracy 和 f1 的字典
        """
        # 标准化预测
        normalized_preds = []
        for pred, options in zip(predictions, options_list):
            normalized_preds.append(extract_option_from_prediction(pred, options))

        normalized_refs = [normalize_option(r) for r in references]

        acc = accuracy(normalized_preds, normalized_refs)
        f1 = f1_score(normalized_preds, normalized_refs)

        return {
            "accuracy": acc,
            "f1": f1,
            "num_samples": len(predictions),
            "predictions": normalized_preds,
            "references": normalized_refs,
        }

    def evaluate_summarization(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict:
        """评测摘要题。"""
        return {
            "rouge_l": rouge_l(predictions, references),
            "bert_score": bert_score(predictions, references),
            "num_samples": len(predictions),
        }
