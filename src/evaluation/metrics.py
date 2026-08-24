"""
评测指标计算 + 5 组对照实验基准
"""
import time
import json
import logging
from typing import List, Dict
from collections import defaultdict

from src.dataset.intent_labels import IntentType, INTENT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)


class EvalMetrics:
    """评测指标"""

    @staticmethod
    def accuracy(predictions: List[str], labels: List[str]) -> float:
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(labels) if labels else 0

    @staticmethod
    def per_class_f1(predictions: List[str], labels: List[str]) -> Dict[str, float]:
        """每个类别的 F1 Score"""
        classes = sorted(set(labels))
        f1_scores = {}

        for cls in classes:
            tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
            fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
            fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores[cls] = round(f1, 4)

        return f1_scores

    @staticmethod
    def confusion_matrix(predictions: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
        """混淆矩阵"""
        classes = sorted(set(labels))
        matrix = {cls: {c: 0 for c in classes} for cls in classes}
        for p, l in zip(predictions, labels):
            matrix[l][p] += 1
        return matrix

    @staticmethod
    def avg_latency(latencies: List[float]) -> dict:
        return {
            "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
        }


class BenchmarkRunner:
    """五组对照实验"""

    def __init__(self, test_data: List[Dict]):
        self.test_data = test_data
        self.results = {}

    def run_deepseek_zero_shot(self, api_key: str) -> dict:
        """DeepSeek-chat zero-shot"""
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        predictions = []
        latencies = []
        for item in self.test_data:
            start = time.time()
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": item["input"]},
                ],
                temperature=0.0, max_tokens=20,
            )
            latencies.append((time.time() - start) * 1000)
            predictions.append(resp.choices[0].message.content.strip())

        labels = [item["output"] for item in self.test_data]
        return self._compute_metrics("DeepSeek zero-shot", predictions, labels, latencies)

    def run_deepseek_few_shot(self, api_key: str) -> dict:
        """DeepSeek-chat 5-shot"""
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        few_shot_text = "以下是一些示例：\n"
        for intent, examples in FEW_SHOT_EXAMPLES.items():
            for q, a in examples[:1]:
                few_shot_text += f"用户: {q}\n分类: {a}\n"
        few_shot_text += "\n现在请分类以下查询：\n"

        predictions = []
        latencies = []
        for item in self.test_data:
            start = time.time()
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": few_shot_text + item["input"]},
                ],
                temperature=0.0, max_tokens=20,
            )
            latencies.append((time.time() - start) * 1000)
            predictions.append(resp.choices[0].message.content.strip())

        labels = [item["output"] for item in self.test_data]
        return self._compute_metrics("DeepSeek 5-shot", predictions, labels, latencies)

    def run_qwen_zero_shot(self, classifier) -> dict:
        """Qwen-7B zero-shot（未微调）"""
        predictions = []
        latencies = []
        for item in self.test_data:
            result = classifier.classify(item["input"])
            predictions.append(result["intent"])
            latencies.append(result["latency_ms"])

        labels = [item["output"] for item in self.test_data]
        return self._compute_metrics("Qwen-7B zero-shot", predictions, labels, latencies)

    def run_qwen_few_shot(self, classifier) -> dict:
        """Qwen-7B 5-shot（未微调）"""
        predictions = []
        latencies = []
        few_shot_prefix = "示例:\n"
        for intent, examples in FEW_SHOT_EXAMPLES.items():
            for q, a in examples[:1]:
                few_shot_prefix += f"用户: {q} → {a}\n"
        few_shot_prefix += "\n"

        for item in self.test_data:
            result = classifier.classify(few_shot_prefix + item["input"])
            predictions.append(result["intent"])
            latencies.append(result["latency_ms"])

        labels = [item["output"] for item in self.test_data]
        return self._compute_metrics("Qwen-7B 5-shot", predictions, labels, latencies)

    def run_qwen_qlora(self, classifier) -> dict:
        """Qwen-7B QLoRA 微调后"""
        predictions = []
        latencies = []
        for item in self.test_data:
            result = classifier.classify(item["input"])
            predictions.append(result["intent"])
            latencies.append(result["latency_ms"])

        labels = [item["output"] for item in self.test_data]
        return self._compute_metrics("Qwen-7B QLoRA", predictions, labels, latencies)

    def _compute_metrics(self, name: str, preds: List[str],
                         labels: List[str], latencies: List[float]) -> dict:
        if not preds:
            return {"name": name, "accuracy": 0, "per_class_f1": {}, "min_f1": ("N/A", 0), "latency": {}}

        acc = EvalMetrics.accuracy(preds, labels)
        f1 = EvalMetrics.per_class_f1(preds, labels)
        latency = EvalMetrics.avg_latency(latencies)

        min_f1_cls, min_f1_val = ("N/A", 0)
        if f1:
            min_f1_cls = min(f1, key=f1.get)
            min_f1_val = f1[min_f1_cls]

        result = {
            "name": name,
            "accuracy": round(acc, 4),
            "per_class_f1": f1,
            "min_f1": (min_f1_cls, min_f1_val),
            "latency": latency,
        }
        self.results[name] = result
        logger.info(f"{name}: acc={acc:.2%}, min_f1={min_f1_cls}={min_f1_val:.2%}, "
                     f"p50={latency['p50_ms']}ms")
        return result

    def print_comparison(self):
        """打印五组对比表"""
        print("\n" + "=" * 80)
        print("  五组对照实验结果")
        print("=" * 80)
        print(f"{'实验组':<22} {'准确率':>8} {'最低F1类别':<12} {'最低F1':>8} {'P50延迟':>10}")
        print("-" * 80)

        order = ["DeepSeek zero-shot", "DeepSeek 5-shot",
                  "Qwen-7B zero-shot", "Qwen-7B 5-shot", "Qwen-7B QLoRA"]

        for name in order:
            if name not in self.results:
                continue
            r = self.results[name]
            min_cls, min_val = r["min_f1"]
            print(f"{name:<22} {r['accuracy']:>7.1%}  {min_cls:<12} {min_val:>7.1%}  "
                  f"{r['latency']['p50_ms']:>7.0f}ms")

        print("-" * 80)

        # 提升对比
        if "DeepSeek zero-shot" in self.results and "Qwen-7B QLoRA" in self.results:
            ds_acc = self.results["DeepSeek zero-shot"]["accuracy"]
            ql_acc = self.results["Qwen-7B QLoRA"]["accuracy"]
            improvement = (ql_acc - ds_acc) / ds_acc * 100
            print(f"\nQLoRA vs DeepSeek zero-shot: {ds_acc:.1%} → {ql_acc:.1%} (+{improvement:.0f}%)")
