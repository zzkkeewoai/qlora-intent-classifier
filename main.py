"""
QLoRA 意图分类器 - 主入口

用法:
  python main.py --build-dataset          构建训练数据集
  python main.py --eval-rule              运行规则分类器 + 评测
  python main.py --eval-api               运行 DeepSeek API 对照 + 评测
  python main.py --server                 启动 FastAPI 推理服务
  python main.py --demo                   演示模式（规则分类器演示）
"""
import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset.builder import build_dataset, save_datasets
from src.dataset.intent_labels import IntentType
from src.inference.gguf_loader import GGUFClassifier
from src.evaluation.metrics import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cmd_build_dataset():
    """构建数据集"""
    print("=" * 60)
    print("  意图分类数据集构建")
    print("=" * 60)

    train, val, test = build_dataset(target_train=2000, augment=True)
    save_datasets(train, val, test, base_dir="data")

    # 类别分布
    from collections import Counter
    for name, data in [("训练集", train), ("验证集", val), ("测试集", test)]:
        dist = Counter(s["intent"] for s in data)
        print(f"\n{name} 分布 ({len(data)} 条):")
        for intent, count in dist.most_common():
            print(f"  {intent}: {count} ({count/len(data)*100:.0f}%)")


def cmd_eval_rule():
    """评测规则分类器"""
    print("=" * 60)
    print("  规则分类器评测")
    print("=" * 60)

    test_path = Config.TEST_DATA
    if not os.path.exists(test_path):
        logger.error(f"测试集不存在: {test_path}，请先运行 --build-dataset")
        return

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 规则分类器测试
    classifier = GGUFClassifier(model_path="")  # 空路径 → 规则模式
    runner = BenchmarkRunner(test_data)

    result = runner.run_qwen_zero_shot(classifier)
    runner.print_comparison()

    # 混淆矩阵
    from src.evaluation.metrics import EvalMetrics
    preds = []
    for item in test_data:
        r = classifier.classify(item["input"])
        preds.append(r["intent"])
    labels = [item["output"] for item in test_data]

    cm = EvalMetrics.confusion_matrix(preds, labels)
    print("\n混淆矩阵:")
    classes = sorted(set(labels))
    header = "         " + "".join(f"{c[:6]:>8}" for c in classes)
    print(header)
    for cls in classes:
        row = f"{cls[:6]:>8}" + "".join(f"{cm[cls][c]:>8}" for c in classes)
        print(row)


def cmd_eval_api():
    """DeepSeek API 对照实验"""
    print("=" * 60)
    print("  DeepSeek API 对照实验")
    print("=" * 60)

    if not Config.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY 未设置，跳过 API 对照实验")
        logger.info("设置方式: set DEEPSEEK_API_KEY=sk-xxx")
        return

    test_path = Config.TEST_DATA
    if not os.path.exists(test_path):
        logger.error(f"测试集不存在: {test_path}")
        return

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    classifier = GGUFClassifier(model_path="")
    runner = BenchmarkRunner(test_data)

    # 1. 规则基线
    runner.run_qwen_zero_shot(classifier)

    # 2. DeepSeek zero-shot
    try:
        runner.run_deepseek_zero_shot(Config.DEEPSEEK_API_KEY)
    except Exception as e:
        logger.error(f"DeepSeek zero-shot 失败: {e}")

    # 3. DeepSeek few-shot
    try:
        runner.run_deepseek_few_shot(Config.DEEPSEEK_API_KEY)
    except Exception as e:
        logger.error(f"DeepSeek few-shot 失败: {e}")

    runner.print_comparison()


def cmd_demo():
    """演示模式"""
    print("=" * 60)
    print("  QLoRA 意图分类器 - 演示模式")
    print("  (规则分类器，模型文件未加载)")
    print("=" * 60)

    classifier = GGUFClassifier(model_path="")

    test_queries = [
        "A项目用了什么技术栈",
        "对比Python和Go的性能差异",
        "帮我生成一份周报发给王明",
        "你好，今天忙吗",
        "那个东西帮我查一下",
        "Redis和Memcached哪个更适合我们的场景",
        "这些Bug按严重程度分个类",
        "项目A的预算是多少",
        "怎么样",
        "帮我看看这个接口的QPS",
    ]

    print(f"\n{'查询':<40} {'意图':<10} {'置信度':>8} {'延迟':>8}")
    print("-" * 70)

    for q in test_queries:
        result = classifier.classify(q)
        print(f"{q[:38]:<40} {result['intent']:<10} {result['confidence']:>7.1%}  "
              f"{result['latency_ms']:>6.0f}ms")


def cmd_server():
    """启动推理服务"""
    import uvicorn
    from src.inference.server import app

    print(f"\nLLM 推理服务: http://localhost:{Config.INFERENCE_PORT}")
    print(f"API 文档: http://localhost:{Config.INFERENCE_PORT}/docs")
    uvicorn.run(app, host=Config.INFERENCE_HOST, port=Config.INFERENCE_PORT)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QLoRA 意图分类器")
    parser.add_argument("--build-dataset", action="store_true", help="构建训练数据集")
    parser.add_argument("--eval-rule", action="store_true", help="规则分类器评测")
    parser.add_argument("--eval-api", action="store_true", help="DeepSeek API 对照实验")
    parser.add_argument("--server", action="store_true", help="启动 FastAPI 推理服务")
    parser.add_argument("--demo", action="store_true", default=True, help="演示模式（默认）")
    args = parser.parse_args()

    if args.build_dataset:
        cmd_build_dataset()
    elif args.eval_rule:
        cmd_eval_rule()
    elif args.eval_api:
        cmd_eval_api()
    elif args.server:
        cmd_server()
    else:
        cmd_demo()


if __name__ == "__main__":
    main()
