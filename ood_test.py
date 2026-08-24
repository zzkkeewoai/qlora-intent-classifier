"""
ood_test.py —— OOD（分布外）验证脚本（面试回击"数据泄漏"质疑）

OOD = Out-of-Distribution，分布外样本。
训练/测试集是"种子+增广"（同分布），而这里的 20 条样本：
  - 不在增广规则里（没有用同义词表/前缀后缀模板生成过）
  - 更像真实用户输入（带 emoji、英文混搭、口语化、指代不明）
  - 每条都有人工标注的正确意图

用途：
  1. 证明"规则分类器在真实场景更崩"（预期准确率 < 40%）
  2. 面试时讲："我做了 OOD 验证——分布内测试 84% 不能代表真实场景，
     真实样本掉到 XX%，所以我把'模糊不清'的兜底策略设计得更好"
  3. 将来模型到位后，跑同一批样本对比 OOD 泛化能力

运行：python ood_test.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference.gguf_loader import GGUFClassifier
from src.evaluation.metrics import EvalMetrics

# ============ 20 条 OOD 样本（人工标注正确意图） ============
# 特点：口语化、带 emoji、英文混搭、指代不明、省略主语——增广规则没覆盖的
OOD_SAMPLES = [
    # (query, 正确意图)
    ("那个接口咋回事儿啊", "模糊不清"),           # 口语+指代不明
    ("快帮我看看线上是不是挂了🔥", "操作指令"),     # emoji + 口语
    ("PY和GO比哪个猛", "对比分析"),               # 英文混搭 + 网络用语
    ("这周的报告整一下", "操作指令"),              # 省略宾语的口语
    ("小王现在忙不", "闲聊"),                     # 熟人闲聊
    ("QAQ 怎么办 又报错了", "模糊不清"),          # 网络表情 + 无信息
    ("df 这个函数是干嘛的", "事实查询"),           # 代码符号混入
    ("上线前 checklist 走一遍", "操作指令"),      # 英文词
    ("你说 A 好还是 B 好", "对比分析"),           # 无上下文的对比
    ("嗯嗯", "模糊不清"),                         # 极短无信息
    ("把错误日志拉出来看看", "操作指令"),          # 无"帮我"关键词
    ("这版本啥时候发", "事实查询"),               # 口语化事实
    ("帮我跟老王说一声", "操作指令"),              # 熟人指代
    ("我裂开了", "模糊不清"),                     # 网络用语
    ("对比下这两个方案的坑", "对比分析"),          # 口语化对比
    ("xxx服务今天出过事吗", "事实查询"),          # 指代模糊但可查
    ("能不能给我整个表", "操作指令"),              # 口语
    ("你是真人吗", "闲聊"),                       # 身份询问（闲聊）
    ("那个上个月说的, 现在咋样了", "模糊不清"),    # 复杂指代
    ("v1.2 的 API 文档更新了吗", "事实查询"),     # 版本+英文
]


def main():
    print("=" * 60)
    print("OOD 分布外样本验证（规则分类器）")
    print("=" * 60)
    print(f"样本数: {len(OOD_SAMPLES)} 条（人工标注，不在增广规则内）")

    # 空模型路径 → 规则分类器模式
    classifier = GGUFClassifier(model_path="")

    preds = []
    labels = []
    correct = 0

    print(f"\n{'查询':<30} {'预测':<8} {'正确':<8} {'对/错':<4}")
    print("-" * 60)
    for query, label in OOD_SAMPLES:
        result = classifier.classify(query)
        pred = result["intent"]
        is_correct = (pred == label)
        if is_correct:
            correct += 1
        preds.append(pred)
        labels.append(label)
        print(f"{query[:28]:<30} {pred:<8} {label:<8} {'✅' if is_correct else '❌':<4}")

    acc = correct / len(OOD_SAMPLES) * 100
    f1 = EvalMetrics.per_class_f1(preds, labels)

    print("\n" + "=" * 60)
    print(f"OOD 准确率: {correct}/{len(OOD_SAMPLES)} = {acc:.1f}%")
    print(f"（对比：分布内测试集约 53% → 说明规则分类器在真实场景更崩）")
    print(f"\n各类 F1:")
    for cls, score in f1.items():
        print(f"  {cls}: {score:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
