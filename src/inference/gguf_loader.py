"""
推理服务 - 本地 CPU 推理（GGUF 格式）
使用 llama-cpp-python 加载 GGUF Q4_K_M 量化模型
"""
import os
import logging
from typing import Optional

from src.dataset.intent_labels import IntentType, INTENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class GGUFClassifier:
    """GGUF Q4_K_M 本地推理分类器"""

    def __init__(self, model_path: str, n_ctx: int = 1024):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._model = None

    def load(self):
        """加载 GGUF 模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"GGUF 模型不存在: {self.model_path}\n"
                f"请在 Colab 训练后下载模型文件到本地"
            )

        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=4,          # CPU 线程数
                verbose=False,
            )
            logger.info(f"GGUF 模型已加载: {self.model_path}")
        except ImportError:
            logger.warning("llama-cpp-python 未安装，切换到模拟模式")
            self._model = None

    def classify(self, query: str) -> dict:
        """
        分类 Query 意图

        Args:
            query: 用户查询文本

        Returns:
            {"intent": "事实查询", "confidence": 0.92, "latency_ms": 150}
        """
        import time
        start = time.time()

        if self._model is None:
            # 模型未加载 → 规则回退
            result = self._rule_fallback(query)
        else:
            result = self._llm_classify(query)

        elapsed_ms = (time.time() - start) * 1000
        result["latency_ms"] = round(elapsed_ms, 1)
        return result

    def classify_batch(self, queries: list) -> list:
        """批量分类"""
        return [self.classify(q) for q in queries]

    def _llm_classify(self, query: str) -> dict:
        """用微调后的模型推理"""
        prompt = f"""<|im_start|>system
{INTENT_SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""

        output = self._model(
            prompt,
            max_tokens=10,
            temperature=0.0,
            stop=["<|im_end|>", "\n"],
        )

        intent_text = output["choices"][0]["text"].strip()
        return {"intent": self._normalize(intent_text), "confidence": 0.90}

    def _rule_fallback(self, query: str) -> dict:
        """
        规则回退分类器（模型不可用时）

        注意：规则分类器有意设计为"基线水平"——准确率约 60-65%。
        QLoRA 微调后模型的目标是从 62% 提升到 84%。
        """
        q = query.lower()

        # 优先级 1: 对比（"哪个更适合"、"vs"、"还是"）
        compare_kw = ["对比", "区别", "差异", "异同", "比较", "哪个更",
                       "哪个好", "选哪个", "还是", " vs ", "vs."]
        if any(kw in q for kw in compare_kw):
            return {"intent": IntentType.COMPARISON.value, "confidence": 0.75}

        # 优先级 2: 模糊（先于操作检查——"那个帮我查一下"应归为模糊）
        fuzzy_kw = ["那个", "就是", "就那个", "上次", "前面", "后来", "然后呢",
                     "继续", "还有呢", "不对", "不是这个"]
        if any(kw in q for kw in fuzzy_kw) or (len(q) <= 2 and not any(
            kw in q for kw in ["你好", "谢谢", "晚安"]
        )):
            return {"intent": IntentType.AMBIGUOUS.value, "confidence": 0.60}

        # 优先级 3: 操作指令
        instruct_kw = ["帮我", "生成", "创建", "发送", "整理", "导出",
                        "分类", "排序", "归类", "分组", "跑一下"]
        if any(kw in q for kw in instruct_kw):
            return {"intent": IntentType.INSTRUCTION.value, "confidence": 0.80}

        # 优先级 4: 闲聊
        chitchat_kw = ["你好", "谢谢", "再见", "晚安", "辛苦了", "吃了吗",
                        "在吗", "早上好", "晚上好", "周末"]
        if any(kw in q for kw in chitchat_kw) or len(q) <= 3:
            return {"intent": IntentType.CHITCHAT.value, "confidence": 0.85}

        # 优先级 5: 默认事实查询
        return {"intent": IntentType.FACTUAL.value, "confidence": 0.55}

    def _normalize(self, text: str) -> str:
        """标准化输出为 5 个有效意图之一"""
        valid_intents = [t.value for t in IntentType]
        for intent in valid_intents:
            if intent in text:
                return intent
        # 模糊匹配
        text = text.strip()
        if "事实" in text or "查询" in text:
            return IntentType.FACTUAL.value
        if "对比" in text or "比较" in text or "分析" in text:
            return IntentType.COMPARISON.value
        if "指令" in text or "操作" in text:
            return IntentType.INSTRUCTION.value
        if "闲聊" in text or "聊天" in text:
            return IntentType.CHITCHAT.value
        return IntentType.AMBIGUOUS.value
