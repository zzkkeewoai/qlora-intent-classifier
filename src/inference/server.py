"""
FastAPI 推理服务
提供意图分类 API 端点
"""
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.gguf_loader import GGUFClassifier
from src.config import Config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="QLoRA 意图分类器",
    description="Qwen2.5-7B QLoRA 微调后的 Query 意图分类服务",
    version="1.0.0",
)

# 全局分类器实例
classifier: GGUFClassifier = None


class ClassifyRequest(BaseModel):
    """分类请求"""
    query: str
    batch_mode: bool = False


class ClassifyResponse(BaseModel):
    """分类响应"""
    query: str
    intent: str
    confidence: float
    latency_ms: float


class BatchClassifyRequest(BaseModel):
    """批量分类请求"""
    queries: list


class BatchClassifyResponse(BaseModel):
    """批量分类响应"""
    results: list


@app.on_event("startup")
async def startup():
    global classifier
    classifier = GGUFClassifier(model_path=Config.GGUF_OUTPUT + "/unsloth.Q4_K_M.gguf")
    try:
        classifier.load()
    except FileNotFoundError as e:
        logger.warning(f"模型文件未找到，使用规则回退模式: {e}")


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    """单个查询分类"""
    if classifier is None:
        raise HTTPException(503, "分类器未初始化")

    result = classifier.classify(req.query)
    return ClassifyResponse(
        query=req.query,
        intent=result["intent"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
    )


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(req: BatchClassifyRequest):
    """批量分类"""
    if classifier is None:
        raise HTTPException(503, "分类器未初始化")

    results = classifier.classify_batch(req.queries)
    return BatchClassifyResponse(results=results)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": classifier._model is not None if classifier else False}
