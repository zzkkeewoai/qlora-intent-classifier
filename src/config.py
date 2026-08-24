"""
QLoRA 意图分类器 - 全局配置
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ============================================================
    # 模型配置
    # ============================================================
    BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    BASE_MODEL_NAME = "Qwen2.5-7B-Instruct"
    LORA_RANK = int(os.getenv("LORA_RANK", "16"))
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]

    # ============================================================
    # 训练配置
    # ============================================================
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
    GRADIENT_ACCUMULATION = int(os.getenv("GRADIENT_ACCUMULATION", "4"))
    MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "3"))
    MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))

    # ============================================================
    # 输出路径
    # ============================================================
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/lora_adapter")
    GGUF_OUTPUT = os.getenv("GGUF_OUTPUT", "./output/gguf_model")

    # ============================================================
    # 数据集路径
    # ============================================================
    TRAIN_DATA = os.getenv("TRAIN_DATA", "data/intent_train.json")
    VAL_DATA = os.getenv("VAL_DATA", "data/intent_val.json")
    TEST_DATA = os.getenv("TEST_DATA", "data/intent_test.json")

    # ============================================================
    # 推理服务
    # ============================================================
    INFERENCE_HOST = os.getenv("INFERENCE_HOST", "0.0.0.0")
    INFERENCE_PORT = int(os.getenv("INFERENCE_PORT", "8020"))

    # ============================================================
    # API 对比（评测用）
    # ============================================================
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
