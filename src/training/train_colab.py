"""
Colab 训练脚本
在 Google Colab T4 16GB 上用 Unsloth 跑 QLoRA 微调 Qwen2.5-7B

使用方法：
  在 Colab 中依次执行以下 Cell：

  Cell 1: !pip install unsloth
  Cell 2: 复制本文件全部内容后执行

适配 Qwen2.5-7B-Instruct，ChatML 格式，5 类意图分类任务
"""
# ============================================================
# Cell 1: 安装依赖（在 Colab 中先执行）
# ============================================================
"""
!pip install unsloth
!pip install --no-deps xformers trl peft accelerate bitsandbytes
"""

# ============================================================
# Cell 2: 加载模型
# ============================================================
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch

# 超参数
max_seq_length = 1024
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.0

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

print(f"模型加载完成: {model.__class__.__name__}")
print(f"显存: {torch.cuda.memory_allocated() / 1024**3:.1f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# Cell 3: 配置 LoRA
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# ============================================================
# Cell 4: 加载并格式化数据集
# ============================================================
from datasets import Dataset
import json

# ChatML 格式化函数
INTENT_SYSTEM = """你是一个查询意图分类器。请将用户查询分类为以下 5 类之一：
1. 事实查询 - 询问具体事实、数据、定义
2. 对比分析 - 要求对两个或多个对象进行比较
3. 操作指令 - 要求执行某个操作
4. 闲聊 - 社交性对话，不涉及具体任务
5. 模糊不清 - 指代不明、信息不完整、无法判断意图
请仅输出意图类别名称，不要输出任何其他内容。"""


def formatting_func(example):
    """将样本转换为 ChatML 格式"""
    return [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]


# 加载数据集（从本地 JSON 文件或 Colab 上传的文件）
import os
train_path = "intent_train.json"

if os.path.exists(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"从本地加载训练集: {len(train_data)} 条")
else:
    # Fallback: 直接内嵌种子数据
    print("本地数据集未找到，使用内嵌种子数据...")
    from train_data_fallback import get_fallback_data
    train_data = get_fallback_data()
    print(f"使用内嵌数据: {len(train_data)} 条")

# 转换为 HuggingFace Dataset
dataset = Dataset.from_list(train_data)
dataset = dataset.map(
    lambda x: {"text": tokenizer.apply_chat_template(
        formatting_func(x), tokenize=False, add_generation_prompt=False
    )},
)

print(f"数据集构建完成: {len(dataset)} 条")
print(f"样例:\n{dataset[0]['text'][:500]}")

# ============================================================
# Cell 5: 训练
# ============================================================
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="./output",
        report_to="none",
    ),
)

print("开始训练...")
trainer_stats = trainer.train()
print(f"训练完成!")
print(f"总步数: {trainer_stats.global_step}")
print(f"训练Loss: {trainer_stats.training_loss:.4f}")

# ============================================================
# Cell 6: 保存模型
# ============================================================
# 保存 LoRA adapter
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")
print("LoRA adapter 已保存到 lora_adapter/")

# 保存为 GGUF 格式（用于本地 CPU 推理）
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
print("GGUF q4_K_M 已保存到 gguf_model/")

# 下载说明
print("\n" + "=" * 60)
print("下载以下文件到本地:")
print("  1. lora_adapter/ → 用于 FastAPI GPU 推理")
print("  2. gguf_model/   → 用于本地 CPU 推理 (llama.cpp)")
print("=" * 60)
