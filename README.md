# QLoRA 大模型领域适配与推理优化

验证一个商业问题：**高频任务能否用开源小模型微调替代大模型 API**，在准确率、延迟、成本三个维度做严格对照。

结论：开源小模型微调处理 **80% 高频任务** + 大模型 API 兜底 **20% 复杂场景**，是高 ROI 架构。

## 技术栈

Python · PyTorch · Unsloth · QLoRA · HuggingFace PEFT · bitsandbytes · llama.cpp · FastAPI

## 完整链路

```
数据构建 → 4-bit 量化微调 → 五组对照评测 → GGUF 量化部署 → FastAPI 服务
```

### 1. 数据工程
- 人工构造 **213 条种子样本**，覆盖 5 类意图（事实查询 / 对比分析 / 操作指令 / 闲聊 / 模糊不清）
- 同义词替换、句式变换、实体名轮换、前后缀随机注入增广至 **2600 条**
- 分层切分：train 1826 / val 387 / test 387，每类含边界 Case
- **OOD 验证**：手工构造 20 条不在增广规则内的真实口语样本（emoji/英文混搭/指代不明），验证分布外泛化

### 2. 参数高效微调（T4 16GB）
- **4-bit NF4 量化** Qwen2.5-7B-Instruct：显存 14GB → 约 5GB，塞进 T4
- LoRA rank=16，挂载全部 7 个注意力投影层，可训练参数仅 **0.1%**
- 3 epoch 约 2 小时（Unsloth 加速）

### 3. 五组对照评测（同一测试集 387 条）
| 实验组 | 准确率 |
|--------|--------|
| 规则分类器基线 | 51.9% |
| Qwen-7B zero-shot | 62% |
| Qwen-7B 5-shot | 71% |
| DeepSeek-chat zero-shot | 76% |
| **Qwen-7B QLoRA 微调** | **84%+** |

含 per-class F1 与混淆矩阵（`src/evaluation/metrics.py`）

### 4. 量化推理部署
- LoRA 合并回 base → `convert_hf_to_gguf.py` 转 GGUF → `llama-quantize` 量化 **q4_K_M**
- 本地 CPU 推理：**p50 0.15s，零成本**
- DeepSeek API：约 1.2s（含网络），持续计费

## 快速开始

```bash
pip install -r requirements.txt

# 构建数据集（213 条种子 → 2600 条）
python main.py --build-dataset

# 规则分类器评测（基线 51.9% 验证）
python main.py --eval-rule

# OOD 分布外验证
python ood_test.py

# DeepSeek API 对照（需配置 DEEPSEEK_API_KEY）
python main.py --eval-api

# 演示模式
python main.py --demo
```

## 目录结构

```
qlora_intent_classifier/
├── src/
│   ├── dataset/       # 种子样本 + 数据增广 + 分层切分
│   ├── training/      # Unsloth QLoRA 训练脚本（Colab）
│   ├── evaluation/    # 评测指标 + 五组对照实验
│   └── inference/     # GGUF 加载 / 规则回退 / FastAPI 服务
├── main.py            # CLI 入口
└── ood_test.py        # OOD 分布外验证
```
