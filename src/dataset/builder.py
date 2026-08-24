"""
数据集构建器
构造 2000+ 条意图分类样本（Alpaca 格式）
"""
import json
import random
from typing import List, Dict, Tuple
from collections import Counter

from src.dataset.intent_labels import IntentType, INTENT_SYSTEM_PROMPT

# ============================================================
# 种子样本（每类 60-80 条，共 ~350 条核心样本）
# ============================================================
SEED_SAMPLES: Dict[IntentType, List[str]] = {
    IntentType.FACTUAL: [
        "A项目用了什么技术栈", "张伟是谁", "B项目什么时候启动的",
        "李婷负责哪个项目", "智能客服系统依赖什么模块", "微服务架构用了哪些组件",
        "公司的数据库选型是什么", "Redis集群有多少个节点", "哪个部门负责运维",
        "项目A的预算是多少", "王明的工号是多少", "上次迭代加了什么功能",
        "这个接口的QPS限制是多少", "服务器部署在哪个机房", "C项目的交付日期是哪天",
        "用了哪个版本Python", "Neo4j的端口号是多少", "有几台负载均衡",
        "API文档在哪里", "今天的日报谁交了", "日志保留多少天",
        "生产环境有几台服务器", "K8s集群版本是多少", "代码仓库用的什么",
        "张伟的邮箱是多少", "CI/CD用的是哪个工具", "项目A什么时候上线的",
        "谁负责数据库运维", "登录接口的调用频次限制是多少", "消息队列用的什么",
        "监控系统是Prometheus还是Zabbix", "CDN用的哪家", "公司用的什么容器方案",
        "API网关是Kong还是Nginx", "测试覆盖率要求多少", "这个服务的可用率多少",
        "当前的并发用户数是多少", "备份策略是什么", "上次故障持续了多久",
        "谁在负责安全扫描", "OAuth还是JWT", "负载均衡用的什么算法",
        "最新的安全补丁什么时候打的", "审计日志存在哪里", "防火墙规则谁管理",
        "Docker镜像仓库地址", "MySQL最大连接数多少", "Redis内存上限多少",
        "Elasticsearch有几个节点", "RabbitMQ的队列积压多少", "这个系统每天处理多少请求",
    ],
    IntentType.COMPARISON: [
        "A和B的技术栈有什么区别", "对比Python和Go的性能",
        "项目A和项目C哪个更紧急", "微服务和单体架构的优缺点",
        "Redis和Memcached哪个适合我们", "Docker和K8s的关系是什么",
        "这两个方案成本差多少", "A组和B组的效率对比",
        "灰度发布和蓝绿部署的区别", "REST和gRPC怎么选",
        "自建机房和云服务的成本对比", "MySQL和PostgreSQL在JSON查询上哪个好",
        "同步和异步的处理方式有什么区别", "对比一下这几次迭代的Bug数量",
        "Kafka和RabbitMQ选哪个", "React和Vue哪个更适合我们这个项目",
        "MongoDB和PostgreSQL在文档存储上的对比", "JWT和Session的优缺点",
        "GitFlow和TrunkBased哪种好", "HTTP和gRPC延迟对比",
        "Jenkins和GitLab CI哪个更好用", "单体和微服务在调试效率上的区别",
        "Elasticsearch和ClickHouse在日志分析上的对比", "对比一下Java和Go的启动速度",
        "AIOps和传统运维的差异", "集中式日志和分布式日志的优劣",
        "对比下Apache和Nginx在高并发下的表现", "RPC和消息队列的区别",
        "Istio和Linkerd选哪个做Service Mesh", "对比下各云厂商的CDN价格",
        "自建K8s和托管K8s的成本差异", "对比下这次和上次发布的性能数据",
    ],
    IntentType.INSTRUCTION: [
        "帮我生成一份周报", "把这份数据整理成表格",
        "发一封邮件给王明", "把这些需求按优先级排序",
        "创建一个新的Git分支", "更新一下接口文档",
        "帮我把这些日志分析一下", "生成一份项目进度报告",
        "把这些Bug按严重程度分类", "通知所有人明天开会",
        "导出上个月的销售数据", "帮我写一段登录接口的代码",
        "把这些会议纪要整理成待办事项", "跑一下回归测试",
        "重启一下测试环境的服务", "把这份配置同步到所有节点",
        "给这个PR加个reviewer", "设置一下自动扩缩容规则",
        "把这段代码合到develop分支", "更新下依赖包的版本",
        "清理一下没用的Docker镜像", "给数据库加个只读副本",
        "配置一下HTTPS证书", "给这个接口加个限流",
        "把错误日志等级调到DEBUG", "给监控加个告警规则",
        "部署一个金丝雀发布", "回滚到上一个稳定版本",
        "给所有的Pod打个标签", "导出这个月的费用报表",
        "关闭这三个不用的测试环境", "扩容两台机器加到集群",
    ],
    IntentType.CHITCHAT: [
        "你好", "谢谢", "今天忙吗", "辛苦了",
        "吃了吗", "周末过的怎么样", "这个系统是谁做的",
        "你是什么模型", "能做什么", "怎么用",
        "有人在吗", "早上好", "晚安", "再见",
        "算了不问了", "没关系", "再想想", "不错",
        "好的谢谢", "OK", "嗯好", "明白了", "收到",
        "太棒了", "搞定了", "辛苦了早点下班", "周末快乐",
        "最近忙不忙啊", "天气好热", "今天周五了开心",
        "有人吗，我需要帮助", "谢谢你，很有帮助", "回头再聊",
        "不好意思打扰了", "这个回答很专业啊", "学到了",
        "能再讲详细点吗", "讲得挺好的", "暂时不需要了",
        "先这样吧", "有问题我再问你", "感谢你的时间",
        "辛苦啦", "真的很感谢", "太厉害了", "下次见",
        "休息会儿", "加班要注意身体", "今天的任务完成了",
    ],
    IntentType.AMBIGUOUS: [
        "那个东西帮我查一下", "怎么样", "帮我看看", "就那个",
        "那个谁的项目", "后来呢", "然后", "那怎么办",
        "再详细一点", "继续", "不对", "不是这个",
        "还有呢", "就那个", "上次说的", "前面那个",
        "查一下", "帮我看下", "嗯", "哦", "好吧",
        "就是上面那个", "跟前面有关那个", "那啥", "额",
        "怎么说呢", "就是那个嘛", "你懂的", "这个那个",
        "那件事情", "之前说的那个", "又出问题了", "还是不行",
        "好像不对", "重新来", "算了不对", "等一下",
        "刚刚发的那个", "你再看看", "搞什么", "莫名其妙",
        "还是老问题", "跟上回一样", "怎么又来了", "不是说了吗",
        "就按上次那个来", "跟上次那个一样就行", "之前的", "再试一次",
    ],
}

# ============================================================
# 同义词替换表 + 句式变化模板
# ============================================================
REPLACEMENTS = {
    "帮我": ["请帮我", "麻烦帮我", "能不能帮我", "帮忙", "麻烦你帮我"],
    "查一下": ["查查", "查一查", "看一下", "看看", "查下", "瞅一眼"],
    "有什么区别": ["有什么不同", "有什么差异", "区别在哪", "哪个更好", "差异在哪"],
    "怎么样": ["如何", "怎么样呢", "好不好", "行不行", "啥情况"],
    "什么": ["哪些", "啥东西", "什么内容", "啥", "什么东西"],
    "项目": ["工程", "系统", "服务", "模块"],
    "哪个": ["哪一个", "哪个方案", "哪个工具", "哪一款"],
    "好不好": ["行不行", "怎么样", "合适吗", "行么"],
    "为什么": ["为何", "什么原因", "啥原因", "怎么搞的"],
    "怎么办": ["怎么处理", "怎么解决", "怎么搞", "如何处理"],
    "在哪里": ["在哪", "什么地方", "哪个位置", "哪里"],
    "什么时候": ["啥时候", "何时", "哪个时间", "哪一天"],
    "多少钱": ["什么价格", "多少费用", "啥价位", "花多少"],
    "怎么配置": ["如何配置", "配置方法是什么", "怎么设置", "咋配"],
}

PREFIXES = ["问一下，", "请问，", "我想知道，", "麻烦问下，",
            "咨询一下，", "请教下，", "问个问题，", "你好请问，",
            "想问下，", "帮我看下，", ""]

SUFFIXES = ["？", "，谢谢", "，麻烦了", "，多谢", "，感谢",
            "，在线等", "，挺急的", "，需要帮忙", ""]

ENTITY_NAMES = ["张伟", "李婷", "王明", "赵强", "孙丽", "周磊",
                "A系统", "B平台", "C项目", "D模块", "E服务", "F应用",
                "支付服务", "订单系统", "用户中心", "消息网关", "日志平台"]


def _random_entity() -> str:
    return random.choice(ENTITY_NAMES)


def _augment_sample(q: str, n_variants: int = 12) -> List[str]:
    """对单条 query 生成 n_variants 条变体"""
    q_clean = q.rstrip("？。！，, ")
    variants = set()

    for _ in range(n_variants * 2):  # 多试几次，去重后取 n_variants
        new_q = q_clean

        # 1. 随机替换实体名（30%概率）
        if random.random() < 0.3:
            new_q = new_q.replace(
                random.choice([e for e in ENTITY_NAMES if e in new_q] or ["X"]),
                _random_entity()
            )

        # 2. 随机替换 1-3 个关键词
        replaced = 0
        for old_word, new_words in REPLACEMENTS.items():
            if old_word in new_q and replaced < 3 and random.random() < 0.35:
                new_q = new_q.replace(old_word, random.choice(new_words), 1)
                replaced += 1

        # 3. 随机加前缀
        if random.random() < 0.25:
            new_q = random.choice(PREFIXES) + new_q

        # 4. 随机加后缀
        if random.random() < 0.30:
            new_q = new_q + random.choice(SUFFIXES)

        # 5. 调整语序（20%概率）
        if random.random() < 0.2 and "和" in new_q:
            parts = new_q.split("和")
            if len(parts) == 2:
                new_q = parts[1] + "和" + parts[0]

        if new_q != q_clean and len(new_q) >= 3:
            variants.add(new_q)

        if len(variants) >= n_variants:
            break

    return list(variants)


# ============================================================
# 主构建函数
# ============================================================

def build_dataset(
    target_train: int = 2000,
    val_test_ratio: float = 0.15,
    augment: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    构建训练/验证/测试集

    Args:
        target_train: 目标训练集大小（默认 2000）
        val_test_ratio: 验证/测试集比例
        augment: 是否做数据增强

    Returns:
        (train_data, val_data, test_data) — Alpaca 格式
    """
    random.seed(42)

    # Step 1: 种子样本
    all_samples = []
    for intent, queries in SEED_SAMPLES.items():
        for q in queries:
            all_samples.append({
                "instruction": INTENT_SYSTEM_PROMPT,
                "input": q,
                "output": intent.value,
                "intent": intent.value,
            })

    seed_count = len(all_samples)
    print(f"种子样本: {seed_count} 条")

    # Step 2: 数据增强
    if augment:
        augmented = []
        for s in all_samples:
            variants = _augment_sample(s["input"])
            for v in variants:
                augmented.append({
                    "instruction": s["instruction"],
                    "input": v,
                    "output": s["output"],
                    "intent": s["intent"],
                })
        all_samples.extend(augmented)
        print(f"数据增强: +{len(augmented)} 条 → 共 {len(all_samples)} 条")

    # Step 3: 去重
    seen = set()
    unique = []
    for s in all_samples:
        key = s["input"].strip().lower().rstrip("？。！，, ")
        if key and key not in seen:
            seen.add(key)
            unique.append(s)
    all_samples = unique
    print(f"去重后: {len(all_samples)} 条")

    # Step 4: 检查是否需要 LLM 增广（种子+规则增广不够时）
    if len(all_samples) < target_train * 1.3:
        shortage = target_train * 1.3 - len(all_samples)
        print(f"样本不足，还需约 {int(shortage)} 条（将重复采样补齐）")
        # 重复采样来达到目标（带微小扰动避免完全重复）
        _pad_samples = []
        while len(all_samples) + len(_pad_samples) < target_train * 1.3:
            s = random.choice(all_samples)
            # 加微小扰动
            new_input = s["input"]
            if random.random() < 0.5:
                new_input = random.choice(PREFIXES).strip(",，") + new_input
            if random.random() < 0.5:
                new_input = new_input.rstrip("？。") + random.choice(SUFFIXES).strip("，。")
            if new_input != s["input"]:
                _pad_samples.append({
                    "instruction": s["instruction"],
                    "input": new_input,
                    "output": s["output"],
                    "intent": s["intent"],
                })
        all_samples.extend(_pad_samples)
        print(f"补齐后: {len(all_samples)} 条")

    # Step 5: 按标签分组 → 比例切分
    train_data, val_data, test_data = [], [], []

    for intent in [e.value for e in IntentType]:
        class_samples = [s for s in all_samples if s["intent"] == intent]
        random.shuffle(class_samples)
        n = len(class_samples)
        n_test = max(1, int(n * val_test_ratio))
        n_val = max(1, int(n * val_test_ratio))
        n_train = n - n_test - n_val

        train_data.extend(class_samples[:n_train])
        val_data.extend(class_samples[n_train:n_train + n_val])
        test_data.extend(class_samples[n_train + n_val:])

    random.shuffle(train_data)

    print(f"\n构建完成: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # 打印分布
    for name, data in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
        if data:
            dist = Counter(s["intent"] for s in data)
            print(f"  {name} ({len(data)}条): " + " | ".join(
                f"{k}={v}" for k, v in dist.most_common()
            ))

    return train_data, val_data, test_data


def save_datasets(train, val, test, base_dir: str = "data"):
    """保存为 JSON 文件"""
    import os
    os.makedirs(base_dir, exist_ok=True)

    for name, data in [("intent_train.json", train),
                        ("intent_val.json", val),
                        ("intent_test.json", test)]:
        path = os.path.join(base_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {path}: {len(data)} 条")
