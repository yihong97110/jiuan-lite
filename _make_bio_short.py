"""制作10条生物学简述问答数据集（回答40字左右），划分train/valid并登记到registry。"""
import json
import time
from pathlib import Path

ROOT = Path(r"E:\all-ex\jiuan-lite")
DATASETS = ROOT / "data" / "datasets"
REGISTRY = ROOT / "data" / "registry" / "datasets.jsonl"

SYSTEM_PROMPT = "你是专业生物学知识助手，请用简练的语言回答，每题约40字，准确概括核心要点。"

# 10条生物学问答：问题 + 约40字回答
SAMPLES = [
    {
        "q": "什么是细胞？",
        "a": "细胞是生物体结构和功能的基本单位，由细胞膜、细胞质和细胞核构成，能独立进行代谢和繁殖。",
    },
    {
        "q": "DNA的作用是什么？",
        "a": "DNA是遗传信息的载体，通过碱基序列储存遗传指令，能复制并传递给子代，决定生物性状。",
    },
    {
        "q": "什么是光合作用？",
        "a": "光合作用是绿色植物利用光能将二氧化碳和水合成有机物并释放氧气的过程，是生态系统的能量基础。",
    },
    {
        "q": "什么是酶？",
        "a": "酶是活细胞产生的生物催化剂，多为蛋白质，能降低反应活化能，具高效性、专一性和反应条件温和的特点。",
    },
    {
        "q": "什么是基因？",
        "a": "基因是携带遗传信息的DNA片段，决定生物性状，通过指导蛋白质合成来表达遗传功能。",
    },
    {
        "q": "什么是自然选择？",
        "a": "自然选择是环境对个体变异进行筛选，适者生存繁殖、不适者被淘汰，是生物进化的主要动力。",
    },
    {
        "q": "什么是细胞呼吸？",
        "a": "细胞呼吸是细胞氧化分解有机物释放能量并生成ATP的过程，分为有氧呼吸和无氧呼吸两种。",
    },
    {
        "q": "什么是病毒？",
        "a": "病毒是无细胞结构的生物，由核酸和蛋白质外壳构成，必须寄生在活细胞内才能繁殖。",
    },
    {
        "q": "什么是生态系统？",
        "a": "生态系统是生物群落与非生物环境相互作用形成的统一整体，包含生产者、消费者和分解者。",
    },
    {
        "q": "什么是蛋白质？",
        "a": "蛋白质是由氨基酸通过肽键连接形成的生物大分子，承担催化、运输、免疫和结构支持等功能。",
    },
]


def to_chat(q: str, a: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def char_count(s: str) -> int:
    # 统计中文字符+英文字母+数字（不含标点空格），更接近"字数"直观感受
    return sum(1 for c in s if c.isalnum())


def main():
    # 校验每条回答字数
    print("=== 回答字数校验 ===")
    for i, s in enumerate(SAMPLES, 1):
        n = char_count(s["a"])
        flag = "✅" if 35 <= n <= 45 else "⚠️"
        print(f"  {flag} #{i} {s['q']} -> {n}字")
    print()

    # 划分：8条train + 2条valid
    train_samples = SAMPLES[:8]
    valid_samples = SAMPLES[8:]

    # 创建数据集目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dataset_id = f"bio-short-v1-生物学简述-{timestamp}"
    out_dir = DATASETS / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== 创建数据集 ===")
    print(f"  dataset_id: {dataset_id}")
    print(f"  目录: {out_dir}")

    # 写 train.jsonl（8条）
    train_path = out_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(to_chat(s["q"], s["a"]), ensure_ascii=False) + "\n")

    # 写 valid.jsonl（2条）
    valid_path = out_dir / "valid.jsonl"
    with open(valid_path, "w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps(to_chat(s["q"], s["a"]), ensure_ascii=False) + "\n")

    # 写 raw_samples.jsonl（instruction/output格式，便于查看标注）
    raw_path = out_dir / "raw_samples.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(SAMPLES, 1):
            f.write(json.dumps({
                "id": i,
                "instruction": s["q"],
                "output": s["a"],
                "type": "生物学简述",
            }, ensure_ascii=False) + "\n")

    print(f"  ✅ train.jsonl: {len(train_samples)} 条")
    print(f"  ✅ valid.jsonl: {len(valid_samples)} 条")
    print(f"  ✅ raw_samples.jsonl: {len(SAMPLES)} 条（含标注）")

    # 登记 registry（追加一行）
    record = {
        "type": "dataset",
        "dataset_id": dataset_id,
        "parent_dataset": None,
        "sample_count": len(SAMPLES),
        "source": "manual/生物学简述40字",
        "registered_at": time.time(),
        "train_count": len(train_samples),
        "valid_count": len(valid_samples),
        "added_count": len(SAMPLES),
    }
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✅ 已登记到 registry")

    # 打印样本预览
    print(f"\n=== 训练集预览（第1条）===")
    print(json.dumps(to_chat(train_samples[0]["q"], train_samples[0]["a"]), ensure_ascii=False, indent=2))
    print(f"\n=== 验证集预览（第1条）===")
    print(json.dumps(to_chat(valid_samples[0]["q"], valid_samples[0]["a"]), ensure_ascii=False, indent=2))

    print(f"\n{'='*50}")
    print(f"完成！数据集ID: {dataset_id}")
    print(f"路径: {out_dir}")
    print(f"可在平台 ① 标·任务 或 ② 训 中选择该数据集进行LoRA微调")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
