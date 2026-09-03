"""第 1 步：蒸馏高考志愿填报 v1 训练集(10条) + valid 集(6条)。
用 DeepSeek API，10 类不同角度的 QA，覆盖选大学/选专业/分数线/调剂。
"""
from __future__ import annotations
import os, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["DEEPSEEK_API_KEY"]
URL = "https://api.deepseek.com/v1/chat/completions"
SYSTEM_PROMPT = "你是经验丰富的高考志愿填报指导老师，帮助高中毕业生根据分数、兴趣和就业前景选择适合的大学和专业。回答要具体、实用、可操作。"

# v1 训练集 10 条：覆盖不同角度
TRAIN_PROMPTS = [
    "我高考分数550分（理科，河南），想学计算机，有哪些性价比较高的大学推荐？",
    "选大学时应该优先考虑学校综合排名还是专业排名？为什么？",
    "我对人工智能很感兴趣，应该选什么专业？本科阶段人工智能专业和计算机专业有什么区别？",
    "高考分数刚过一本线，是选普通一本的冷门专业还是二本院校的王牌专业？",
    "填报志愿时如何合理设置冲稳保的梯度？每个梯度差多少分合适？",
    "服从专业调剂有什么利弊？什么情况下建议服从调剂？",
    "文科生580分（湖北），想学法学，五院四系和综合类大学法学专业怎么选？",
    "大学专业选错了入学后还能转专业吗？转专业的难度和要求是什么？",
    "从就业前景看，目前哪些专业最值得报考？哪些专业就业形势严峻？",
    "地域对大学选择有多大影响？在北上广读普通一本和在三四线读211哪个更划算？",
]

# valid 集 6 条：held-out，不参与训练
VALID_PROMPTS = [
    "我分数530分（理科，广东），在省内读还是去外省读更好？",
    "新高考模式下选科对专业选择有多大限制？物理和历史对应的专业范围有什么区别？",
    "本科提前批有哪些类型？什么学生适合报提前批？",
    "大学宿舍条件和城市消费水平应该纳入选大学的考量吗？",
    "公费师范生和定向医学生值得报考吗？有什么利弊？",
    "考研难度大的专业在本科报考时应该回避吗？",
]

def call_deepseek(question: str) -> str:
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请针对以下问题给出专业、实用的回答，控制在150字以内：\n{question}"},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def build_sample(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

def main():
    ds_id = "gaokao-v1-20260820"
    out_dir = ROOT / "data" / "datasets" / ds_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"蒸馏 v1 训练集: {len(TRAIN_PROMPTS)} 条")
    print("=" * 60)

    train_samples = []
    for i, q in enumerate(TRAIN_PROMPTS, 1):
        print(f"\n[{i}/{len(TRAIN_PROMPTS)}] Q: {q[:50]}")
        try:
            ans = call_deepseek(q)
            print(f"  A: {ans[:60]}...")
            train_samples.append(build_sample(q, ans))
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as fh:
        for s in train_samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n训练集写入: {len(train_samples)} 条 -> {out_dir / 'train.jsonl'}")

    print("\n" + "=" * 60)
    print(f"蒸馏 valid 集: {len(VALID_PROMPTS)} 条")
    print("=" * 60)

    valid_samples = []
    for i, q in enumerate(VALID_PROMPTS, 1):
        print(f"\n[{i}/{len(VALID_PROMPTS)}] Q: {q[:50]}")
        try:
            ans = call_deepseek(q)
            print(f"  A: {ans[:60]}...")
            valid_samples.append(build_sample(q, ans))
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")

    with open(out_dir / "valid.jsonl", "w", encoding="utf-8") as fh:
        for s in valid_samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nvalid 集写入: {len(valid_samples)} 条 -> {out_dir / 'valid.jsonl'}")

    # 登记血缘
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from jiuan import registry
        registry.register_dataset(ds_id, parent=None, sample_count=len(train_samples),
                                  source="distill:deepseek/deepseek-v4-flash")
        registry.mark_eval_dataset(ds_id)  # valid 也要标记 held-out
        print(f"\n血缘登记完成: {ds_id}")
    except Exception as e:
        print(f"\n血缘登记跳过: {e}")

if __name__ == "__main__":
    main()
