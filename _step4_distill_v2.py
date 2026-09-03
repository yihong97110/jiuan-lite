"""第 4 步：针对 v1 的 4 个 gap 主题蒸馏 10 条补数据，形成 v2。
v2 = v1(10条) + 补充(10条) = 20 条。
"""
from __future__ import annotations
import os, json, urllib.request, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["DEEPSEEK_API_KEY"]
URL = "https://api.deepseek.com/v1/chat/completions"
SYSTEM_PROMPT = "你是经验丰富的高考志愿填报指导老师，帮助高中毕业生根据分数、兴趣和就业前景选择适合的大学和专业。回答要具体、实用、可操作。"

# 针对 v1 的 4 个 gap 主题，每个 2-3 条
GAP_PROMPTS = [
    # gap1: 省内vs外省 (3条)
    "广东理科考生分数不高，留在省内读和去外省读各有什么优劣？从就业认可度、生活成本、人脉积累角度分析。",
    "高考分数中等，想去外省读大学开拓视野，但担心回本省就业不便，这个顾虑合理吗？如何权衡？",
    "哪些情况建议优先考虑省外大学？哪些情况建议留在本省？",

    # gap2: 新高考选科限制 (3条)
    "新高考3+1+2模式下，首选物理可以报哪些专业？首选历史呢？具体举例说明。",
    "新高考选科要求中的'物理+化学'绑定是什么意思？对专业选择有什么影响？",
    "如果高中选了历史+政治+地理，大学能报理工科专业吗？有哪些限制？",

    # gap3: 本科提前批 (2条)
    "本科提前批包括哪些院校类型？军校、公安、公费师范、免费医学定向各有什么报考条件？",
    "什么类型的学生适合报提前批？提前批没录取会影响后续批次吗？",

    # gap4: 公费师范/定向医学生 (2条)
    "公费师范生的协议内容是什么？毕业后必须回生源地任教几年？违约有什么后果？",
    "定向医学生和普通临床医学专业有什么区别？定向服务期满后能自由择业吗？",
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
        URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def build_sample(q, a):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}

def main():
    v2_id = "gaokao-v2-20260820"
    out_dir = ROOT / "data" / "datasets" / v2_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 复制 v1 训练集
    v1_train = ROOT / "data" / "datasets" / "gaokao-v1-20260820" / "train.jsonl"
    v1_valid = ROOT / "data" / "datasets" / "gaokao-v1-20260820" / "valid.jsonl"
    shutil.copy2(v1_train, out_dir / "train.jsonl")
    shutil.copy2(v1_valid, out_dir / "valid.jsonl")

    print("=" * 60)
    print(f"蒸馏 v2 补数据: {len(GAP_PROMPTS)} 条")
    print("=" * 60)

    new_samples = []
    for i, q in enumerate(GAP_PROMPTS, 1):
        print(f"\n[{i}/{len(GAP_PROMPTS)}] Q: {q[:50]}")
        try:
            ans = call_deepseek(q)
            print(f"  A: {ans[:60]}...")
            new_samples.append(build_sample(q, ans))
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")

    # 追加到 v2 训练集
    with open(out_dir / "train.jsonl", "a", encoding="utf-8") as fh:
        for s in new_samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    total = sum(1 for _ in open(out_dir / "train.jsonl", encoding="utf-8"))
    print(f"\nv2 训练集: {total} 条 (v1=10 + 新增={len(new_samples)})")

    # 登记血缘
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from jiuan import registry
        registry.register_dataset(v2_id, parent="gaokao-v1-20260820",
                                  sample_count=total,
                                  source="distill:deepseek/deepseek-v4-flash")
        print(f"血缘登记完成: {v2_id} (parent=gaokao-v1-20260820)")
    except Exception as e:
        print(f"血缘登记跳过: {e}")

if __name__ == "__main__":
    main()
