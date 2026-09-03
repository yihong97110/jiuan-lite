"""为 v2 训练集生成 bad case 主题的 QA 样本。
针对 v1 评测发现的 judge<=2 bad case：
  - AED 自动体外除颤器使用（v1 judge=1，答成 CPR）
  - 厨房油锅起火扑救（v1 judge=4 但 v4 退步到 judge=2，答"浇水"危险错误）
  - 台风预警信号颜色（v1 judge=2，遗漏蓝色）
每主题生成 5 条同主题不同角度的 QA，避免与 valid 集完全相同。
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

THEMES = [
    {
        "topic": "AED自动体外除颤器",
        "instruction": (
            "针对'AED自动体外除颤器使用'主题生成5条不同角度的QA对，"
            "覆盖：操作步骤、电极片粘贴位置、心律分析时注意事项、放电时机、与CPR的衔接。"
            "每条要明确AED是除颤设备而非按压设备，避免与CPR混淆。"
            "严格输出JSON数组格式：[{\"question\":\"...\",\"answer\":\"...\"}, ...]，5条，"
            "每条answer不超过100字。"
        ),
    },
    {
        "topic": "厨房油锅起火扑救",
        "instruction": (
            "针对'厨房油锅起火扑救'主题生成5条不同角度的QA对，"
            "覆盖：正确扑救方法、为什么不能用水、锅盖盖灭原理、蔬菜降温法、关闭燃气阀门顺序。"
            "每条必须明确'严禁用水浇灭油火'。"
            "严格输出JSON数组格式：[{\"question\":\"...\",\"answer\":\"...\"}, ...]，5条，"
            "每条answer不超过100字。"
        ),
    },
    {
        "topic": "台风预警信号",
        "instruction": (
            "针对'台风预警信号'主题生成5条不同角度的QA对，"
            "覆盖：四种颜色（蓝、黄、橙、红）对应等级、风力等级、防范措施、预警升级含义。"
            "每条必须完整列出蓝色、黄色、橙色、红色四级。"
            "严格输出JSON数组格式：[{\"question\":\"...\",\"answer\":\"...\"}, ...]，5条，"
            "每条answer不超过100字。"
        ),
    },
]

SYSTEM_PROMPT = "你是应急管理领域的专业助手，回答需准确、简洁、可执行。"


def call_deepseek(instruction: str) -> str:
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "你是应急管理领域专家，严格按要求输出JSON。"},
            {"role": "user", "content": instruction},
        ],
        "max_tokens": 1500,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def parse_qa(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def main() -> None:
    all_samples = []
    for theme in THEMES:
        print(f"\n=== 生成主题: {theme['topic']} ===")
        try:
            raw = call_deepseek(theme["instruction"])
            qa_list = parse_qa(raw)
            print(f"  解析到 {len(qa_list)} 条")
            for qa in qa_list:
                sample = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": qa["question"]},
                        {"role": "assistant", "content": qa["answer"]},
                    ]
                }
                all_samples.append(sample)
                print(f"  Q: {qa['question'][:40]}")
        except Exception as e:
            print(f"  ❌ 主题失败: {type(e).__name__}: {e}")

    out_path = ROOT / "data" / "datasets" / "iter-v2-badcase-fix-20260820" / "train.jsonl"
    with open(out_path, "a", encoding="utf-8") as fh:
        for s in all_samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n追加 {len(all_samples)} 条样本到 {out_path}")
    total = sum(1 for _ in open(out_path, encoding="utf-8"))
    print(f"v2 训练集总计: {total} 条")


if __name__ == "__main__":
    main()
