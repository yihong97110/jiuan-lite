"""蒸馏：调用外部大模型API自动生成SFT训练数据。

用户输入模型名称、平台、API key，自动生成50条多类型问答对，
保存为jsonl格式，可直接用于数据标注和训练。

改进：
1. 调用前先测试API连通性，失败立即报错
2. 每条生成实时写入日志（包含问题和回答内容）
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable

from ..common import DATASETS, ROOT
from .. import registry

# 问题类型模板（50条 = 10类 × 5条/类）
QUESTION_TYPES = [
    {
        "type": "概念解释",
        "prompt": "生成5个关于{domain}的概念解释类问题，要求覆盖核心术语和基础概念。",
    },
    {
        "type": "原理机制",
        "prompt": "生成5个关于{domain}的原理机制类问题，要求涉及工作原理和内在逻辑。",
    },
    {
        "type": "流程步骤",
        "prompt": "生成5个关于{domain}的流程步骤类问题，要求包含操作流程和实施步骤。",
    },
    {
        "type": "案例分析",
        "prompt": "生成5个关于{domain}的案例分析类问题，要求结合实际场景进行分析。",
    },
    {
        "type": "对比分析",
        "prompt": "生成5个关于{domain}的对比分析类问题，要求比较不同方案的异同。",
    },
    {
        "type": "问题排查",
        "prompt": "生成5个关于{domain}的问题排查类问题，要求涉及常见故障和解决方法。",
    },
    {
        "type": "最佳实践",
        "prompt": "生成5个关于{domain}的最佳实践类问题，要求包含行业标准和推荐做法。",
    },
    {
        "type": "法规标准",
        "prompt": "生成5个关于{domain}的法规标准类问题，要求涉及相关法律法规和标准规范。",
    },
    {
        "type": "技术应用",
        "prompt": "生成5个关于{domain}的技术应用类问题，要求涉及具体技术工具和应用场景。",
    },
    {
        "type": "综合评估",
        "prompt": "生成5个关于{domain}的综合评估类问题，要求涉及效果评估和改进建议。",
    },
]

SYSTEM_PROMPT = "你是一个专业的数据标注专家。请严格按照JSON格式输出，不要包含其他内容。"


def _resolve_base_url(platform: str, base_url: str = "") -> str:
    """根据平台确定API端点。"""
    if base_url:
        return base_url.rstrip("/")
    if platform == "deepseek":
        return "https://api.deepseek.com/v1"
    elif platform == "openai":
        return "https://api.openai.com/v1"
    elif platform == "volcengine":
        return "https://ark.cn-beijing.volces.com/api/v3"
    return "https://api.deepseek.com/v1"


def _call_api(platform: str, model: str, api_key: str, messages: list, base_url: str = "") -> str:
    """调用外部模型API，返回文本回答。"""
    url = f"{_resolve_base_url(platform, base_url)}/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"]


def _test_api_connection(platform: str, model: str, api_key: str, base_url: str) -> None:
    """测试API连通性，失败立即抛出详细错误。"""
    try:
        reply = _call_api(
            platform, model, api_key,
            [{"role": "user", "content": "你好，请回复'API连接成功'"}],
            base_url,
        )
        return reply  # 连接成功
    except urllib.error.HTTPError as e:
        # 读取错误详情
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        if e.code == 401:
            raise ValueError(f"API认证失败(401)：API Key不正确或已过期\n详情: {err_body}")
        elif e.code == 404:
            raise ValueError(f"模型不存在(404)：模型名'{model}'不正确\n详情: {err_body}")
        elif e.code == 429:
            raise ValueError(f"请求频率超限(429)：请稍后重试\n详情: {err_body}")
        else:
            raise ValueError(f"API请求失败(HTTP {e.code})\n详情: {err_body}")
    except urllib.error.URLError as e:
        raise ValueError(f"无法连接到API服务器：{e.reason}\n请检查网络或base_url是否正确")
    except Exception as e:
        raise ValueError(f"API连接测试失败: {e}")


def _generate_batch(
    platform: str, model: str, api_key: str, domain: str,
    q_type: str, prompt: str, base_url: str,
    log: Callable[[str], None],
    batch_idx: int, total_batches: int,
    count: int = 5,
) -> list[dict]:
    """生成一批问答对（一个类型，count条），实时输出每条内容到日志。"""
    full_prompt = prompt.format(domain=domain)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""{full_prompt}

请为每个问题生成专业的回答。输出格式为JSON数组，每个元素包含question和answer字段：
[
  {{"question": "问题内容", "answer": "专业回答"}},
  ...
]

要求：
1. 生成{count}个问题
2. 每个问题要有明确的答案
3. 答案要专业、准确、简洁
4. 只输出JSON，不要有其他文字"""},
    ]

    raw = _call_api(platform, model, api_key, messages, base_url)

    # 解析JSON（容错处理）
    import re
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            items = json.loads(match.group())
        else:
            items = []

    # 转为统一格式，实时输出每条内容
    results = []
    for j, item in enumerate(items[:count]):
        if isinstance(item, dict) and "question" in item and "answer" in item:
            q = item["question"]
            a = item["answer"]
            results.append({
                "instruction": q,
                "output": a,
                "type": q_type,
            })
            # 实时输出每条QA到日志
            log(f"  [{batch_idx}/{total_batches}] {q_type} #{j+1}:")
            log(f"    Q: {q[:80]}{'...' if len(q)>80 else ''}")
            log(f"    A: {a[:80]}{'...' if len(a)>80 else ''}")

    return results


def run(params: dict, log: Callable[[str], None]) -> dict:
    """蒸馏主入口：调用外部API生成50条问答对，保存为数据集。"""
    platform = params.get("platform", "deepseek")
    model = params.get("model", "deepseek-chat")
    api_key = params.get("api_key", "")
    domain = params.get("domain", "应急管理")
    base_url = params.get("base_url", "")
    dataset_name = params.get("name", "distill")
    rename = params.get("rename", "")

    if not api_key:
        raise ValueError("API key不能为空")

    # === 第0步：测试API连通性 ===
    log("=" * 50)
    log("第0步：测试API连通性")
    log(f"  平台: {platform}")
    log(f"  模型: {model}")
    log(f"  端点: {_resolve_base_url(platform, base_url)}")
    log("  正在测试连接...")

    reply = _test_api_connection(platform, model, api_key, base_url)
    log(f"  ✅ API连接成功！模型回复: {reply[:50]}")
    log("=" * 50)

    # === 第1步：开始蒸馏 ===
    log(f"开始蒸馏：领域={domain}")
    log(f"将生成 {len(QUESTION_TYPES)} 类 × 5条 = {len(QUESTION_TYPES)*5} 条问答对")
    log("-" * 50)

    all_samples = []
    total_batches = len(QUESTION_TYPES)

    for i, qt in enumerate(QUESTION_TYPES):
        q_type = qt["type"]
        prompt = qt["prompt"]
        log(f"[{i+1}/{total_batches}] 生成「{q_type}」类问题...")

        try:
            batch = _generate_batch(
                platform, model, api_key, domain,
                q_type, prompt, base_url, log,
                batch_idx=i+1, total_batches=total_batches,
                count=5,
            )
            all_samples.extend(batch)
            log(f"  -> ✅ 生成 {len(batch)} 条")
        except Exception as e:
            log(f"  -> ❌ 生成失败: {e}")

    if not all_samples:
        raise RuntimeError("未能生成任何问答对，请检查API key和模型名称")

    log("-" * 50)
    log(f"蒸馏完成：共生成 {len(all_samples)} 条问答对")

    # === 第2步：保存数据集 ===
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dataset_id = f"{dataset_name}-{timestamp}"
    if rename:
        dataset_id = f"{rename}-{timestamp}"

    out_dir = DATASETS / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始蒸馏数据（instruction/output格式）
    raw_path = out_dir / "raw_samples.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 转为chat格式并切分train/valid（90/10）
    import random
    rng = random.Random(42)
    rng.shuffle(all_samples)

    n_valid = max(1, int(len(all_samples) * 0.1))
    valid_samples = all_samples[:n_valid]
    train_samples = all_samples[n_valid:]

    SYSTEM_PROMPT_DOMAIN = f"你是{domain}领域的专业助手，回答需准确、简洁、可执行。"

    def _to_chat(rec):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_DOMAIN},
                {"role": "user", "content": rec["instruction"]},
                {"role": "assistant", "content": rec["output"]},
            ]
        }

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(_to_chat(s), ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps(_to_chat(s), ensure_ascii=False) + "\n")

    # 登记数据集血缘
    registry.register_dataset(
        dataset_id,
        None,
        len(all_samples),
        source=f"distill:{platform}/{model}",
        extra={
            "train_count": len(train_samples),
            "valid_count": len(valid_samples),
            "platform": platform,
            "model": model,
            "domain": domain,
            "types": [qt["type"] for qt in QUESTION_TYPES],
        },
    )

    # === 第3步：输出汇总 ===
    log("=" * 50)
    log("数据集已保存:")
    log(f"  dataset_id: {dataset_id}")
    log(f"  路径: {out_dir}")
    log(f"  train.jsonl: {len(train_samples)} 条")
    log(f"  valid.jsonl: {len(valid_samples)} 条")
    log(f"  raw_samples.jsonl: {len(all_samples)} 条（原始格式）")

    # 按类型统计
    type_stats = {}
    for s in all_samples:
        t = s.get("type", "未知")
        type_stats[t] = type_stats.get(t, 0) + 1

    log("-" * 50)
    log("各类型统计:")
    for t, c in type_stats.items():
        log(f"  {t}: {c} 条")
    log("=" * 50)

    return {
        "dataset_id": dataset_id,
        "dataset_path": str(out_dir),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "raw_path": str(raw_path),
        "total_count": len(all_samples),
        "train_count": len(train_samples),
        "valid_count": len(valid_samples),
        "type_stats": type_stats,
        "platform": platform,
        "model": model,
        "domain": domain,
        "samples": [  # 返回所有生成的QA，前端可直接展示
            {
                "type": s.get("type", ""),
                "question": s["instruction"],
                "answer": s["output"],
            }
            for s in all_samples
        ],
    }
