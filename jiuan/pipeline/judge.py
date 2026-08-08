# -*- coding: utf-8 -*-
"""评测后端：LLM-as-Judge（通过 OpenAI 兼容 HTTP 服务打分）。

对标久安「模型评测系统」：用一个更强的裁判模型给待测模型的回答打分，
弥补 ROUGE/BLEU 只做字面匹配、无法衡量语义与事实正确性的缺陷。

裁判端点走 OpenAI 兼容协议（可指向本地 vLLM、Ollama 或云端 API），
与推理后端 vllm_backend 复用同一套调用方式；未配置或不可达时优雅回退，
由 evaluate.py 决定是否降级到 ROUGE/BLEU。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import time
from typing import Callable, Optional

JUDGE_SYSTEM = (
    "你是客观、宽容且领域中立的答案评审专家。请依据问题和参考答案，从"
    "事实正确性、相关性、完整性和可执行性四个维度综合评估待测回答的质量。"
    "只输出 JSON，不要解释。格式：{\"score\": <0到5的整数>, \"reason\": \"<简短中文理由>\"}。"
    "评分应以是否正确回答核心问题为主，不要求逐字覆盖参考答案，也不要因表达简略、措辞不同或"
    "遗漏非关键扩展知识过度扣分；除非问题明确询问身份或权限，否则不要把身份声明作为评分条件。"
    "评分标准：5=核心结论与关键机制准确，表达足以使用；4=基本正确，仅有非关键遗漏；"
    "3=核心方向正确但有明显遗漏或轻微混淆；2=关键结论存在较大偏差；1=几乎不相关；0=错误或有害。"
)


def _judge_cfg(cfg: dict) -> dict:
    return cfg.get("judge", {}) or {}


def _resolve_key(jc: dict) -> str:
    """解析裁判 API Key：config.judge.api_key 若为真实值优先，否则读环境变量。

    - 避免把云端密钥明文写进会提交的配置文件。
    - 环境变量名可用 config.judge.api_key_env 指定，默认 JIUAN_JUDGE_API_KEY，
      并回退兼容 ARK_API_KEY（火山方舟）。
    """
    k = jc.get("api_key")
    if k and str(k).strip() not in ("", "EMPTY"):
        return str(k)
    env_name = jc.get("api_key_env", "JIUAN_JUDGE_API_KEY")
    return os.environ.get(env_name) or os.environ.get("ARK_API_KEY") or ""


def available(cfg: dict) -> bool:
    """判定裁判服务是否可用。

    - 必须配置 base_url，且能解析出 api_key（本地无鉴权服务可在 config 显式写 EMPTY）。
    - probe=false（如火山方舟等不暴露 /models 的云端）时，只要 base_url+key 就视为可用。
    - probe=true（默认，适合本地 vLLM）时探测 /models：200→可用；401/403→鉴权失败不可用；
      404/405→服务不提供该接口但可用。
    """
    jc = _judge_cfg(cfg)
    base_url = jc.get("base_url")
    if not base_url:
        return False
    key = _resolve_key(jc)
    require_key = jc.get("require_key", True)
    if require_key and not key:
        return False
    if not jc.get("probe", True):
        return True
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=jc.get("timeout", 5)) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 405):
            return True  # 服务不提供 /models，但 chat 接口可能可用
        return False
    except Exception:
        return False


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from plain or fenced model output."""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for pos, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _post_chat(cfg: dict, payload: dict) -> str:
    jc = _judge_cfg(cfg)
    base_url = jc.get("base_url", "").rstrip("/")
    api_key = _resolve_key(jc) or "EMPTY"
    data = json.dumps(payload).encode("utf-8")
    attempts = max(1, int(jc.get("max_retries", 3)))
    for attempt in range(attempts):
        req = urllib.request.Request(
            base_url + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=jc.get("timeout", 60)) as resp:
                body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt + 1 >= attempts:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"裁判服务 HTTP {exc.code}: {detail or exc.reason}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
            time.sleep(min(delay, 10.0))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"连接裁判服务失败({base_url}): {exc}") from exc
    raise RuntimeError("裁判服务请求失败")


def score_one(cfg: dict, prompt: str, reference: str, prediction: str, log: Callable) -> dict:
    """调用裁判模型给单条打分，返回 {score:0-5, reason:str}。失败抛异常由上层处理。"""
    jc = _judge_cfg(cfg)
    model = jc.get("model", "judge")
    user = (
        f"【问题】\n{prompt}\n\n"
        f"【参考答案】\n{reference}\n\n"
        f"【待测回答】\n{prediction}\n\n"
        "请打分并只输出 JSON。"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    content = _post_chat(cfg, payload)
    parsed = _extract_json(content) or {}
    raw = parsed.get("score", 0)
    try:
        score = int(round(float(raw)))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(5, score))
    return {"score": score, "reason": str(parsed.get("reason", ""))[:200]}


def score_batch(cfg: dict, samples: list[dict], log: Callable) -> list[dict]:
    """Score a whole evaluation batch in one request to avoid Ark plan rate limits."""
    if not samples:
        return []
    jc = _judge_cfg(cfg)
    compact = [
        {
            "index": i,
            "question": str(item.get("prompt") or "")[:1200],
            "reference": str(item.get("reference") or "")[:1800],
            "prediction": str(item.get("prediction") or "")[:1800],
        }
        for i, item in enumerate(samples, 1)
    ]
    payload = {
        "model": jc.get("model", "judge"),
        "messages": [
            {
                "role": "system",
                "content": (
                    JUDGE_SYSTEM
                    + " 本次需要批量评分。只输出 JSON："
                    + '{"results":[{"index":1,"score":0,"reason":"理由"}]}。'
                ),
            },
            {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": min(4096, max(512, len(samples) * 180)),
    }
    content = _post_chat(cfg, payload)
    parsed = _extract_json(content) or {}
    raw_results = parsed.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError(f"Judge 批量返回无法解析为 results 数组: {content[:180]}")
    by_index = {}
    for position, item in enumerate(raw_results, 1):
        try:
            idx = int(item.get("index", position))
            score = max(0, min(5, int(round(float(item.get("score", 0))))))
        except (AttributeError, TypeError, ValueError):
            continue
        by_index[idx] = {"score": score, "reason": str(item.get("reason") or "")[:200]}
    # Some compatible APIs return zero-based indexes even when prompted otherwise.
    if 0 in by_index and len(by_index) == len(samples):
        by_index = {idx + 1: value for idx, value in by_index.items()}
    missing = [i for i in range(1, len(samples) + 1) if i not in by_index]
    if missing:
        raise RuntimeError(f"Judge 批量返回缺少样本索引: {missing}")
    log(f"Judge 单次批量请求完成 {len(samples)} 条评分")
    return [by_index[i] for i in range(1, len(samples) + 1)]
