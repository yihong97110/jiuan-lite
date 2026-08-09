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
    "你是客观、宽容且领域中立的答案评审专家。请依据问题和参考答案，从以下四个维度独立打分（0-100整数）：\n"
    "1. accuracy（事实准确性）：回答中的事实陈述是否正确无误。\n"
    "2. completeness（完整性）：是否覆盖了问题的关键要点和必要细节。\n"
    "3. relevance（相关性）：回答是否切题，没有跑题或冗余内容。\n"
    "4. hallucination（幻觉检测）：0=无幻觉（所有断言可被参考答案支撑），1=存在幻觉（有无法证实的断言）。\n"
    "并给出 overall（综合评分，0-100整数，加权平均）。\n"
    "只输出 JSON，不要解释。格式：\n"
    '{"accuracy": <0-100>, "completeness": <0-100>, "relevance": <0-100>, "hallucination": <0或1>, "overall": <0-100>, "reason": "<简短中文理由>"}\n'
    "评分标准（以 overall 为例）：90+=优秀，70-89=良好，50-69=及格，30-49=较差，<30=极差。\n"
    "不要求逐字覆盖参考答案；表达简略或措辞不同不应过度扣分；除非问题明确询问身份或权限，否则不要把身份声明作为评分条件。"
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
    """调用裁判模型给单条打分，返回四维度分数+幻觉标记+综合分。失败抛异常由上层处理。"""
    jc = _judge_cfg(cfg)
    model = jc.get("model", "judge")
    user = (
        f"【问题】\n{prompt}\n\n"
        f"【参考答案】\n{reference}\n\n"
        f"【待测回答】\n{prediction}\n\n"
        "请逐维度打分并只输出 JSON。"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    content = _post_chat(cfg, payload)
    parsed = _extract_json(content) or {}

    def _clamp(v, lo=0, hi=100):
        try:
            return max(lo, min(hi, int(round(float(v)))))
        except (TypeError, ValueError):
            return 0

    result = {
        "accuracy": _clamp(parsed.get("accuracy")),
        "completeness": _clamp(parsed.get("completeness")),
        "relevance": _clamp(parsed.get("relevance")),
        "hallucination": 1 if str(parsed.get("hallucination", "0")).strip() in ("1", "true", "True") else 0,
        "overall": _clamp(parsed.get("overall")),
        "reason": str(parsed.get("reason", ""))[:200],
    }
    return result


def score_batch(cfg: dict, samples: list[dict], log: Callable) -> list[dict]:
    """Score a whole evaluation batch in one request to avoid Ark plan rate limits.

    Returns a list of dicts, each containing:
    accuracy, completeness, relevance, hallucination, overall, reason
    """
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
                    + '{"results":[{"index":1,"accuracy":0,"completeness":0,"relevance":0,"hallucination":0,"overall":0,"reason":"理由"}]}。'
                ),
            },
            {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": min(16384, max(1024, len(samples) * 600)),
    }
    content = _post_chat(cfg, payload)
    parsed = _extract_json(content) or {}
    raw_results = parsed.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError(f"Judge 批量返回无法解析为 results 数组: {content[:180]}")

    def _clamp(v, lo=0, hi=100):
        try:
            return max(lo, min(hi, int(round(float(v)))))
        except (TypeError, ValueError):
            return 0

    by_index = {}
    for position, item in enumerate(raw_results, 1):
        try:
            idx = int(item.get("index", position))
        except (AttributeError, TypeError, ValueError):
            continue
        by_index[idx] = {
            "accuracy": _clamp(item.get("accuracy")),
            "completeness": _clamp(item.get("completeness")),
            "relevance": _clamp(item.get("relevance")),
            "hallucination": 1 if str(item.get("hallucination", "0")).strip() in ("1", "true", "True") else 0,
            "overall": _clamp(item.get("overall")),
            "reason": str(item.get("reason") or "")[:200],
        }
    # Some compatible APIs return zero-based indexes even when prompted otherwise.
    if 0 in by_index and len(by_index) == len(samples):
        by_index = {idx + 1: value for idx, value in by_index.items()}
    missing = [i for i in range(1, len(samples) + 1) if i not in by_index]
    if missing:
        raise RuntimeError(f"Judge 批量返回缺少样本索引: {missing}")
    log(f"Judge 单次批量请求完成 {len(samples)} 条评分")
    return [by_index[i] for i in range(1, len(samples) + 1)]
