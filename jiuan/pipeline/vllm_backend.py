"""推理后端：vLLM（通过 OpenAI 兼容 HTTP 服务）。

生产推荐模式：用 `vllm serve <model>` 起一个 OpenAI 兼容服务，
jiuan-lite 侧只做一个轻量 HTTP 客户端，解耦、天然支持高并发/连续批处理，
token 计量直接采用服务返回的 usage 字段（对齐久安「计量计费」）。

约束：vLLM 仅支持 Linux + GPU。本机(Windows/CPU)不能真跑 vLLM，
但本后端为纯 HTTP 客户端，可对任意 OpenAI 兼容端点工作（便于本机用 stub 验证逻辑，
上 Linux+GPU 后把配置指向真实 vLLM 服务即可，无需改业务代码）。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ..common import load_config, model_path

SYSTEM_PROMPT = "你是应急管理领域的专业助手，回答需准确、简洁、可执行。"


def _vllm_cfg(cfg: dict) -> dict:
    return cfg.get("vllm", {}) or {}


def available(cfg: Optional[dict] = None) -> bool:
    """探测 vLLM OpenAI 服务是否可达（GET {base_url}/models）。"""
    cfg = cfg or load_config()
    base_url = _vllm_cfg(cfg).get("base_url", "http://127.0.0.1:8001/v1")
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _served_model_name(cfg: dict, model_id: str, model_dir: Optional[Path]) -> str:
    """vLLM 服务里注册的模型名。
    - 若配置显式给了 served_model_name，用它；
    - 否则：微调产物用其权重目录名(或 base)，让运维在 vLLM 侧用 --served-model-name 对齐。
    """
    vc = _vllm_cfg(cfg)
    if vc.get("served_model_name"):
        return vc["served_model_name"]
    if model_dir is not None:
        # 约定：vLLM 侧以该 model_id 作为 served-model-name 或 LoRA 名
        return model_dir.name
    return model_path(cfg)


def generate(model_dir: Optional[Path], model_id: str, prompt: str, cfg: dict, log: Callable):
    """调用 vLLM 的 /chat/completions，返回 (answer, prompt_tokens, completion_tokens)。"""
    vc = _vllm_cfg(cfg)
    icfg = cfg["infer"]
    base_url = vc.get("base_url", "http://127.0.0.1:8001/v1").rstrip("/")
    api_key = vc.get("api_key", "EMPTY")
    served = _served_model_name(cfg, model_id, model_dir)

    payload = {
        "model": served,
        "messages": [
            {"role": "system", "content": icfg.get("system_prompt") or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": icfg["max_new_tokens"],
        "temperature": icfg["temperature"] if icfg.get("do_sample", False) else 0.0,
    }
    if icfg.get("do_sample", False):
        payload["top_p"] = icfg["top_p"]

    log(f"[vLLM] POST {base_url}/chat/completions (model={served})")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=vc.get("timeout", 120)) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as exc:  # noqa: PERF203
        raise RuntimeError(f"连接 vLLM 服务失败({base_url}): {exc}") from exc

    answer = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {}) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return answer, prompt_tokens, completion_tokens
