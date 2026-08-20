"""推：推理并做 token 计量（对标久安「计量计费」埋点）。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from ..common import MODELS, apply_infer_overrides, load_config, mock_mode, model_path
from . import vllm_backend

SYSTEM_PROMPT = "你是应急管理领域的专业助手，回答需准确、简洁、可执行。"

# 简单进程内缓存，避免重复加载真实模型
_REAL_CACHE: dict = {}


def _resolve_model_dir(model_id: str) -> Optional[Path]:
    if model_id in ("base", "", None):
        return None
    d = MODELS / model_id
    return d if d.exists() else None


def _model_meta(model_dir: Optional[Path]) -> dict:
    if not model_dir:
        return {}
    path = model_dir / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _apply_model_contract(cfg: dict, model_dir: Optional[Path], params: Optional[dict]) -> dict:
    """Use the model's own identity unless the caller explicitly overrides it."""
    if params and str(params.get("system_prompt") or "").strip():
        return cfg
    system_prompt = str(_model_meta(model_dir).get("system_prompt") or "").strip()
    if not system_prompt:
        return cfg
    return {**cfg, "infer": {**cfg.get("infer", {}), "system_prompt": system_prompt}}


def _estimate_tokens(text: str) -> int:
    """离线粗估：中文按字、英文按词，仅用于 mock 计量演示。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    non_cjk = len([w for w in "".join(
        " " if "\u4e00" <= ch <= "\u9fff" else ch for ch in text
    ).split()])
    return max(1, cjk + non_cjk)


def clear_gpu_cache():
    """清理推理模型缓存和 GPU 显存。

    在自动化迭代中，每轮训练/评测后调用此函数，
    确保上一轮的模型从 GPU 释放，避免下一轮 OOM。
    """
    global _REAL_CACHE
    _REAL_CACHE = {}
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _mock_generate(model_dir: Optional[Path], prompt: str, log: Callable) -> str:
    if model_dir and (model_dir / "memory.json").exists():
        memory = json.loads((model_dir / "memory.json").read_text(encoding="utf-8"))
        if prompt.strip() in memory:
            log("[MOCK] 命中微调记忆")
            return memory[prompt.strip()]
        # 简单最长公共前缀匹配
        for q, a in memory.items():
            if q[:6] and q[:6] in prompt:
                log("[MOCK] 近似命中微调记忆")
                return a
    log("[MOCK] 未命中，返回占位回答")
    return f"[mock 回答] 关于「{prompt}」，请补充更多上下文。（启用真实模型请设 JIUAN_MOCK=0）"


def _is_lora_adapter(weights_dir: Path) -> bool:
    return (weights_dir / "adapter_config.json").exists()


def _has_full_model(weights_dir: Path) -> bool:
    return any(
        (weights_dir / f).exists()
        for f in ("config.json", "model.safetensors", "pytorch_model.bin")
    )


def _load_real(model_dir: Optional[Path], cfg: dict, log: Callable):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = str(model_dir or "base")
    if key not in _REAL_CACHE:
        import torch

        # 加载新模型前，先清理 GPU 缓存（避免上一轮训练/推理的残留显存导致 OOM）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        base = model_path(cfg)
        if model_dir and (model_dir / "meta.json").exists():
            try:
                meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
                base = meta.get("base_model_path") or meta.get("base") or base
            except Exception:
                pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_dir = (model_dir / "weights") if model_dir else None

        if weights_dir and _is_lora_adapter(weights_dir):
            # LoRA(hf / LLaMA-Factory)：加载基座 + adapter
            from peft import PeftModel

            log(f"加载基座 {base} + LoRA adapter {weights_dir}（设备 {device}）")
            tok_src = weights_dir if (weights_dir / "tokenizer_config.json").exists() else base
            tok = AutoTokenizer.from_pretrained(tok_src)
            model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
            model = PeftModel.from_pretrained(model, str(weights_dir))
        elif weights_dir and _has_full_model(weights_dir):
            # 全量微调产物
            log(f"加载全量微调模型：{weights_dir}（设备 {device}）")
            tok = AutoTokenizer.from_pretrained(weights_dir)
            model = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype="auto")
        else:
            # 未微调或找不到产物，回退基座
            log(f"加载基座模型：{base}（设备 {device}）")
            tok = AutoTokenizer.from_pretrained(base)
            model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")

        model = model.to(device)
        model.eval()
        _REAL_CACHE[key] = (tok, model, device)
    return _REAL_CACHE[key]


def _real_generate(model_dir: Optional[Path], prompt: str, cfg: dict, log: Callable):
    tok, model, device = _load_real(model_dir, cfg, log)
    system_prompt = cfg.get("infer", {}).get("system_prompt") or SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    icfg = cfg["infer"]
    do_sample = bool(icfg.get("do_sample", False))
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    gen_kwargs = dict(
        max_new_tokens=icfg["max_new_tokens"],
        do_sample=do_sample,
        repetition_penalty=icfg.get("repetition_penalty", 1.0),
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    # 仅在采样时传采样参数，避免 greedy 下触发无效 flag 警告 / 输出跑偏
    if do_sample:
        gen_kwargs["temperature"] = icfg["temperature"]
        gen_kwargs["top_p"] = icfg["top_p"]

    import torch

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[1]:]
    completion_tokens = int(gen.shape[0])
    answer = tok.decode(gen, skip_special_tokens=True).strip()
    return answer, prompt_tokens, completion_tokens


def _real_generate_batch(model_dir: Optional[Path], prompts: list, cfg: dict, log: Callable, on_item=None):
    """批量真实推理：模型只加载一次，逐条 generate。

    避免评测时每条重新 _load_real 的开销；tokenizer 也只初始化一次。
    （为稳定起见仍逐条 forward；后续可改真正的 padding batch，vLLM 则天然支持）
    """
    tok, model, device = _load_real(model_dir, cfg, log)
    icfg = cfg["infer"]
    do_sample = bool(icfg.get("do_sample", False))
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    gen_kwargs = dict(
        max_new_tokens=icfg["max_new_tokens"],
        do_sample=do_sample,
        repetition_penalty=icfg.get("repetition_penalty", 1.0),
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = icfg["temperature"]
        gen_kwargs["top_p"] = icfg["top_p"]

    import torch

    out = []
    system_prompt = cfg.get("infer", {}).get("system_prompt") or SYSTEM_PROMPT
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            gen_out = model.generate(**inputs, **gen_kwargs)
        gen = gen_out[0][inputs["input_ids"].shape[1]:]
        completion_tokens = int(gen.shape[0])
        answer = tok.decode(gen, skip_special_tokens=True).strip()
        out.append((answer, prompt_tokens, completion_tokens))
        if on_item:
            on_item(len(out), len(prompts))
    return out


def generate_batch(
    model_id: str,
    prompts: list,
    log: Callable[[str], None],
    backend: Optional[str] = None,
    params: Optional[dict] = None,
    on_item: "Optional[Callable[[int, int], None]]" = None,
) -> list:
    """批量推理：相比逐条 generate() 只加载一次模型。

    返回 list[dict]，结构与 generate() 一致。on_item(i, total) 用于上报进度。
    """
    cfg = load_config()
    if params:
        cfg = apply_infer_overrides(cfg, params)
    model_dir = _resolve_model_dir(model_id)
    cfg = _apply_model_contract(cfg, model_dir, params)
    chosen = _resolve_backend(cfg, backend, model_dir)
    total = len(prompts)
    log(f"批量推理后端: {chosen}（{total} 条，模型仅加载一次）")

    results = []
    if chosen == "transformers":
        triples = _real_generate_batch(model_dir, prompts, cfg, log, on_item=on_item)
        metering = "tokenizer"
        for i, (answer, pt, ct) in enumerate(triples, 1):
            results.append(_pack(model_id, chosen, prompts[i - 1], answer, pt, ct, metering, 0.0))
        return results

    # mock / vllm：逐条走现有路径（vllm 服务端本身已支持并发）
    for i, prompt in enumerate(prompts, 1):
        results.append(generate(model_id, prompt, log=lambda _m: None, backend=backend, params=params))
        if on_item:
            on_item(i, total)
    return results


def _pack(model_id, chosen, prompt, answer, prompt_tokens, completion_tokens, metering, latency):
    return {
        "model_id": model_id,
        "mode": "mock" if chosen == "mock" else "real",
        "backend": chosen,
        "prompt": prompt,
        "answer": answer,
        "latency_s": latency,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "metering": metering,
        },
    }


def _resolve_backend(cfg: dict, params_backend: Optional[str], model_dir: Optional[Path] = None) -> str:
    """解析推理后端：auto|transformers|vllm|mock。

    - mock 环境（未装 torch/transformers 或 JIUAN_MOCK=1）→ mock。
    - mock 训练产物（memory.json）→ mock。
    - 显式指定则用指定后端。
    - auto：vLLM 服务可达则用 vllm，否则回退 transformers。
    """
    if mock_mode():
        return "mock"
    if model_dir and (model_dir / "memory.json").exists() and str(params_backend or "auto").lower() in ("", "auto"):
        return "mock"
    backend = (params_backend or cfg.get("infer", {}).get("backend") or "auto").lower()
    if backend == "auto":
        return "vllm" if vllm_backend.available(cfg) else "transformers"
    return backend


def generate(
    model_id: str,
    prompt: str,
    log: Callable[[str], None],
    backend: Optional[str] = None,
    params: Optional[dict] = None,
) -> dict:
    cfg = load_config()
    if params:
        cfg = apply_infer_overrides(cfg, params)
    model_dir = _resolve_model_dir(model_id)
    cfg = _apply_model_contract(cfg, model_dir, params)
    chosen = _resolve_backend(cfg, backend, model_dir)
    log(f"推理后端: {chosen}")
    t0 = time.time()
    if chosen == "mock":
        answer = _mock_generate(model_dir, prompt, log)
        # 计量埋点：mock 下用离线估算，明确标注 estimated
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(answer)
        metering = "estimated"
    elif chosen == "vllm":
        answer, prompt_tokens, completion_tokens = vllm_backend.generate(
            model_dir, model_id, prompt, cfg, log
        )
        metering = "vllm-usage"
    elif chosen == "transformers":
        answer, prompt_tokens, completion_tokens = _real_generate(
            model_dir, prompt, cfg, log
        )
        metering = "tokenizer"
    else:
        raise ValueError(f"未知推理后端: {chosen}")
    latency = round(time.time() - t0, 3)
    return {
        "model_id": model_id,
        "mode": "mock" if chosen == "mock" else "real",
        "backend": chosen,
        "prompt": prompt,
        "answer": answer,
        "latency_s": latency,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "metering": metering,
        },
    }


def run(params: dict, log: Callable[[str], None]) -> dict:
    return generate(
        params.get("model_id", "base"),
        params["prompt"],
        log,
        backend=params.get("backend"),
        params=params,
    )



