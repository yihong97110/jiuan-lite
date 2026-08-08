"""Shared helpers: paths, config loading, mock detection."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REGISTRY = DATA / "registry"
DATASETS = DATA / "datasets"
MODELS = DATA / "models"
REPORTS = DATA / "reports"
DB_PATH = DATA / "jiuan.db"

for _p in (DATASETS, MODELS, REPORTS, REGISTRY):
    _p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "qwen0.5b.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_path(cfg: dict) -> str:
    """优先返回已下载到本地的模型目录，否则返回 HF 名称（联网拉取）。"""
    local = cfg.get("model", {}).get("local_dir")
    if local:
        p = Path(local)
        if not p.is_absolute():
            p = ROOT / p
        if (p / "config.json").exists():
            return str(p)
    return cfg["model"]["name"]


def resolve_device(cfg: dict) -> str:
    """解析训练/推理设备：auto|cpu|cuda。

    - auto：检测到 CUDA 用 cuda，否则 cpu。
    - cuda：显式要求 GPU；若环境无 CUDA 则回退 cpu 并留待调用方记录告警。
    """
    want = str(cfg.get("train", {}).get("device", "auto")).lower()
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if want == "cpu":
        return "cpu"
    if want == "cuda":
        return "cuda" if has_cuda else "cpu"
    return "cuda" if has_cuda else "cpu"


def resolve_precision(cfg: dict, device: str) -> str:
    """解析训练精度：CPU 强制 fp32；GPU 上 auto 优先 bf16(支持时)否则 fp16。"""
    want = str(cfg.get("train", {}).get("precision", "auto")).lower()
    if device == "cpu":
        return "fp32"
    if want in ("fp16", "bf16", "fp32"):
        return want
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16"
    except Exception:
        pass
    return "fp16"


def mock_mode() -> bool:
    """默认 REAL：只要 torch+transformers 可用就走真实模型。

    - 默认(JIUAN_MOCK 未设或非 "1")：尝试真实模型；torch/transformers 不可用则自动回退 mock。
    - JIUAN_MOCK=1：强制 mock（无需依赖，用于纯演示链路）。
    """
    if os.environ.get("JIUAN_MOCK", "0") == "1":
        return True
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return False
    except Exception:
        return True


def apply_train_overrides(cfg: dict, params: dict) -> dict:
    """把请求/前端传入的训练超参覆盖到 config（None/空串表示不覆盖）。"""
    train = dict(cfg.get("train", {}))
    lora = dict(train.get("lora", {}))
    model = dict(cfg.get("model", {}))
    base_model_path = params.get("base_model_path") or params.get("model_local_dir")
    base_model_name = params.get("base_model_name")
    if base_model_path:
        model["local_dir"] = str(base_model_path)
        model["name"] = str(base_model_name or base_model_path)
    elif base_model_name:
        model["name"] = str(base_model_name)
    for key in ("method", "epochs", "device", "precision", "lr",
                "batch_size", "grad_accum", "max_seq_len"):
        v = params.get(key)
        if v is not None and v != "":
            train[key] = v
    for pkey, ckey in (("lora_r", "r"), ("lora_alpha", "alpha"), ("lora_dropout", "dropout")):
        v = params.get(pkey)
        if v is not None and v != "":
            lora[ckey] = v
    train["lora"] = lora
    return {**cfg, "model": model, "train": train}


def apply_infer_overrides(cfg: dict, params: dict) -> dict:
    """把请求/前端传入的推理参数覆盖到 config.infer（None/空串表示不覆盖）。"""
    infer = dict(cfg.get("infer", {}))
    for key in ("max_new_tokens", "do_sample", "temperature", "top_p", "system_prompt"):
        v = params.get(key)
        if v is not None and v != "":
            infer[key] = v
    return {**cfg, "infer": infer}
