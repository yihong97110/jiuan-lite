"""Validate a jiuan-lite installation before real model training."""
from __future__ import annotations

import argparse
import importlib
import os
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def _module_status(names: list[str]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
            present.append(name)
        except Exception:
            missing.append(name)
    return present, missing


def _resolve_local_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 jiuan-lite 部署和真实训练前置条件")
    parser.add_argument("--config", default="configs/qwen0.5b.yaml")
    parser.add_argument("--require-real", action="store_true", help="缺少真实训练条件时返回非零")
    args = parser.parse_args()

    config_path = _resolve_local_dir(args.config)
    if not config_path.exists():
        print(f"[FAIL] 配置不存在: {config_path}")
        return 2
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    print(f"项目目录: {ROOT}")
    print(f"配置文件: {config_path}")
    print(f"HF 模型名: {model_cfg.get('name') or '(未配置)'}")

    runtime_names = ["fastapi", "uvicorn", "pydantic", "yaml"]
    train_names = ["torch", "transformers", "datasets", "accelerate", "peft"]
    _, runtime_missing = _module_status(runtime_names)
    _, train_missing = _module_status(train_names)
    print("控制面依赖: " + ("OK" if not runtime_missing else "缺少 " + ", ".join(runtime_missing)))
    print("训练依赖: " + ("OK" if not train_missing else "缺少 " + ", ".join(train_missing)))

    local_raw = str(model_cfg.get("local_dir") or "").strip()
    local_ready = False
    if local_raw:
        local_dir = _resolve_local_dir(local_raw)
        config_ok = (local_dir / "config.json").is_file()
        weights_ok = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("pytorch_model*.bin"))
        tokenizer_ok = any((local_dir / name).is_file() for name in (
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model"
        ))
        local_ready = config_ok and weights_ok and tokenizer_ok
        print(f"本地模型目录: {local_dir}")
        print(f"模型文件: config={'OK' if config_ok else '缺失'}, "
              f"weights={'OK' if weights_ok else '缺失'}, tokenizer={'OK' if tokenizer_ok else '缺失'}")
    else:
        print("本地模型目录: 未配置，将尝试通过 model.name 在线读取")

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    writable = False
    try:
        with tempfile.NamedTemporaryFile(prefix="deploy-check-", dir=data_dir, delete=True):
            writable = True
    except OSError as exc:
        print(f"data 写权限: FAIL ({exc})")
    else:
        print("data 写权限: OK")

    cuda_ok = True
    requested_device = str(train_cfg.get("device") or "auto").lower()
    if not train_missing:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        print(f"PyTorch: {torch.__version__}; CUDA available={cuda_available}")
        if requested_device == "cuda" and not cuda_available:
            cuda_ok = False
            print("[FAIL] 配置要求 cuda，但当前 PyTorch 未检测到 CUDA")

    print(f"训练后端: {train_cfg.get('backend', 'auto')}; 设备: {requested_device}; "
          f"方法: {train_cfg.get('method', 'lora')}")
    print(f"训练权重输出目录: {ROOT / 'data' / 'models' / '<model_id>' / 'weights'}")
    print(f"JIUAN_MOCK={os.environ.get('JIUAN_MOCK', '(未设置，默认 0)')}")

    failures = []
    if runtime_missing:
        failures.append("控制面依赖不完整")
    if args.require_real and train_missing:
        failures.append("真实训练依赖不完整")
    if args.require_real and not local_ready:
        failures.append("本地基底模型不完整；请运行 scripts/fetch_model.py 或修改 model.local_dir")
    if not writable:
        failures.append("data 目录不可写")
    if not cuda_ok:
        failures.append("CUDA 配置与环境不匹配")

    if failures:
        print("\n部署检查未通过：")
        for item in failures:
            print(f"- {item}")
        return 1
    print("\n部署检查通过。" + ("可以开始真实训练。" if args.require_real else "控制面可以启动。"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

