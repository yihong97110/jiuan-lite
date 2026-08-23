"""训：SFT / LoRA 微调（Qwen2.5-0.5B），支持多训练后端。

后端选择（params.backend，缺省取 config.train.backend，默认 auto）：
- mock         : 未装训练框架时的离线兜底，记忆训练集供推理演示。
- hf           : 内置 transformers+peft 训练。
- llamafactory : 二次整合 LLaMA-Factory（成熟开源件）。
- auto         : mock 模式→mock；否则优先 llamafactory(若可用)，回退 hf。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .. import registry
from ..common import (
    DATASETS,
    MODELS,
    ROOT,
    apply_train_overrides,
    load_config,
    mock_mode,
    model_path,
    resolve_device,
    resolve_precision,
)
from . import lf_backend


def _read_dataset(dataset_id: str, split: str = "train") -> list[dict]:
    if not dataset_id or dataset_id in ("__none__", "none", "null"):
        avail = sorted(p.name for p in DATASETS.iterdir() if p.is_dir())
        raise ValueError(
            "未指定有效的 dataset_id（请先在①标·数据准备生成数据集，再选择）。"
            f"当前可用数据集: {avail}"
        )
    path = DATASETS / dataset_id / f"{split}.jsonl"
    if not path.exists():
        avail = sorted(p.name for p in DATASETS.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"数据集不存在: {dataset_id}（缺少 {split}.jsonl）。当前可用数据集: {avail}"
        )
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_backend(cfg: dict, params: dict) -> str:
    backend = (params.get("backend") or cfg["train"].get("backend") or "auto").lower()
    if mock_mode():
        return "mock"
    if backend == "auto":
        return "llamafactory" if lf_backend.available() else "hf"
    return backend


def _has_full_model(weights_dir: Path) -> bool:
    return any(
        (weights_dir / name).exists()
        for name in ("config.json", "model.safetensors", "pytorch_model.bin")
    )


def _mock_train(cfg: dict, params: dict, rows: list[dict], out_dir: Path, log: Callable):
    epochs = params.get("epochs") or cfg["train"]["epochs"]
    method = params.get("method") or cfg["train"]["method"]
    log(f"[MOCK] 未启用真实训练框架，模拟 {method} 训练 {epochs} epoch")
    loss = None
    # "记忆"训练样本作为可复用知识（演示推理能读到微调效果）
    memory = {}
    for r in rows:
        msgs = r["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        memory[user.strip()] = asst.strip()
        loss = round(2.0 / (len(memory) + 1), 4)
        log(f"[MOCK] step {len(memory)} loss={loss}")
    (out_dir / "memory.json").write_text(
        json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"mode": "mock", "backend": "mock", "steps": len(rows), "final_loss": loss}


def _hf_train(cfg: dict, params: dict, dataset_id: str, out_dir: Path, log: Callable, progress: "Callable | None" = None):
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    model_name = model_path(cfg)
    init_model_id = str(params.get("init_model_id") or "").strip()
    init_model_dir = MODELS / init_model_id if init_model_id else None
    init_weights = init_model_dir / "weights" if init_model_dir else None
    init_meta = {}
    if init_model_dir:
        meta_path = init_model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"续训模型不存在或缺少 meta.json: {init_model_id}")
        init_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if init_meta.get("mode") == "mock":
            raise ValueError(f"模型 {init_model_id} 是 mock 产物，不能作为 HF/LoRA 续训权重")
        model_name = init_meta.get("base_model_path") or init_meta.get("base") or model_name
    method = params.get("method") or cfg["train"]["method"]
    epochs = params.get("epochs") or cfg["train"]["epochs"]
    max_len = cfg["train"]["max_seq_len"]

    # 设备/精度：支持 config 显式 cpu|cuda|auto；params.device 可临时覆盖
    if params.get("device"):
        cfg = {**cfg, "train": {**cfg["train"], "device": params["device"]}}
    device = resolve_device(cfg)
    precision = resolve_precision(cfg, device)
    want = str(cfg["train"].get("device", "auto")).lower()
    if want == "cuda" and device == "cpu":
        log("[hf] 警告：请求 cuda 但本机无 GPU，已回退 CPU")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(precision, torch.float32)
    log(f"[hf] 设备: {device}  精度: {precision}")

    tokenizer_source = init_weights if init_weights and (init_weights / "tokenizer_config.json").exists() else model_name
    log(f"[hf] 加载 tokenizer/model: {tokenizer_source} / {model_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full_init = bool(init_weights and _has_full_model(init_weights))
    model_source = init_weights if full_init else model_name
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=dtype)
    model = model.to(device)

    if method == "lora":
        if init_weights and (init_weights / "adapter_config.json").exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(init_weights), is_trainable=True)
            log(f"[hf] 从模型 {init_model_id} 的 LoRA adapter 继续训练")
        else:
            from peft import LoraConfig, get_peft_model

            lc = cfg["train"]["lora"]
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                    target_modules=lc["target_modules"], task_type="CAUSAL_LM",
                ),
            )
            log("[hf] 已启用新 LoRA")

    ds = load_dataset(
        "json", data_files=str(DATASETS / dataset_id / "train.jsonl")
    )["train"]

    def _format(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        # 不做 max_length padding，交给 collator 动态 padding
        out = tok(text, truncation=True, max_length=max_len)
        # labels 与 input 对齐；padding 位由 collator 置为 -100
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(_format, remove_columns=ds.column_names)

    # DataCollatorForSeq2Seq 动态 padding，并把 labels 的 padding 位设为 -100
    collator = DataCollatorForSeq2Seq(
        tok, model=model, label_pad_token_id=-100, padding=True
    )

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=float(cfg["train"]["lr"]),
        logging_steps=1,
        save_steps=cfg["train"]["save_steps"],
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds, data_collator=collator
    )

    if progress is not None:
        from transformers import TrainerCallback

        class _ProgressCb(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                if state.max_steps:
                    progress(f"{int(state.global_step)}/{int(state.max_steps)} steps")

        trainer.add_callback(_ProgressCb())

    log("[hf] 开始训练…")
    result = trainer.train()
    if progress is not None:
        progress("done")
    model.save_pretrained(out_dir / "weights")
    tok.save_pretrained(out_dir / "weights")

    # 训练完成后显式释放 GPU 显存，避免后续推理 OOM
    del model
    del trainer
    del ds
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log("[hf] GPU 显存已清理")

    return {
        "mode": "real",
        "backend": "hf",
        "method": method,
        "device": device,
        "precision": precision,
        "train_loss": result.training_loss,
    }


def run(params: dict, log: Callable[[str], None], progress: "Callable[[str], None] | None" = None) -> dict:
    # 根据前端选择的基座模型加载对应 config（0.5b / 7b）
    base_model = (params.get("base_model") or "").lower()
    if base_model == "7b":
        cfg_path = ROOT / "configs" / "qwen7b.yaml"
        if cfg_path.exists():
            cfg = load_config(cfg_path)
            log(f"基座模型: Qwen2.5-7B-Instruct (config={cfg_path.name})")
        else:
            # 回退：用默认 config 但覆盖模型路径为7B
            cfg = load_config()
            cfg.setdefault("model", {})["local_dir"] = "/root/autodl-tmp/models/qwen2.5-7b-instruct"
            log(f"基座模型: Qwen2.5-7B-Instruct (路径覆盖，未找到 {cfg_path.name})")
    elif base_model == "0.5b":
        cfg = load_config(ROOT / "configs" / "qwen0.5b.yaml")
        log("基座模型: Qwen2.5-0.5B-Instruct")
    else:
        cfg = load_config()
        log(f"基座模型: config 默认 ({cfg.get('model', {}).get('name', '?')})")
    cfg = apply_train_overrides(cfg, params)
    dataset_id = params["dataset_id"]
    name = params.get("name", "qwen0.5b-sft")
    rows = _read_dataset(dataset_id, "train")
    log(f"数据集 {dataset_id}：训练集 {len(rows)} 条样本")

    backend = _resolve_backend(cfg, params)
    log(f"训练后端: {backend}")

    model_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = MODELS / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "mock":
        stats = _mock_train(cfg, params, rows, out_dir, log)
    elif backend == "llamafactory":
        if not lf_backend.available():
            raise RuntimeError(
                "指定 backend=llamafactory 但未检测到 LLaMA-Factory，"
                "请 pip install llamafactory，或改用 backend=hf"
            )
        stats = lf_backend.train(cfg, params, dataset_id, out_dir, log, progress=progress)
    elif backend == "hf":
        stats = _hf_train(cfg, params, dataset_id, out_dir, log, progress=progress)
    else:
        raise ValueError(f"未知训练后端: {backend}")

    # 从训练数据提取领域身份（若 params 未显式指定）
    # 训练数据的 messages[0] 通常是 system 消息，记录了模型应扮演的领域角色
    domain_prompt = params.get("system_prompt") or ""
    if not domain_prompt:
        try:
            rows = _read_dataset(dataset_id, "train")
            for row in rows:
                msgs = row.get("messages") or []
                if msgs and msgs[0].get("role") == "system":
                    domain_prompt = msgs[0].get("content", "")
                    break
        except Exception:
            pass

    meta = {
        "model_id": model_id,
        "base": model_path(cfg),
        "base_model_path": model_path(cfg),
        "dataset_id": dataset_id,
        "parent_model_id": params.get("init_model_id") or None,
        # Keep the project contract with the weights so inference does not
        # fall back to another domain's global system prompt.
        "system_prompt": domain_prompt,
        "domain_direction": params.get("domain_direction") or "",
        "knowledge_collection": params.get("knowledge_collection") or "",
        "project_id": params.get("project_id") or "",
        "created_at": time.time(),
        **stats,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    registry.register_model(meta)
    log(f"训练完成，产物登记为 {model_id}（backend={stats.get('backend')}）")
    return meta



