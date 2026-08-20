"""训练后端：LLaMA-Factory 二次整合。

思路（对标久安「模型训练系统」用成熟开源件替换手写实现）：
- 把我们已清洗的 messages 格式数据集，登记成 LLaMA-Factory 的 sharegpt 数据集；
- 生成 LLaMA-Factory 的训练配置 yaml；
- 通过 `llamafactory-cli train <config>` 子进程执行，日志实时回传；
- 产物(LoRA adapter 或全量权重)落到 out_dir/weights，供推理阶段加载。

未安装 llamafactory 时 available() 返回 False，train.py 会回退到 hf 后端或 mock。
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import yaml

from ..common import DATASETS, MODELS, model_path, resolve_device, resolve_precision  # noqa: F401

# 兼容 dict-repr("'loss': 1.23") 与 key=val("loss=1.23") 两种输出
_LOSS_RE = re.compile(r"['\"]?loss['\"]?\s*[:=]\s*([0-9]+\.?[0-9]*)")
# 进度：形如 " 12/58 " 或 "'epoch': 1.0"
_STEP_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
_EPOCH_RE = re.compile(r"['\"]?epoch['\"]?\s*[:=]\s*([0-9]+\.?[0-9]*)")


def available() -> bool:
    """检测 LLaMA-Factory 是否可用（Python 包或 CLI 任一存在即可）。"""
    if importlib.util.find_spec("llamafactory") is not None:
        return True
    return shutil.which("llamafactory-cli") is not None


def _cli_command() -> list[str]:
    """优先用 CLI，其次退化为 `python -m llamafactory.cli`。"""
    exe = shutil.which("llamafactory-cli")
    if exe:
        return [exe]
    return [sys.executable, "-m", "llamafactory.cli"]


def _child_env() -> dict:
    """为子进程构造环境变量：确保能导入 llamafactory 与本项目。

    - 把当前解释器的 site-packages / 项目根加入 PYTHONPATH，
      避免退化为 `python -m llamafactory.cli` 时子进程 sys.path 缺失。
    - 离线/推理相关开关从父进程透传。
    """
    env = os.environ.copy()
    from ..common import ROOT

    extra_paths = [str(ROOT)]
    for p in sys.path:
        if p and p not in extra_paths:
            extra_paths.append(p)
    prev = env.get("PYTHONPATH", "")
    if prev:
        extra_paths.append(prev)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return env


def _write_dataset_info(run_dir: Path, dataset_id: str, ds_name: str) -> None:
    """把 messages 格式登记为 LLaMA-Factory sharegpt 数据集。"""
    train_file = DATASETS / dataset_id / "train.jsonl"
    dataset_info = {
        ds_name: {
            "file_name": str(train_file.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    (run_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_config(cfg: dict, params: dict, run_dir: Path, ds_name: str, out_weights: Path) -> Path:
    method = params.get("method") or cfg["train"]["method"]
    epochs = params.get("epochs") or cfg["train"]["epochs"]
    tcfg = cfg["train"]
    if params.get("device"):
        cfg = {**cfg, "train": {**cfg["train"], "device": params["device"]}}
    device = resolve_device(cfg)
    precision = resolve_precision(cfg, device)
    lf = {
        "model_name_or_path": model_path(cfg),
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora" if method == "lora" else "full",
        "dataset": ds_name,
        "dataset_dir": str(run_dir.resolve()),
        "template": cfg.get("llamafactory", {}).get("template", "qwen"),
        "cutoff_len": tcfg["max_seq_len"],
        "per_device_train_batch_size": int(params.get("batch_size") or tcfg["batch_size"]),
        "gradient_accumulation_steps": int(params.get("grad_accum") or tcfg["grad_accum"]),
        "learning_rate": float(params.get("lr") or tcfg["lr"]),
        "num_train_epochs": float(epochs),
        "logging_steps": 1,
        "save_steps": tcfg["save_steps"],
        "output_dir": str(out_weights.resolve()),
        "overwrite_output_dir": True,
        "report_to": "none",
        "trust_remote_code": True,
    }
    # 精度：GPU 上按解析结果开 fp16/bf16；CPU 保持 fp32（不设标志）
    if device == "cuda":
        if precision == "bf16":
            lf["bf16"] = True
        elif precision == "fp16":
            lf["fp16"] = True
    else:
        # LLaMA-Factory 在无 GPU 时强制单卡 CPU
        lf["use_cpu"] = True
    if lf["finetuning_type"] == "lora":
        lc = tcfg["lora"]
        lf["lora_rank"] = lc["r"]
        lf["lora_alpha"] = lc["alpha"]
        lf["lora_dropout"] = lc["dropout"]
        lf["lora_target"] = ",".join(lc["target_modules"])
    cfg_path = run_dir / "train_config.yaml"
    cfg_path.write_text(yaml.safe_dump(lf, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg_path


def _terminate(proc: subprocess.Popen, log: Callable[[str], None]) -> None:
    """优雅终止子进程：先 terminate，超时再 kill，避免孤儿进程。"""
    if proc.poll() is not None:
        return
    try:
        log("[LF] 检测到中断，正在终止 LLaMA-Factory 子进程…")
    except Exception:
        pass
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    except Exception:
        pass


def train(
    cfg: dict,
    params: dict,
    dataset_id: str,
    out_dir: Path,
    log: Callable[[str], None],
    progress: "Callable[[str], None] | None" = None,
) -> dict:
    method = params.get("method") or cfg["train"]["method"]
    run_dir = out_dir / "lf_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_weights = out_dir / "weights"
    out_weights.mkdir(parents=True, exist_ok=True)

    ds_name = f"jiuan_{dataset_id.replace('-', '_')}"
    _write_dataset_info(run_dir, dataset_id, ds_name)
    cfg_path = _build_config(cfg, params, run_dir, ds_name, out_weights)
    log(f"LLaMA-Factory 配置已生成: {cfg_path.name}（finetuning={method}）")

    command = _cli_command() + ["train", str(cfg_path)]
    log(f"启动 LLaMA-Factory: {' '.join(command)}")

    final_loss: Optional[float] = None
    total_steps: Optional[int] = None
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(run_dir),
        env=_child_env(),  # 确保子进程能导入 llamafactory / 本项目
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            m = _LOSS_RE.search(line)
            if m:
                try:
                    final_loss = float(m.group(1))
                except ValueError:
                    pass
            # 解析进度：优先 step 形式，其次 epoch
            if progress:
                sm = _STEP_RE.search(line)
                if sm:
                    cur, tot = sm.group(1), sm.group(2)
                    total_steps = int(tot)
                    progress(f"{cur}/{tot} steps")
                else:
                    em = _EPOCH_RE.search(line)
                    if em:
                        progress(f"epoch {em.group(1)}")
            # 只回传关键行，避免日志过多
            if m or any(k in line for k in ("epoch", "error", "Error", "Traceback", "saved", "Saving")):
                log(f"[LF] {line[:200]}")
    except BaseException:
        # 任务被取消/异常：优雅终止子进程，避免变孤儿进程
        _terminate(proc, log)
        raise
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"LLaMA-Factory 训练失败，退出码 {ret}")
    if progress:
        progress("done")

    # 记录基座信息，便于推理阶段加载 LoRA adapter
    (out_weights / "jiuan_base.json").write_text(
        json.dumps({"base": cfg["model"]["name"], "finetuning": method}, ensure_ascii=False),
        encoding="utf-8",
    )
    log("LLaMA-Factory 训练完成，产物已保存")
    return {
        "mode": "real",
        "backend": "llamafactory",
        "method": method,
        "train_loss": final_loss,
        "weights_dir": str(out_weights),
    }


