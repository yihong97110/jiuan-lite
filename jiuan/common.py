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


def _load_env_file() -> None:
    """加载项目根目录 .env 到 os.environ（不覆盖已存在的环境变量）。

    无 dotenv 依赖，手工解析 KEY=VALUE 行。修复：直接命令行跑 train/evaluate
    等脚本时 judge API key 不会从 .env 进入环境变量，导致 judge 静默降级。
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


_load_env_file()

for _p in (DATASETS, MODELS, REPORTS, REGISTRY):
    _p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict:
    # 优先级：显式path > 环境变量JIUAN_CONFIG > 默认0.5b
    # 环境变量让worker子进程继承app进程的配置选择
    if path is None:
        env_path = os.environ.get("JIUAN_CONFIG")
        path = Path(env_path) if env_path else ROOT / "configs" / "qwen0.5b.yaml"
    path = Path(path)
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


# 平台 → 默认 base_url 映射（与 distill.py _resolve_base_url 一致，本文件内定义避免循环导入）
_PLATFORM_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
}
# 平台 → .env 变量名映射（volcengine 复用 judge.py 已有的 ARK_API_KEY 回退）
_PLATFORM_ENV_KEY = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "volcengine": "ARK_API_KEY",
    "custom": "CUSTOM_JUDGE_API_KEY",
}


def _persist_judge_key_to_env(platform: str, api_key: str) -> str:
    """把 API Key 写入 .env 文件 + os.environ，返回 env 变量名。

    用户填一次 Key 后长期保存，后续评测 judge.py._resolve_key 走 env 回退自动读到。
    """
    env_name = _PLATFORM_ENV_KEY.get(platform, "CUSTOM_JUDGE_API_KEY")
    os.environ[env_name] = api_key  # 当前进程立即生效
    env_path = ROOT / ".env"
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        lines = [l for l in lines if not l.startswith(f"{env_name}=")]  # 去旧的同名行
    lines.append(f"{env_name}={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_name


def apply_judge_overrides(cfg: dict, params: dict) -> dict:
    """把请求/前端传入的 judge 配置覆盖到 config.judge + Key 持久化到 .env。

    - 仅当 params 提供 judge_platform 或 judge_base_url 时才介入，避免无配置请求误伤 yaml 默认值。
    - platform→base_url 自动映射；custom 平台必须显式给 judge_base_url。
    - judge_api_key 非空时写入 .env 长期保存 + 设到 cfg 让本次任务也用。
    """
    platform = params.get("judge_platform")
    base_url_param = params.get("judge_base_url")
    if not platform and not base_url_param:
        return cfg  # 未传任何 judge 配置，保持 yaml 原样（向后兼容）
    judge = dict(cfg.get("judge", {}) or {})
    # 平台→base_url 映射；custom 用显式 base_url
    if platform == "custom" and base_url_param:
        judge["base_url"] = base_url_param.rstrip("/")
    elif platform:
        judge["base_url"] = _PLATFORM_BASE_URLS.get(platform, "")
    elif base_url_param:
        judge["base_url"] = base_url_param.rstrip("/")
    # 模型名
    model = params.get("judge_model")
    if model:
        judge["model"] = model
    # API Key：写入 .env 长期保存 + 设到 cfg 让本次任务也用
    api_key = params.get("judge_api_key")
    if api_key:
        env_name = _persist_judge_key_to_env(platform or "custom", api_key)
        judge["api_key"] = api_key         # 本次任务直接用
        judge["api_key_env"] = env_name    # 让 judge.py 后续走 env 回退
    # 火山方舟 /models 接口语义不同，关掉 probe 避免误判不可用
    if platform == "volcengine":
        judge.setdefault("probe", False)
    return {**cfg, "judge": judge}


# ---------------------------------------------------------------------------
# API Key 管理：统一读写 .env，支持前端动态配置
# ---------------------------------------------------------------------------

# 所有受管理的 Key 定义：env变量名 -> 显示名/描述
API_KEY_DEFS = {
    "DEEPSEEK_API_KEY": {"label": "DeepSeek API Key", "desc": "Judge 评测裁判 + 对话端点", "platform": "deepseek"},
    "ARK_API_KEY": {"label": "火山方舟 API Key", "desc": "火山方舟 DeepSeek 对话端点", "platform": "volcengine"},
    "OPENAI_API_KEY": {"label": "OpenAI API Key", "desc": "OpenAI GPT 端点（可选）", "platform": "openai"},
}


def get_api_keys() -> dict:
    """读取所有 API Key 的状态（脱敏，只显示是否已配置 + 前后4位）。"""
    env_path = ROOT / ".env"
    env_values = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env_values[k.strip()] = v.strip()

    result = {}
    for env_name, info in API_KEY_DEFS.items():
        val = os.environ.get(env_name, "") or env_values.get(env_name, "")
        if val and len(val) > 8:
            masked = val[:4] + "*" * (len(val) - 8) + val[-4:]
        elif val:
            masked = "****"
        else:
            masked = ""
        result[env_name] = {
            "label": info["label"],
            "desc": info["desc"],
            "platform": info["platform"],
            "configured": bool(val),
            "preview": masked,
        }
    return result


def save_api_key(env_name: str, api_key: str) -> dict:
    """保存单个 API Key 到 .env 文件 + 当前进程环境变量。"""
    if env_name not in API_KEY_DEFS:
        return {"error": f"未知的 Key 类型: {env_name}"}
    if not api_key or not api_key.strip():
        return {"error": "API Key 不能为空"}

    api_key = api_key.strip()
    os.environ[env_name] = api_key

    env_path = ROOT / ".env"
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        lines = [l for l in lines if not l.startswith(f"{env_name}=")]
    lines.append(f"{env_name}={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"env_name": env_name, "saved": True}
