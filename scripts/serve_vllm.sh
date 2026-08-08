#!/usr/bin/env bash
# serve_vllm.sh —— 在 Linux + NVIDIA GPU 上启动 vLLM 的 OpenAI 兼容推理服务。
#
# 用法:
#   ./scripts/serve_vllm.sh                       # 起 base 模型(Qwen2.5-0.5B)
#   ./scripts/serve_vllm.sh <model_path> <name>   # 起指定模型，served-model-name=<name>
#
# 说明:
# - vLLM 仅支持 Linux + CUDA；本脚本不在 Windows 上运行。
# - 服务端口 8001，与 jiuan 控制面(8000)分离；jiuan 通过 configs 里的 vllm.base_url 调用它。
# - LoRA 产物: vLLM 支持 --enable-lora + --lora-modules name=path 动态挂载 adapter。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-$ROOT/data/registry/base/qwen2.5-0.5b-instruct}"
SERVED_NAME="${2:-qwen2.5-0.5b-instruct}"
PORT="${VLLM_PORT:-8001}"

echo "=== 启动 vLLM OpenAI 服务 ==="
echo "  模型      : $MODEL_PATH"
echo "  served名  : $SERVED_NAME"
echo "  端点      : http://127.0.0.1:${PORT}/v1"

# 全量模型/基座:
exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9

# —— 如需挂载 LoRA adapter（训练产物在 data/models/<id>/weights），改用下面这段： ——
# BASE="$ROOT/data/registry/base/qwen2.5-0.5b-instruct"
# ADAPTER="$ROOT/data/models/<model_id>/weights"
# exec vllm serve "$BASE" \
#   --enable-lora \
#   --lora-modules "<model_id>=$ADAPTER" \
#   --served-model-name qwen-base \
#   --port "$PORT" --dtype auto --max-model-len 4096
