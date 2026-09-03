#!/usr/bin/env bash
# boot_server.sh —— AutoDL GPU 服务器开机一键拉起（vLLM 8001 + jiuan 平台 8000）。
#
# 用法（服务器上）:
#   bash scripts/boot_server.sh          # 幂等：已在运行的组件会跳过
#
# 组件:
#   1) vLLM OpenAI 服务(8001)：基座 qwen2.5-7b-instruct + LoRA gaokao7b-v3
#   2) jiuan-lite 平台(8000)：REAL 模式（JIUAN_MOCK=0，infer.backend=auto 自动走 vLLM）
#
# 已知坑（都已在脚本内处理）:
#   - flashinfer JIT 需要 ninja：启动前 export PATH 含 /root/miniconda3/bin
#   - vLLM 0.27 挂 LoRA 必须显式 --enable-lora
#   - 平台与 vLLM 必须用同一个 python 解释器（/root/miniconda3/bin/python3）
set -u

export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # 服务器连不上 huggingface.co，必须进程启动前设镜像（import 时固化）
DELIVERY="${DELIVERY:-/root/autodl-tmp/delivery-test/jiuan-lite}"
LOGDIR="${LOGDIR:-/root/autodl-tmp}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/qwen2.5-7b-instruct}"
LORA_ID="${LORA_ID:-gaokao7b-v3-20260823-223535}"
LORA_WEIGHTS="$DELIVERY/data/models/$LORA_ID/weights"
VLLM_PORT="${VLLM_PORT:-8001}"
APP_PORT="${APP_PORT:-8000}"

wait_http() {  # wait_http <url> <名字> <最长秒>
  local url="$1" name="$2" max="${3:-180}" i=0
  while [ $i -lt $max ]; do
    if curl -s -m 3 "$url" >/dev/null 2>&1; then echo "✓ $name 就绪 (${i}s)"; return 0; fi
    sleep 5; i=$((i+5))
  done
  echo "✗ $name 在 ${max}s 内未就绪，请查日志"; return 1
}

echo "=== 1/2 vLLM (端口 $VLLM_PORT) ==="
if curl -s -m 3 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
  echo "已在运行，跳过"
else
  [ -d "$LORA_WEIGHTS" ] || { echo "✗ LoRA 权重不存在: $LORA_WEIGHTS"; exit 1; }
  nohup /root/miniconda3/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --enable-lora \
    --lora-modules "$LORA_ID=$LORA_WEIGHTS" \
    --port "$VLLM_PORT" --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 --max-model-len 2048 --max-loras 4 \
    > "$LOGDIR/vllm$VLLM_PORT.log" 2>&1 &
  echo "启动中（约1-3分钟），日志: $LOGDIR/vllm$VLLM_PORT.log"
  wait_http "http://127.0.0.1:$VLLM_PORT/v1/models" "vLLM" 300
fi

echo ""
echo "=== 2/2 jiuan-lite 平台 (端口 $APP_PORT, REAL 模式) ==="
if curl -s -m 3 "http://127.0.0.1:$APP_PORT/health" >/dev/null 2>&1; then
  echo "已在运行，跳过"
else
  cd "$DELIVERY" || { echo "✗ 平台目录不存在: $DELIVERY"; exit 1; }
  JIUAN_MOCK=0 JIUAN_HOST=0.0.0.0 JIUAN_PORT="$APP_PORT" \
    nohup /root/miniconda3/bin/python3 -m jiuan.app --config configs/qwen7b.yaml \
    > "$LOGDIR/app$APP_PORT.log" 2>&1 &
  echo "启动中，日志: $LOGDIR/app$APP_PORT.log"
  wait_http "http://127.0.0.1:$APP_PORT/health" "平台" 120
fi

echo ""
echo "=== 状态汇总 ==="
curl -s -m 5 "http://127.0.0.1:$APP_PORT/health" && echo ""
curl -s -m 5 "http://127.0.0.1:$VLLM_PORT/v1/models" | /root/miniconda3/bin/python3 -c \
  "import json,sys; d=json.load(sys.stdin); print('vLLM models:', [m['id'] for m in d['data']])" 2>/dev/null \
  || echo "(vLLM 模型列表解析失败，不影响使用)"
echo "完成。访问平台: http://127.0.0.1:$APP_PORT/"
