#!/usr/bin/env bash
# jiuan-lite Linux 一键启动（不带基底模型版）
# 用法: ./start.sh [配置名 qwen0.5b|qwen7b]，默认 qwen0.5b
# 首次使用请先: 1) pip install -r requirements.txt -r requirements-agent.txt
#               2) cp .env.example .env 并填入 DEEPSEEK_API_KEY
set -e
cd "$(dirname "$0")"
CFG="${1:-qwen0.5b}"
if [ ! -f .env ]; then
  echo "[提示] 未发现 .env，正在从模板创建，请编辑填入 DEEPSEEK_API_KEY"
  cp .env.example .env
fi
echo "启动配置: configs/${CFG}.yaml  (http://127.0.0.1:8000)"
exec python -m jiuan.app --config "configs/${CFG}.yaml"
