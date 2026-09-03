@echo off
REM jiuan-lite Windows 一键启动（不带基底模型版）
REM 用法: start.bat [配置名 qwen0.5b^|qwen7b]，默认 qwen0.5b
REM 首次使用请先: 1) pip install -r requirements.txt -r requirements-agent.txt
REM               2) copy .env.example .env 并填入 DEEPSEEK_API_KEY
setlocal
cd /d "%~dp0"
set CFG=%1
if "%CFG%"=="" set CFG=qwen0.5b
if not exist .env (
  echo [提示] 未发现 .env，正在从模板创建，请编辑填入 DEEPSEEK_API_KEY
  copy .env.example .env >nul
)
echo 启动配置: configs/%CFG%.yaml  (http://127.0.0.1:8000)
python -m jiuan.app --config configs/%CFG%.yaml
pause
