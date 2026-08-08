<#
.SYNOPSIS
  jiuan-lite 一键工作流编排（Windows / PowerShell）。
  把「建环境 -> 装依赖 -> 下模型 -> 起服务 -> 全链路自检」固化，避免重复踩坑。

.USAGE
  .\workflow.ps1 setup      # 建 conda 环境 + 装依赖（控制面 + 训练 + LLaMA-Factory）
  .\workflow.ps1 model      # 下载 Qwen2.5-0.5B 到 data/registry/base（走 hf-mirror）
  .\workflow.ps1 serve      # 后台启动控制面（REAL + 离线）
  .\workflow.ps1 run        # 调 /workflow 跑一键全链路并打印结果
  .\workflow.ps1 stop       # 停掉占用 8000 端口的服务
  .\workflow.ps1 status     # 查看健康与最近任务
  .\workflow.ps1 all        # setup -> model -> serve -> run 一条龙

.NOTES
  - 全程 UTF-8，避免中文 prompt 变 ?????（这是之前踩过的坑）。
  - REAL 模式依赖本地已下载模型 + 离线环境变量。
#>
param(
  [Parameter(Position=0)]
  [ValidateSet('setup','model','serve','run','stop','status','all')]
  [string]$Command = 'status'
)

$ErrorActionPreference = 'Stop'
# —— 全程 UTF-8（关键：否则中文请求体会乱码）——
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName     = 'jiuan'
$EnvPy       = "$env:USERPROFILE\anaconda3\envs\$EnvName\python.exe"
$BaseUrl     = 'http://127.0.0.1:8000'
$ModelDir    = Join-Path $ProjectRoot 'data\registry\base\qwen2.5-0.5b-instruct'

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

function Set-RealEnv {
  $env:JIUAN_MOCK = '0'
  $env:HF_HUB_OFFLINE = '1'
  $env:TRANSFORMERS_OFFLINE = '1'
}

function Invoke-Api($Method, $Path, $BodyObj) {
  $uri = "$BaseUrl$Path"
  if ($BodyObj) {
    # 显式 UTF-8 字节体，规避 PowerShell 默认编码把中文发成 ?????
    $json  = $BodyObj | ConvertTo-Json -Depth 6
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    return Invoke-RestMethod -Method $Method -Uri $uri -ContentType 'application/json; charset=utf-8' -Body $bytes
  }
  return Invoke-RestMethod -Method $Method -Uri $uri
}

function Wait-Task($TaskId, $MaxSec = 600) {
  for ($i = 0; $i -lt $MaxSec; $i++) {
    $t = Invoke-Api GET "/tasks/$TaskId"
    if ($t.status -in @('succeeded','failed')) { return $t }
    Start-Sleep -Seconds 1
  }
  return $t
}

function Ensure-Conda {
  if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'conda 未找到，请先安装 Anaconda/Miniconda 并把 conda 加入 PATH'
  }
}

function Do-Setup {
  Ensure-Conda
  Write-Step "创建 conda 环境 $EnvName (python 3.10)"
  $exists = (& conda env list) -match "^$EnvName\s"
  if ($exists) { Write-Host "环境已存在，跳过创建" } else { & conda create -y -n $EnvName python=3.10 }

  Write-Step "安装控制面依赖"
  & conda run -n $EnvName --no-capture-output python -m pip install -r "$ProjectRoot\requirements.txt"

  Write-Step "安装 CPU 版 torch"
  & conda run -n $EnvName --no-capture-output python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

  Write-Step "安装训练依赖 + LLaMA-Factory"
  & conda run -n $EnvName --no-capture-output python -m pip install -r "$ProjectRoot\requirements-train.txt"

  Write-Host "`nsetup 完成" -ForegroundColor Green
}

function Do-Model {
  Write-Step "下载 Qwen2.5-0.5B 到本地 (hf-mirror)"
  if ((Test-Path (Join-Path $ModelDir 'model.safetensors')) -and (Test-Path (Join-Path $ModelDir 'config.json'))) {
    Write-Host "模型已存在，跳过下载"; return
  }
  $env:HF_ENDPOINT = 'https://hf-mirror.com'
  & conda run -n $EnvName --no-capture-output python "$ProjectRoot\scripts\fetch_model.py"
  & conda run -n $EnvName --no-capture-output python "$ProjectRoot\scripts\fetch_small.py"
  Write-Host "`nmodel 完成" -ForegroundColor Green
}

function Do-Stop {
  $conns = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
  if ($conns) {
    $conns | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
    Write-Host "已停止 8000 端口服务"
  } else {
    Write-Host "8000 端口无服务在运行"
  }
}

function Do-Serve {
  if (-not (Test-Path $EnvPy)) { throw "找不到环境 python: $EnvPy，请先 .\workflow.ps1 setup" }
  Do-Stop
  Set-RealEnv
  Write-Step "后台启动控制面 (REAL + 离线)"
  $p = Start-Process -FilePath $EnvPy -ArgumentList '-m','jiuan.app' -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
  $p.Id | Out-File (Join-Path $ProjectRoot '.server_pid') -Encoding ascii
  for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    try { $h = Invoke-Api GET '/health'; Write-Host "health: $($h | ConvertTo-Json -Compress)  PID=$($p.Id)" -ForegroundColor Green; return } catch {}
  }
  throw '服务启动超时，请检查依赖是否安装完整'
}

function Do-Run {
  Write-Step "调用 /workflow 跑一键全链路（标->训->推->评）"
  $sub = Invoke-Api POST '/workflow' @{ source='data/samples.jsonl'; name='workflow'; backend='auto' }
  Write-Host "workflow task_id: $($sub.task_id)"
  $t = Wait-Task $sub.task_id 900
  Write-Host "`n状态: $($t.status)" -ForegroundColor $(if ($t.status -eq 'succeeded') {'Green'} else {'Red'})
  if ($t.status -eq 'succeeded') {
    $r = $t.result
    Write-Host "  数据集 : $($r.dataset_id)"
    Write-Host "  模型   : $($r.model_id)  (backend=$($r.backend))"
    Write-Host "  train_loss: $($r.train_loss)"
    Write-Host "  评测   : reliable=$($r.reliable)  metrics=$($r.metrics | ConvertTo-Json -Compress)"
    Write-Host "`n步骤日志:" -ForegroundColor Cyan
    $t.logs | ForEach-Object { Write-Host "  $_" }
  } else {
    Write-Host "错误:`n$($t.error)" -ForegroundColor Red
  }
}

function Do-Status {
  try {
    $h = Invoke-Api GET '/health'
    Write-Host "health: $($h | ConvertTo-Json -Compress)" -ForegroundColor Green
    Write-Host "`n最近任务:"
    (Invoke-Api GET '/tasks') | Select-Object -First 8 |
      ForEach-Object { Write-Host ("  {0}  {1,-9}  {2}" -f $_.id, $_.status, $_.stage) }
  } catch {
    Write-Host "服务未运行（先 .\workflow.ps1 serve）" -ForegroundColor Yellow
  }
}

switch ($Command) {
  'setup'  { Do-Setup }
  'model'  { Do-Model }
  'serve'  { Do-Serve }
  'run'    { Do-Run }
  'stop'   { Do-Stop }
  'status' { Do-Status }
  'all'    { Do-Setup; Do-Model; Do-Serve; Do-Run }
}
