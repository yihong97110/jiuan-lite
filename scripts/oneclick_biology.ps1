param(
  [switch]$ForceTraining,
  [switch]$SkipInference,
  [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$defaultPython = "C:\Users\21888\anaconda3\envs\jiuan\python.exe"
$python = if ($env:JIUAN_PYTHON) { $env:JIUAN_PYTHON } elseif (Test-Path $defaultPython) { $defaultPython } else { "python" }

function U([string]$Base64) {
  return [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Base64))
}

function Read-Utf8Json($Response) {
  $stream = $Response.RawContentStream
  $stream.Position = 0
  $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8)
  return ($reader.ReadToEnd() | ConvertFrom-Json)
}

function Get-Json([string]$Uri) {
  $resp = Invoke-WebRequest -Uri $Uri -Method Get -UseBasicParsing -TimeoutSec 30
  return Read-Utf8Json $resp
}

function Post-Json([string]$Uri, [string]$Body) {
  $bytes = [Text.Encoding]::UTF8.GetBytes($Body)
  $resp = Invoke-WebRequest -Uri $Uri -Method Post -UseBasicParsing -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec 30
  return Read-Utf8Json $resp
}

function Test-Health {
  try {
    Get-Json "$BaseUrl/health" | Out-Null
    return $true
  } catch {
    return $false
  }
}

if (-not (Test-Health)) {
  Write-Host (U "5pyN5Yqh5pyq5ZCv5Yqo77yM5q2j5Zyo5ZCv5YqoIGppdWFuLWxpdGUuLi4=")
  $proc = Start-Process -FilePath $python -ArgumentList "-m","jiuan.app" -WorkingDirectory $root -WindowStyle Hidden -PassThru
  Set-Content -LiteralPath (Join-Path $root ".server_pid") -Value $proc.Id
  Start-Sleep -Seconds 4
}

if (-not (Test-Health)) {
  throw "$(U "5peg5rOV6L+e5o6l") $BaseUrl, $(U "6K+35qOA5p+lIFB5dGhvbiDnjq/looPlkoznq6/lj6PljaDnlKjjgII=")"
}

$body = @{
  source_name = (U "55Sf54mp55+l6K+G")
  collection = "biology"
  iteration_prefix = "bio-v"
  run_biology_if_missing = $true
  force_biology_run = [bool]$ForceTraining
  run_inference_probe = -not [bool]$SkipInference
  train_backend = "hf"
  train_device = "cpu"
  train_epochs = 1
  eval_max_samples = 11
  breadth_max_samples = 11
  use_judge = "auto"
  poll_interval = 5
} | ConvertTo-Json

Write-Host (U "5ZCv5Yqo5LiA6ZSu5L+d6YCa5omT5YyFLi4u")
$status = Post-Json "$BaseUrl/agent/oneclick/start" $body

while ($status.active) {
  $msg = if ($status.message) { $status.message } else { "running" }
  Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg)
  Start-Sleep -Seconds 5
  $status = Get-Json "$BaseUrl/agent/oneclick/status"
}

Write-Host "$(U "5a6M5oiQ77ya")$($status.message)"
Write-Host "$(U "5pyA57uI5qih5Z6L77ya")$($status.final_model_id)"
Write-Host "$(U "5pyA57uI5pWw5o2u6ZuG77ya")$($status.final_dataset_id)"
Write-Host "$(U "5oql5ZGK77ya")$($status.package_report_md)"
if ($status.problems -and $status.problems.Count -gt 0) {
  Write-Host (U "6Zeu6aKY5YiG5p6Q77ya")
  foreach ($p in $status.problems) {
    Write-Host ("- {0}: {1}" -f $p.kind, $p.detail)
  }
}
