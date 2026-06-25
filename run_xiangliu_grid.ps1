$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\ZH\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
Set-Location $Root
Write-Host "Xiangliu Grid v0.3.0: http://127.0.0.1:8765"
Write-Host "Keep this window open while using the tool."
& $Python "$Root\server.py" 8765
