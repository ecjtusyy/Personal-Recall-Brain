$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "尚未安装，请先双击“安装与下载模型.bat”。" }
& $python -m second_brain.cli --config config.toml scan
Read-Host "扫描结束，按回车退出"

