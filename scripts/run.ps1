$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "尚未安装，请先双击“安装与下载模型.bat”。" }
$env:SECOND_BRAIN_CONFIG = (Resolve-Path -LiteralPath "config.toml").Path
& $python -m streamlit run "src\second_brain\ui\app.py" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false

