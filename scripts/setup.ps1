$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "[1/4] 检查 Python 3.11+" -ForegroundColor Cyan
$python = $null
foreach ($candidate in @("py", "python")) {
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($command -and $command.Source -notlike "*WindowsApps*") {
        $python = $command.Source
        break
    }
}
if (-not $python) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) { throw "未找到 Python，也未找到 winget。请安装 Python 3.11 或 3.12 后重试。" }
    Write-Host "正在为当前用户安装 Python 3.12…"
    winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
    $python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (-not (Test-Path -LiteralPath $python)) { throw "Python 安装后仍未找到，请重开终端再运行本脚本。" }
}

Write-Host "[2/4] 创建独立运行环境" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $python -m venv .venv
}
$venvPython = (Resolve-Path -LiteralPath ".venv\Scripts\python.exe").Path
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[dev]"

Write-Host "[3/4] 初始化安全配置" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath "config.toml")) {
    Copy-Item -LiteralPath "config.example.toml" -Destination "config.toml"
}
& $venvPython -m second_brain.cli --config config.toml init

Write-Host "[4/4] 下载 Intel OpenVINO 核心 INT4 模型（约 1.2GB）" -ForegroundColor Cyan
& $venvPython -m second_brain.cli --config config.toml download-models --profile core

Write-Host "安装完成。双击“启动第二大脑.bat”即可使用。" -ForegroundColor Green
Read-Host "按回车退出"

