$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$python = ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "尚未安装，请先双击 '安装与下载模型.bat'。" }
$env:SECOND_BRAIN_CONFIG = (Resolve-Path -LiteralPath "config.toml").Path
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$arguments = @(
    "-m", "streamlit", "run", "src\second_brain\ui\app.py",
    "--server.address", "127.0.0.1", "--server.port", "8501",
    "--server.headless", "true", "--browser.gatherUsageStats", "false"
)
$server = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -NoNewWindow -PassThru
$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if ($server.HasExited) { break }
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8501/_stcore/health" -TimeoutSec 2
        if ($response.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    throw "第二大脑服务未能在 60 秒内启动，请查看上方错误。"
}
if (-not $env:SECOND_BRAIN_SKIP_BROWSER) { Start-Process "http://127.0.0.1:8501" }
$server.WaitForExit()
exit $server.ExitCode

