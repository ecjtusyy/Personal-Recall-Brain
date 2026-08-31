from pathlib import Path


def test_powershell_scripts_have_utf8_bom_for_windows_powershell_51():
    scripts = Path(__file__).parents[1] / "scripts"
    for path in scripts.glob("*.ps1"):
        assert path.read_bytes().startswith(b"\xef\xbb\xbf"), f"{path.name} 缺少 UTF-8 BOM"
        text = path.read_text(encoding="utf-8-sig")
        assert "“" not in text and "”" not in text, f"{path.name} 包含 PowerShell 会解释为引号的弯引号"


def test_run_script_disables_streamlit_first_run_prompt():
    root = Path(__file__).parents[1]
    text = (root / "scripts" / "run.ps1").read_text(encoding="utf-8-sig")
    assert '$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"' in text
    assert '"--server.headless", "true"' in text
    assert 'Start-Process "http://127.0.0.1:8501"' in text
