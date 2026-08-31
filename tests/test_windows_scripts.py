from pathlib import Path


def test_powershell_scripts_have_utf8_bom_for_windows_powershell_51():
    scripts = Path(__file__).parents[1] / "scripts"
    for path in scripts.glob("*.ps1"):
        assert path.read_bytes().startswith(b"\xef\xbb\xbf"), f"{path.name} 缺少 UTF-8 BOM"
        text = path.read_text(encoding="utf-8-sig")
        assert "“" not in text and "”" not in text, f"{path.name} 包含 PowerShell 会解释为引号的弯引号"
