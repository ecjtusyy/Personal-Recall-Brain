from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from second_brain.dates import parse_date_query, resolve_event_date


def test_filename_full_date_has_high_confidence():
    result = resolve_event_date(Path("2026-08-21 申论.docx"), 2026)
    assert result.value.isoformat() == "2026-08-21"
    assert result.source == "filename"
    assert result.confidence > 0.9


def test_short_date_uses_study_year():
    result = resolve_event_date(Path("8.21！！！申论开始.docx"), 2026)
    assert result.value.isoformat() == "2026-08-21"
    assert result.source == "filename"


def test_invalid_filename_date_falls_back_to_core_property():
    result = resolve_event_date(
        Path("2.30 错误日期.docx"),
        2026,
        {"created": datetime(2025, 9, 1, 10, 0)},
    )
    assert result.value.isoformat() == "2025-09-01"
    assert result.source == "document_created"


def test_filesystem_fallback():
    result = resolve_event_date(Path("无日期.docx"), 2026, stat=SimpleNamespace(st_ctime=0, st_mtime=0))
    assert result.source == "filesystem"


def test_parse_date_query():
    assert parse_date_query("8.21 那天学了什么", 2026) == ("2026-08-21", "2026-08-21")

