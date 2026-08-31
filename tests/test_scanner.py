import os
import shutil
from dataclasses import replace
from pathlib import Path

from docx import Document

from second_brain.config import AppConfig, IngestionConfig, ModelConfig
from second_brain.db import open_db
from second_brain.ingestion.scanner import Scanner


def make_config(tmp_path: Path, source: Path) -> AppConfig:
    return AppConfig(
        config_path=tmp_path / "config.toml",
        study_year=2026,
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        device="CPU",
        source_roots=(source,),
        ingestion=IngestionConfig(extensions=(".docx",), extract_docx_images=False, enable_ocr=False, enable_asr=False),
        models=ModelConfig(agent_enabled=False),
    )


def write_docx(path: Path, text: str) -> None:
    doc = Document()
    doc.add_paragraph(text)
    doc.save(path)


def test_scanner_is_incremental_and_never_modifies_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "8.21 申论.docx"
    write_docx(path, "公文写作和主体分析")
    before = path.stat().st_mtime_ns
    config = make_config(tmp_path, source)
    config.ensure_runtime_dirs()
    conn = open_db(config.db_path)
    scanner = Scanner(config, conn)
    first = scanner.scan()
    second = scanner.scan()
    assert first.files_changed == 1
    assert second.files_changed == 0
    assert second.files_skipped == 1
    assert path.stat().st_mtime_ns == before
    row = conn.execute("SELECT path, event_date, status FROM documents").fetchone()
    assert row["path"] == str(path.resolve())
    assert row["event_date"] == "2026-08-21"
    assert row["status"] == "ready"


def test_scanner_reindexes_only_changed_document(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    one, two = source / "1.1 一.docx", source / "1.2 二.docx"
    write_docx(one, "内容一")
    write_docx(two, "内容二")
    config = make_config(tmp_path, source)
    config.ensure_runtime_dirs()
    conn = open_db(config.db_path)
    scanner = Scanner(config, conn)
    scanner.scan()
    write_docx(two, "内容二已更新")
    stats = scanner.scan()
    assert stats.files_changed == 1
    assert stats.files_skipped == 1


def test_corrupt_document_does_not_abort_scan(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "坏文件.docx").write_bytes(b"not a zip")
    write_docx(source / "好文件.docx", "正常内容")
    config = make_config(tmp_path, source)
    config.ensure_runtime_dirs()
    conn = open_db(config.db_path)
    stats = Scanner(config, conn).scan()
    assert stats.files_failed == 1
    assert stats.files_changed == 1
    assert conn.execute("SELECT COUNT(*) FROM documents WHERE status='ready'").fetchone()[0] == 1

