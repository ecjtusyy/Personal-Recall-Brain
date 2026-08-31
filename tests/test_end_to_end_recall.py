from pathlib import Path

from docx import Document

from second_brain.config import AppConfig, IngestionConfig, ModelConfig
from second_brain.db import open_db
from second_brain.ingestion.scanner import Scanner
from second_brain.retrieval.lexical import search_memory


def test_word_recall_workflow_end_to_end(tmp_path: Path):
    source = tmp_path / "资料"
    source.mkdir()
    target = source / "8.21！！！申论开始.docx"
    unrelated = source / "8.22 英语.docx"
    for path, text in ((target, "申论中的公文写作需要做好主体分析。"), (unrelated, "英语单词复习。")):
        doc = Document()
        doc.add_paragraph(text)
        doc.save(path)
    config = AppConfig(
        config_path=tmp_path / "config.toml", study_year=2026, data_dir=tmp_path / "data",
        model_dir=tmp_path / "models", device="CPU", source_roots=(source,),
        ingestion=IngestionConfig(extensions=(".docx",), extract_docx_images=False, enable_ocr=False, enable_asr=False),
        models=ModelConfig(agent_enabled=False),
    )
    config.ensure_runtime_dirs()
    conn = open_db(config.db_path)
    scanner = Scanner(config, conn)
    first = scanner.scan()
    results = search_memory(conn, "我之前复习过申论吗？")
    second = scanner.scan()
    assert first.files_changed == 2
    assert second.files_changed == 0
    assert results[0].event_date == "2026-08-21"
    assert results[0].path == str(target.resolve())
    assert "公文写作" in results[0].snippet

