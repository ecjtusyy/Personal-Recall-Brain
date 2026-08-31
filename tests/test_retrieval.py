from pathlib import Path

from docx import Document

from second_brain.agent.tools import MemoryTools
from second_brain.config import AppConfig, IngestionConfig, ModelConfig
from second_brain.db import open_db
from second_brain.ingestion.scanner import Scanner
from second_brain.retrieval.lexical import extract_topic, is_progress_query, search_memory
from second_brain.retrieval.timeline import get_timeline


def make_brain(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for filename, content in (
        ("8.21！！！申论开始.docx", "今天复习申论，重点是公文写作、主体分析和材料分析。"),
        ("8.22 高等数学.docx", "今天复习一致收敛与最小多项式。"),
    ):
        doc = Document()
        doc.add_paragraph(content)
        doc.save(source / filename)
    config = AppConfig(
        config_path=tmp_path / "config.toml", study_year=2026, data_dir=tmp_path / "data",
        model_dir=tmp_path / "models", device="CPU", source_roots=(source,),
        ingestion=IngestionConfig(extensions=(".docx",), extract_docx_images=False, enable_ocr=False, enable_asr=False),
        models=ModelConfig(agent_enabled=False),
    )
    config.ensure_runtime_dirs()
    conn = open_db(config.db_path)
    Scanner(config, conn).scan()
    return config, conn


def test_natural_question_extracts_topic():
    assert extract_topic("我以前什么时候复习过申论吗？") == "申论"
    assert extract_topic("我的高等代数的学习路线是什么情况？") == "高等代数"
    assert is_progress_query("我的高等代数学习到什么阶段？")


def test_chinese_lexical_search_returns_evidence(tmp_path):
    _config, conn = make_brain(tmp_path)
    result = search_memory(conn, "申论")[0]
    assert result.event_date == "2026-08-21"
    assert "公文写作" in result.snippet
    assert result.path.endswith("8.21！！！申论开始.docx")


def test_title_match_and_filters(tmp_path):
    _config, conn = make_brain(tmp_path)
    results = search_memory(conn, "高等数学", file_type="docx")
    assert results[0].filename == "8.22 高等数学.docx"
    assert search_memory(conn, "申论", start_date="2026-08-22") == []


def test_timeline_exact_date(tmp_path):
    _config, conn = make_brain(tmp_path)
    results = get_timeline(conn, "2026-08-21")
    assert len(results) == 1
    assert results[0].filename == "8.21！！！申论开始.docx"


def test_structured_agent_tools_and_source_guard(tmp_path):
    config, conn = make_brain(tmp_path)
    tools = MemoryTools(config, conn)
    result = tools.search_memory("我在哪个 Word 写过公文写作？")[0]
    assert result["filename"] == "8.21！！！申论开始.docx"
    assert tools.get_evidence(result["chunk_id"])["path"] == result["path"]
    assert tools.open_source(result["document_id"])["ok"] is True
    assert tools.status()["documents"]["ready"] == 2


def test_trace_topic_returns_chronological_documents(tmp_path):
    config, conn = make_brain(tmp_path)
    tools = MemoryTools(config, conn)
    results = tools.trace_topic("高等数学")
    assert [item["event_date"] for item in results] == ["2026-08-22"]
