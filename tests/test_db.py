import sqlite3

from second_brain.db import SCHEMA_VERSION, open_db


def test_schema_and_fts_are_created(tmp_path):
    conn = open_db(tmp_path / "brain.db")
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert {
        "documents", "chunks", "assets", "memory_cards", "ingestion_runs", "embeddings", "chunks_fts",
        "episodes", "concepts", "concept_evidence", "concept_states", "consolidation_runs", "analysis_cache",
    } <= names
    conn.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "brain.db"
    open_db(path).close()
    conn = open_db(path)
    assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
    conn.close()


def test_version_one_assets_are_upgraded_without_losing_ocr_text(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute("CREATE TABLE assets(id INTEGER PRIMARY KEY, ocr_text TEXT)")
    legacy.execute("INSERT INTO assets(ocr_text) VALUES('已经识别的文字')")
    legacy.commit()
    legacy.close()

    conn = open_db(path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(assets)")}
    assert {"ocr_status", "ocr_error", "ocr_attempted_at"} <= columns
    assert conn.execute("SELECT ocr_status FROM assets").fetchone()[0] == "done"
    conn.close()
