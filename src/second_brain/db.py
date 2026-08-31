from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 2


def _configure(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            sha256 TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            title TEXT,
            event_date TEXT,
            date_source TEXT,
            date_confidence REAL,
            created_at TEXT,
            modified_at TEXT,
            indexed_at TEXT,
            status TEXT NOT NULL,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_documents_event_date ON documents(event_date);
        CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(file_type);
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            block_index INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            source_kind TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            start_seconds REAL,
            end_seconds REAL,
            UNIQUE(document_id, source_kind, block_index, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            order_index INTEGER,
            original_name TEXT,
            stored_path TEXT NOT NULL,
            mime_type TEXT,
            sha256 TEXT,
            context_before TEXT,
            context_after TEXT,
            ocr_text TEXT,
            ocr_confidence REAL,
            ocr_status TEXT NOT NULL DEFAULT 'pending',
            ocr_error TEXT,
            ocr_attempted_at TEXT,
            vlm_caption TEXT,
            vlm_status TEXT,
            UNIQUE(document_id, stored_path)
        );
        CREATE TABLE IF NOT EXISTS memory_cards (
            document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
            topic TEXT,
            subtopics_json TEXT,
            activity_type TEXT,
            summary TEXT,
            keywords_json TEXT,
            confidence REAL,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id INTEGER PRIMARY KEY,
            started_at TEXT,
            finished_at TEXT,
            files_seen INTEGER DEFAULT 0,
            files_changed INTEGER DEFAULT 0,
            files_failed INTEGER DEFAULT 0,
            files_skipped INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector BLOB NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    asset_columns = {row["name"] for row in conn.execute("PRAGMA table_info(assets)")}
    if "ocr_status" not in asset_columns:
        conn.execute("ALTER TABLE assets ADD COLUMN ocr_status TEXT NOT NULL DEFAULT 'pending'")
    if "ocr_error" not in asset_columns:
        conn.execute("ALTER TABLE assets ADD COLUMN ocr_error TEXT")
    if "ocr_attempted_at" not in asset_columns:
        conn.execute("ALTER TABLE assets ADD COLUMN ocr_attempted_at TEXT")
    conn.execute(
        "UPDATE assets SET ocr_status=CASE WHEN COALESCE(ocr_text, '') <> '' THEN 'done' ELSE 'pending' END "
        "WHERE ocr_status IS NULL OR ocr_status = ''"
    )
    conn.execute(
        "UPDATE assets SET ocr_status='done' "
        "WHERE ocr_status='pending' AND COALESCE(ocr_text, '') <> ''"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_ocr_status ON assets(ocr_status)")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "document_id UNINDEXED, chunk_id UNINDEXED, filename, title, content, "
            "tokenize='trigram case_sensitive 0')"
        )
    except sqlite3.OperationalError:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "document_id UNINDEXED, chunk_id UNINDEXED, filename, title, content, "
            "tokenize='unicode61')"
        )
    conn.execute(
        "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def open_db(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    _configure(conn)
    migrate(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks_fts")
    conn.execute(
        """
        INSERT INTO chunks_fts(document_id, chunk_id, filename, title, content)
        SELECT d.id, c.id, d.filename, COALESCE(d.title, ''), c.content
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'ready'
        """
    )
    conn.commit()
