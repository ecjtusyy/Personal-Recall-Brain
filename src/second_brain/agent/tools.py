from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..retrieval.hybrid import hybrid_search
from ..retrieval.timeline import get_timeline as timeline_query


class MemoryTools:
    def __init__(self, config: AppConfig, conn: sqlite3.Connection) -> None:
        self.config = config
        self.conn = conn

    def search_memory(self, query: str, start_date: str | None = None, end_date: str | None = None,
                      file_type: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
        return [item.as_dict() for item in hybrid_search(self.conn, query, start_date, end_date, file_type, limit)]

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT d.*, m.topic, m.activity_type, m.summary, m.keywords_json FROM documents d "
            "LEFT JOIN memory_cards m ON m.document_id=d.id WHERE d.id=?", (document_id,),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["chunks"] = [dict(chunk) for chunk in self.conn.execute(
            "SELECT id, source_kind, content, start_seconds, end_seconds FROM chunks WHERE document_id=? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()]
        return result

    def get_evidence(self, chunk_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT c.id AS chunk_id, c.content, c.source_kind, d.id AS document_id,
                   d.filename, d.path, d.event_date, d.date_source
            FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?
            """, (chunk_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_timeline(self, start_date: str, end_date: str | None = None,
                     topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [item.as_dict() for item in timeline_query(self.conn, start_date, end_date, topic, limit)]

    def open_source(self, document_id: int) -> dict[str, Any]:
        row = self.conn.execute("SELECT path FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            return {"ok": False, "error": "文档不存在"}
        path = Path(row["path"]).resolve(strict=False)
        allowed = any(path == root or root in path.parents for root in self.config.source_roots)
        if not allowed:
            return {"ok": False, "error": "来源路径不在允许的只读资料目录中"}
        return {"ok": path.exists(), "path": str(path), "error": None if path.exists() else "源文件当前不可见"}

    def status(self) -> dict[str, Any]:
        counts = self.conn.execute(
            "SELECT COUNT(*) AS total, SUM(status='ready') AS ready, SUM(status='failed') AS failed, "
            "SUM(status='missing') AS missing FROM documents"
        ).fetchone()
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        latest = self.conn.execute("SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT 1").fetchone()
        return {"documents": dict(counts), "chunks": chunks, "latest_run": dict(latest) if latest else None}

