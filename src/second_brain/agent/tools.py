from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..retrieval.hybrid import hybrid_search
from ..retrieval.semantic import OpenVINOEmbedder, SemanticIndex
from ..retrieval.timeline import get_timeline as timeline_query


class MemoryTools:
    def __init__(self, config: AppConfig, conn: sqlite3.Connection) -> None:
        self.config = config
        self.conn = conn
        embedding_path = config.model_dir / config.models.embedding_id.split("/")[-1]
        self.semantic = SemanticIndex(conn, OpenVINOEmbedder(embedding_path, config.device)) if config.models.semantic_enabled else None

    def search_memory(self, query: str, start_date: str | None = None, end_date: str | None = None,
                      file_type: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
        return [item.as_dict() for item in hybrid_search(
            self.conn, query, start_date, end_date, file_type, limit, self.semantic,
        )]

    def build_semantic_index(self, limit: int | None = None) -> dict[str, Any]:
        if self.semantic is None:
            return {"ok": False, "error": "语义检索未启用", "indexed": 0}
        if not self.semantic.available:
            return {"ok": False, "error": "OpenVINO 语义模型尚未下载", "indexed": 0}
        return {"ok": True, "indexed": self.semantic.index_missing(limit)}

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

    def analyze_image(self, asset_id: int) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        if not row:
            return {"ok": False, "error": "图片资产不存在"}
        if row["vlm_caption"]:
            return {"ok": True, "cached": True, "description": row["vlm_caption"]}
        if not self.config.models.vision_enabled:
            return {"ok": False, "error": "按需视觉分析未启用"}
        path = Path(row["stored_path"]).resolve(strict=False)
        allowed_roots = (*self.config.source_roots, self.config.assets_dir.resolve(strict=False))
        if not any(path == root or root in path.parents for root in allowed_roots):
            return {"ok": False, "error": "图片路径不在允许范围"}
        from .qwen_vision_adapter import OpenVINOVisionEngine

        model_path = self.config.model_dir / self.config.models.vision_id.split("/")[-1]
        engine = OpenVINOVisionEngine(model_path, self.config.device)
        if not engine.available:
            return {"ok": False, "error": "OpenVINO 视觉模型尚未下载"}
        try:
            result = engine.analyze(path)
        finally:
            engine.unload()
        description = str(result["description"])
        self.conn.execute("UPDATE assets SET vlm_caption=?, vlm_status='ready' WHERE id=?", (description, asset_id))
        content_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
        chunk_index = self.conn.execute(
            "SELECT COALESCE(MAX(chunk_index), -1) + 1 FROM chunks WHERE document_id=?", (row["document_id"],)
        ).fetchone()[0]
        cursor = self.conn.execute(
            "INSERT INTO chunks(document_id, block_index, chunk_index, source_kind, content, content_hash) VALUES(?, ?, ?, 'vlm', ?, ?)",
            (row["document_id"], 200000 + asset_id, chunk_index, description, content_hash),
        )
        document = self.conn.execute("SELECT filename, title FROM documents WHERE id=?", (row["document_id"],)).fetchone()
        self.conn.execute(
            "INSERT INTO chunks_fts(document_id, chunk_id, filename, title, content) VALUES(?, ?, ?, ?, ?)",
            (row["document_id"], cursor.lastrowid, document["filename"], document["title"] or "", description),
        )
        self.conn.commit()
        return {"ok": True, "cached": False, "description": description}

    def status(self) -> dict[str, Any]:
        counts = self.conn.execute(
            "SELECT COUNT(*) AS total, SUM(status='ready') AS ready, SUM(status='failed') AS failed, "
            "SUM(status='missing') AS missing, SUM(status='ignored') AS ignored FROM documents"
        ).fetchone()
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        ocr = self.conn.execute(
            "SELECT COUNT(*) AS total, SUM(ocr_status='pending') AS pending, "
            "SUM(ocr_status='done') AS done, SUM(ocr_status='failed') AS failed, "
            "SUM(COALESCE(ocr_text, '') <> '') AS with_text FROM assets"
        ).fetchone()
        latest = self.conn.execute("SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT 1").fetchone()
        return {
            "documents": dict(counts),
            "chunks": chunks,
            "images": dict(ocr),
            "latest_run": dict(latest) if latest else None,
        }
