from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image, ImageStat

from ..chunking import chunk_blocks
from ..config import AppConfig
from ..db import transaction
from ..memory.cards import build_memory_card
from ..models import TextBlock
from .image_pipeline import OpenVINOOCREngine


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EnrichmentStats:
    selected: int = 0
    completed: int = 0
    with_text: int = 0
    skipped_blank: int = 0
    failed: int = 0


class ImageEnricher:
    """Resumable, per-image OCR enrichment; source material is always read-only."""

    def __init__(
        self,
        config: AppConfig,
        conn: sqlite3.Connection,
        progress: Callable[[str], None] | None = None,
        ocr_engine: OpenVINOOCREngine | None = None,
    ) -> None:
        self.config = config
        self.conn = conn
        self.progress = progress or (lambda _message: None)
        self._ocr = ocr_engine

    def enrich(self, limit: int = 200, retry_failed: bool = False) -> EnrichmentStats:
        statuses = ("pending", "failed") if retry_failed else ("pending",)
        placeholders = ",".join("?" for _ in statuses)
        rows = self.conn.execute(
            f"""
            SELECT a.*, d.filename, d.title
            FROM assets a JOIN documents d ON d.id=a.document_id
            WHERE d.status='ready' AND a.ocr_status IN ({placeholders})
            ORDER BY a.id LIMIT ?
            """,
            (*statuses, max(1, limit)),
        ).fetchall()
        stats = EnrichmentStats(selected=len(rows))
        if not rows:
            return stats
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        for row in rows:
            path = Path(row["stored_path"]).resolve(strict=False)
            try:
                self._assert_allowed(path)
                if self._is_blank_or_tiny(path):
                    self._store_result(row, "", 0.0, "done", None)
                    stats = replace(stats, completed=stats.completed + 1, skipped_blank=stats.skipped_blank + 1)
                    continue
                self.progress(f"OCR：{row['filename']} / {row['original_name']}")
                result = self._engine().ocr(path)
                self._store_result(row, result.text, result.confidence, "done", None)
                stats = replace(
                    stats,
                    completed=stats.completed + 1,
                    with_text=stats.with_text + (1 if result.text else 0),
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self._store_result(row, "", 0.0, "failed", str(exc)[:2000])
                stats = replace(stats, failed=stats.failed + 1)
                self.progress(f"OCR 失败：{row['original_name']}（{exc}）")
        return stats

    def _engine(self) -> OpenVINOOCREngine:
        if self._ocr is None:
            self._ocr = OpenVINOOCREngine()
        return self._ocr

    def _assert_allowed(self, path: Path) -> None:
        roots = (*self.config.source_roots, self.config.assets_dir.resolve(strict=False))
        if not any(path == root or root in path.parents for root in roots):
            raise ValueError("图片路径不在只读资料目录或项目缓存目录内")
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")

    @staticmethod
    def _is_blank_or_tiny(path: Path) -> bool:
        with Image.open(path) as image:
            if image.width < 48 or image.height < 24 or image.width * image.height < 4096:
                return True
            gray = image.convert("L")
            low, high = ImageStat.Stat(gray).extrema[0]
            return high - low < 8

    def _store_result(
        self,
        row: sqlite3.Row,
        text: str,
        confidence: float,
        status: str,
        error: str | None,
    ) -> None:
        text = text.strip()
        block_index = 300000 + int(row["id"])
        with transaction(self.conn):
            old_ids = [
                old["id"] for old in self.conn.execute(
                    "SELECT id FROM chunks WHERE document_id=? AND source_kind='ocr' AND block_index=?",
                    (row["document_id"], block_index),
                ).fetchall()
            ]
            for chunk_id in old_ids:
                self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
            self.conn.execute(
                "DELETE FROM chunks WHERE document_id=? AND source_kind='ocr' AND block_index=?",
                (row["document_id"], block_index),
            )
            if status == "done" and text:
                chunks = chunk_blocks([TextBlock(block_index, text, "ocr")], source_kind="ocr")
                for chunk in chunks:
                    cursor = self.conn.execute(
                        "INSERT INTO chunks(document_id, block_index, chunk_index, source_kind, content, content_hash) "
                        "VALUES(?, ?, ?, 'ocr', ?, ?)",
                        (row["document_id"], chunk.block_index, chunk.chunk_index, chunk.content, chunk.content_hash),
                    )
                    self.conn.execute(
                        "INSERT INTO chunks_fts(document_id, chunk_id, filename, title, content) VALUES(?, ?, ?, ?, ?)",
                        (row["document_id"], cursor.lastrowid, row["filename"], row["title"] or "", chunk.content),
                    )
            self.conn.execute(
                "UPDATE assets SET ocr_text=?, ocr_confidence=?, ocr_status=?, ocr_error=?, ocr_attempted_at=? WHERE id=?",
                (text, confidence, status, error, _utcnow(), row["id"]),
            )
            build_memory_card(self.conn, int(row["document_id"]))
