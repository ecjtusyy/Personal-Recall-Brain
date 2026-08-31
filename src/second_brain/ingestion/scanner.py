from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..chunking import chunk_blocks
from ..config import AppConfig
from ..dates import resolve_event_date
from ..db import transaction
from ..fingerprint import sha256_bytes, sha256_file
from ..models import Chunk, ScanStats, TextBlock
from .audio_pipeline import OpenVINOASREngine
from .image_pipeline import OpenVINOOCREngine
from .text_parser import parse_text_source


IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".flac"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_source_files(config: AppConfig):
    extensions = set(config.ingestion.extensions)
    for root in config.source_roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            for filename in files:
                path = Path(current) / filename
                if path.suffix.lower() in extensions:
                    yield path


class Scanner:
    def __init__(
        self,
        config: AppConfig,
        conn: sqlite3.Connection,
        progress: Callable[[str], None] | None = None,
        ocr_engine: OpenVINOOCREngine | None = None,
        asr_engine: OpenVINOASREngine | None = None,
    ) -> None:
        self.config = config
        self.conn = conn
        self.progress = progress or (lambda _message: None)
        self._ocr = ocr_engine
        asr_path = config.model_dir / config.models.asr_id.split("/")[-1]
        self._asr = asr_engine or OpenVINOASREngine(asr_path, config.device)

    def scan(self) -> ScanStats:
        run_id = self.conn.execute("INSERT INTO ingestion_runs(started_at) VALUES(?)", (_utcnow(),)).lastrowid
        self.conn.commit()
        stats = ScanStats()
        seen_paths: set[str] = set()
        for path in iter_source_files(self.config):
            stats = replace(stats, files_seen=stats.files_seen + 1)
            canonical = str(path.resolve(strict=False))
            seen_paths.add(canonical)
            try:
                if path.stat().st_size > self.config.ingestion.max_file_mb * 1024 * 1024:
                    stats = replace(stats, files_skipped=stats.files_skipped + 1)
                    continue
                fingerprint = sha256_file(path)
                existing = self.conn.execute("SELECT id, sha256, status FROM documents WHERE path=?", (canonical,)).fetchone()
                if existing and existing["sha256"] == fingerprint and existing["status"] == "ready":
                    stats = replace(stats, files_skipped=stats.files_skipped + 1)
                    continue
                self.progress(f"正在索引：{path.name}")
                self._index_one(path, canonical, fingerprint)
                stats = replace(stats, files_changed=stats.files_changed + 1)
            except Exception as exc:
                self._record_failure(path, canonical, str(exc))
                stats = replace(stats, files_failed=stats.files_failed + 1)
        for row in self.conn.execute("SELECT id, path FROM documents WHERE status='ready'").fetchall():
            if row["path"] not in seen_paths and any(str(row["path"]).lower().startswith(str(root).lower()) for root in self.config.source_roots):
                self.conn.execute("UPDATE documents SET status='missing', error='源文件当前不可见' WHERE id=?", (row["id"],))
        self.conn.execute(
            "UPDATE ingestion_runs SET finished_at=?, files_seen=?, files_changed=?, files_failed=?, files_skipped=? WHERE id=?",
            (_utcnow(), stats.files_seen, stats.files_changed, stats.files_failed, stats.files_skipped, run_id),
        )
        self.conn.commit()
        return stats

    def _upsert_document(self, path: Path, canonical: str, fingerprint: str, parsed, date_info) -> int:
        stat = path.stat()
        created = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_ctime)).isoformat()
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
        self.conn.execute(
            """
            INSERT INTO documents(path, sha256, filename, file_type, title, event_date, date_source,
                                  date_confidence, created_at, modified_at, indexed_at, status, error)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'indexing', NULL)
            ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, filename=excluded.filename,
                file_type=excluded.file_type, title=excluded.title, event_date=excluded.event_date,
                date_source=excluded.date_source, date_confidence=excluded.date_confidence,
                modified_at=excluded.modified_at, indexed_at=excluded.indexed_at, status='indexing', error=NULL
            """,
            (
                canonical, fingerprint, path.name, path.suffix.lower().lstrip("."), parsed.title,
                date_info.value.isoformat(), date_info.source, date_info.confidence,
                created, modified, _utcnow(),
            ),
        )
        return int(self.conn.execute("SELECT id FROM documents WHERE path=?", (canonical,)).fetchone()[0])

    def _index_one(self, path: Path, canonical: str, fingerprint: str) -> None:
        initial = path.stat()
        parsed = parse_text_source(path, self.config.ingestion.extract_docx_images)
        date_info = resolve_event_date(path, self.config.study_year, parsed.core_properties, initial)
        with transaction(self.conn):
            document_id = self._upsert_document(path, canonical, fingerprint, parsed, date_info)
            self.conn.execute("DELETE FROM chunks_fts WHERE document_id=?", (document_id,))
            self.conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            self.conn.execute("DELETE FROM assets WHERE document_id=?", (document_id,))
            chunks = chunk_blocks(parsed.text_blocks)
            self._store_assets(document_id, fingerprint, parsed.assets, chunks)
            suffix = path.suffix.lower()
            if suffix in IMAGE_TYPES:
                self._store_standalone_image(document_id, path, chunks)
            elif suffix in AUDIO_TYPES:
                self._store_audio(path, chunks)
            for chunk in chunks:
                cursor = self.conn.execute(
                    "INSERT INTO chunks(document_id, block_index, chunk_index, source_kind, content, content_hash) VALUES(?, ?, ?, ?, ?, ?)",
                    (document_id, chunk.block_index, chunk.chunk_index, chunk.source_kind, chunk.content, chunk.content_hash),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts(document_id, chunk_id, filename, title, content) VALUES(?, ?, ?, ?, ?)",
                    (document_id, cursor.lastrowid, path.name, parsed.title or path.stem, chunk.content),
                )
            final = path.stat()
            if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                raise RuntimeError("索引期间源文件发生变化，请稍后重试")
            self.conn.execute("UPDATE documents SET status='ready', error=NULL WHERE id=?", (document_id,))

    def _ocr_engine(self) -> OpenVINOOCREngine:
        if self._ocr is None:
            self._ocr = OpenVINOOCREngine()
        return self._ocr

    def _store_assets(self, document_id: int, fingerprint: str, assets, chunks: list[Chunk]) -> None:
        if not assets:
            return
        folder = self.config.assets_dir / fingerprint[:16]
        folder.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            safe_name = f"{asset.order_index:04d}_{Path(asset.original_name).name}"
            stored = folder / safe_name
            stored.write_bytes(asset.data)
            asset_hash = sha256_bytes(asset.data)
            ocr_text = ""
            confidence = 0.0
            if self.config.ingestion.enable_ocr and stored.suffix.lower() in IMAGE_TYPES:
                try:
                    result = self._ocr_engine().ocr(stored)
                    ocr_text, confidence = result.text, result.confidence
                except Exception as exc:
                    self.progress(f"图片 OCR 暂缓：{asset.original_name}（{exc}）")
            self.conn.execute(
                """INSERT INTO assets(document_id, order_index, original_name, stored_path, mime_type, sha256,
                                      context_before, context_after, ocr_text, ocr_confidence, vlm_status)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (document_id, asset.order_index, asset.original_name, str(stored), asset.mime_type, asset_hash,
                 asset.context_before, asset.context_after, ocr_text, confidence, "pending"),
            )
            if ocr_text:
                block = 100000 + asset.order_index
                chunks.extend(chunk_blocks([TextBlock(block, ocr_text, "ocr")], source_kind="ocr"))

    def _store_standalone_image(self, document_id: int, path: Path, chunks: list[Chunk]) -> None:
        ocr_text = ""
        confidence = 0.0
        if self.config.ingestion.enable_ocr:
            try:
                result = self._ocr_engine().ocr(path)
                ocr_text, confidence = result.text, result.confidence
            except Exception as exc:
                self.progress(f"图片 OCR 暂缓：{path.name}（{exc}）")
        self.conn.execute(
            "INSERT INTO assets(document_id, order_index, original_name, stored_path, mime_type, sha256, ocr_text, ocr_confidence, vlm_status) VALUES(?, 0, ?, ?, ?, ?, ?, ?, ?)",
            (document_id, path.name, str(path), None, sha256_file(path), ocr_text, confidence, "pending"),
        )
        if ocr_text:
            chunks.extend(chunk_blocks([TextBlock(0, ocr_text, "ocr")], source_kind="ocr"))

    def _store_audio(self, path: Path, chunks: list[Chunk]) -> None:
        if not self.config.ingestion.enable_asr:
            return
        try:
            segments = self._asr.transcribe(path)
        except Exception as exc:
            self.progress(f"音频转写暂缓：{path.name}（{exc}）")
            return
        for index, segment in enumerate(segments):
            content = f"[{segment.start:.1f}s-{segment.end:.1f}s] {segment.text}"
            chunks.extend(chunk_blocks([TextBlock(index, content, "asr")], source_kind="asr"))

    def _record_failure(self, path: Path, canonical: str, error: str) -> None:
        try:
            fingerprint = sha256_file(path)
        except Exception:
            fingerprint = "unavailable"
        self.conn.execute(
            """
            INSERT INTO documents(path, sha256, filename, file_type, title, indexed_at, status, error)
            VALUES(?, ?, ?, ?, ?, ?, 'failed', ?)
            ON CONFLICT(path) DO UPDATE SET indexed_at=excluded.indexed_at, status='failed', error=excluded.error
            """,
            (canonical, fingerprint, path.name, path.suffix.lower().lstrip("."), path.stem, _utcnow(), error[:2000]),
        )
        self.conn.commit()

