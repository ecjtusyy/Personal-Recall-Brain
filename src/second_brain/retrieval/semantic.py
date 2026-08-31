from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..models import SearchResult


class OpenVINOEmbedder:
    """Feature extraction using only OpenVINO IR model and tokenizer graphs."""

    def __init__(self, model_path: Path, device: str = "CPU") -> None:
        self.model_path = model_path
        self.device = device
        self.model_id = model_path.name
        self._tokenizer = None
        self._model = None

    @property
    def available(self) -> bool:
        return (self.model_path / "openvino_model.xml").exists() and (self.model_path / "openvino_tokenizer.xml").exists()

    def _load(self) -> None:
        if self._model is not None:
            return
        import openvino as ov

        core = ov.Core()
        self._tokenizer = core.compile_model(str(self.model_path / "openvino_tokenizer.xml"), self.device)
        self._model = core.compile_model(str(self.model_path / "openvino_model.xml"), self.device)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.available:
            raise FileNotFoundError(f"OpenVINO 语义模型尚未下载: {self.model_path}")
        self._load()
        tokenized = self._tokenizer(texts)
        by_name = {port.any_name: value for port, value in tokenized.items()}
        inputs = {}
        for model_input in self._model.inputs:
            name = model_input.any_name
            if name in by_name:
                inputs[name] = by_name[name]
        outputs = self._model(inputs)
        array = np.asarray(next(iter(outputs.values())), dtype=np.float32)
        if array.ndim == 3:
            mask = np.asarray(by_name.get("attention_mask"))
            if mask.size:
                indexes = np.maximum(mask.sum(axis=1).astype(int) - 1, 0)
                array = array[np.arange(array.shape[0]), indexes]
            else:
                array = array[:, -1, :]
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None


class SemanticIndex:
    def __init__(self, conn: sqlite3.Connection, embedder: OpenVINOEmbedder) -> None:
        self.conn = conn
        self.embedder = embedder

    @property
    def available(self) -> bool:
        return self.embedder.available

    def index_missing(self, limit: int | None = None) -> int:
        sql = """
            SELECT c.id, c.content FROM chunks c
            LEFT JOIN embeddings e ON e.chunk_id=c.id AND e.model_id=?
            WHERE e.chunk_id IS NULL ORDER BY c.id
        """
        params: list[object] = [self.embedder.model_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        count = 0
        for offset in range(0, len(rows), 8):
            batch = rows[offset : offset + 8]
            vectors = self.embedder.embed([row["content"] for row in batch])
            for row, vector in zip(batch, vectors, strict=True):
                self.conn.execute(
                    """
                    INSERT INTO embeddings(chunk_id, model_id, dimensions, vector, updated_at)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET model_id=excluded.model_id,
                        dimensions=excluded.dimensions, vector=excluded.vector, updated_at=excluded.updated_at
                    """,
                    (row["id"], self.embedder.model_id, int(vector.size), vector.astype(np.float32).tobytes(),
                     datetime.now(timezone.utc).isoformat()),
                )
                count += 1
            self.conn.commit()
        return count

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        query_vector = self.embedder.embed([query])[0]
        rows = self.conn.execute(
            """
            SELECT e.vector, e.dimensions, c.id AS chunk_id, c.content, c.source_kind,
                   d.id AS document_id, d.title, d.filename, d.path, d.file_type,
                   d.event_date, d.date_source
            FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
            JOIN documents d ON d.id=c.document_id
            WHERE e.model_id=? AND d.status='ready'
            """, (self.embedder.model_id,),
        ).fetchall()
        scored: list[SearchResult] = []
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32, count=row["dimensions"])
            if vector.size != query_vector.size:
                continue
            similarity = float(np.dot(query_vector, vector))
            scored.append(SearchResult(
                document_id=row["document_id"], chunk_id=row["chunk_id"],
                title=row["title"] or row["filename"], filename=row["filename"], path=row["path"],
                file_type=row["file_type"], event_date=row["event_date"], date_source=row["date_source"],
                snippet=row["content"][:240].replace("\n", " "), score=similarity * 100.0,
                source_kind=row["source_kind"],
            ))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

