from __future__ import annotations

import sqlite3

from ..models import SearchResult
from .lexical import search_memory


def get_timeline(conn: sqlite3.Connection, start_date: str, end_date: str | None = None,
                 topic: str | None = None, limit: int = 100) -> list[SearchResult]:
    end = end_date or start_date
    if topic:
        results = search_memory(conn, topic, start_date=start_date, end_date=end, limit=limit)
        return sorted(results, key=lambda item: (item.event_date or "", item.filename, item.chunk_id))
    rows = conn.execute(
        """
        SELECT d.id AS document_id, c.id AS chunk_id, d.title, d.filename, d.path,
               d.file_type, d.event_date, d.date_source, c.content, c.source_kind
        FROM documents d JOIN chunks c ON c.document_id=d.id
        WHERE d.status='ready' AND d.event_date BETWEEN ? AND ?
        ORDER BY d.event_date, d.filename, c.chunk_index LIMIT ?
        """, (start_date, end, limit),
    ).fetchall()
    return [SearchResult(
        document_id=int(row["document_id"]), chunk_id=int(row["chunk_id"]),
        title=row["title"] or row["filename"], filename=row["filename"], path=row["path"],
        file_type=row["file_type"], event_date=row["event_date"], date_source=row["date_source"],
        snippet=row["content"][:240].replace("\n", " "), score=50.0, source_kind=row["source_kind"],
    ) for row in rows]

