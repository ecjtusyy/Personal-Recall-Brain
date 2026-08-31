from __future__ import annotations

import sqlite3

from ..models import SearchResult
from .lexical import search_memory
from .semantic import SemanticIndex


def hybrid_search(conn: sqlite3.Connection, query: str, start_date: str | None = None,
                  end_date: str | None = None, file_type: str | None = None,
                  limit: int = 20, semantic: SemanticIndex | None = None) -> list[SearchResult]:
    lexical = search_memory(conn, query, start_date, end_date, file_type, limit)
    if semantic is None or not semantic.available:
        return lexical
    try:
        semantic_results = semantic.search(query, limit)
    except Exception:
        return lexical
    merged: dict[int, SearchResult] = {item.chunk_id: item for item in lexical}
    scores = {item.chunk_id: 1.0 / (60 + rank) for rank, item in enumerate(lexical, 1)}
    for rank, item in enumerate(semantic_results, 1):
        scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (60 + rank)
        merged.setdefault(item.chunk_id, item)
    return sorted(merged.values(), key=lambda item: scores[item.chunk_id], reverse=True)[:limit]
