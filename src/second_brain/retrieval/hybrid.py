from __future__ import annotations

import sqlite3
from dataclasses import replace

from ..models import SearchResult
from .lexical import extract_topic, search_memory
from .semantic import SemanticIndex


def _metadata_search(
    conn: sqlite3.Connection,
    query: str,
    start_date: str | None,
    end_date: str | None,
    file_type: str | None,
    limit: int,
) -> list[SearchResult]:
    topic = extract_topic(query) or query.strip()
    if not topic:
        return []
    filters = ["(concepts.name LIKE ? OR concepts.subject LIKE ?)"]
    pattern = f"%{topic}%"
    params: list[object] = [pattern, pattern]
    if start_date:
        filters.append("documents.event_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("documents.event_date <= ?")
        params.append(end_date)
    if file_type:
        filters.append("documents.file_type = ?")
        params.append(file_type.lower().lstrip("."))
    rows = conn.execute(
        f"""
        SELECT documents.id AS document_id, chunks.id AS chunk_id, documents.title,
               documents.filename, documents.path, documents.file_type, documents.event_date,
               documents.date_source, chunks.content, chunks.source_kind,
               concepts.exposure_count
        FROM concepts
        JOIN concept_evidence ON concept_evidence.concept_id=concepts.id
        JOIN chunks ON chunks.id=concept_evidence.chunk_id
        JOIN documents ON documents.id=chunks.document_id
        WHERE {' AND '.join(filters)} AND documents.status='ready'
        ORDER BY concepts.exposure_count DESC, documents.event_date DESC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return [
        SearchResult(
            document_id=int(row["document_id"]), chunk_id=int(row["chunk_id"]),
            title=row["title"] or row["filename"], filename=row["filename"], path=row["path"],
            file_type=row["file_type"], event_date=row["event_date"], date_source=row["date_source"],
            snippet=str(row["content"])[:300].replace("\n", " "),
            score=float(row["exposure_count"]), source_kind=row["source_kind"],
        )
        for row in rows
    ]


def hybrid_search(conn: sqlite3.Connection, query: str, start_date: str | None = None,
                  end_date: str | None = None, file_type: str | None = None,
                  limit: int = 20, semantic: SemanticIndex | None = None) -> list[SearchResult]:
    candidate_limit = max(limit * 3, 30)
    lexical = search_memory(conn, query, start_date, end_date, file_type, candidate_limit)
    metadata = _metadata_search(conn, query, start_date, end_date, file_type, candidate_limit)
    semantic_results: list[SearchResult] = []
    if semantic is not None and semantic.available:
        try:
            semantic_results = semantic.search(query, candidate_limit)
        except Exception:
            semantic_results = []
    rankings = ((lexical, 1.0), (semantic_results, 0.9), (metadata, 1.15))
    merged: dict[int, SearchResult] = {}
    scores: dict[int, float] = {}
    for ranking, weight in rankings:
        for rank, item in enumerate(ranking, 1):
            if start_date and (not item.event_date or item.event_date < start_date):
                continue
            if end_date and (not item.event_date or item.event_date > end_date):
                continue
            if file_type and item.file_type != file_type.lower().lstrip("."):
                continue
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight / (60 + rank)
            merged.setdefault(item.chunk_id, item)
    ranked = sorted(
        merged.values(),
        key=lambda item: (-scores[item.chunk_id], item.event_date or "9999-99-99", item.chunk_id),
    )
    # Keep enough variety for the controller instead of filling the context with
    # adjacent chunks from one long diary document.
    per_document: dict[int, int] = {}
    diversified: list[SearchResult] = []
    for item in ranked:
        if per_document.get(item.document_id, 0) >= 2:
            continue
        per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
        diversified.append(replace(item, score=scores[item.chunk_id] * 10000.0))
        if len(diversified) >= limit:
            break
    return diversified
