from __future__ import annotations

import re
import sqlite3

from ..models import SearchResult


QUESTION_NOISE = (
    "我的", "学习路线", "学习路径", "学习进展", "学习进度", "学习情况", "掌握情况",
    "是什么情况", "什么情况", "目前情况", "当前情况", "现在情况", "整体情况",
    "进展如何", "进度如何", "学得怎么样", "学到哪里", "到什么阶段",
    "我以前", "我之前", "之前", "以前", "什么时候", "哪一天", "那天", "学了什么",
    "复习过", "学习过", "写过", "记录过", "在哪里", "在哪个", "哪份", "是否",
    "有没有", "了吗", "吗", "请帮我", "帮我", "word", "Word", "文档", "文件", "的",
)
PROGRESS_MARKERS = (
    "路线", "路径", "进展", "进度", "情况", "阶段", "脉络", "轨迹", "总结", "梳理",
    "学到哪里", "学得怎么样",
)
DATE_TEXT = re.compile(r"(?<!\d)(?:20\d{2}[年./_-])?\d{1,2}[月./_-]\d{1,2}日?(?!\d)")


def extract_topic(query: str) -> str:
    text = DATE_TEXT.sub(" ", query)
    for noise in sorted(QUESTION_NOISE, key=len, reverse=True):
        text = text.replace(noise, " ")
    text = re.sub(r"[^\w\u3400-\u9fff]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split()).strip()


def is_progress_query(query: str) -> bool:
    return any(marker in query for marker in PROGRESS_MARKERS)


def _snippet(content: str, query: str, width: int = 180) -> str:
    terms = [term for term in query.split() if term]
    positions = [content.lower().find(term.lower()) for term in terms]
    positions = [pos for pos in positions if pos >= 0]
    pos = min(positions) if positions else 0
    start = max(0, pos - width // 3)
    end = min(len(content), start + width)
    return ("…" if start else "") + content[start:end].replace("\n", " ") + ("…" if end < len(content) else "")


def _row_result(row: sqlite3.Row, query: str, score: float) -> SearchResult:
    return SearchResult(
        document_id=int(row["document_id"]), chunk_id=int(row["chunk_id"]),
        title=row["title"] or row["filename"], filename=row["filename"], path=row["path"],
        file_type=row["file_type"], event_date=row["event_date"], date_source=row["date_source"],
        snippet=_snippet(row["content"], query), score=float(score), source_kind=row["source_kind"],
    )


def search_memory(
    conn: sqlite3.Connection,
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    file_type: str | None = None,
    limit: int = 20,
) -> list[SearchResult]:
    topic = extract_topic(query) or query.strip()
    filters = ["d.status='ready'"]
    params: list[object] = []
    if start_date:
        filters.append("d.event_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("d.event_date <= ?")
        params.append(end_date)
    if file_type:
        filters.append("d.file_type = ?")
        params.append(file_type.lower().lstrip("."))
    where = " AND ".join(filters)
    base_columns = """
        SELECT d.id AS document_id, c.id AS chunk_id, d.title, d.filename, d.path,
               d.file_type, d.event_date, d.date_source, c.content, c.source_kind
        FROM chunks c JOIN documents d ON d.id=c.document_id
    """
    found: dict[int, SearchResult] = {}
    if len(topic.replace(" ", "")) >= 3:
        fts_query = '"' + topic.replace('"', '""') + '"'
        try:
            sql = f"""
                SELECT d.id AS document_id, c.id AS chunk_id, d.title, d.filename, d.path,
                       d.file_type, d.event_date, d.date_source, c.content, c.source_kind,
                       bm25(chunks_fts, 0.0, 0.0, 5.0, 7.0, 1.0) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.id=CAST(chunks_fts.chunk_id AS INTEGER)
                JOIN documents d ON d.id=c.document_id
                WHERE chunks_fts MATCH ? AND {where}
                ORDER BY rank LIMIT ?
            """
            for row in conn.execute(sql, [fts_query, *params, limit]).fetchall():
                result = _row_result(row, topic, 100.0 - float(row["rank"]))
                found[result.chunk_id] = result
        except sqlite3.OperationalError:
            pass
    terms = [term for term in topic.split() if len(term) >= 2] or ([topic] if topic else [])
    if terms:
        like_parts: list[str] = []
        like_params: list[object] = []
        for term in terms:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            like_parts.append("(c.content LIKE ? ESCAPE '\\' OR d.filename LIKE ? ESCAPE '\\' OR COALESCE(d.title,'') LIKE ? ESCAPE '\\')")
            like_params.extend([pattern, pattern, pattern])
        sql = f"{base_columns} WHERE {where} AND ({' OR '.join(like_parts)}) LIMIT ?"
        for row in conn.execute(sql, [*params, *like_params, max(limit * 3, 30)]).fetchall():
            title_hits = sum(term.lower() in f"{row['filename']} {row['title'] or ''}".lower() for term in terms)
            content_hits = sum(term.lower() in row["content"].lower() for term in terms)
            result = _row_result(row, topic, 60.0 + title_hits * 15.0 + content_hits * 8.0)
            previous = found.get(result.chunk_id)
            if previous is None or result.score > previous.score:
                found[result.chunk_id] = result
    return sorted(found.values(), key=lambda item: (-item.score, item.event_date or "9999"))[:limit]
