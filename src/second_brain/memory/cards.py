from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone


def _activity(text: str) -> str:
    rules = (
        ("review", ("复习", "回顾", "错题", "背诵")),
        ("plan", ("计划", "明天", "安排", "待办")),
        ("idea", ("想法", "思考", "灵感", "反思")),
        ("project", ("项目", "论文", "代码", "实验")),
    )
    for label, words in rules:
        if any(word in text for word in words):
            return label
    return "unknown"


def _keywords(title: str) -> list[str]:
    parts = re.split(r"[\s,，、。；;：:!！?？()（）\[\]【】._-]+", title)
    return [part for part in parts if 2 <= len(part) <= 20][:12]


def build_memory_card(conn: sqlite3.Connection, document_id: int) -> None:
    doc = conn.execute("SELECT title, filename FROM documents WHERE id=?", (document_id,)).fetchone()
    chunks = conn.execute("SELECT content FROM chunks WHERE document_id=? ORDER BY chunk_index LIMIT 3", (document_id,)).fetchall()
    if not doc:
        return
    title = (doc["title"] or doc["filename"]).strip()
    text = "\n".join(row["content"] for row in chunks)
    summary = text[:360].strip()
    keywords = _keywords(f"{title} {doc['filename']}")
    conn.execute(
        """
        INSERT INTO memory_cards(document_id, topic, subtopics_json, activity_type, summary,
                                 keywords_json, confidence, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET topic=excluded.topic, subtopics_json=excluded.subtopics_json,
            activity_type=excluded.activity_type, summary=excluded.summary, keywords_json=excluded.keywords_json,
            confidence=excluded.confidence, updated_at=excluded.updated_at
        """,
        (document_id, title[:160], "[]", _activity(f"{title}\n{text}"), summary,
         json.dumps(keywords, ensure_ascii=False), 0.72 if summary else 0.4, datetime.now(timezone.utc).isoformat()),
    )

