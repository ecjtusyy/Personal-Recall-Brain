from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..retrieval.hybrid import hybrid_search
from ..retrieval.semantic import OpenVINOEmbedder, SemanticIndex
from ..retrieval.timeline import get_timeline as timeline_query


SUBJECT_HEADINGS = (
    "高等代数", "高代", "数学分析", "数分", "高等数学", "英语", "申论", "行测", "政治",
    "计算机", "教资", "教育学",
)
TOPIC_ALIASES = {
    "高等代数": ("高等代数", "高代"),
    "数学分析": ("数学分析", "数分"),
}
ALGEBRA_SIGNALS = (
    "多项式", "矩阵", "行列式", "线性空间", "子空间", "线性变换", "映射", "特征值",
    "特征向量", "最小多项式", "相似", "合同", "二次型", "Jordan", "约尔当", "秩", "维数",
    "核", "像空间", "互素", "整除", "不变因子",
)


def _topic_terms(topic: str) -> tuple[str, ...]:
    return TOPIC_ALIASES.get(topic, (topic,))


def _topic_chunk_score(content: str, topic: str) -> int:
    terms = _topic_terms(topic)
    topic_hits = sum(content.count(term) for term in terms)
    if not topic_hits:
        return 0
    score = topic_hits * 30
    score += sum(word in content for word in ("证明", "题目", "不会", "理解", "掌握", "复习", "继续", "总结", "基础")) * 4
    if topic in {"高等代数", "高代"}:
        score += sum(word in content for word in ALGEBRA_SIGNALS) * 5
    return score


def _focused_excerpt(content: str, topic: str, width: int = 360) -> str:
    terms = _topic_terms(topic)
    positions = [(match.start(), term) for term in terms for match in re.finditer(re.escape(term), content, re.I)]
    if not positions:
        return content[:width].replace("\n", " ").strip()
    candidates: list[tuple[int, str]] = []
    for position, matched_term in positions:
        end = min(len(content), position + width)
        for heading in SUBJECT_HEADINGS:
            if heading in terms:
                continue
            next_heading = content.find(heading, position + len(matched_term) + 8, end)
            if next_heading >= 0:
                end = min(end, next_heading)
        excerpt = content[max(0, position - 25):end].replace("\n", " ").strip()
        signal = sum(word in excerpt for word in ("证明", "题目", "不会", "理解", "掌握", "复习", "继续", "总结", "卡", "基础"))
        score = signal * 10 + sum(excerpt.count(term) for term in terms) * 4 + min(position, 200) // 50
        if topic in {"高等代数", "高代"}:
            score += sum(word in excerpt for word in ALGEBRA_SIGNALS) * 8
        candidates.append((score, excerpt))
    return max(candidates, key=lambda item: item[0])[1]


class MemoryTools:
    def __init__(self, config: AppConfig, conn: sqlite3.Connection) -> None:
        self.config = config
        self.conn = conn
        embedding_path = config.model_dir / config.models.embedding_id.split("/")[-1]
        self.semantic = SemanticIndex(
            conn, OpenVINOEmbedder(embedding_path, config.models.embedding_device)
        ) if config.models.semantic_enabled else None

    def search_memory(self, query: str, start_date: str | None = None, end_date: str | None = None,
                      file_type: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
        return [item.as_dict() for item in hybrid_search(
            self.conn, query, start_date, end_date, file_type, limit, self.semantic,
        )]

    def search_semantic(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        if self.semantic is None or not self.semantic.available:
            return []
        return [item.as_dict() for item in self.semantic.search(query, limit)]

    def search_topic(self, topic: str, limit: int = 12) -> list[dict[str, Any]]:
        return self.trace_topic(topic, limit)

    def trace_topic(self, topic: str, limit: int = 12) -> list[dict[str, Any]]:
        """Return one strong, chronologically sampled memory per document for a topic."""
        candidates = self.search_memory(topic, limit=max(60, limit * 6))
        documents: dict[int, dict[str, Any]] = {}
        for item in candidates:
            previous = documents.get(item["document_id"])
            if previous is None or item["score"] > previous["score"]:
                documents[item["document_id"]] = dict(item)
        memories: list[dict[str, Any]] = []
        for document_id, item in documents.items():
            chunks = self.conn.execute(
                "SELECT id, content, source_kind FROM chunks WHERE document_id=?",
                (document_id,),
            ).fetchall()
            if not chunks:
                continue
            best_chunk = max(chunks, key=lambda row: _topic_chunk_score(str(row["content"]), topic))
            if _topic_chunk_score(str(best_chunk["content"]), topic) == 0:
                item["snippet"] = f"文件标题记录：{item['title']}"
                item["source_kind"] = "metadata"
                memories.append(item)
                continue
            item["chunk_id"] = int(best_chunk["id"])
            item["source_kind"] = str(best_chunk["source_kind"])
            item["snippet"] = _focused_excerpt(str(best_chunk["content"]), topic)
            memories.append(item)
        personal = [item for item in memories if item["file_type"] in {"docx", "md", "txt"}]
        if len(personal) >= 3:
            memories = personal
        memories.sort(key=lambda item: (item.get("event_date") or "9999-99-99", item["filename"]))
        if len(memories) <= limit:
            return memories
        indexes = sorted({round(index * (len(memories) - 1) / (limit - 1)) for index in range(limit)})
        return [memories[index] for index in indexes]

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

    def get_episode(self, event_date: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM episodes WHERE event_date=?", (event_date,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["subjects"] = json.loads(result.pop("subjects_json"))
        result["source_document_ids"] = json.loads(result.pop("source_document_ids_json"))
        result["sources"] = [
            dict(item) for item in self.conn.execute(
                f"SELECT id, filename, path, event_date FROM documents WHERE id IN ({','.join('?' for _ in result['source_document_ids'])})",
                result["source_document_ids"],
            ).fetchall()
        ] if result["source_document_ids"] else []
        return result

    def get_concept_state(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        pattern = f"%{query.strip()}%"
        rows = self.conn.execute(
            """
            SELECT c.id, c.name, c.subject, c.first_seen, c.last_seen, c.exposure_count,
                   s.state, s.current_summary, s.remaining_problem, s.confidence, s.updated_at
            FROM concepts c JOIN concept_states s ON s.concept_id=c.id
            WHERE c.name LIKE ? OR c.subject LIKE ?
            ORDER BY CASE WHEN c.name=? OR c.subject=? THEN 0 ELSE 1 END,
                     c.exposure_count DESC, c.last_seen DESC
            LIMIT ?
            """,
            (pattern, pattern, query, query, limit),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["evidence"] = [
                dict(evidence) for evidence in self.conn.execute(
                    """
                    SELECT ce.chunk_id, ce.event_date, c.content AS snippet,
                           d.id AS document_id, d.filename, d.path, d.file_type,
                           d.date_source, c.source_kind
                    FROM concept_evidence ce
                    JOIN chunks c ON c.id=ce.chunk_id
                    JOIN documents d ON d.id=c.document_id
                    WHERE ce.concept_id=?
                    ORDER BY ce.event_date DESC, ce.chunk_id DESC LIMIT 4
                    """,
                    (row["id"],),
                ).fetchall()
            ]
            results.append(item)
        return results

    def get_topic_profile(self, topic: str) -> dict[str, Any]:
        states = self.get_concept_state(topic, 20)
        episodes = [
            self.get_episode(row["event_date"]) for row in self.conn.execute(
                """
                SELECT event_date FROM episodes
                WHERE subjects_json LIKE ? OR summary LIKE ?
                ORDER BY event_date LIMIT 40
                """,
                (f"%{topic}%", f"%{topic}%"),
            ).fetchall()
        ]
        return {
            "topic": topic,
            "concept_states": states,
            "episodes": [episode for episode in episodes if episode],
        }

    @staticmethod
    def analysis_key(question: str, evidence_ids: list[int], model_id: str) -> str:
        payload = json.dumps(
            {"question": question.strip(), "evidence_ids": sorted(evidence_ids), "model_id": model_id},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_cached_analysis(self, question: str, evidence_ids: list[int], model_id: str) -> str | None:
        key = self.analysis_key(question, evidence_ids, model_id)
        row = self.conn.execute("SELECT answer FROM analysis_cache WHERE request_hash=?", (key,)).fetchone()
        return str(row["answer"]) if row else None

    def cache_analysis(
        self,
        question: str,
        intent: str,
        subject: str | None,
        answer: str,
        evidence_ids: list[int],
        model_id: str,
    ) -> None:
        key = self.analysis_key(question, evidence_ids, model_id)
        self.conn.execute(
            """
            INSERT INTO analysis_cache(request_hash, question, intent, subject, answer,
                                       evidence_ids_json, model_id, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_hash) DO UPDATE SET answer=excluded.answer, created_at=excluded.created_at
            """,
            (
                key, question, intent, subject, answer, json.dumps(evidence_ids),
                model_id, datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

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

        model_path = self.config.reasoner_model_path
        engine = OpenVINOVisionEngine(model_path, self.config.models.reasoner_device)
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
        memory = self.conn.execute(
            "SELECT (SELECT COUNT(*) FROM episodes) AS episodes, "
            "(SELECT COUNT(*) FROM concepts) AS concepts, "
            "(SELECT COUNT(*) FROM concept_states WHERE state='blocked') AS blocked"
        ).fetchone()
        semantic = self.conn.execute(
            "SELECT model_id, COUNT(*) AS indexed FROM embeddings GROUP BY model_id ORDER BY indexed DESC"
        ).fetchall()
        return {
            "documents": dict(counts),
            "chunks": chunks,
            "images": dict(ocr),
            "latest_run": dict(latest) if latest else None,
            "memory": dict(memory),
            "semantic_indexes": [dict(row) for row in semantic],
        }
