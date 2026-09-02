from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from ..db import transaction


SUBJECT_CONCEPTS: dict[str, tuple[str, ...]] = {
    "高等代数": (
        "线性空间", "向量空间", "线性映射", "矩阵", "行列式", "特征值", "特征向量",
        "相似", "合同", "二次型", "Jordan", "若尔当", "多项式", "内积空间",
    ),
    "数学分析": (
        "极限", "连续", "一致连续", "导数", "微分", "积分", "级数", "函数列",
        "一致收敛", "偏导数", "重积分", "曲线积分", "曲面积分", "微分方程",
    ),
    "英语": (
        "英语", "单词", "词汇", "长难句", "阅读理解", "完形填空", "翻译", "作文",
        "语法", "句式", "逻辑", "词组",
    ),
    "申论": (
        "申论", "归纳概括", "综合分析", "提出对策", "贯彻执行", "公文写作", "大作文",
        "材料分析", "中心论点", "分论点",
    ),
    "行测": (
        "行测", "言语理解", "判断推理", "资料分析", "数量关系", "常识判断",
        "图形推理", "定义判断", "类比推理",
    ),
    "政治": (
        "政治", "马克思主义", "毛中特", "史纲", "思修", "社会主义", "辩证法",
        "唯物主义", "时政",
    ),
    "计算机": (
        "计算机", "数据结构", "算法", "操作系统", "计算机网络", "数据库",
        "程序设计", "Python", "C++",
    ),
}

_BLOCKED = ("不会", "不懂", "没懂", "卡住", "卡在", "没思路", "困难", "薄弱", "错误", "做错", "忘了")
_PROGRESS = ("理解", "明白", "掌握", "串起来", "会做", "解决", "总结", "复习", "回顾")
_MASTERED = ("完全掌握", "熟练掌握", "已经掌握", "可以独立", "融会贯通")
_AMBIGUOUS_CONTEXT: dict[str, tuple[str, ...]] = {
    "合同": ("矩阵", "二次型", "正定", "标准形", "线性代数", "高等代数"),
    "相似": ("矩阵", "线性变换", "特征值", "对角化", "高等代数"),
}


@dataclass(frozen=True)
class ConsolidationStats:
    documents_seen: int
    episodes_written: int
    concepts_written: int
    evidence_links: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def detect_subjects(text: str) -> list[str]:
    scores: list[tuple[int, str]] = []
    folded = text.casefold()
    for subject, concepts in SUBJECT_CONCEPTS.items():
        score = int(subject.casefold() in folded) * 3
        score += sum(1 for concept in concepts if concept.casefold() in folded)
        if score:
            scores.append((score, subject))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [subject for _, subject in scores]


def detect_concepts(text: str) -> list[tuple[str, str]]:
    folded = text.casefold()
    found: list[tuple[str, str]] = []
    for subject, concepts in SUBJECT_CONCEPTS.items():
        for concept in concepts:
            if concept.casefold() not in folded:
                continue
            context = _AMBIGUOUS_CONTEXT.get(concept)
            if context and not any(word.casefold() in folded for word in context):
                continue
            if concept.casefold() in folded:
                found.append((subject, concept))
    return found


def _activity(text: str) -> str:
    for name, words in (
        ("review", ("复习", "回顾", "背诵", "错题")),
        ("practice", ("练习", "刷题", "做题", "测试")),
        ("planning", ("计划", "安排", "待办", "明天")),
        ("learning", ("学习", "理解", "课程", "笔记")),
    ):
        if any(word in text for word in words):
            return name
    return "study"


def _state(evidence: list[str], exposure_count: int) -> tuple[str, str, str, float]:
    latest = evidence[-3:]
    joined = "\n".join(latest)
    blocked_lines = [_compact(line, 160) for line in latest if any(word in line for word in _BLOCKED)]
    blocked = sum(joined.count(word) for word in _BLOCKED)
    progress = sum(joined.count(word) for word in _PROGRESS)
    mastered = any(word in joined for word in _MASTERED)
    if blocked > progress:
        state = "blocked"
    elif mastered and exposure_count >= 2:
        state = "mastered"
    elif exposure_count >= 3 or progress:
        state = "reinforcing"
    else:
        state = "learning"
    confidence = min(0.96, 0.48 + 0.12 * math.log2(exposure_count + 1))
    summary = _compact("；".join(latest), 480)
    remaining = "；".join(blocked_lines) if blocked_lines else ""
    return state, summary, remaining, round(confidence, 3)


def consolidate_memory(conn: sqlite3.Connection) -> ConsolidationStats:
    """Rebuild derived memories from indexed chunks without touching source files."""
    started = _now()
    run_id = conn.execute("INSERT INTO consolidation_runs(started_at) VALUES(?)", (started,)).lastrowid
    conn.commit()
    rows = conn.execute(
        """
        SELECT d.id AS document_id, d.event_date, d.filename, COALESCE(d.title, '') AS title,
               c.id AS chunk_id, c.content
        FROM documents d
        LEFT JOIN chunks c ON c.document_id=d.id
        WHERE d.status='ready'
        ORDER BY d.event_date, d.id, c.chunk_index
        """
    ).fetchall()
    documents = {int(row["document_id"]) for row in rows}
    episodes: dict[str, dict[str, object]] = {}
    concept_hits: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        event_date = row["event_date"] or "unknown"
        text = f"{row['title']} {row['filename']} {row['content'] or ''}"
        episode = episodes.setdefault(
            event_date,
            {"documents": set(), "titles": [], "subjects": set(), "samples": [], "activity_text": []},
        )
        episode["documents"].add(int(row["document_id"]))
        if row["title"] or row["filename"]:
            episode["titles"].append(row["title"] or row["filename"])
        episode["subjects"].update(detect_subjects(text))
        if row["content"] and len(episode["samples"]) < 6:
            episode["samples"].append(_compact(row["content"], 180))
        episode["activity_text"].append(text[:300])
        if row["chunk_id"] is None:
            continue
        # Titles help classify an episode, but concepts must be grounded in the
        # chunk itself. Otherwise a long Word title gets repeated into every
        # unrelated paragraph and pollutes concept state.
        for subject, concept in detect_concepts(str(row["content"] or "")):
            concept_hits[(subject, concept)].append(
                {"chunk_id": int(row["chunk_id"]), "date": event_date, "text": _compact(row["content"] or text, 260)}
            )

    now = _now()
    evidence_links = 0
    try:
        with transaction(conn):
            conn.execute("DELETE FROM episodes")
            conn.execute("DELETE FROM concepts")
            for event_date, data in episodes.items():
                subjects = sorted(data["subjects"])
                titles = list(dict.fromkeys(data["titles"]))[:6]
                samples = list(dict.fromkeys(data["samples"]))[:3]
                summary = _compact(
                    ("；".join(titles) + ("。要点：" + "；".join(samples) if samples else "")),
                    720,
                )
                document_ids = sorted(data["documents"])
                digest = hashlib.sha256(
                    json.dumps([event_date, subjects, summary, document_ids], ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                conn.execute(
                    """
                    INSERT INTO episodes(event_date, activity, subjects_json, summary,
                                         source_document_ids_json, content_hash, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_date, _activity(" ".join(data["activity_text"])),
                        json.dumps(subjects, ensure_ascii=False), summary,
                        json.dumps(document_ids), digest, now,
                    ),
                )
            for (subject, name), hits in sorted(concept_hits.items()):
                dates = sorted({str(hit["date"]) for hit in hits})
                cursor = conn.execute(
                    """
                    INSERT INTO concepts(name, subject, first_seen, last_seen, exposure_count, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (name, subject, dates[0], dates[-1], len(dates), now),
                )
                concept_id = int(cursor.lastrowid)
                unique_hits: dict[int, dict[str, object]] = {}
                for hit in hits:
                    unique_hits[int(hit["chunk_id"])] = hit
                for hit in unique_hits.values():
                    conn.execute(
                        "INSERT INTO concept_evidence(concept_id, chunk_id, event_date, relevance) VALUES(?, ?, ?, 1.0)",
                        (concept_id, hit["chunk_id"], hit["date"]),
                    )
                    evidence_links += 1
                ordered_text = [str(hit["text"]) for hit in sorted(unique_hits.values(), key=lambda hit: (hit["date"], hit["chunk_id"]))]
                state, summary, remaining, confidence = _state(ordered_text, len(dates))
                conn.execute(
                    """
                    INSERT INTO concept_states(concept_id, state, current_summary, remaining_problem, confidence, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (concept_id, state, summary, remaining, confidence, now),
                )
            conn.execute(
                """
                UPDATE consolidation_runs
                SET finished_at=?, documents_seen=?, episodes_written=?, concepts_written=?
                WHERE id=?
                """,
                (now, len(documents), len(episodes), len(concept_hits), run_id),
            )
    except Exception as exc:
        conn.execute(
            "UPDATE consolidation_runs SET finished_at=?, error=? WHERE id=?",
            (_now(), str(exc)[:2000], run_id),
        )
        conn.commit()
        raise
    return ConsolidationStats(len(documents), len(episodes), len(concept_hits), evidence_links)
