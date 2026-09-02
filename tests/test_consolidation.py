from second_brain.db import open_db
from second_brain.memory.consolidation import consolidate_memory, detect_subjects


def _document(conn, name, date, text):
    cursor = conn.execute(
        """
        INSERT INTO documents(path, sha256, filename, file_type, title, event_date, status)
        VALUES(?, ?, ?, 'docx', ?, ?, 'ready')
        """,
        (f"D:/readonly/{name}", name, name, name, date),
    )
    document_id = cursor.lastrowid
    chunk = conn.execute(
        """
        INSERT INTO chunks(document_id, block_index, chunk_index, source_kind, content, content_hash)
        VALUES(?, 0, 0, 'paragraph', ?, ?)
        """,
        (document_id, text, name),
    )
    return document_id, chunk.lastrowid


def test_detect_subjects_uses_specific_learning_terms():
    assert detect_subjects("今天复习线性空间和特征值") == ["高等代数"]
    assert "申论" in detect_subjects("申论归纳概括与公文写作")


def test_consolidation_builds_episode_concept_state_and_evidence(tmp_path):
    conn = open_db(tmp_path / "brain.db")
    _document(conn, "8.20.docx", "2026-08-20", "高等代数线性空间这里卡住了，没思路。")
    _document(conn, "8.21.docx", "2026-08-21", "复习线性空间后理解了基与维数的关系。")
    conn.commit()

    stats = consolidate_memory(conn)

    assert stats.documents_seen == 2
    assert stats.episodes_written == 2
    concept = conn.execute(
        """
        SELECT c.name, c.subject, c.first_seen, c.last_seen, c.exposure_count,
               s.state, s.current_summary
        FROM concepts c JOIN concept_states s ON s.concept_id=c.id
        WHERE c.name='线性空间'
        """
    ).fetchone()
    assert tuple(concept[:5]) == ("线性空间", "高等代数", "2026-08-20", "2026-08-21", 2)
    assert concept["state"] == "reinforcing"
    assert "理解" in concept["current_summary"]
    assert conn.execute("SELECT COUNT(*) FROM concept_evidence").fetchone()[0] == 2
    conn.close()


def test_consolidation_is_rebuildable_without_changing_source_memory(tmp_path):
    conn = open_db(tmp_path / "brain.db")
    _document(conn, "note.docx", "2026-08-30", "数学分析极限复习总结")
    conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    consolidate_memory(conn)
    consolidate_memory(conn)

    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == before
    assert conn.execute("SELECT COUNT(*) FROM concepts WHERE name='极限'").fetchone()[0] == 1
    conn.close()


def test_concepts_ignore_repeated_title_and_ambiguous_everyday_words(tmp_path):
    conn = open_db(tmp_path / "brain.db")
    document_id, _chunk_id = _document(conn, "高等代数.docx", "2026-08-30", "今天讨论商业合作，要写好合同。")
    conn.execute("UPDATE documents SET title='高等代数 矩阵' WHERE id=?", (document_id,))
    conn.commit()
    consolidate_memory(conn)
    assert conn.execute("SELECT COUNT(*) FROM concepts WHERE name='矩阵'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM concepts WHERE name='合同'").fetchone()[0] == 0

    conn.execute(
        "UPDATE chunks SET content='研究二次型对应矩阵的合同变换' WHERE document_id=?",
        (document_id,),
    )
    conn.commit()
    consolidate_memory(conn)
    assert conn.execute("SELECT COUNT(*) FROM concepts WHERE name='矩阵'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM concepts WHERE name='合同'").fetchone()[0] == 1
    conn.close()
