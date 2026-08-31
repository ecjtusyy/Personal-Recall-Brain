from second_brain.db import open_db


def test_schema_and_fts_are_created(tmp_path):
    conn = open_db(tmp_path / "brain.db")
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert {"documents", "chunks", "assets", "memory_cards", "ingestion_runs", "embeddings", "chunks_fts"} <= names
    conn.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "brain.db"
    open_db(path).close()
    conn = open_db(path)
    assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == "1"
    conn.close()

