from pathlib import Path


def test_streamlit_does_not_cache_sqlite_connections():
    app = Path(__file__).parents[1] / "src" / "second_brain" / "ui" / "app.py"
    source = app.read_text(encoding="utf-8")
    assert "conn = open_db(config.db_path)" in source
    assert "return config, conn" not in source
    assert "agent_runtime" not in source
    assert "resident_model" in source
    assert "conn.close()" in source
    assert "on_click=open_document" not in source
