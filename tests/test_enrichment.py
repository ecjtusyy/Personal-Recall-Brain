from pathlib import Path

from PIL import Image, ImageDraw

from second_brain.config import AppConfig, IngestionConfig, ModelConfig
from second_brain.db import open_db
from second_brain.ingestion.enrichment import ImageEnricher
from second_brain.models import OCRResult


class FakeOCR:
    def __init__(self, text: str = "申论公文写作") -> None:
        self.text = text
        self.calls = 0

    def ocr(self, _path: Path) -> OCRResult:
        self.calls += 1
        return OCRResult(self.text, 0.93)


def make_runtime(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = AppConfig(
        config_path=tmp_path / "config.toml",
        study_year=2026,
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        device="CPU",
        source_roots=(source,),
        ingestion=IngestionConfig(),
        models=ModelConfig(agent_enabled=False),
    )
    config.ensure_runtime_dirs()
    conn = open_db(config.db_path)
    image_path = config.assets_dir / "page.png"
    image = Image.new("RGB", (300, 120), "white")
    ImageDraw.Draw(image).text((20, 40), "exam notes", fill="black")
    image.save(image_path)
    conn.execute(
        "INSERT INTO documents(path, sha256, filename, file_type, title, status) "
        "VALUES(?, 'doc', '笔记.docx', 'docx', '笔记', 'ready')",
        (str(source / "笔记.docx"),),
    )
    document_id = conn.execute("SELECT id FROM documents").fetchone()[0]
    conn.execute(
        "INSERT INTO assets(document_id, original_name, stored_path, sha256, ocr_status) "
        "VALUES(?, 'page.png', ?, 'image', 'pending')",
        (document_id, str(image_path)),
    )
    conn.commit()
    return config, conn


def test_image_enrichment_is_resumable_and_searchable(tmp_path):
    config, conn = make_runtime(tmp_path)
    fake = FakeOCR()
    first = ImageEnricher(config, conn, ocr_engine=fake).enrich(limit=10)
    second = ImageEnricher(config, conn, ocr_engine=fake).enrich(limit=10)
    assert first.completed == 1
    assert first.with_text == 1
    assert second.selected == 0
    assert fake.calls == 1
    asset = conn.execute("SELECT ocr_status, ocr_text FROM assets").fetchone()
    assert asset["ocr_status"] == "done"
    assert asset["ocr_text"] == "申论公文写作"
    chunk = conn.execute("SELECT source_kind, content FROM chunks").fetchone()
    assert chunk["source_kind"] == "ocr"
    assert chunk["content"] == "申论公文写作"


def test_tiny_images_are_skipped_without_loading_ocr(tmp_path):
    config, conn = make_runtime(tmp_path)
    image_path = Path(conn.execute("SELECT stored_path FROM assets").fetchone()[0])
    Image.new("RGB", (10, 10), "white").save(image_path)
    fake = FakeOCR()
    stats = ImageEnricher(config, conn, ocr_engine=fake).enrich(limit=10)
    assert stats.skipped_blank == 1
    assert fake.calls == 0
    assert conn.execute("SELECT ocr_status FROM assets").fetchone()[0] == "done"
