from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ..models import ParsedDocument, TextBlock
from .docx_parser import parse_docx


def parse_text_source(path: Path, include_docx_assets: bool = True) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return parse_docx(path, include_docx_assets)
    if suffix == ".pdf":
        reader = PdfReader(path)
        blocks = [
            TextBlock(index, text.strip(), "pdf")
            for index, page in enumerate(reader.pages)
            if (text := (page.extract_text() or "").strip())
        ]
        title = str(reader.metadata.title) if reader.metadata and reader.metadata.title else path.stem
        return ParsedDocument(title=title, text_blocks=blocks)
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = [TextBlock(i, part.strip()) for i, part in enumerate(text.split("\n\n")) if part.strip()]
        return ParsedDocument(title=path.stem, text_blocks=blocks)
    return ParsedDocument(title=path.stem)

